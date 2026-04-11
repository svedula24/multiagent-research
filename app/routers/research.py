import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import db_cursor
from app.graph.research_graph import graph
from app.models.research_run import ResearchRunRequest, ResearchRunResponse

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/research", tags=["research"])


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

async def _run_graph(run_id: int, query: str, competitors: list[str]) -> None:
    config = {"configurable": {"thread_id": str(run_id)}}
    initial_state = {
        "run_id": run_id,
        "query": query,
        "competitors": competitors,
        "web_output": None,
        "review_output": None,
        "sales_output": None,
        "draft": None,
        "confidence": None,
        "rejection_count": 0,
        "feedback": None,
        "status": "running",
    }
    try:
        await graph.ainvoke(initial_state, config=config)
        logger.info("Graph run completed for run_id=%d", run_id)
    except Exception as exc:
        logger.error("Graph run failed for run_id=%d: %s", run_id, type(exc).__name__)
        with db_cursor() as cur:
            cur.execute(
                "UPDATE research_runs SET status = 'failed', updated_at = NOW() WHERE id = %s",
                (run_id,),
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/run", response_model=ResearchRunResponse, status_code=202)
@limiter.limit("20/minute")
async def create_research_run(
    request: Request,
    body: ResearchRunRequest,
    background_tasks: BackgroundTasks,
) -> ResearchRunResponse:
    """
    Submit a research request. Returns a run_id immediately.
    Agents run asynchronously — poll GET /research/{run_id} for results.
    """
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_runs (query, competitors, status)
            VALUES (%s, %s, 'running')
            RETURNING id
            """,
            (body.query, body.competitors),
        )
        run_id = cur.fetchone()[0]

    logger.info("Created research run run_id=%d", run_id)
    background_tasks.add_task(_run_graph, run_id, body.query, body.competitors)

    return ResearchRunResponse(run_id=run_id, status="running")


@router.get("/{run_id}")
@limiter.limit("60/minute")
async def get_research_run(request: Request, run_id: int) -> dict:
    """
    Poll for run status and draft recommendation.
    Streamlit calls this every few seconds until status is 'pending_approval' or terminal.
    """
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, query, competitors, status, rejection_count, created_at, updated_at "
            "FROM research_runs WHERE id = %s",
            (run_id,),
        )
        run_row = cur.fetchone()

    if not run_row:
        raise HTTPException(status_code=404, detail=f"Research run {run_id} not found")

    run = {
        "run_id": run_row[0],
        "query": run_row[1],
        "competitors": run_row[2],
        "status": run_row[3],
        "rejection_count": run_row[4],
        "created_at": run_row[5].isoformat(),
        "updated_at": run_row[6].isoformat(),
    }

    # Attach draft recommendation if available
    with db_cursor() as cur:
        cur.execute(
            "SELECT summary, findings_by_source, confidence, status, rejection_count "
            "FROM draft_recommendations WHERE run_id = %s",
            (run_id,),
        )
        draft_row = cur.fetchone()

    if draft_row:
        findings = draft_row[1]
        if isinstance(findings, str):
            findings = json.loads(findings)
        run["draft"] = {
            "summary": draft_row[0],
            "findings_by_source": findings,
            "confidence": float(draft_row[2]) if draft_row[2] else None,
            "confidence_pct": f"{float(draft_row[2]) * 100:.1f}%" if draft_row[2] else None,
            "status": draft_row[3],
            "rejection_count": draft_row[4],
        }

    # Attach per-agent statuses
    with db_cursor() as cur:
        cur.execute(
            "SELECT source, status, confidence, error FROM worker_outputs WHERE run_id = %s",
            (run_id,),
        )
        agent_rows = cur.fetchall()

    run["agents"] = [
        {
            "source": r[0],
            "status": r[1],
            "confidence": float(r[2]) if r[2] else None,
            "error": r[3],
        }
        for r in agent_rows
    ]

    return run
