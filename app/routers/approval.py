import logging

from fastapi import APIRouter, HTTPException, Request
from langgraph.types import Command
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.database import db_cursor
from app.graph.research_graph import graph
from app.models.recommendation import RejectionRequest

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter(prefix="/research", tags=["approval"])


def _get_run_or_404(run_id: int) -> dict:
    with db_cursor() as cur:
        cur.execute(
            "SELECT id, status, rejection_count FROM research_runs WHERE id = %s",
            (run_id,),
        )
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Research run {run_id} not found")
    return {"run_id": row[0], "status": row[1], "rejection_count": row[2]}


@router.post("/{run_id}/approve", status_code=200)
@limiter.limit("20/minute")
async def approve_research_run(request: Request, run_id: int) -> dict:
    """
    Approve the draft recommendation.
    Resumes the paused graph → finalise node writes to final_recommendations.
    """
    run = _get_run_or_404(run_id)

    if run["status"] != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not pending approval (current status: {run['status']})",
        )

    config = {"configurable": {"thread_id": str(run_id)}}
    try:
        await graph.ainvoke(Command(resume={"action": "approve"}), config=config)
        logger.info("run_id=%d approved and finalised", run_id)
    except Exception as exc:
        logger.error("Approval failed for run_id=%d: %s", run_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Failed to finalise recommendation")

    return {"run_id": run_id, "status": "completed"}


@router.post("/{run_id}/reject", status_code=200)
@limiter.limit("20/minute")
async def reject_research_run(
    request: Request, run_id: int, body: RejectionRequest
) -> dict:
    """
    Reject the draft recommendation with optional PM feedback.
    Triggers re-synthesis using the original agent outputs (agents do NOT re-run).
    Maximum 3 rejection cycles before the run is marked failed.
    """
    run = _get_run_or_404(run_id)

    if run["status"] != "pending_approval":
        raise HTTPException(
            status_code=409,
            detail=f"Run {run_id} is not pending approval (current status: {run['status']})",
        )

    if run["rejection_count"] >= 3:
        raise HTTPException(
            status_code=422,
            detail=f"Run {run_id} has reached the maximum of 3 rejection cycles",
        )

    config = {"configurable": {"thread_id": str(run_id)}}
    try:
        await graph.ainvoke(
            Command(resume={"action": "reject", "feedback": body.feedback}),
            config=config,
        )
        logger.info(
            "run_id=%d rejected (cycle %d/3), re-synthesis triggered",
            run_id,
            run["rejection_count"] + 1,
        )
    except Exception as exc:
        logger.error("Rejection failed for run_id=%d: %s", run_id, type(exc).__name__)
        raise HTTPException(status_code=500, detail="Failed to trigger re-synthesis")

    return {
        "run_id": run_id,
        "status": "running",
        "rejection_count": run["rejection_count"] + 1,
    }
