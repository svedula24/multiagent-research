import json
import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict

from app.agents import manager
from app.database import db_cursor
from app.models.recommendation import DraftRecommendation
from app.models.worker_output import WorkerOutput
from app.services.synthesis import synthesize

logger = logging.getLogger(__name__)

MAX_REJECTIONS = 3

# Retry policy for nodes that call external services (LLM or DB writes).
# Agents have their own internal retry — this covers transient node-level failures.
_DB_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=1.0,
    backoff_factor=2.0,          # 1s → 2s → 4s
    retry_on=(Exception,),
)

_LLM_RETRY = RetryPolicy(
    max_attempts=3,
    initial_interval=2.0,
    backoff_factor=2.0,          # 2s → 4s → 8s — LLM rate limits need a longer gap
    retry_on=(Exception,),
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    run_id: int
    query: str
    competitors: list[str]
    web_output: Optional[dict]      # WorkerOutput serialised to dict for state
    review_output: Optional[dict]
    sales_output: Optional[dict]
    draft: Optional[dict]           # DraftRecommendation serialised to dict
    confidence: Optional[float]
    rejection_count: int
    feedback: Optional[str]
    status: str                     # running | pending_approval | completed | failed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_to_dict(o: WorkerOutput) -> dict:
    return o.model_dump(mode="json")


def _dict_to_output(d: dict) -> WorkerOutput:
    return WorkerOutput.model_validate(d)


def _draft_to_dict(d: DraftRecommendation) -> dict:
    return d.model_dump(mode="json")


def _update_run_status(run_id: int, status: str) -> None:
    with db_cursor() as cur:
        cur.execute(
            "UPDATE research_runs SET status = %s, updated_at = NOW() WHERE id = %s",
            (status, run_id),
        )


def _save_worker_outputs(run_id: int, outputs: list[WorkerOutput]) -> None:
    with db_cursor() as cur:
        for o in outputs:
            cur.execute(
                """
                INSERT INTO worker_outputs (run_id, source, status, findings, confidence, error)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    run_id,
                    o.source,
                    o.status,
                    json.dumps(o.findings),
                    o.confidence,
                    o.error,
                ),
            )


def _save_draft(draft: DraftRecommendation) -> None:
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO draft_recommendations
                (run_id, summary, findings_by_source, confidence, status, rejection_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                summary            = EXCLUDED.summary,
                findings_by_source = EXCLUDED.findings_by_source,
                confidence         = EXCLUDED.confidence,
                status             = EXCLUDED.status,
                rejection_count    = EXCLUDED.rejection_count,
                updated_at         = NOW()
            """,
            (
                draft.run_id,
                draft.summary,
                json.dumps(draft.findings_by_source),
                draft.confidence,
                draft.status,
                draft.rejection_count,
            ),
        )


def _save_final(run_id: int, draft: DraftRecommendation) -> None:
    content = json.dumps(
        {
            "summary": draft.summary,
            "findings_by_source": draft.findings_by_source,
        }
    )
    with db_cursor() as cur:
        cur.execute(
            """
            INSERT INTO final_recommendations (run_id, content, confidence)
            VALUES (%s, %s, %s)
            """,
            (run_id, content, draft.confidence),
        )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def dispatch_agents(state: ResearchState) -> dict[str, Any]:
    """Run all three worker agents in parallel via the manager."""
    run_id = state["run_id"]
    logger.info("dispatch_agents: run_id=%d", run_id)

    outputs = await manager.dispatch(state["query"], state["competitors"])
    web_out, review_out, sales_out = outputs

    _save_worker_outputs(run_id, [web_out, review_out, sales_out])

    return {
        "web_output": _output_to_dict(web_out),
        "review_output": _output_to_dict(review_out),
        "sales_output": _output_to_dict(sales_out),
        "status": "running",
    }


async def run_synthesis(state: ResearchState) -> dict[str, Any]:
    """Synthesise worker outputs into a DraftRecommendation."""
    run_id = state["run_id"]
    rejection_count = state.get("rejection_count", 0)
    feedback = state.get("feedback")

    outputs = [
        _dict_to_output(state["web_output"]),
        _dict_to_output(state["review_output"]),
        _dict_to_output(state["sales_output"]),
    ]

    logger.info("run_synthesis: run_id=%d rejection_count=%d", run_id, rejection_count)

    draft = await synthesize(
        run_id=run_id,
        query=state["query"],
        competitors=state["competitors"],
        outputs=outputs,
        feedback=feedback,
        rejection_count=rejection_count,
    )

    _save_draft(draft)
    _update_run_status(run_id, "pending_approval")

    return {
        "draft": _draft_to_dict(draft),
        "confidence": draft.confidence,
        "status": "pending_approval",
    }


def await_approval(state: ResearchState) -> Command:
    """
    Pause execution here and surface the draft to the PM.
    Resumes when the approve/reject endpoint sends a Command.

    Expected resume payload: {"action": "approve" | "reject", "feedback": str | None}
    """
    draft = state["draft"]
    logger.info("await_approval: run_id=%d — waiting for PM decision", state["run_id"])

    decision: dict = interrupt({"draft": draft, "run_id": state["run_id"]})

    action = decision.get("action")
    if action == "approve":
        return Command(goto="finalize", update={"status": "approved"})
    elif action == "reject":
        return Command(
            goto="re_synthesize",
            update={
                "feedback": decision.get("feedback"),
                "rejection_count": state.get("rejection_count", 0) + 1,
                "status": "running",
            },
        )
    else:
        logger.error("await_approval: unknown action %r for run_id=%d", action, state["run_id"])
        return Command(goto=END, update={"status": "failed"})


def re_synthesize(state: ResearchState) -> dict[str, Any]:
    """Gate: block re-synthesis if max rejections reached, else route back."""
    if state.get("rejection_count", 0) >= MAX_REJECTIONS:
        logger.warning(
            "run_id=%d reached max rejections (%d), marking failed",
            state["run_id"],
            MAX_REJECTIONS,
        )
        _update_run_status(state["run_id"], "failed")
        return {"status": "failed"}
    # Otherwise let the graph route back to run_synthesis
    return {}


def finalize(state: ResearchState) -> dict[str, Any]:
    """Write the approved draft to final_recommendations and mark run completed."""
    run_id = state["run_id"]
    draft = DraftRecommendation.model_validate(state["draft"])

    _save_final(run_id, draft)
    _update_run_status(run_id, "completed")

    logger.info("finalize: run_id=%d completed", run_id)
    return {"status": "completed"}


def route_after_resynthesize(state: ResearchState) -> str:
    if state.get("status") == "failed":
        return END
    return "run_synthesis"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("dispatch_agents", dispatch_agents, retry=_DB_RETRY)   # retry DB write after agents succeed
    g.add_node("run_synthesis", run_synthesis, retry=_LLM_RETRY)     # retry Gemini synthesis call
    g.add_node("await_approval", await_approval)                      # no external calls — interrupt only
    g.add_node("re_synthesize", re_synthesize)                        # pure gate logic — no external calls
    g.add_node("finalize", finalize, retry=_DB_RETRY)                 # retry final DB write

    g.add_edge(START, "dispatch_agents")
    g.add_edge("dispatch_agents", "run_synthesis")
    g.add_edge("run_synthesis", "await_approval")
    # await_approval uses Command to route dynamically to finalize or re_synthesize
    g.add_edge("finalize", END)
    g.add_conditional_edges("re_synthesize", route_after_resynthesize)

    return g


# Compiled graph — single instance shared across the app, mainly for human in the loop
checkpointer = MemorySaver()
graph = build_graph().compile(checkpointer=checkpointer)
