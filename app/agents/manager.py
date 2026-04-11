import asyncio
import logging

from app.agents import review_agent, sales_agent, web_agent
from app.models.recommendation import DraftRecommendation
from app.models.worker_output import WorkerOutput
from app.services.synthesis import synthesize

logger = logging.getLogger(__name__)


async def run(
    run_id: int,
    query: str,
    competitors: list[str],
    feedback: str | None = None,
    rejection_count: int = 0,
    existing_outputs: list[WorkerOutput] | None = None,
) -> tuple[DraftRecommendation, list[WorkerOutput]]:
    """
    Orchestrate the three worker agents in parallel, then synthesise their outputs.

    On re-synthesis (after PM rejection), pass existing_outputs to skip re-running agents.

    Returns:
        (DraftRecommendation, list[WorkerOutput])
    """
    if existing_outputs is not None:
        # Re-synthesis path: reuse agent outputs, only re-run synthesis with feedback
        logger.info(
            "Re-synthesising run_id=%d (rejection %d/3) with feedback=%s",
            run_id,
            rejection_count,
            bool(feedback),
        )
        outputs = existing_outputs
    else:
        # Fresh run: dispatch all three agents in parallel
        logger.info("Dispatching agents for run_id=%d query_len=%d", run_id, len(query))
        web_task = web_agent.run(competitors)
        review_task = review_agent.run(query)
        sales_task = sales_agent.run(query)

        outputs: list[WorkerOutput] = list(
            await asyncio.gather(web_task, review_task, sales_task)
        )

        succeeded = sum(1 for o in outputs if o.status == "success")
        logger.info(
            "Agents completed for run_id=%d: %d/3 succeeded", run_id, succeeded
        )

    draft = synthesize(
        run_id=run_id,
        query=query,
        competitors=competitors,
        outputs=outputs,
        feedback=feedback,
        rejection_count=rejection_count,
    )

    return draft, outputs
