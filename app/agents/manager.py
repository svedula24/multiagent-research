import asyncio
import logging

from app.agents import review_agent, sales_agent, web_agent
from app.models.worker_output import WorkerOutput

logger = logging.getLogger(__name__)


async def dispatch(
    query: str,
    competitors: list[str],
) -> list[WorkerOutput]:
    """
    Dispatch all three worker agents in parallel and return their outputs.
    The LangGraph state machine calls this from the dispatch_agents node.
    """
    logger.info(
        "Dispatching agents — query_len=%d competitors=%d",
        len(query),
        len(competitors),
    )

    web_out, review_out, sales_out = await asyncio.gather(
        web_agent.run(competitors),
        review_agent.run(query),
        sales_agent.run(query),
    )

    succeeded = sum(1 for o in [web_out, review_out, sales_out] if o.status == "success")
    logger.info("Agents completed: %d/3 succeeded", succeeded)

    return [web_out, review_out, sales_out]
