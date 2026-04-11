import asyncio
import logging
from datetime import datetime, timezone

from app.models.worker_output import WorkerOutput
from app.services.confidence import web_confidence
from app.tools.tavily_tool import search

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_RETRY_DELAYS = [0, 2, 4]


async def run(competitors: list[str]) -> WorkerOutput:
    """
    Gather recent competitor activity via Tavily web search.
    Runs searches for all competitors and aggregates results.
    """
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            results = await asyncio.wait_for(_fetch(competitors), timeout=_TIMEOUT)
            confidence = web_confidence(results)
            logger.info("web_agent success on attempt %d, %d results", attempt, len(results))
            return WorkerOutput(
                source="web",
                status="success",
                findings={"results": results},
                confidence=confidence,
                retrieved_at=datetime.now(timezone.utc),
            )
        except asyncio.TimeoutError:
            logger.warning("web_agent timed out on attempt %d", attempt)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="web",
                    status="timeout",
                    findings={},
                    confidence=0.0,
                    error="Agent timed out after 30 seconds",
                    retrieved_at=datetime.now(timezone.utc),
                )
        except Exception as exc:
            logger.warning("web_agent attempt %d failed: %s", attempt, type(exc).__name__)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="web",
                    status="failed",
                    findings={},
                    confidence=0.0,
                    error=str(type(exc).__name__),
                    retrieved_at=datetime.now(timezone.utc),
                )

    # Unreachable but satisfies type checker
    return WorkerOutput(
        source="web", status="failed", findings={}, confidence=0.0,
        error="Unknown failure", retrieved_at=datetime.now(timezone.utc),
    )


async def _fetch(competitors: list[str]) -> list[dict]:
    """Build per-competitor queries and gather Tavily results."""
    all_results = []
    for competitor in competitors:
        query = f'"{competitor}" product announcement OR feature launch OR update'
        # search() is sync — run in thread pool to avoid blocking the event loop
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda q=query: search(q, max_results=5)
        )
        for r in results:
            r["competitor"] = competitor
        all_results.extend(results)
    return all_results
