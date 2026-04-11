import logging
from typing import Any

from langchain_community.tools.tavily_search import TavilySearchResults

from app.config import get_settings

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Run a Tavily web search and return structured results.

    Returns a list of dicts with keys: title, url, content, score.
    Returns an empty list on any failure so callers can treat it as a soft error.
    """
    settings = get_settings()
    try:
        tool = TavilySearchResults(
            max_results=max_results,
            tavily_api_key=settings.tavily_api_key,
        )
        raw: list[dict] = tool.invoke(query)
        results = []
        for item in raw:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "score": item.get("score", 0.0),
                }
            )
        logger.info("Tavily search returned %d results for query length=%d", len(results), len(query))
        return results
    except Exception as exc:
        logger.error("Tavily search failed: %s", type(exc).__name__)
        raise
