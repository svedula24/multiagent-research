import asyncio
import logging
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.models.worker_output import WorkerOutput
from app.services.confidence import review_confidence
from app.services.embedding import embed_text
from app.tools.pgvector_tool import similarity_search

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_RETRY_DELAYS = [0, 2, 4]

_REVIEW_PROMPT = """\
You are analysing customer reviews for a product team.

## Research Context
{query}

## Retrieved Customer Reviews
{reviews}

Extract the following from the reviews above:
1. Top 3 pain points customers are experiencing
2. Top 3 feature requests customers are asking for
3. Overall sentiment: positive, neutral, or negative

Respond ONLY with valid JSON in this exact format:
{{
  "pain_points": ["<point 1>", "<point 2>", "<point 3>"],
  "feature_requests": ["<request 1>", "<request 2>", "<request 3>"],
  "overall_sentiment": "positive" | "neutral" | "negative"
}}
"""


async def run(query: str) -> WorkerOutput:
    """
    Embed the query, retrieve semantically similar reviews via pgvector,
    then use Gemini to extract pain points, feature requests, and sentiment.
    """
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            output = await asyncio.wait_for(_fetch(query), timeout=_TIMEOUT)
            logger.info("review_agent success on attempt %d", attempt)
            return output
        except asyncio.TimeoutError:
            logger.warning("review_agent timed out on attempt %d", attempt)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="reviews",
                    status="timeout",
                    findings={},
                    confidence=0.0,
                    error="Agent timed out after 30 seconds",
                    retrieved_at=datetime.now(timezone.utc),
                )
        except Exception as exc:
            logger.warning("review_agent attempt %d failed: %s", attempt, type(exc).__name__)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="reviews",
                    status="failed",
                    findings={},
                    confidence=0.0,
                    error=str(type(exc).__name__),
                    retrieved_at=datetime.now(timezone.utc),
                )

    return WorkerOutput(
        source="reviews", status="failed", findings={}, confidence=0.0,
        error="Unknown failure", retrieved_at=datetime.now(timezone.utc),
    )


async def _fetch(query: str) -> WorkerOutput:
    import json

    loop = asyncio.get_running_loop()

    # Embed query and DB search are sync — run in thread pool
    embedding = await loop.run_in_executor(None, embed_text, query)
    rows = await loop.run_in_executor(None, lambda: similarity_search(embedding, top_k=10))

    similarity_scores = [float(r["similarity"]) for r in rows]
    confidence = review_confidence(similarity_scores)

    review_text = "\n\n".join(
        f"[Review {i+1}] Rating: {r['rating']}/5 | Category: {r['category']}\n{r['content']}"
        for i, r in enumerate(rows)
    )

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.worker_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )

    # Use ainvoke directly — Gemini SDK is async-native, run_in_executor causes issues
    prompt = _REVIEW_PROMPT.format(query=query, reviews=review_text)
    response = await llm.ainvoke(prompt)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)

    return WorkerOutput(
        source="reviews",
        status="success",
        findings={
            "pain_points": parsed.get("pain_points", []),
            "feature_requests": parsed.get("feature_requests", []),
            "overall_sentiment": parsed.get("overall_sentiment", "neutral"),
            "reviews_retrieved": len(rows),
        },
        confidence=confidence,
        retrieved_at=datetime.now(timezone.utc),
    )
