import asyncio
import json
import logging
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.models.worker_output import WorkerOutput
from app.services.confidence import sales_confidence
from app.tools.sql_tool import run_query

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_RETRY_DELAYS = [0, 2, 4]

_SALES_PROMPT = """\
You are a product analyst interpreting internal sales data.

## Research Context
{query}

## Sales Data
{sales_data}

Interpret the data and surface the key business signals:
1. Revenue trend over the last 3 months (month-over-month change)
2. Current churn rate vs 3-month average — is it improving or worsening?
3. Top performing feature by adoption
4. Most common reason for lost deals
5. Latest NPS score and what it indicates

Respond ONLY with valid JSON in this exact format:
{{
  "revenue_trend": "<interpretation of MoM revenue change>",
  "churn_analysis": "<current vs avg churn and direction>",
  "top_feature": "<feature name and why it's winning>",
  "lost_deal_reason": "<most common reason and implication>",
  "nps_insight": "<NPS score and what it means>",
  "key_signal": "<single most important insight for the PM>"
}}
"""

# Parameterised queries — no user input is interpolated into SQL
_RECENT_MONTHS_SQL = """
    SELECT period, revenue, churn_rate, new_customers,
           lost_deals_reason, top_feature, nps_score, segment
    FROM sales_reports
    ORDER BY created_at DESC
    LIMIT %s;
"""

_AVG_CHURN_SQL = """
    SELECT AVG(churn_rate) AS avg_churn
    FROM (
        SELECT churn_rate FROM sales_reports
        ORDER BY created_at DESC
        LIMIT 3
    ) recent;
"""


async def run(query: str) -> WorkerOutput:
    """
    Query sales_reports for recent metrics, then use Gemini to interpret trends.
    """
    for attempt, delay in enumerate(_RETRY_DELAYS, start=1):
        if delay:
            await asyncio.sleep(delay)
        try:
            output = await asyncio.wait_for(_fetch(query), timeout=_TIMEOUT)
            logger.info("sales_agent success on attempt %d", attempt)
            return output
        except asyncio.TimeoutError:
            logger.warning("sales_agent timed out on attempt %d", attempt)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="sales",
                    status="timeout",
                    findings={},
                    confidence=0.0,
                    error="Agent timed out after 30 seconds",
                    retrieved_at=datetime.now(timezone.utc),
                )
        except Exception as exc:
            logger.warning("sales_agent attempt %d failed: %s", attempt, type(exc).__name__)
            if attempt == len(_RETRY_DELAYS):
                return WorkerOutput(
                    source="sales",
                    status="failed",
                    findings={},
                    confidence=0.0,
                    error=str(type(exc).__name__),
                    retrieved_at=datetime.now(timezone.utc),
                )

    return WorkerOutput(
        source="sales", status="failed", findings={}, confidence=0.0,
        error="Unknown failure", retrieved_at=datetime.now(timezone.utc),
    )


async def _fetch(query: str) -> WorkerOutput:
    loop = asyncio.get_running_loop()

    # DB queries are sync — run in thread pool
    rows = await loop.run_in_executor(None, lambda: run_query(_RECENT_MONTHS_SQL, (3,)))
    avg_rows = await loop.run_in_executor(None, lambda: run_query(_AVG_CHURN_SQL, ()))

    if not rows:
        raise ValueError("No sales data found in database")

    confidence = sales_confidence(rows[0])
    avg_churn = float(avg_rows[0]["avg_churn"]) if avg_rows and avg_rows[0]["avg_churn"] else None

    sales_data = {
        "recent_months": rows,
        "avg_churn_last_3_months": avg_churn,
    }

    settings = get_settings()
    llm = ChatGoogleGenerativeAI(
        model=settings.worker_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )

    prompt = _SALES_PROMPT.format(
        query=query,
        sales_data=json.dumps(sales_data, indent=2, default=str),
    )

    # Use ainvoke directly — Gemini SDK is async-native, run_in_executor causes issues
    response = await llm.ainvoke(prompt)
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    parsed = json.loads(raw)

    return WorkerOutput(
        source="sales",
        status="success",
        findings={
            **parsed,
            "months_analysed": len(rows),
            "avg_churn_last_3_months": avg_churn,
        },
        confidence=confidence,
        retrieved_at=datetime.now(timezone.utc),
    )
