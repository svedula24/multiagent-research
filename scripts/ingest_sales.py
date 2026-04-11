"""
One-time script to generate 12 synthetic monthly sales reports using Gemini
and insert them into sales_reports.

Idempotent: skips if 12+ rows already exist.

Usage:
    python scripts/ingest_sales.py
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.database import close_pool, db_cursor, init_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COUNT = 12

_PROMPT = """
Generate exactly 12 synthetic monthly sales reports for a B2B SaaS product management tool,
covering the last 12 months in order from oldest to most recent (e.g. 2025-05 through 2026-04).

Each report must have these exact fields:
- period: string in format "YYYY-MM" (e.g. "2025-05")
- revenue: decimal, realistic MRR between 80000 and 200000, with a general growth trend
- churn_rate: decimal percentage between 1.5 and 8.0 (e.g. 3.2 means 3.2%)
- new_customers: integer between 10 and 80
- lost_deals_reason: a short string naming the top reason (e.g. "pricing", "missing features", "competitor", "no budget")
- top_feature: the most adopted feature that month (e.g. "roadmap view", "integrations", "analytics dashboard")
- nps_score: integer between 20 and 72
- segment: primary customer segment that month — one of "enterprise", "mid-market", "smb"

Make the data internally consistent and tell a plausible business story with some variance.
Return ONLY a valid JSON array of 12 objects. No markdown, no explanation — raw JSON array only.
"""


def generate_reports(llm: ChatGoogleGenerativeAI) -> list[dict]:
    logger.info("Generating %d synthetic sales reports with Gemini...", TARGET_COUNT)
    response = llm.invoke(_PROMPT)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    reports = json.loads(raw)
    assert isinstance(reports, list), "Expected a JSON array"
    logger.info("Generated %d reports", len(reports))
    return reports


def insert_reports(reports: list[dict]) -> None:
    logger.info("Inserting %d sales reports...", len(reports))
    for report in reports:
        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO sales_reports
                    (period, revenue, churn_rate, new_customers,
                     lost_deals_reason, top_feature, nps_score, segment)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    report["period"],
                    float(report["revenue"]),
                    float(report["churn_rate"]),
                    int(report["new_customers"]),
                    report["lost_deals_reason"],
                    report["top_feature"],
                    int(report["nps_score"]),
                    report["segment"],
                ),
            )
        logger.info("  Inserted report for %s", report["period"])

    logger.info("All sales reports inserted successfully.")


def main() -> None:
    settings = get_settings()
    init_pool(settings.database_url)

    try:
        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sales_reports")
            count = cur.fetchone()[0]

        if count >= TARGET_COUNT:
            logger.info("sales_reports already has %d rows — skipping ingestion.", count)
            return

        llm = ChatGoogleGenerativeAI(
            model=settings.worker_model,
            google_api_key=settings.google_api_key,
            temperature=0.7,
        )

        reports = generate_reports(llm)
        insert_reports(reports)

        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM sales_reports")
            final_count = cur.fetchone()[0]
        logger.info("Done. sales_reports now has %d rows.", final_count)

    finally:
        close_pool()


if __name__ == "__main__":
    main()
