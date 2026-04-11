"""
One-time script to generate 50 synthetic customer reviews using Gemini,
embed them with OpenAI text-embedding-3-small, and insert into customer_reviews.

Idempotent: skips if 50+ rows already exist.

Usage:
    python scripts/ingest_reviews.py
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.database import close_pool, db_cursor, init_pool
from app.services.embedding import embed_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COUNT = 50

_PROMPT = """
Generate exactly 50 synthetic customer reviews for a B2B SaaS product management tool.
Reviews should be realistic, varied, and cover a range of sentiments and topics.

Each review must have:
- content: 1-3 sentences of realistic customer feedback
- rating: integer 1-5
- category: one of "onboarding", "performance", "features", "support", "pricing"

Return ONLY a valid JSON array of 50 objects with keys: content, rating, category.
No markdown, no explanation — raw JSON array only.
"""


def generate_reviews(llm: ChatGoogleGenerativeAI) -> list[dict]:
    logger.info("Generating %d synthetic reviews with Gemini...", TARGET_COUNT)
    response = llm.invoke(_PROMPT)
    raw = response.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    reviews = json.loads(raw)
    assert isinstance(reviews, list), "Expected a JSON array"
    logger.info("Generated %d reviews", len(reviews))
    return reviews


def embed_and_insert(reviews: list[dict]) -> None:
    logger.info("Embedding and inserting %d reviews...", len(reviews))
    for i, review in enumerate(reviews, start=1):
        content = review["content"]
        rating = int(review["rating"])
        category = review["category"]

        vector = embed_text(content)
        vector_literal = "[" + ",".join(str(x) for x in vector) + "]"

        with db_cursor() as cur:
            cur.execute(
                """
                INSERT INTO customer_reviews (content, rating, category, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                (content, rating, category, vector_literal),
            )

        if i % 10 == 0:
            logger.info("  Inserted %d / %d reviews", i, len(reviews))

    logger.info("All reviews inserted successfully.")


def main() -> None:
    settings = get_settings()
    init_pool(settings.database_url)

    try:
        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer_reviews")
            count = cur.fetchone()[0]

        if count >= TARGET_COUNT:
            logger.info("customer_reviews already has %d rows — skipping ingestion.", count)
            return

        llm = ChatGoogleGenerativeAI(
            model=settings.worker_model,
            google_api_key=settings.google_api_key,
            temperature=0.9,
        )

        reviews = generate_reviews(llm)
        embed_and_insert(reviews)

        with db_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM customer_reviews")
            final_count = cur.fetchone()[0]
        logger.info("Done. customer_reviews now has %d rows.", final_count)

    finally:
        close_pool()


if __name__ == "__main__":
    main()
