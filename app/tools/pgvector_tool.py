import logging
from typing import Any

from app.database import db_cursor

logger = logging.getLogger(__name__)

# Parameterised query using pgvector cosine distance operator (<=>)
_SIMILARITY_SQL = """
    SELECT
        id,
        content,
        rating,
        category,
        created_at,
        1 - (embedding <=> %s::vector) AS similarity
    FROM customer_reviews
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""


def similarity_search(embedding: list[float], top_k: int = 10) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most semantically similar reviews using cosine similarity.

    Args:
        embedding: 1536-dimensional query vector from OpenAI text-embedding-3-small.
        top_k: number of results to return.

    Returns:
        List of dicts with keys: id, content, rating, category, created_at, similarity.
    """
    # pgvector expects the vector literal as a string, e.g. '[0.1, 0.2, ...]'
    vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"

    try:
        with db_cursor() as cur:
            cur.execute(_SIMILARITY_SQL, (vector_literal, vector_literal, top_k))
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]

        results = [dict(zip(cols, row)) for row in rows]
        logger.info("pgvector similarity search returned %d rows", len(results))
        return results
    except Exception as exc:
        logger.error("pgvector similarity search failed: %s", type(exc).__name__)
        raise
