import logging
from typing import Any

from app.database import db_cursor

logger = logging.getLogger(__name__)


def run_query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    """
    Execute a parameterised SQL query and return rows as a list of dicts.

    Args:
        sql:    SQL string with %s placeholders — never f-strings or concatenation.
        params: Tuple of values to bind to the placeholders.

    Returns:
        List of row dicts keyed by column name. Empty list if no rows.

    Raises:
        ValueError: if params is not a tuple (defence against accidental string interpolation).
        Exception:  re-raises any psycopg2 errors after logging.
    """
    if not isinstance(params, tuple):
        raise ValueError(
            "sql_tool.run_query requires params to be a tuple. "
            "Never pass user input via string interpolation."
        )

    try:
        with db_cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            cols = [desc[0] for desc in cur.description]

        results = [dict(zip(cols, row)) for row in rows]
        logger.info("SQL query returned %d rows", len(results))
        return results
    except Exception as exc:
        logger.error("SQL query failed: %s", type(exc).__name__)
        raise
