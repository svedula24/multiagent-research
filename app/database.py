import logging
from contextlib import contextmanager
from urllib.parse import urlparse

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

logger = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None


def _parse_dsn(database_url: str) -> dict:
    parsed = urlparse(database_url)
    return {
        "dbname": parsed.path.lstrip("/"),
        "user": parsed.username,
        "password": parsed.password,
        "host": parsed.hostname,
        "port": parsed.port or 5432,
    }


def init_pool(database_url: str, minconn: int = 1, maxconn: int = 10) -> None:
    global _pool
    dsn = _parse_dsn(database_url)
    _pool = ThreadedConnectionPool(minconn, maxconn, **dsn)
    logger.info("Database connection pool initialised (min=%d, max=%d)", minconn, maxconn)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed")


def get_conn() -> psycopg2.extensions.connection:
    if _pool is None:
        raise RuntimeError("Connection pool has not been initialised. Call init_pool() first.")
    return _pool.getconn()


def release_conn(conn: psycopg2.extensions.connection) -> None:
    if _pool is not None:
        _pool.putconn(conn)


@contextmanager
def db_cursor():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_conn(conn)
