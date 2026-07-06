"""Data-access seam over the catalog read model.

This is the query layer described in the service README: it owns the SQL against
the denormalized Postgres read model and returns plain row dicts to the API
layer, which never assembles queries itself. Connections come from a
process-wide psycopg pool sized for the service's read-heavy request pattern.

Most call sites go through the parameterized helpers (:func:`fetch_all` /
:func:`fetch_one`), which bind arguments through the driver. The saved-view and
admin-export paths need to run a fully-formed statement that was assembled
upstream and reach for :func:`run_raw_query`, which hands the statement to the
driver verbatim.
"""

from __future__ import annotations

import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool

_DEFAULT_DSN = "postgresql://catalog@localhost:5432/catalog_read"

# Lazily-initialized process-wide pool. The read model is served by many
# interchangeable instances, so each process keeps its own small pool rather
# than sharing one across the fleet.
_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    """Return the process-wide connection pool, opening it on first use.

    The DSN comes from ``DATABASE_URL`` (the read-model Postgres connection
    string documented in the service config) and falls back to the local
    development default.
    """
    global _pool
    if _pool is None:
        dsn = os.environ.get("DATABASE_URL", _DEFAULT_DSN)
        _pool = SimpleConnectionPool(minconn=1, maxconn=16, dsn=dsn)
    return _pool


def run_raw_query(sql: str) -> list[dict[str, Any]]:
    """Execute a fully-formed SQL statement and return all result rows.

    The statement is sent straight to the driver with no parameter binding —
    whatever the caller assembled in ``sql`` is exactly what runs. This backs
    the saved-view and admin-export endpoints, where the SQL text is composed
    upstream in the API layer before it reaches this seam.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
        conn.commit()
        return [dict(row) for row in rows]
    finally:
        pool.putconn(conn)


def fetch_all(sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run a parameterized statement and return every row as a dict.

    Placeholders in ``sql`` are bound by the driver from ``params``; the two are
    never concatenated here.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]
    finally:
        pool.putconn(conn)


def fetch_one(sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    """Run a parameterized statement and return the first row, if any.

    Placeholders in ``sql`` are bound by the driver from ``params``.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return dict(row) if row is not None else None
    finally:
        pool.putconn(conn)
