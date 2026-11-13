"""Check migrated timestamp columns against PostgreSQL's timezone-aware type.

Inspect all ``_at`` and ``timestamp`` columns. ORM metadata checks cannot catch
a migration that changes the type (``timestamps-are-aware``)."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

OUTRIDER_TABLES = (
    "installations",
    "installation_repositories",
    "severity_policies",
    "reviews",
    "audit_events",
    "findings",
    "llm_call_content",
    "anomalies",
    "purge_audit",
)


async def test_all_timestamp_columns_use_timestamptz(migrated_db: str) -> None:
    """Every _at or `timestamp` column across the 9 Outrider tables is TIMESTAMPTZ."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT table_name, column_name, data_type "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' "
                    "AND table_name = ANY(:tables) "
                    "AND (column_name LIKE '%_at' OR column_name = 'timestamp') "
                    "ORDER BY table_name, column_name"
                ),
                {"tables": list(OUTRIDER_TABLES)},
            )
            rows = list(result)

        # Sanity: we expect at least one such column per table that has
        # one (purge_audit has timestamp; severity_policies has created_at;
        # etc.). If this drops to zero, the query is wrong, not the schema.
        assert len(rows) >= 9, f"Expected at least 9 timestamp-bearing columns; found: {rows}"

        non_aware = [
            (table, column, dtype)
            for (table, column, dtype) in rows
            if dtype != "timestamp with time zone"
        ]
        assert non_aware == [], f"Found timestamp columns NOT timezone-aware: {non_aware}"
    finally:
        await engine.dispose()
