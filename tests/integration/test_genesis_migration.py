"""Genesis migration smoke test: upgrade head + downgrade base round-trip.

Backs the ``alembic migration applies cleanly against a fresh DB`` claim
in the schema-layer spec. Two assertions in two tests:

  - upgrade head produces all 9 outrider tables, both PG ENUM types, the
    append-only trigger function and its two trigger attachments, and
    the v1.0.0 row in severity_policies.
  - downgrade base produces a clean state — only ``alembic_version``
    survives (alembic owns this table; it remains after downgrade by
    design with no rows). No leftover content tables, ENUM types,
    plpgsql functions, or triggers.

These tests use ``fresh_db`` (not ``migrated_db``) because they exercise
the migration itself; they need to drive ``alembic_runner`` directly.
"""

from collections.abc import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

AlembicRunner = Callable[[str, str, str], Awaitable[None]]

EXPECTED_OUTRIDER_TABLES = {
    "anomalies",
    "audit_events",
    "findings",
    "installation_repositories",
    "installations",
    "llm_call_content",
    "purge_audit",
    "reviews",
    "severity_policies",
}
EXPECTED_ENUM_TYPES = {"review_status_enum", "anomaly_status_enum"}
EXPECTED_FUNCTIONS = {"outrider_audit_append_only_guard"}
EXPECTED_TRIGGERS = {
    "trg_audit_events_append_only",
    "trg_purge_audit_append_only",
}
GENESIS_REVISION = "af138edd4b57"


async def test_genesis_upgrade_creates_full_schema(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """upgrade head against a fresh DB produces the full V1 schema."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            tables_result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = {row[0] for row in tables_result}
            assert EXPECTED_OUTRIDER_TABLES.issubset(tables), (
                f"missing tables: {EXPECTED_OUTRIDER_TABLES - tables}"
            )
            assert "alembic_version" in tables

            enums_result = await conn.execute(
                text(
                    "SELECT typname FROM pg_type "
                    "WHERE typtype = 'e' AND typnamespace = 'public'::regnamespace"
                )
            )
            assert {row[0] for row in enums_result} == EXPECTED_ENUM_TYPES

            functions_result = await conn.execute(
                text(
                    "SELECT proname FROM pg_proc "
                    "WHERE pronamespace = 'public'::regnamespace "
                    "AND proname = ANY(:names)"
                ),
                {"names": list(EXPECTED_FUNCTIONS)},
            )
            assert {row[0] for row in functions_result} == EXPECTED_FUNCTIONS

            triggers_result = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND tgname = ANY(:names)"
                ),
                {"names": list(EXPECTED_TRIGGERS)},
            )
            assert {row[0] for row in triggers_result} == EXPECTED_TRIGGERS

            seed_result = await conn.execute(text("SELECT version FROM severity_policies"))
            assert [row[0] for row in seed_result] == ["1.0.0"]

            revision_result = await conn.execute(text("SELECT version_num FROM alembic_version"))
            assert revision_result.scalar_one() == GENESIS_REVISION
    finally:
        await engine.dispose()


async def test_genesis_downgrade_round_trips_clean(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """upgrade head + downgrade base leaves no Outrider-owned objects behind."""
    await alembic_runner("upgrade", "head", fresh_db)
    await alembic_runner("downgrade", "base", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            tables_result = await conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            )
            tables = {row[0] for row in tables_result}
            assert EXPECTED_OUTRIDER_TABLES.isdisjoint(tables), (
                f"downgrade left Outrider tables behind: {EXPECTED_OUTRIDER_TABLES & tables}"
            )
            # alembic_version remains by design; alembic's own table is
            # not part of our migration's create/drop cycle.
            assert tables == {"alembic_version"}

            enums_result = await conn.execute(
                text(
                    "SELECT typname FROM pg_type "
                    "WHERE typtype = 'e' AND typnamespace = 'public'::regnamespace"
                )
            )
            leftover_enums = {row[0] for row in enums_result}
            assert leftover_enums == set(), f"leftover ENUMs: {leftover_enums}"

            functions_result = await conn.execute(
                text(
                    "SELECT proname FROM pg_proc "
                    "WHERE pronamespace = 'public'::regnamespace "
                    "AND proname = ANY(:names)"
                ),
                {"names": list(EXPECTED_FUNCTIONS)},
            )
            leftover_functions = {row[0] for row in functions_result}
            assert leftover_functions == set(), f"leftover plpgsql functions: {leftover_functions}"

            triggers_result = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND tgname = ANY(:names)"
                ),
                {"names": list(EXPECTED_TRIGGERS)},
            )
            leftover_triggers = {row[0] for row in triggers_result}
            assert leftover_triggers == set(), f"leftover triggers: {leftover_triggers}"

            revision_count = await conn.execute(text("SELECT COUNT(*) FROM alembic_version"))
            assert revision_count.scalar_one() == 0
    finally:
        await engine.dispose()
