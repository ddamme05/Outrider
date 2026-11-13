"""Check append-only protections on audit_events and purge_audit.

Use a migrated PostgreSQL database to inspect the triggers and verify that
UPDATE, DELETE, and TRUNCATE are rejected without a privileged bypass."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine


async def test_append_only_trigger_objects_exist(migrated_db: str) -> None:
    """The plpgsql function and both triggers exist with the documented names.

    Schema-state assertion. Behavioral counterparts are below.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            function_row = await conn.execute(
                text(
                    "SELECT proname FROM pg_proc "
                    "WHERE proname = 'outrider_audit_append_only_guard' "
                    "AND pronamespace = 'public'::regnamespace"
                )
            )
            assert function_row.scalar_one() == "outrider_audit_append_only_guard"

            audit_trigger = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger "
                    "WHERE tgname = 'trg_audit_events_append_only' AND NOT tgisinternal"
                )
            )
            assert audit_trigger.scalar_one() == "trg_audit_events_append_only"

            purge_trigger = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger "
                    "WHERE tgname = 'trg_purge_audit_append_only' AND NOT tgisinternal"
                )
            )
            assert purge_trigger.scalar_one() == "trg_purge_audit_append_only"

            # Statement-level BEFORE TRUNCATE guards (migration c22e2864d3d8) — the
            # companion the row-level triggers above can't provide (TRUNCATE skips them).
            truncate_triggers = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger "
                    "WHERE tgname IN ('trg_audit_events_no_truncate', "
                    "'trg_purge_audit_no_truncate') AND NOT tgisinternal "
                    "ORDER BY tgname"
                )
            )
            assert truncate_triggers.scalars().all() == [
                "trg_audit_events_no_truncate",
                "trg_purge_audit_no_truncate",
            ]
    finally:
        await engine.dispose()


# Each parameterized case inserts a valid row (the trigger fires only on
# UPDATE/DELETE; INSERT is intentionally allowed), then attempts the named
# mutation and asserts it raises with the trigger's RAISE EXCEPTION
# message. The match pattern keys on the function's literal output:
# "append-only table <tablename>: <op> not permitted".
_INSERT_AUDIT = text(
    "INSERT INTO audit_events (review_id, event_type, payload) "
    "VALUES (gen_random_uuid(), 'TestEvent', '{}'::jsonb) "
    "RETURNING event_id"
)
_INSERT_PURGE_AUDIT = text(
    "INSERT INTO purge_audit "
    "(installation_id, target_table, rows_affected, purge_role) "
    "VALUES (1, 'reviews', 0, 'test_role') "
    "RETURNING id"
)


@pytest.mark.parametrize(
    ("table", "insert_stmt", "id_column", "update_column", "operation"),
    [
        ("audit_events", _INSERT_AUDIT, "event_id", "event_type", "UPDATE"),
        ("audit_events", _INSERT_AUDIT, "event_id", "event_type", "DELETE"),
        ("purge_audit", _INSERT_PURGE_AUDIT, "id", "purge_role", "UPDATE"),
        ("purge_audit", _INSERT_PURGE_AUDIT, "id", "purge_role", "DELETE"),
    ],
)
async def test_append_only_trigger_blocks_mutation(
    migrated_db: str,
    table: str,
    insert_stmt,
    id_column: str,
    update_column: str,
    operation: str,
) -> None:
    """UPDATE and DELETE on an append-only table raise via the trigger.

    The error class is ``sqlalchemy.exc.ProgrammingError`` because PostgreSQL's
    ``RAISE EXCEPTION`` from a trigger surfaces through psycopg as a
    `RaiseException` and SQLAlchemy maps that to ProgrammingError. The
    `match` pattern keys on the trigger function's literal output so the
    assertion can't be satisfied by an unrelated DB error.

    ``update_column`` is unused for the DELETE cases; carrying it on every
    parametrize tuple keeps the SQL construction below uniform across the
    two operations.
    """
    engine = create_async_engine(migrated_db)
    try:
        # Step 1: insert succeeds (trigger does not fire on INSERT).
        async with engine.begin() as conn:
            inserted = await conn.execute(insert_stmt)
            row_id = inserted.scalar_one()

        # Step 2: mutation raises via the trigger's RAISE EXCEPTION.
        if operation == "UPDATE":
            mutation_stmt = text(
                f"UPDATE {table} SET {update_column} = 'overwritten' "  # noqa: S608
                f"WHERE {id_column} = :id"
            )
        else:
            mutation_stmt = text(f"DELETE FROM {table} WHERE {id_column} = :id")  # noqa: S608

        with pytest.raises(
            ProgrammingError,
            match=f"append-only table {table}: {operation} not permitted",
        ):
            async with engine.begin() as conn:
                await conn.execute(mutation_stmt, {"id": row_id})
    finally:
        await engine.dispose()


@pytest.mark.parametrize("table", ["audit_events", "purge_audit"])
async def test_append_only_trigger_blocks_truncate(migrated_db: str, table: str) -> None:
    """TRUNCATE ... CASCADE on an append-only table raises via the statement trigger.

    Row-level triggers do NOT fire for TRUNCATE, so the row-level UPDATE/DELETE guard
    above leaves a hole: a table-owner TRUNCATE would erase the entire audit log /
    purge trail. The ``BEFORE TRUNCATE FOR EACH STATEMENT`` trigger (migration
    ``c22e2864d3d8``) closes it, reusing the same ``outrider_audit_append_only_guard``
    function — so ``TG_OP`` is ``TRUNCATE`` and the message keys on the same literal.

    Why CASCADE: ``audit_events`` is referenced by foreign keys (findings /
    llm_call_content), so a bare ``TRUNCATE audit_events`` raises Postgres's
    feature-not-supported FK error (psycopg ``NotSupportedError``) BEFORE the trigger
    even fires. CASCADE is precisely the path the FK can't stop — a determined
    ``TRUNCATE audit_events CASCADE`` would otherwise wipe the log AND its referencing
    rows — so it is the path the trigger uniquely guards. Exercising CASCADE proves the
    trigger (not the incidental FK) is doing the protecting. No row insert is needed;
    the statement trigger fires regardless of contents.
    """
    engine = create_async_engine(migrated_db)
    try:
        with pytest.raises(
            ProgrammingError,
            match=f"append-only table {table}: TRUNCATE not permitted",
        ):
            async with engine.begin() as conn:
                await conn.execute(text(f"TRUNCATE {table} CASCADE"))  # noqa: S608
    finally:
        await engine.dispose()
