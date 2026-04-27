"""Append-only trigger behavioral test.

Backs `audit-events-append-only` and the schema-layer spec's append-only
guarantee on both `audit_events` and `purge_audit`. Per
DECISIONS.md#012 + #014, the trigger is absolute: there is no
privileged-role bypass, and `purge_audit` carries the same trigger as
`audit_events` so the forensic trail of which content rows were purged
on which `installation.deleted` event survives untouched.

Three assertion concepts, six concrete cases:

  - Introspection: function + both triggers exist with expected names.
    Overlaps deliberately with
    test_genesis_migration.py::test_genesis_upgrade_creates_full_schema —
    that test asserts the post-upgrade schema state at the introspection
    layer; this test is the behavioral counterpart, asserting the
    trigger's protective effect at the constraint-enforcement layer.
    Both layers are required: a trigger that exists but doesn't fire
    correctly fails this test; a trigger whose existence isn't asserted
    fails the genesis test.
  - UPDATE raises on audit_events and on purge_audit.
  - DELETE raises on audit_events and on purge_audit.

Uses ``migrated_db`` (not ``fresh_db``) — the test does not need to
drive alembic itself, so it asks the fixture for a DB-at-head. This is
the first test that exercises ``migrated_db``; if the fixture is broken,
this is where it would surface.
"""

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
    ("table", "insert_stmt", "id_column", "operation"),
    [
        ("audit_events", _INSERT_AUDIT, "event_id", "UPDATE"),
        ("audit_events", _INSERT_AUDIT, "event_id", "DELETE"),
        ("purge_audit", _INSERT_PURGE_AUDIT, "id", "UPDATE"),
        ("purge_audit", _INSERT_PURGE_AUDIT, "id", "DELETE"),
    ],
)
async def test_append_only_trigger_blocks_mutation(
    migrated_db: str, table: str, insert_stmt, id_column: str, operation: str
) -> None:
    """UPDATE and DELETE on an append-only table raise via the trigger.

    The error class is ``sqlalchemy.exc.ProgrammingError`` because PostgreSQL's
    ``RAISE EXCEPTION`` from a trigger surfaces through psycopg as a
    `RaiseException` and SQLAlchemy maps that to ProgrammingError. The
    `match` pattern keys on the trigger function's literal output so the
    assertion can't be satisfied by an unrelated DB error.
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
                f"UPDATE {table} SET purge_role = 'overwritten' "  # noqa: S608
                f"WHERE {id_column} = :id"
                if table == "purge_audit"
                else f"UPDATE {table} SET event_type = 'overwritten' "  # noqa: S608
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
