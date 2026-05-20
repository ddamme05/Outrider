"""severity_policies append-only trigger — behavioral test.

Backs §0c of specs/2026-05-19-analyze-foundation.md. Mirrors the shape
of test_audit_append_only_trigger.py: introspection plus UPDATE/DELETE
behavioral cases. The trigger guards against post-startup mutation of
the policy row; the startup fingerprint check is once-at-lifespan, and
a concurrent process or out-of-band UPDATE on `severity_policies` would
otherwise tamper with the DB row while findings keep writing under the
live mapping, causing replay divergence.

Three concepts:

  - Introspection: function + trigger exist with documented names.
  - UPDATE on the genesis-seeded '1.0.0' row raises via the trigger.
  - DELETE on the genesis-seeded '1.0.0' row raises via the trigger.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.ext.asyncio import create_async_engine


async def test_append_only_trigger_objects_exist(migrated_db: str) -> None:
    """The plpgsql function and both triggers exist with the documented names.

    Two triggers: row-level UPDATE/DELETE guard + statement-level TRUNCATE
    guard. Row-level triggers don't fire for TRUNCATE; without the
    statement-level companion an operator with table-owner privileges
    could TRUNCATE the seeded '1.0.0' row out from under the lifespan
    fingerprint check. Audit finding §0c-adv-M1.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            function_row = await conn.execute(
                text(
                    "SELECT proname FROM pg_proc "
                    "WHERE proname = 'outrider_severity_policies_append_only_guard' "
                    "AND pronamespace = 'public'::regnamespace"
                )
            )
            assert function_row.scalar_one() == "outrider_severity_policies_append_only_guard"

            trigger_names = await conn.execute(
                text(
                    "SELECT tgname FROM pg_trigger "
                    "WHERE tgname IN ("
                    "  'trg_severity_policies_append_only', "
                    "  'trg_severity_policies_no_truncate'"
                    ") AND NOT tgisinternal "
                    "ORDER BY tgname"
                )
            )
            assert [row[0] for row in trigger_names] == [
                "trg_severity_policies_append_only",
                "trg_severity_policies_no_truncate",
            ]
    finally:
        await engine.dispose()


@pytest.mark.parametrize("operation", ["UPDATE", "DELETE"])
async def test_append_only_trigger_blocks_mutation(
    migrated_db: str,
    operation: str,
) -> None:
    """UPDATE and DELETE on severity_policies raise via the trigger.

    Targets the genesis-seeded '1.0.0' row (no INSERT step needed; the row
    exists post-migration). Mirrors test_audit_append_only_trigger.py's
    behavioral pattern; the trigger's RAISE EXCEPTION surfaces as a
    `RaiseException` via psycopg, which SQLAlchemy maps to
    `ProgrammingError`. The match pattern keys on the function's literal
    output prefix so the assertion can't be satisfied by an unrelated
    DB error.
    """
    engine = create_async_engine(migrated_db)
    try:
        if operation == "UPDATE":
            stmt = text("UPDATE severity_policies SET policy = '{}'::jsonb WHERE version = '1.0.0'")
        else:
            stmt = text("DELETE FROM severity_policies WHERE version = '1.0.0'")

        with pytest.raises(
            ProgrammingError,
            match=f"append-only table severity_policies: {operation} not permitted",
        ):
            async with engine.begin() as conn:
                await conn.execute(stmt)
    finally:
        await engine.dispose()
