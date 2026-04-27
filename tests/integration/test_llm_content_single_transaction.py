"""LLMCallEvent + LLM_CALL_CONTENT single-transaction insert constraint.

Per ``DECISIONS.md#016``: the LLMCallEvent audit row insert and the
LLM_CALL_CONTENT row insert happen in the same DB transaction or
neither happens. The replay tool's mode distinction (full /
metadata-only / hybrid-refused) depends on this — a present-audit/
missing-content pair would be ambiguous between "purged per retention"
(correct) and "insert failed" (a third state the dashboard cannot
distinguish from the first).

The constraint is enforced at the application layer (the agent's LLM
call wrapper inserts both rows in one transaction), not by a DB-level
constraint. This test verifies the transactional behavior the
application code MUST honor: if the second insert fails, the first
must not be visible afterward.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_INSTALLATION_ID = 12345


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, account_type, "
                " permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"id": _INSTALLATION_ID},
        )


_INSERT_AUDIT_AND_CONTENT_ATOMIC = """
INSERT INTO audit_events (review_id, event_type, payload)
VALUES (gen_random_uuid(), 'LLMCallEvent', '{}'::jsonb)
RETURNING event_id;
"""


async def test_audit_plus_content_in_same_transaction_both_commit(
    migrated_db: str,
) -> None:
    """Happy path: both rows insert and commit together."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine)

        async with engine.begin() as conn:
            audit_result = await conn.execute(
                text(
                    "INSERT INTO audit_events (review_id, event_type, payload) "
                    "VALUES (gen_random_uuid(), 'LLMCallEvent', '{}'::jsonb) "
                    "RETURNING event_id"
                )
            )
            event_id = audit_result.scalar_one()

            await conn.execute(
                text(
                    "INSERT INTO llm_call_content "
                    "(event_id, installation_id, prompt, completion, retention_expires_at) "
                    "VALUES (:event_id, :installation_id, 'p', 'c', "
                    "NOW() + INTERVAL '90 days')"
                ),
                {"event_id": event_id, "installation_id": _INSTALLATION_ID},
            )

        async with engine.connect() as conn:
            audit_count = await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE event_id = :id"),
                {"id": event_id},
            )
            content_count = await conn.execute(
                text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :id"),
                {"id": event_id},
            )
            assert audit_count.scalar_one() == 1
            assert content_count.scalar_one() == 1
    finally:
        await engine.dispose()


async def test_failure_in_second_insert_rolls_back_first(migrated_db: str) -> None:
    """Failure path: if the content insert fails, the audit row must not survive.

    Failure is induced by referencing a non-existent installation_id, which
    hits the FK constraint on llm_call_content.installation_id (RESTRICT).
    """
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine)

        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                audit_result = await conn.execute(
                    text(
                        "INSERT INTO audit_events (review_id, event_type, payload) "
                        "VALUES (gen_random_uuid(), 'LLMCallEvent', '{}'::jsonb) "
                        "RETURNING event_id"
                    )
                )
                event_id = audit_result.scalar_one()

                # installation_id 99999 does not exist; FK violation.
                await conn.execute(
                    text(
                        "INSERT INTO llm_call_content "
                        "(event_id, installation_id, prompt, completion, "
                        " retention_expires_at) "
                        "VALUES (:event_id, 99999, 'p', 'c', "
                        "NOW() + INTERVAL '90 days')"
                    ),
                    {"event_id": event_id},
                )

        # Transaction rolled back — neither row should be visible.
        async with engine.connect() as conn:
            audit_count = await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE event_type = 'LLMCallEvent'")
            )
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()
