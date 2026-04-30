"""Integration-tier test for the eval-harness is_eval=True propagation contract.

Verifies that values constructed by the eval-harness factories propagate
end-to-end to actual `reviews` and `audit_events` rows in Postgres. Crosses
the DB/audit boundary, so it lives in the integration tier per
`docs/conventions.md`'s "integration if the feature crosses subsystems"
rule.

Note: this test does NOT consume the `is_eval_injection` autouse fixture
from `tests/eval/conftest.py` — different conftest tree. It tests the
propagation contract directly via explicit SELECTs after insert.
"""

from typing import Any

import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from tests.eval.fixtures import FindingEventFactory, ReviewFactory


@pytest_asyncio.fixture
async def session(migrated_db: str) -> Any:
    """Async session scoped to a single migrated test DB."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as s:
        yield s
    await engine.dispose()


async def _insert_installation(session: AsyncSession, installation_id: int) -> None:
    """Seed an installations row so the reviews FK target exists."""
    await session.execute(
        text(
            "INSERT INTO installations "
            "(installation_id, app_slug, account_id, account_login, "
            "account_type, permissions_at_install) "
            "VALUES (:installation_id, 'outrider-test', 99, 'eval-account', "
            "'User', '{}'::jsonb)"
        ),
        {"installation_id": installation_id},
    )
    await session.commit()


async def test_review_factory_inserts_with_is_eval_true(session: AsyncSession) -> None:
    """A factory-built review row inserts with is_eval=True."""
    row = ReviewFactory.create()
    await _insert_installation(session, row["installation_id"])

    await session.execute(
        text(
            "INSERT INTO reviews ("
            "id, installation_id, repo_id, pr_number, head_sha, status, "
            "files_examined, files_traced_beyond_diff, llm_calls_made, "
            "total_input_tokens, total_output_tokens, total_cost_usd, "
            "wall_clock_seconds, is_eval, retention_expires_at"
            ") VALUES ("
            ":id, :installation_id, :repo_id, :pr_number, :head_sha, "
            "CAST(:status AS review_status_enum), "
            ":files_examined, :files_traced_beyond_diff, :llm_calls_made, "
            ":total_input_tokens, :total_output_tokens, :total_cost_usd, "
            ":wall_clock_seconds, :is_eval, :retention_expires_at"
            ")"
        ),
        row,
    )
    await session.commit()

    result = await session.execute(
        text("SELECT is_eval FROM reviews WHERE id = :id"),
        {"id": row["id"]},
    )
    is_eval = result.scalar_one()
    assert is_eval is True


async def test_finding_event_factory_inserts_with_is_eval_true(
    session: AsyncSession,
) -> None:
    """A factory-built FindingEvent inserts with is_eval=True on the audit row.

    Constructs the Pydantic event, dumps to JSON-mode payload (excluding
    sequence_number per the row/payload split), inserts into audit_events
    with the row-level is_eval column populated, then asserts.
    """
    event = FindingEventFactory.create()
    payload = event.model_dump(mode="json", exclude={"sequence_number"})

    await session.execute(
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES ("
            ":event_id, :review_id, :event_type, :timestamp, :is_eval, "
            "CAST(:payload AS jsonb)"
            ")"
        ),
        {
            "event_id": event.event_id,
            "review_id": event.review_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp,
            "is_eval": event.is_eval,
            "payload": __import__("json").dumps(payload),
        },
    )
    await session.commit()

    result = await session.execute(
        text("SELECT is_eval FROM audit_events WHERE event_id = :event_id"),
        {"event_id": event.event_id},
    )
    assert result.scalar_one() is True
