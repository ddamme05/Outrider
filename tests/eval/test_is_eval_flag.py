"""Integration-tier test for the eval-harness is_eval=True propagation contract.

Verifies that values constructed by the eval-harness factories propagate
end-to-end to actual `reviews` and `audit_events` rows in Postgres. Crosses
the DB/audit boundary; lives under tests/eval/ alongside the harness
infrastructure it tests so cross-tier imports aren't needed (the
production/test boundary stays at `pythonpath = ["src"]` per
`docs/conventions.md`).

The eval_db fixture's teardown integrity gate ALSO runs after this test;
both checks fire — the explicit assertions in this test and the gate's
UNION-over-6-tables (analyze_file_cache joined per
specs/2026-06-11-file-hash-analyze-cache.md). Belt + suspenders.
"""

from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from outrider.db.models import AuditEvent, Installation, Review

from .fixtures import FindingEventFactory, ReviewFactory


@pytest_asyncio.fixture
async def session(eval_db: str) -> Any:
    """Async session scoped to a single fresh eval DB.

    Uses `eval_db` from `tests/eval/conftest.py` — alembic-upgraded fresh
    DB with the integrity-gate teardown baked in.
    """
    engine = create_async_engine(eval_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    async with sessionmaker() as s:
        yield s
    await engine.dispose()


async def _insert_installation(session: AsyncSession, installation_id: int) -> None:
    """Seed an installations row so the reviews FK target exists."""
    session.add(
        Installation(
            installation_id=installation_id,
            app_slug="outrider-test",
            account_id=99,
            account_login="eval-account",
            account_type="User",
            permissions_at_install={},
        )
    )
    await session.commit()


async def test_review_factory_inserts_with_is_eval_true(session: AsyncSession) -> None:
    """A factory-built review row inserts with is_eval=True."""
    row = ReviewFactory.create()
    await _insert_installation(session, row["installation_id"])

    session.add(Review(**row))
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

    session.add(
        AuditEvent(
            event_id=event.event_id,
            review_id=event.review_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            is_eval=event.is_eval,
            payload=payload,
        )
    )
    await session.commit()

    result = await session.execute(
        text("SELECT is_eval FROM audit_events WHERE event_id = :event_id"),
        {"event_id": event.event_id},
    )
    assert result.scalar_one() is True


async def test_integrity_gate_flags_a_non_eval_row(session: AsyncSession) -> None:
    """The eval_db integrity gate raises on an is_eval=False row the factory rejects.

    Plants a review with is_eval=False by mutating the factory dict AFTER
    construction — `_reject_is_eval_false` only guards the create() kwargs, so
    this models the "direct insertion forgot the flag" bug the gate exists to
    catch. No other test exercises the gate's raise path (every factory refuses
    to produce a violating row), so without this the `IS DISTINCT FROM TRUE`
    predicate and the gate mechanism go untested.
    """
    from .conftest import _assert_no_is_eval_violations

    row = ReviewFactory.create()
    row["is_eval"] = False  # bypass the factory guard to plant a violation
    await _insert_installation(session, row["installation_id"])
    session.add(Review(**row))
    await session.commit()

    conn = await session.connection()
    try:
        with pytest.raises(AssertionError, match="is_eval discipline violation"):
            await _assert_no_is_eval_violations(conn)
    finally:
        # Remove the planted row in a finally so a failed assertion can't leave it
        # behind — eval_db's own teardown gate (same helper) would otherwise also
        # raise and mask the primary failure.
        await session.execute(text("DELETE FROM reviews WHERE id = :id"), {"id": row["id"]})
        await session.commit()
