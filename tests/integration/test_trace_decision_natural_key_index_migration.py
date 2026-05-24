"""Trace-decision natural-key partial unique index — migration smoke test.

Per `specs/2026-05-23-trace-node.md` M7 (a) + DECISIONS.md#026. The
8f2a4c1e7b3d migration adds:

    CREATE UNIQUE INDEX uq_audit_events_trace_decision_natural_key
        ON audit_events (review_id, (payload->>'source_finding_id'))
        WHERE event_type = 'trace_decision';

This test pins:

  1. `alembic upgrade head` creates the index on a fresh DB.
  2. The index enforces uniqueness — inserting two TraceDecisionEvent
     rows with the same (review_id, source_finding_id) tuple raises
     IntegrityError on the second INSERT.
  3. The partial WHERE clause restricts the constraint to
     `event_type = 'trace_decision'` only — other event types with the
     same (review_id, payload->>'source_finding_id') tuple do NOT
     collide (the partial index doesn't apply to them).
  4. `alembic downgrade -1` removes the index cleanly.
"""

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

AlembicRunner = Callable[[str, str, str], Awaitable[None]]

INDEX_NAME = "uq_audit_events_trace_decision_natural_key"
MIGRATION_REVISION = "8f2a4c1e7b3d"
PRIOR_REVISION = "3d03bca7f2be"


async def test_migration_creates_partial_unique_index(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """After `alembic upgrade head`, the partial unique index exists with
    the expected definition (event_type='trace_decision' WHERE clause +
    review_id + payload->>'source_finding_id' lookup columns)."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            # pg_indexes carries the indexdef as a parseable text form.
            row = await conn.execute(
                sa.text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
            indexdef = row.scalar_one()
            # Index definition checks (Postgres normalizes the DDL):
            assert "UNIQUE" in indexdef
            assert "review_id" in indexdef
            assert "source_finding_id" in indexdef
            # WHERE clause partial-index filter:
            assert "trace_decision" in indexdef.lower()
    finally:
        await engine.dispose()


async def test_index_enforces_natural_key_uniqueness(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """Two TraceDecisionEvent-shaped rows with the same
    (review_id, source_finding_id) tuple — second INSERT raises
    IntegrityError. Pins the load-bearing M7(a) safety net."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        # Use raw connection + raw INSERT — this test is about DB-level
        # uniqueness, NOT about the persister-side code path (that's
        # Group 4). Construct minimal valid payloads.
        review_id = uuid4()
        source_finding_id = str(uuid4())
        first_event_id = uuid4()
        second_event_id = uuid4()
        timestamp = datetime.now(UTC)

        # Reflect the audit_events table from the migrated DB.
        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            # First insert succeeds.
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=first_event_id,
                    review_id=review_id,
                    event_type="trace_decision",
                    timestamp=timestamp,
                    sequence_number=1,
                    is_eval=False,
                    payload={"source_finding_id": source_finding_id, "marker": "first"},
                )
            )

        # Second insert with same (review_id, source_finding_id) but
        # different event_id MUST raise IntegrityError on the partial
        # unique index. Test uses a fresh transaction so the first
        # commit persists and the second sees the conflicting state.
        async with engine.begin() as conn:
            with pytest.raises(sa.exc.IntegrityError):
                await conn.execute(
                    audit_events_table.insert().values(
                        event_id=second_event_id,
                        review_id=review_id,
                        event_type="trace_decision",
                        timestamp=timestamp,
                        sequence_number=2,
                        is_eval=False,
                        payload={
                            "source_finding_id": source_finding_id,
                            "marker": "second",
                        },
                    )
                )
    finally:
        await engine.dispose()


async def test_index_partial_where_does_not_restrict_other_event_types(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """The partial WHERE clause restricts the unique constraint to
    `event_type='trace_decision'`. Other event types with the same
    (review_id, source_finding_id) tuple do NOT collide. Verifies the
    partial-index narrow scope per #026 mode mixing — non-trace events
    stay under event_id-PK idempotency, not natural-key."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        review_id = uuid4()
        source_finding_id = str(uuid4())
        timestamp = datetime.now(UTC)

        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            # Two finding events with the same (review_id, source_finding_id)
            # tuple — partial index doesn't apply because event_type !=
            # 'trace_decision'. Both inserts must succeed.
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="finding",
                    timestamp=timestamp,
                    sequence_number=1,
                    is_eval=False,
                    payload={"source_finding_id": source_finding_id, "marker": "first"},
                )
            )
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="finding",
                    timestamp=timestamp,
                    sequence_number=2,
                    is_eval=False,
                    payload={"source_finding_id": source_finding_id, "marker": "second"},
                )
            )
            # If we reach here without IntegrityError, the partial index
            # correctly excluded these non-trace_decision rows.
    finally:
        await engine.dispose()


async def test_index_on_conflict_do_nothing_returns_no_rows(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """The persister-side code path uses `postgresql_insert(...).on_conflict
    _do_nothing(...)` against this index per #026 point 2(b). Pin the
    expected on-conflict behavior: second INSERT returns zero rows
    (not IntegrityError) when explicitly handled via on_conflict_do_nothing."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        review_id = uuid4()
        source_finding_id = str(uuid4())
        timestamp = datetime.now(UTC)

        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="trace_decision",
                    timestamp=timestamp,
                    sequence_number=1,
                    is_eval=False,
                    payload={"source_finding_id": source_finding_id, "marker": "first"},
                )
            )

            # Use postgresql_insert with on_conflict_do_nothing — the
            # persister will use this idiom. Note we point at the
            # partial index name via `index_elements` + `index_where`.
            stmt = (
                pg_insert(audit_events_table)
                .values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="trace_decision",
                    timestamp=timestamp,
                    sequence_number=2,
                    is_eval=False,
                    payload={"source_finding_id": source_finding_id, "marker": "second"},
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        audit_events_table.c.review_id,
                        sa.text("(payload->>'source_finding_id')"),
                    ],
                    index_where=sa.text("event_type = 'trace_decision'"),
                )
                .returning(audit_events_table.c.event_id)
            )
            result = await conn.execute(stmt)
            rows = result.fetchall()
            # No rows returned = conflict path fired (existing row not
            # overwritten, no IntegrityError raised).
            assert rows == []
    finally:
        await engine.dispose()


async def test_downgrade_drops_index(fresh_db: str, alembic_runner: AlembicRunner) -> None:
    """After `alembic downgrade -1`, the partial unique index is gone."""
    await alembic_runner("upgrade", "head", fresh_db)
    await alembic_runner("downgrade", PRIOR_REVISION, fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                sa.text(
                    "SELECT count(*)::int FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
            count = row.scalar_one()
            assert count == 0
    finally:
        await engine.dispose()
