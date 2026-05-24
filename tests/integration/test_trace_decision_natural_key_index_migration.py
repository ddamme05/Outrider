"""Trace-decision natural-key partial unique index — migration smoke test.

Per `specs/2026-05-23-trace-node.md` M7 (a) + DECISIONS.md#026. The
8f2a4c1e7b3d migration adds:

    CREATE UNIQUE INDEX uq_audit_events_trace_decision_natural_key
        ON audit_events (review_id, (payload->>'source_finding_id'))
        WHERE event_type = 'trace_decision'
              AND payload ? 'source_finding_id';

This test pins:

  1. `alembic upgrade head` creates the index on a fresh DB; the
     normalized `pg_indexes.indexdef` matches the expected canonical
     form (column order, expression shape, partial-WHERE shape).
  2. The index enforces uniqueness — inserting two TraceDecisionEvent
     rows with the same (review_id, source_finding_id) tuple raises
     IntegrityError on the second INSERT.
  3. The partial WHERE clause restricts the constraint to
     `event_type = 'trace_decision'` only — other event types with the
     same (review_id, payload->>'source_finding_id') tuple do NOT
     collide (the partial index doesn't apply to them).
  4. The `payload ? 'source_finding_id'` conjunct in the WHERE excludes
     key-missing rows from the index — duplicates with NULL natural
     keys do not silently collide nor get admitted as distinct.
  5. `postgresql_insert(...).on_conflict_do_nothing(...)` returns zero
     rows on natural-key conflict (the persister-side contract).
  6. `alembic downgrade -1` removes the index cleanly; upgrade →
     downgrade → upgrade cycles cleanly via the IF NOT EXISTS guard.
"""

import re
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
    the expected canonical definition. Substring checks are not enough —
    a wrong index (missing UNIQUE, swapped column order, missing the
    payload-key WHERE conjunct) would pass. Pin via regex against the
    canonical `pg_indexes.indexdef` form Postgres normalizes to."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                sa.text(
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
            indexdef = row.scalar_one()
            # Canonical-form regex against Postgres's normalized DDL.
            # Components pinned: UNIQUE, btree, column order
            # (review_id first, payload->>'source_finding_id' second),
            # both WHERE conjuncts (event_type + payload-key existence).
            canonical_pattern = re.compile(
                r"CREATE UNIQUE INDEX uq_audit_events_trace_decision_natural_key "
                r"ON public\.audit_events "
                r"USING btree "
                r"\(review_id, \(\(payload ->> 'source_finding_id'::text\)\)\) "
                r"WHERE "
                r"\(\(event_type = 'trace_decision'::text\) "
                r"AND \(payload \? 'source_finding_id'::text\)\)"
            )
            assert canonical_pattern.fullmatch(indexdef), (
                f"indexdef did not match canonical form; got: {indexdef!r}"
            )
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
                    index_where=sa.text(
                        "event_type = 'trace_decision' AND payload ? 'source_finding_id'"
                    ),
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


async def test_index_excludes_rows_missing_payload_key(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """The WHERE clause includes `payload ? 'source_finding_id'` so
    trace_decision rows whose payload lacks the natural-key field are
    excluded from the partial index entirely. This is defense-in-depth
    against the silent-NULL-collision footgun: `payload->>'absent'`
    returns NULL, and Postgres treats NULLs as distinct in unique
    indexes (pre-PG15 default), which would otherwise admit duplicate
    NULL-keyed rows. With the `?`-conjunct, such rows fall outside the
    index — no uniqueness check fires, no silent admission as distinct.
    Pydantic's `TraceDecisionEvent.source_finding_id: UUID` is the
    application-side guarantee; this test pins the DB-side floor."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        review_id = uuid4()
        timestamp = datetime.now(UTC)

        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            # Two trace_decision rows with NO source_finding_id key in
            # payload. Under the hardened WHERE clause they fall outside
            # the partial index, so both inserts succeed. (This documents
            # the intentional behavior — Pydantic is the actual gate
            # ensuring trace_decision rows always carry the key in
            # production; the DB floor handles the manual-SQL /
            # schema-drift case by exclusion rather than silent
            # admission.)
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="trace_decision",
                    timestamp=timestamp,
                    sequence_number=1,
                    is_eval=False,
                    payload={"marker": "no-source-finding-id"},
                )
            )
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="trace_decision",
                    timestamp=timestamp,
                    sequence_number=2,
                    is_eval=False,
                    payload={"marker": "also-no-source-finding-id"},
                )
            )
            # Both inserts succeed = key-missing rows are excluded from
            # the partial index per the `payload ? '...'` conjunct.
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


async def test_upgrade_downgrade_upgrade_cycles_cleanly(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """The IF NOT EXISTS guard on upgrade + IF EXISTS guard on downgrade
    keep upgrade → downgrade → upgrade idempotent. Realistic redeploy /
    rollback / re-deploy scenarios hit this path; the original downgrade
    test only exercised upgrade → downgrade."""
    await alembic_runner("upgrade", "head", fresh_db)
    await alembic_runner("downgrade", PRIOR_REVISION, fresh_db)
    await alembic_runner("upgrade", "head", fresh_db)

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
            assert count == 1
    finally:
        await engine.dispose()
