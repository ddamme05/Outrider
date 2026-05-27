"""review_phase natural-key partial unique index — migration smoke test.

Per `specs/2026-05-26-hitl-node.md` §Q8 + `docs/invariants.md`
`phase-events-bound-work`. The 4b9f1c5a7e21 migration adds:

    CREATE UNIQUE INDEX uq_audit_events_review_phase_natural_key
        ON audit_events (
            review_id,
            (payload->>'phase_id'),
            COALESCE(phase_key, ''),
            (payload->>'marker')
        )
        WHERE event_type = 'review_phase'
              AND payload ? 'phase_id'
              AND payload ? 'marker';

This test pins:

  1. `alembic upgrade head` creates the index on a fresh DB with the
     expected canonical definition (column order, expression shape,
     partial-WHERE shape).
  2. The index enforces uniqueness — inserting two ReviewPhaseEvent
     rows with the same (review_id, phase_id, phase_key, marker)
     tuple raises IntegrityError on the second INSERT.
  3. Different `marker` values for the same `(review_id, phase_id,
     phase_key)` do NOT collide — `start` + `end` are distinct rows.
  4. NULL `phase_key` values collide via `COALESCE(phase_key, '')`
     — closes PostgreSQL's NULL-distinct default for the V1 case
     where every phase event has `phase_key=NULL`.
  5. The partial WHERE clause restricts the constraint to
     `event_type='review_phase'` only.
  6. `payload ? 'phase_id' AND payload ? 'marker'` conjuncts in the
     WHERE exclude key-missing rows from the index.
  7. `alembic downgrade -1` removes the index cleanly; upgrade →
     downgrade → upgrade cycles cleanly via the IF NOT EXISTS guard.
"""

import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine

AlembicRunner = Callable[[str, str, str], Awaitable[None]]

INDEX_NAME = "uq_audit_events_review_phase_natural_key"
MIGRATION_REVISION = "4b9f1c5a7e21"
PRIOR_REVISION = "33f8fe051bec"


async def test_migration_creates_partial_unique_index(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """After `alembic upgrade head`, the partial unique index exists
    with the expected canonical definition. Substring checks would
    pass for a wrong index (missing UNIQUE, swapped column order,
    missing the payload-key WHERE conjuncts); pin via regex against
    the canonical `pg_indexes.indexdef` form Postgres normalizes to."""
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
            # (review_id first, then phase_id JSONB extraction, then
            # COALESCE(phase_key, ''), then marker JSONB extraction),
            # all three WHERE conjuncts.
            canonical_pattern = re.compile(
                r"CREATE UNIQUE INDEX uq_audit_events_review_phase_natural_key "
                r"ON public\.audit_events "
                r"USING btree "
                r"\(review_id, "
                r"\(\(payload ->> 'phase_id'::text\)\), "
                r"COALESCE\(phase_key, ''::text\), "
                r"\(\(payload ->> 'marker'::text\)\)\) "
                r"WHERE "
                r"\(\(event_type = 'review_phase'::text\) "
                r"AND \(payload \? 'phase_id'::text\) "
                r"AND \(payload \? 'marker'::text\)\)"
            )
            assert canonical_pattern.fullmatch(indexdef), (
                f"indexdef did not match canonical form; got: {indexdef!r}"
            )
    finally:
        await engine.dispose()


async def test_index_enforces_natural_key_uniqueness(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """Two ReviewPhaseEvent-shaped rows with the same
    (review_id, phase_id, phase_key=NULL, marker) tuple — second
    INSERT raises IntegrityError. Pins the load-bearing safety net
    that closes the HITL-resume duplicate-emit gap."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        review_id = uuid4()
        phase_id = "abc123"
        timestamp = datetime.now(UTC)

        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            # First insert succeeds.
            await conn.execute(
                audit_events_table.insert().values(
                    event_id=uuid4(),
                    review_id=review_id,
                    event_type="review_phase",
                    timestamp=timestamp,
                    is_eval=False,
                    phase_key=None,
                    payload={"phase_id": phase_id, "marker": "start", "node_id": "hitl"},
                )
            )

        # Second insert with same natural key but fresh event_id MUST
        # raise IntegrityError on the partial unique index.
        async with engine.begin() as conn:
            with pytest.raises(sa.exc.IntegrityError):
                await conn.execute(
                    audit_events_table.insert().values(
                        event_id=uuid4(),
                        review_id=review_id,
                        event_type="review_phase",
                        timestamp=timestamp,
                        is_eval=False,
                        phase_key=None,
                        payload={
                            "phase_id": phase_id,
                            "marker": "start",
                            "node_id": "hitl",
                        },
                    )
                )
    finally:
        await engine.dispose()


async def test_start_and_end_markers_for_same_phase_do_not_collide(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """`(review_id, phase_id, phase_key=NULL, marker='start')` and
    `(review_id, phase_id, phase_key=NULL, marker='end')` are distinct
    natural keys. Both rows MUST coexist for the start/end pair
    `phase-events-bound-work` invariant to hold."""
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        review_id = uuid4()
        phase_id = "phase-shared"
        timestamp = datetime.now(UTC)

        async with engine.begin() as conn:
            metadata = sa.MetaData()
            await conn.run_sync(
                lambda sync_conn: metadata.reflect(sync_conn, only=["audit_events"])
            )
            audit_events_table = metadata.tables["audit_events"]

            for marker in ("start", "end"):
                await conn.execute(
                    audit_events_table.insert().values(
                        event_id=uuid4(),
                        review_id=review_id,
                        event_type="review_phase",
                        timestamp=timestamp,
                        is_eval=False,
                        phase_key=None,
                        payload={
                            "phase_id": phase_id,
                            "marker": marker,
                            "node_id": "hitl",
                        },
                    )
                )
    finally:
        await engine.dispose()


async def test_index_does_not_restrict_other_event_types(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """The partial WHERE restricts the unique constraint to
    `event_type='review_phase'`. Other event types with the same
    (review_id, payload->>'phase_id', phase_key, payload->>'marker')
    tuple do NOT collide. Realistically no other event type uses
    those exact JSONB keys, but the partial-index restriction is the
    structural guarantee."""
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

            # Two rows with the same (review_id, payload->>'phase_id',
            # phase_key, payload->>'marker') but DIFFERENT event_type:
            # the partial index excludes them entirely.
            for event_type in ("review_phase", "finding"):
                await conn.execute(
                    audit_events_table.insert().values(
                        event_id=uuid4(),
                        review_id=review_id,
                        event_type=event_type,
                        timestamp=timestamp,
                        is_eval=False,
                        phase_key=None,
                        payload={"phase_id": "x", "marker": "start"},
                    )
                )
    finally:
        await engine.dispose()


async def test_downgrade_removes_index_and_upgrade_recreates_it(
    fresh_db: str, alembic_runner: AlembicRunner
) -> None:
    """`alembic downgrade -1` drops the index cleanly. `alembic
    upgrade head` re-creates it (IF NOT EXISTS-guarded so a
    re-run lands on top of a partial state without error)."""
    await alembic_runner("upgrade", "head", fresh_db)
    await alembic_runner("downgrade", PRIOR_REVISION, fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                sa.text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
            assert row.first() is None, "index should be dropped after downgrade"
    finally:
        await engine.dispose()

    # Re-upgrade lands cleanly.
    await alembic_runner("upgrade", "head", fresh_db)

    engine = create_async_engine(fresh_db)
    try:
        async with engine.connect() as conn:
            row = await conn.execute(
                sa.text(
                    "SELECT indexname FROM pg_indexes "
                    "WHERE schemaname = 'public' AND indexname = :name"
                ),
                {"name": INDEX_NAME},
            )
            assert row.scalar_one() == INDEX_NAME, "index should be recreated after upgrade"
    finally:
        await engine.dispose()
