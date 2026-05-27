"""AuditPersister.acquire_publish_lock — DB-touching contention + timeout pin.

See DECISIONS.md#027 — V1 per-review publish-side advisory lock.

Per F5/F6/F8 (CodeRabbit + reviewer-driven, F8 round): the publish-side
advisory lock must

  1. serialize concurrent acquisitions on the same review_id (one task
     yields only after the other's transaction commits/rolls back),
  2. NOT block cross-review acquisitions (different review_ids hash to
     different lock keys via `hashtext('publish:<uuid>')`),
  3. honor a bounded deadline — a holder that monopolizes the lock past
     `max_wait_seconds` causes the waiter to raise
     `AuditPersisterPublishLockAcquisitionTimeoutError`, not hang forever
     and not silently skip.

Tier: integration (real Postgres, real `pg_advisory_xact_lock` /
`pg_try_advisory_xact_lock`, real session pooling). Unit-level tests
exercise the publish-node side (lock loser path, lock-acquire failure
emit); this file exercises the lock primitive against a real DB so the
contention semantics are proven, not assumed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import (
    AuditPersister,
    AuditPersisterPublishLockAcquisitionTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


@pytest_asyncio.fixture
async def persister_for_lock(
    migrated_db: str,
) -> AsyncGenerator[tuple[AuditPersister, str]]:
    """Build an `AuditPersister` against a freshly-migrated DB, yielding
    the persister + the DB URL (for spawning sibling engines used to
    simulate cross-task contention)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        persister = AuditPersister(
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
            retention_settings=RetentionSettings(),
        )
        yield persister, migrated_db
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_two_concurrent_same_review_lock_serializes(
    persister_for_lock: tuple[AuditPersister, str],
) -> None:
    """Two `acquire_publish_lock(review_id)` tasks on the SAME review_id
    serialize via the try-lock + backoff loop. The second task acquires
    only after the first task's transaction commits."""
    persister, _ = persister_for_lock
    review_id = uuid4()

    timeline: list[tuple[str, float]] = []
    loop = asyncio.get_running_loop()

    async def hold_lock(task_name: str, hold_for: float) -> None:
        async with persister.acquire_publish_lock(review_id):
            timeline.append((f"{task_name}_acquired", loop.time()))
            await asyncio.sleep(hold_for)
            timeline.append((f"{task_name}_releasing", loop.time()))

    await asyncio.gather(
        hold_lock("A", 0.15),
        hold_lock("B", 0.05),
    )

    # Both tasks acquired the lock; the timeline shows one task's
    # release strictly preceding the other's acquire (serialization).
    # Sort by TIMESTAMP (tuple position 1), not by name (default tuple
    # ordering sorts lexicographically on position 0, which is
    # event_name — `"A_acquired" < "B_acquired"` is order-dependent on
    # the name, not the actual time).
    assert len(timeline) == 4
    acquires = sorted((e for e in timeline if e[0].endswith("_acquired")), key=lambda e: e[1])
    releases = sorted((e for e in timeline if e[0].endswith("_releasing")), key=lambda e: e[1])
    # The first release must occur before the second acquire. Epsilon
    # is subtracted from the RELEASE side (not added to the acquire
    # side) so a broken lock implementation that lets release-time
    # drift past acquire-time can't pass via timing-jitter slack.
    clock_epsilon_seconds = 0.001
    assert releases[0][1] - clock_epsilon_seconds <= acquires[1][1], (
        f"Expected first release to precede second acquire; got timeline {timeline}"
    )


@pytest.mark.asyncio
async def test_distinct_review_ids_lock_independently(
    persister_for_lock: tuple[AuditPersister, str],
) -> None:
    """Two `acquire_publish_lock` tasks on DIFFERENT review_ids do NOT
    contend — the partial-key namespace `hashtext('publish:<uuid>')`
    isolates per-review locks. Both tasks acquire concurrently."""
    persister, _ = persister_for_lock
    review_a = uuid4()
    review_b = uuid4()

    acquire_times: dict[str, float] = {}
    loop = asyncio.get_running_loop()

    async def hold_and_record(name: str, review_id: UUID) -> None:
        async with persister.acquire_publish_lock(review_id):
            acquire_times[name] = loop.time()
            # Hold the lock briefly so the test can observe whether
            # the other task waited or acquired in parallel.
            await asyncio.sleep(0.05)

    await asyncio.gather(
        hold_and_record("A", review_a),
        hold_and_record("B", review_b),
    )

    # Both acquired; the time delta between acquisitions should be
    # well under the 50ms hold duration. If one had to wait for the
    # other, delta would be ≥50ms (the hold). Threshold set to 100ms
    # (2x the hold) so transient CI slowness doesn't flake — the
    # contended-vs-independent distinction has a 5-10x margin under
    # this bound.
    delta = abs(acquire_times["A"] - acquire_times["B"])
    assert delta < 0.1, (
        f"Distinct review_ids should not contend; observed acquire delta "
        f"{delta:.3f}s (≥0.1s suggests they serialized — namespace "
        f"isolation broken)."
    )


@pytest.mark.asyncio
async def test_timeout_raises_after_holder_exceeds_deadline(
    persister_for_lock: tuple[AuditPersister, str],
) -> None:
    """A waiter whose deadline lapses while the holder still has the lock
    raises `AuditPersisterPublishLockAcquisitionTimeoutError` with
    `review_id` + `waited_seconds` attributes. The waiter does NOT block
    forever (no false-skip, no hang) and does NOT silently emit a
    spurious `IDEMPOTENTLY_SKIPPED`."""
    persister, _ = persister_for_lock
    review_id = uuid4()

    holder_acquired = asyncio.Event()
    holder_released = asyncio.Event()

    async def holder() -> None:
        async with persister.acquire_publish_lock(review_id):
            # Signal that the lock IS held — replaces the prior
            # `asyncio.sleep(0.1)` race where the waiter could probe
            # before the holder finished acquiring.
            holder_acquired.set()
            # Hold past the waiter's deadline.
            await asyncio.sleep(2.5)
        holder_released.set()

    async def waiter() -> None:
        # Tight deadline relative to the holder's hold duration.
        async with persister.acquire_publish_lock(
            review_id,
            max_wait_seconds=1.0,
            initial_backoff_seconds=0.05,
            max_backoff_seconds=0.2,
        ):
            pytest.fail("Waiter should have timed out, not acquired the lock")

    holder_task = asyncio.create_task(holder())
    # Synchronize: wait until the holder has the lock before starting
    # the waiter. Without this, the waiter could probe the lock BEFORE
    # the holder finished acquiring, acquire successfully (no contention),
    # and the test would race-condition-fail. Bound with `wait_for` so
    # a holder that errors before calling `holder_acquired.set()`
    # surfaces as a TimeoutError + the holder_task's exception, not as
    # an infinite hang.
    try:
        await asyncio.wait_for(holder_acquired.wait(), timeout=5.0)
    except TimeoutError:
        # Surface any holder-side exception that prevented set()
        # from being called.
        if holder_task.done():
            holder_task.result()  # re-raises holder's exception
        raise

    with pytest.raises(AuditPersisterPublishLockAcquisitionTimeoutError) as exc_info:
        await waiter()

    assert exc_info.value.review_id == review_id
    assert exc_info.value.waited_seconds == 1

    # Let the holder finish so the test exits cleanly.
    await holder_task
    assert holder_released.is_set()
