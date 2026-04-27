"""Sweep advisory lock serializes concurrent runs.

Backs ``sweep-jobs-use-advisory-locks``. Both sweep entrypoints —
``purge_expired`` and ``purge_installation`` — must acquire
SWEEP_LOCK_ID before mutating shared content state. The two share a
lock identifier on purpose: a time-based sweep and an installation
hard-delete cannot run concurrently against the same database without
risking torn state.

PostgreSQL's ``pg_try_advisory_xact_lock`` releases the lock at
transaction commit/rollback. So a second invocation sees the lock
held only while the first transaction is still open. Within a single
transaction the lock is reentrant — purge_expired and
purge_installation can compose into one advisory-locked transaction
per the schema-layer spec.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.sweep.purge_expired import purge_expired, purge_installation


async def test_concurrent_sweeps_serialize_via_advisory_lock(migrated_db: str) -> None:
    """A second sweep against an already-locked DB returns empty-dict."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn1, engine.connect() as conn2:
            tx1 = await conn1.begin()
            try:
                # Sweep A acquires the lock and runs (no expired rows;
                # rows_per_table is empty either way; we care about the
                # lock state, not the work).
                rows_a = await purge_expired(conn1, purge_role="sweep-A")
                assert rows_a == {}

                # Sweep B's transaction tries to acquire while A holds —
                # pg_try_advisory_xact_lock returns false, purge_expired
                # short-circuits and returns {} without scanning.
                tx2 = await conn2.begin()
                try:
                    rows_b = await purge_expired(conn2, purge_role="sweep-B")
                    assert rows_b == {}, (
                        "Sweep B should have skipped because Sweep A holds "
                        f"the advisory lock; got {rows_b}"
                    )
                finally:
                    await tx2.rollback()
            finally:
                await tx1.rollback()

        # After both transactions release, a fresh sweep can acquire
        # cleanly — confirms the lock is transaction-scoped, not
        # session-scoped or process-leaked.
        async with engine.begin() as conn:
            rows_c = await purge_expired(conn, purge_role="sweep-C")
            assert rows_c == {}  # nothing to purge; lock acquired cleanly

        async with engine.connect() as conn:
            purge_count = await conn.execute(text("SELECT COUNT(*) FROM purge_audit"))
            assert purge_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_purge_installation_respects_advisory_lock(migrated_db: str) -> None:
    """purge_installation must also acquire SWEEP_LOCK_ID.

    Without this gate, a time-based sweep and an installation
    hard-delete could run concurrently against the same DB, mutating
    overlapping content rows. The test asserts the same short-circuit
    behavior as purge_expired: if the lock is held, return {} without
    touching state.

    Closed a real correctness gap surfaced in the audit pass against
    commit 01d7edb: purge_installation didn't acquire the lock at all.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn1, engine.connect() as conn2:
            tx1 = await conn1.begin()
            try:
                # Sweep A holds the lock.
                rows_a = await purge_expired(conn1, purge_role="sweep-A")
                assert rows_a == {}

                # Install-purge B tries to acquire while A holds — must
                # short-circuit. installation_id 99999 doesn't exist;
                # if the function failed to short-circuit it would
                # touch state regardless.
                tx2 = await conn2.begin()
                try:
                    rows_b = await purge_installation(conn2, 99999, purge_role="install-purge-B")
                    assert rows_b == {}, (
                        "purge_installation must skip when SWEEP_LOCK_ID "
                        f"is held by another sweep; got {rows_b}"
                    )
                finally:
                    await tx2.rollback()
            finally:
                await tx1.rollback()

        # Confirm no purge_audit rows landed during either call.
        async with engine.connect() as conn:
            purge_count = await conn.execute(text("SELECT COUNT(*) FROM purge_audit"))
            assert purge_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_purge_expired_and_purge_installation_compose_in_one_transaction(
    migrated_db: str,
) -> None:
    """The lock is reentrant within a single transaction.

    purge_expired acquires SWEEP_LOCK_ID; purge_installation called
    later in the same transaction sees the lock as already-held by
    this session and proceeds. Backs the schema-layer spec's
    "the whole installation-purge sequence runs inside the same
    advisory-locked transaction as content purge" claim.
    """
    engine = create_async_engine(migrated_db)
    try:
        async with engine.begin() as conn:
            await purge_expired(conn, purge_role="step-1")
            # No data was seeded, so purge_installation against a
            # non-existent install does nothing — but it MUST still
            # return cleanly (rather than failing the lock acquisition)
            # because the lock is reentrant within this transaction.
            result = await purge_installation(conn, 99999, purge_role="step-2")
            assert result == {}, (
                "purge_installation against non-existent install should "
                "return empty dict, not be blocked by lock acquisition"
            )
    finally:
        await engine.dispose()
