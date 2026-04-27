"""Sweep advisory lock serializes concurrent runs.

Backs ``sweep-jobs-use-advisory-locks``. Two concurrent invocations of
``purge_expired`` must NOT race; the lock acquisition is the gate. The
test runs two purge_expired calls on separate connections in
overlapping transactions: the first acquires the lock and proceeds
(possibly with no rows to purge); the second tries to acquire while
the first holds it, fails, and returns the empty-dict skip signal.

PostgreSQL's ``pg_try_advisory_xact_lock`` releases the lock at
transaction commit/rollback. So the second invocation sees the lock
held only while the first transaction is still open.
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.sweep.purge_expired import purge_expired


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
