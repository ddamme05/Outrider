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
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.sweep.purge_expired import purge_expired, purge_installation

_INSTALLATION_ID = 12345


async def _seed_expired_review_for_tombstoned_install(engine: AsyncEngine) -> None:
    """Tombstoned installation with one expired review.

    Lets the compose test exercise both sweep entrypoints against real
    data: purge_expired sweeps the expired review, purge_installation
    then hard-deletes the installation row.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install, tombstoned_at, "
                " purge_after_at) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb, "
                " NOW() - INTERVAL '7 days', NOW() - INTERVAL '1 day')"
            ),
            {"id": _INSTALLATION_ID},
        )
        await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'completed', "
                "  NOW() - INTERVAL '1 day'"
                ")"
            ),
            {"id": _INSTALLATION_ID},
        )


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


async def test_concurrent_sweep_skips_despite_purgeable_data_present(
    migrated_db: str,
) -> None:
    """Sweep-B returns {} via the lock even when purgeable data EXISTS.

    Disambiguates the empty dict in the concurrent-sweep test above: there
    both sweeps run against empty data, so sweep-B's {} could be a lock-skip
    OR a no-data skip. Here an expired review is committed before either
    transaction opens (so it is visible to sweep-B's separate connection
    under READ COMMITTED), sweep-A purges it inside the uncommitted
    transaction that holds the lock, and sweep-B still returns {} — proving
    the empty dict is lock-based, not data-based.
    """
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_review_for_tombstoned_install(engine)

        async with engine.connect() as conn1, engine.connect() as conn2:
            tx1 = await conn1.begin()
            try:
                # Sweep A acquires the lock and purges the seeded review
                # (uncommitted in tx1; the xact lock is held until tx1 ends).
                rows_a = await purge_expired(conn1, purge_role="sweep-A")
                assert rows_a == {"reviews": 1}, (
                    f"Sweep A should have purged the seeded expired review; got {rows_a}"
                )

                # Sweep B's transaction still SEES the expired review (tx1's
                # delete is uncommitted under READ COMMITTED), yet cannot
                # acquire the lock, so it short-circuits to {} without
                # scanning — the empty dict is a lock-skip, not no-data.
                tx2 = await conn2.begin()
                try:
                    rows_b = await purge_expired(conn2, purge_role="sweep-B")
                    assert rows_b == {}, (
                        "Sweep B must skip via the advisory lock even though a "
                        f"purgeable expired review exists; got {rows_b}"
                    )
                finally:
                    await tx2.rollback()
            finally:
                await tx1.rollback()
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


async def test_compose_sweeps_purge_real_data_in_one_transaction(
    migrated_db: str,
) -> None:
    """Reentrant compose actually purges real data in one transaction.

    Stronger counterpart to the previous test. Seeds an expired review
    on a tombstoned installation, then runs purge_expired followed by
    purge_installation in the same advisory-locked transaction.

    Asserts both sweeps did real work (not just lock acquisition):
      - purge_expired returns the expired-content row count
      - purge_installation returns empty (content already purged) AND
        the installations row is gone after commit
      - per-table purge_audit row is written by purge_expired
    """
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_review_for_tombstoned_install(engine)

        async with engine.begin() as conn:
            expired_rows = await purge_expired(conn, purge_role="step-1")
            install_rows = await purge_installation(conn, _INSTALLATION_ID, purge_role="step-2")

        assert expired_rows == {"reviews": 1}, (
            f"purge_expired should report the expired review; got {expired_rows}"
        )
        assert install_rows == {}, (
            "content was already swept by purge_expired; purge_installation "
            f"should report no further content rows; got {install_rows}"
        )

        async with engine.connect() as conn:
            review_count = await conn.execute(text("SELECT COUNT(*) FROM reviews"))
            assert review_count.scalar_one() == 0, "expired review must be gone"

            install_count = await conn.execute(
                text("SELECT COUNT(*) FROM installations WHERE installation_id = :id"),
                {"id": _INSTALLATION_ID},
            )
            assert install_count.scalar_one() == 0, (
                "purge_installation must hard-delete the installations row"
            )

            purge_rows = await conn.execute(
                text("SELECT target_table, purge_role FROM purge_audit ORDER BY target_table")
            )
            rows = list(purge_rows)
            assert rows == [("reviews", "step-1")], (
                f"expected one purge_audit row from step-1; got {rows}"
            )
    finally:
        await engine.dispose()
