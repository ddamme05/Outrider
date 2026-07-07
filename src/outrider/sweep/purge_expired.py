# See DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted
"""Retention sweep — purge content past its TTL.

Per DECISIONS.md#012/#014: retention purge applies to the four content
tables (reviews, findings, llm_call_content, and — per
specs/2026-06-11-file-hash-analyze-cache.md — analyze_file_cache) only.
audit_events is append-only forever (its trigger blocks DELETE
entirely), and purge_audit is the forensic trail of the purges
themselves.

Two operations:

  - ``purge_expired(conn)`` — time-based sweep: deletes rows whose
    ``retention_expires_at`` has passed. Used by APScheduler /
    sweep/runner.py (when written) every few minutes.
  - ``purge_installation(conn, installation_id)`` — installation
    lifecycle hard-delete after the grace window. Deletes all content
    for the installation, then deletes the installation row itself
    (INSTALLATION_REPOSITORIES cascades via the FK action declared in
    migration 0001).

Both functions take an existing AsyncConnection and DO NOT manage their
own transaction. The caller wraps them in ``async with engine.begin()``
or similar — that boundary is what makes the advisory lock useful and
what makes multi-table purge atomic. If the caller doesn't hold a
transaction, the lock is process-scoped and the multi-table failure
mode is exposed.

Strict deletion order is load-bearing: ``analyze_file_cache`` →
``llm_call_content`` → ``findings`` → ``reviews``. Reversing the order
risks a child-of-reviews CASCADE (``findings`` or ``analyze_file_cache``)
silently dropping rows without writing the per-table purge_audit row.
``test_retention_sweep`` asserts the per-table purge_audit count, which
catches order-reversal bugs.
"""

import logging
from typing import Final

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

# Advisory-lock identifier for the sweep job. PostgreSQL int64; chosen
# arbitrarily within Outrider's lock-id namespace. Changing this value
# would let two sweeps run concurrently against the same database,
# defeating the lock — do not change without a coordinated migration of
# every running scheduler.
SWEEP_LOCK_ID: Final[int] = 0x4F55545244520001

# Strict deletion order — child tables first so the
# `findings.review_id → reviews.id` and
# `analyze_file_cache.source_review_id → reviews.id` CASCADEs don't
# silently drop child rows without a per-table purge_audit row.
# `analyze_file_cache` joined per specs/2026-06-11-file-hash-analyze-cache.md
# (the cache payload is user-code-derived content under the #014 regime).
_RETENTION_TABLES: Final[tuple[str, ...]] = (
    "analyze_file_cache",
    "llm_call_content",
    "findings",
    "reviews",
)

# Reviews in these statuses are NOT eligible for time-based purge even if their
# `retention_expires_at` has passed: a 'running' review hasn't completed (mid-graph;
# deleting it strands LangGraph checkpoints); an 'awaiting_approval' review is the
# HITL-paused state; and an 'awaiting_approval_expired' review is the post-timeout
# REMEDIATION state — still decidable, NOT a dead end: per spec.md ("the remediation
# path is explicit"), `POST /reviews/{id}/decide` accepts decisions on expired reviews
# and publishes immediately, and the resume gate (`mark_running`) admits it. Purging
# any of the three would silently close the human-decision path the HITL gate
# (output boundary #6) guarantees. Operators who need to force-delete stuck reviews
# can use a separate maintenance action; the automated sweep MUST preserve resumable
# state. Child tables (`llm_call_content`, `findings`) follow their own retention TTL
# independently per `DECISIONS.md#012/#014` — purging content while a parent review is
# still active is the documented retention semantics.
_REVIEWS_ACTIVE_STATUSES: Final[tuple[str, ...]] = (
    "running",
    "awaiting_approval",
    "awaiting_approval_expired",
)

# Sentinel installation_id for time-based sweeps that aren't scoped to
# a particular install. purge_audit.installation_id is a loose `bigint`
# reference (no FK), so 0 is safe — installations.installation_id
# values come from GitHub and are positive.
_GLOBAL_SWEEP_INSTALLATION_ID: Final[int] = 0


async def _try_acquire_lock(conn: AsyncConnection) -> bool:
    """Try to acquire the transaction-scoped advisory lock; return True on success."""
    result = await conn.execute(
        text("SELECT pg_try_advisory_xact_lock(:lock_id)"),
        {"lock_id": SWEEP_LOCK_ID},
    )
    return bool(result.scalar_one())


async def _write_purge_audit(
    conn: AsyncConnection,
    *,
    installation_id: int,
    target_table: str,
    rows_affected: int,
    purge_role: str,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO purge_audit "
            "(installation_id, target_table, rows_affected, purge_role) "
            "VALUES (:installation_id, :target_table, :rows_affected, :purge_role)"
        ),
        {
            "installation_id": installation_id,
            "target_table": target_table,
            "rows_affected": rows_affected,
            "purge_role": purge_role,
        },
    )


async def purge_expired(
    conn: AsyncConnection,
    *,
    purge_role: str = "sweep",
) -> dict[str, int]:
    """Time-based retention sweep: delete content past its TTL.

    Acquires the sweep advisory lock first. If another sweep already
    holds it, returns an empty dict without scanning. Otherwise deletes
    in strict order across the four retention tables and writes one
    purge_audit row per target table that had at least one row purged.

    Returns dict of {target_table: rows_affected} for tables that had
    rows. Empty dict means nothing was purged (no expired rows OR lock
    was held by another sweep).
    """
    if not await _try_acquire_lock(conn):
        logger.info("sweep_skipped: advisory lock held by another sweep")
        return {}

    rows_per_table: dict[str, int] = {}

    for table in _RETENTION_TABLES:
        # Table names come from a controlled allow-list (_RETENTION_TABLES);
        # they are not user input. Bandit S608 flagged for the f-string but
        # parameterizing identifiers is not supported in SQL.
        if table == "reviews":
            # Reviews in 'running' or 'awaiting_approval' must not be
            # purged — see `_REVIEWS_ACTIVE_STATUSES` docstring.
            # The active-statuses list is inlined as SQL literals (not
            # bound parameters) because `reviews.status` is a custom
            # `review_status_enum` Postgres type; SQLAlchemy-bound
            # VARCHARs can't compare to the enum without a CAST, which
            # adds complexity for no benefit — the constant is a fixed
            # module-level tuple and there's no injection vector.
            active_sql = ", ".join(f"'{s}'" for s in _REVIEWS_ACTIVE_STATUSES)
            # noqa is on the f-string line per ruff's line-anchoring;
            # table comes from `_RETENTION_TABLES` allowlist; statuses
            # come from `_REVIEWS_ACTIVE_STATUSES` module-level constant.
            sql_text = f"DELETE FROM {table} WHERE retention_expires_at < NOW() AND status NOT IN ({active_sql})"  # noqa: S608, E501
            result = await conn.execute(text(sql_text))
        else:
            result = await conn.execute(
                text(f"DELETE FROM {table} WHERE retention_expires_at < NOW()")  # noqa: S608
            )
        count = result.rowcount or 0
        if count > 0:
            rows_per_table[table] = count

    for table, count in rows_per_table.items():
        await _write_purge_audit(
            conn,
            installation_id=_GLOBAL_SWEEP_INSTALLATION_ID,
            target_table=table,
            rows_affected=count,
            purge_role=purge_role,
        )

    return rows_per_table


async def purge_installation(
    conn: AsyncConnection,
    installation_id: int,
    *,
    purge_role: str = "sweep",
) -> dict[str, int]:
    """Installation lifecycle hard-delete.

    Acquires the same SWEEP_LOCK_ID as purge_expired so the two sweep
    paths cannot race with each other. Returns an empty dict (without
    touching any data) if the lock is held by another sweep — the
    caller can retry on the next scheduled tick.

    Note on idempotency: ``pg_try_advisory_xact_lock`` succeeds if the
    same transaction already holds the lock (transaction-scoped locks
    are reentrant within the holding session). So if the caller already
    acquired the lock via purge_expired earlier in the same transaction,
    this call's acquisition is a no-op success — the documented "single
    advisory-locked transaction" pattern from the schema-layer spec
    composes cleanly.

    Deletes all content rows scoped to this installation in strict
    order (analyze_file_cache → llm_call_content → findings →
    reviews), writes per-table purge_audit rows, then deletes the
    installations row itself AND writes a `target_table="installations"`
    evidence row for that delete — so even a zero-content install leaves
    a forensic record it was purged. installation_repositories cascades
    automatically via the FK action declared in migration 0001.
    purge_audit rows survive the installation hard-delete (loose
    reference, no FK).

    Returns rows_per_table for content tables (does not include the
    installations row itself, which is by definition deleted).
    """
    if not await _try_acquire_lock(conn):
        logger.info(
            "install_purge_skipped: advisory lock held by another sweep (installation_id=%s)",
            installation_id,
        )
        return {}

    rows_per_table: dict[str, int] = {}

    # Deliberately NO `_REVIEWS_ACTIVE_STATUSES` filter here (unlike
    # `purge_expired` above). An installation hard-delete is the
    # revocation path: the user removed the GitHub App, so all of
    # their data must go — including any in-flight `running` or
    # `awaiting_approval` reviews. Preserving those would leave
    # orphan reviews that can never resume (the installation is gone)
    # and never publish (no token available). The asymmetry between
    # the two functions is intentional: time-based sweep protects
    # active reviews from TTL deletion; installation-purge terminates
    # everything by definition.
    for table in _RETENTION_TABLES:
        result = await conn.execute(
            text(f"DELETE FROM {table} WHERE installation_id = :id"),  # noqa: S608
            {"id": installation_id},
        )
        count = result.rowcount or 0
        if count > 0:
            rows_per_table[table] = count

    for table, count in rows_per_table.items():
        await _write_purge_audit(
            conn,
            installation_id=installation_id,
            target_table=table,
            rows_affected=count,
            purge_role=purge_role,
        )

    install_deleted = await conn.execute(
        text("DELETE FROM installations WHERE installation_id = :id"),
        {"id": installation_id},
    )

    # Always audit the install-row hard-delete itself, even when the install had ZERO
    # content rows (the per-table loop above wrote no audit). Without this a zero-content
    # install is hard-deleted with NO purge_audit evidence, so #012 forensics ("did the
    # uninstall purge complete for X?") can't be answered. target_table="installations".
    deleted_count = install_deleted.rowcount or 0
    if deleted_count > 0:
        await _write_purge_audit(
            conn,
            installation_id=installation_id,
            target_table="installations",
            rows_affected=deleted_count,
            purge_role=purge_role,
        )

    return rows_per_table


async def purge_expired_installations(
    conn: AsyncConnection,
    *,
    purge_role: str = "install-purge",
) -> dict[str, dict[str, int]]:
    """Hard-delete installations whose #012 grace window has expired.

    Arc B2 fix (DECISIONS.md#065 / #012): `purge_installation` existed but had NO
    scheduled caller, so tombstoned installs were never actually hard-deleted and
    #012 retention silently never completed on uninstall. This selects the expired
    tombstones (`tombstoned_at IS NOT NULL AND purge_after_at < NOW()` — BOTH
    predicates: the `tombstoned_at IS NOT NULL` guard prevents a stray
    `purge_after_at` on a live/reinstalled install from deleting live data) and calls
    `purge_installation` for each.

    `purge_role` defaults to ``"install-purge"`` — DISTINCT from the time-based sweep's
    ``"sweep"`` — so #012 lifecycle hard-deletes are separable from routine TTL purges
    in the `purge_audit` forensic trail (the exact question #012 auditing asks).

    Acquires the sweep advisory lock FIRST, like its siblings (`purge_expired`,
    `purge_installation`): if another sweep holds it, returns `{}` WITHOUT scanning —
    rather than running the SELECT and returning `{install_id: {}}` per due install,
    which would be indistinguishable from "purged, zero content rows." In the
    `run_all_sweeps` path the lock is already held by this tick's transaction, so the
    acquire is a reentrant no-op success. Returns `{installation_id: rows_per_table}`
    for each purged install (empty if none are due OR the lock was held elsewhere).
    """
    if not await _try_acquire_lock(conn):
        logger.info("install_purge_skipped: advisory lock held by another sweep")
        return {}

    # `FOR UPDATE` row-locks each due install for this transaction, closing a TOCTOU
    # window the advisory lock does NOT cover: the advisory lock serializes sweep-vs-sweep,
    # but a NON-sweep writer (a future reinstall webhook clearing `tombstoned_at`) is not a
    # sweep and never holds SWEEP_LOCK_ID. Without the row lock, such a writer could commit
    # a tombstone-clear between this SELECT and the per-install content DELETE, so live data
    # of a just-reinstalled install would be purged. With it, that writer either commits
    # BEFORE the SELECT (excluded by `tombstoned_at IS NOT NULL`) or blocks until this purge
    # commits and then no-ops (the row is gone). Belt-and-suspenders today (no such writer
    # exists yet); required before the reinstall handler lands.
    due = await conn.execute(
        text(
            "SELECT installation_id FROM installations "
            "WHERE tombstoned_at IS NOT NULL AND purge_after_at < NOW() "
            "FOR UPDATE"
        )
    )
    purged: dict[str, dict[str, int]] = {}
    for (installation_id,) in due.fetchall():
        purged[str(installation_id)] = await purge_installation(
            conn, installation_id, purge_role=purge_role
        )
    return purged
