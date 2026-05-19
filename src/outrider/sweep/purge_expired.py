# See DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted
"""Retention sweep — purge content past its TTL.

Per DECISIONS.md#012/#014: retention purge applies to the three content
tables (reviews, findings, llm_call_content) only. audit_events is
append-only forever (its trigger blocks DELETE entirely), and
purge_audit is the forensic trail of the purges themselves.

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

Strict deletion order is load-bearing: ``llm_call_content`` → ``findings``
→ ``reviews``. Reversing the order risks the ``findings → reviews``
CASCADE silently dropping findings without writing the per-table
purge_audit row. ``test_retention_sweep`` asserts the per-table
purge_audit count, which catches order-reversal bugs.
"""

import logging
from typing import Final

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

logger = logging.getLogger(__name__)

# Advisory-lock identifier for the sweep job. PostgreSQL int64; chosen
# arbitrarily within Outrider's lock-id namespace. Changing this value
# would let two sweeps run concurrently against the same database,
# defeating the lock — do not change without a coordinated migration of
# every running scheduler.
SWEEP_LOCK_ID: Final[int] = 0x4F55545244520001

# Strict deletion order — child tables first so the
# `findings.review_id → reviews.id` CASCADE doesn't silently drop
# findings without a per-table purge_audit row.
_RETENTION_TABLES: Final[tuple[str, ...]] = (
    "llm_call_content",
    "findings",
    "reviews",
)

# Reviews in these statuses are NOT eligible for time-based purge even
# if their `retention_expires_at` has passed: a 'running' review hasn't
# completed (mid-graph; deleting it strands LangGraph checkpoints), and
# an 'awaiting_approval' review is the HITL-paused state — purging
# would prevent the human-decision resume path entirely. Operators who
# need to force-delete stuck reviews can use a separate maintenance
# action; the automated sweep MUST preserve active state. Child tables
# (`llm_call_content`, `findings`) follow their own retention TTL
# independently per `DECISIONS.md#012/#014` — purging content while a
# parent review is still active is the documented retention semantics.
_REVIEWS_ACTIVE_STATUSES: Final[tuple[str, ...]] = ("running", "awaiting_approval")

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
    in strict order across the three retention tables and writes one
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
            sql = (
                f"DELETE FROM {table} WHERE retention_expires_at < NOW() "  # noqa: S608
                f"AND status NOT IN :active_statuses"
            )
            active = list(_REVIEWS_ACTIVE_STATUSES)
            result = await conn.execute(
                text(sql).bindparams(bindparam("active_statuses", expanding=True, value=active))
            )
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
    order (llm_call_content → findings → reviews), writes per-table
    purge_audit rows, then deletes the installations row itself.
    installation_repositories cascades automatically via the FK action
    declared in migration 0001. purge_audit rows survive the
    installation hard-delete (loose reference, no FK).

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

    await conn.execute(
        text("DELETE FROM installations WHERE installation_id = :id"),
        {"id": installation_id},
    )

    return rows_per_table
