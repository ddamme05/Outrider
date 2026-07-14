# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
# See DECISIONS.md#067 (session-scoped advisory lock exception to sweep-jobs-use-advisory-locks)
"""Reconcile janitor: sync the local install cache against GitHub's authoritative list.

Per `#065` the local `installations` table is a CACHE; webhook events can be MISSED (lost /
out-of-order / delivered during downtime), so this periodic janitor is the retention safety net
(`#012`). It calls `GET /app/installations` (App-JWT — GitHub is the authority) and reconciles:

  - a local install GitHub NO LONGER lists  → TOMBSTONE it (a missed `installation.deleted`, so its
    data still purges after the grace window — otherwise `#012` retention silently never completes).
  - a TOMBSTONED local install GitHub DOES list → CLEAR the tombstone (LIVE-CONFIRMED restore — the
    counterpart to the intake `created` handler NOT clearing tombstones blindly, which would let a
    stale redelivered `created` cancel a purge).

Single-runner across replicas via a SESSION-scoped advisory lock (`#067`) held across the network
call: `pg_try_advisory_lock` on a dedicated connection, `pg_advisory_unlock` in a `finally`. The
transaction-scoped `SWEEP_LOCK_ID` form the time-based sweeps use cannot span the GitHub call, which
is exactly the `#067` exception. Idempotent: a skipped tick (lock held) or a re-run reconciles the
same way. `list_installation_ids` RAISES on any failure, so the janitor NEVER reconciles against a
partial or empty-by-error list (which would wrongly tombstone live installs).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from outrider.db.models.installations import Installation
from outrider.github.authz import list_installation_ids

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from outrider.github.credentials import GitHubCredentialProvider

logger = logging.getLogger(__name__)

# SESSION-scoped advisory lock id for the reconcile janitor (`#067`) — distinct from the
# transaction-scoped `SWEEP_LOCK_ID` (0x4F55545244520001) used by the time-based sweeps.
RECONCILE_LOCK_ID: Final[int] = 0x4F55545244520002

# `#012` tombstone grace window. Duplicated from
# `api/webhooks/installation_events._TOMBSTONE_GRACE` (the webhook `deleted` handler) to avoid a
# sweep→webhook layering dependency; the two MUST stay equal. FUP-226 unifies both into a single
# `pydantic-settings` value, retiring this duplication.
_TOMBSTONE_GRACE: Final[timedelta] = timedelta(days=30)


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of one reconcile tick."""

    skipped_lock_held: bool = False
    tombstoned: int = 0
    restored: int = 0


async def reconcile_installations(
    engine: AsyncEngine,
    provider: GitHubCredentialProvider,
) -> ReconcileResult:
    """Run one reconcile tick under the `#067` session-scoped advisory lock.

    Returns counts; a no-op `ReconcileResult(skipped_lock_held=True)` if another runner holds the
    lock. The GitHub list runs OUTSIDE any write transaction (the `#067` rationale) and raises on
    failure — a failed tick aborts WITHOUT reconciling, never tombstoning against partial data.

    The lock lives on a DEDICATED connection: `pg_try_advisory_lock` acquires a SESSION-scoped lock,
    then we COMMIT immediately so no transaction stays open across the `GET /app/installations` call
    (a lingering txn would pin the connection through the network stall — the exact thing `#067`'s
    session-scoped form exists to avoid). The lock persists on the connection across the commit +
    the network call; the reconcile WRITES run on a separate short-lived session/transaction.
    """
    sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
        engine, expire_on_commit=False
    )
    async with engine.connect() as lock_conn:
        acquired = (
            await lock_conn.execute(
                text("SELECT pg_try_advisory_lock(:lock_id)"),
                {"lock_id": RECONCILE_LOCK_ID},
            )
        ).scalar_one()
        # Commit the implicit (autobegin) transaction the SELECT opened so it does NOT stay open
        # across the network call. The session-scoped lock is on the connection, not the txn, so
        # it survives the commit (and the network call, and until pg_advisory_unlock / conn close).
        await lock_conn.commit()
        if not acquired:
            logger.info("reconcile janitor: advisory lock held by another runner; skipping tick")
            return ReconcileResult(skipped_lock_held=True)
        try:
            github_ids = await list_installation_ids(provider)
            return await _apply_reconcile(sessionmaker, github_ids)
        finally:
            # Release the session-scoped lock explicitly (connection close is the crash backstop).
            await lock_conn.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"),
                {"lock_id": RECONCILE_LOCK_ID},
            )
            await lock_conn.commit()


async def _apply_reconcile(
    sessionmaker: async_sessionmaker[AsyncSession], github_ids: set[int]
) -> ReconcileResult:
    """Apply the two reconcile writes in one short transaction (the lock is held across it)."""
    now = datetime.now(UTC)
    async with sessionmaker() as session, session.begin():
        # Tombstone installs GitHub no longer lists (missed `installation.deleted`). The
        # `tombstoned_at IS NULL` guard is idempotent and preserves an existing grace deadline.
        # An empty `github_ids` (GitHub lists none — every install uninstalled) tombstones every
        # non-tombstoned local install, which is correct: `NOT IN ()` matches all.
        tombstone_result = await session.execute(
            update(Installation)
            .where(
                Installation.tombstoned_at.is_(None),
                Installation.installation_id.notin_(github_ids),
            )
            .values(tombstoned_at=now, purge_after_at=now + _TOMBSTONE_GRACE)
        )
        tombstoned = getattr(tombstone_result, "rowcount", 0) or 0

        # Live-confirmed restore: clear the tombstone on installs GitHub DOES list.
        restore_result = await session.execute(
            update(Installation)
            .where(
                Installation.tombstoned_at.is_not(None),
                Installation.installation_id.in_(github_ids),
            )
            .values(tombstoned_at=None, purge_after_at=None)
        )
        restored = getattr(restore_result, "rowcount", 0) or 0

    if tombstoned and not github_ids:
        logger.warning(
            "reconcile janitor: GitHub lists ZERO installs; tombstoned %d local install(s) — "
            "expected only if the App was fully uninstalled",
            tombstoned,
        )
    logger.info("reconcile janitor: tombstoned=%d restored=%d", tombstoned, restored)
    return ReconcileResult(tombstoned=tombstoned, restored=restored)
