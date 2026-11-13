"""Refresh local installation state from GitHub before retention cleanup.

Only confirmed results mark an installation deleted or restore it. Failed or
partial fetches leave local state unchanged. A session lock covers the network
work without holding a database transaction; purge uses only a successful
reconciliation result.

See DECISIONS.md#012, #065, #067 and sweep-jobs-use-advisory-locks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from sqlalchemy import text, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from outrider.db.models.installations import Installation
from outrider.github.authz import list_installation_ids
from outrider.github.installation_policy import (
    INSTALLATION_TOMBSTONE_GRACE as _TOMBSTONE_GRACE,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from outrider.github.credentials import GitHubCredentialProvider

logger = logging.getLogger(__name__)

# SESSION-scoped advisory lock id for the reconcile janitor (`#067`) — distinct from the
# transaction-scoped `SWEEP_LOCK_ID` (0x4F55545244520001) used by the time-based sweeps.
RECONCILE_LOCK_ID: Final[int] = 0x4F55545244520002


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
    """Reconcile once under the `#067` session-scoped advisory lock.

    Lock contention returns a skipped result. Commit the lock probe's implicit
    transaction before the GitHub call; the session lock survives that commit.
    List failures propagate before any cache writes. Successful writes use a
    separate short transaction, then the lock is explicitly released.
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
        # End the probe transaction before network I/O. The session-scoped lock
        # survives commit and remains held until the explicit advisory unlock.
        await lock_conn.commit()
        if not acquired:
            logger.info("reconcile janitor: advisory lock held by another runner; skipping tick")
            return ReconcileResult(skipped_lock_held=True)
        try:
            github_ids = await list_installation_ids(provider)
            return await _apply_reconcile(sessionmaker, github_ids)
        finally:
            # Release the session lock before returning this connection to the pool.
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
