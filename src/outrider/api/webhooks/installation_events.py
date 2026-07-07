# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Install-event dispatch handlers (Arc B2, `DECISIONS.md#012` + `#065`).

Under `#065` the local `installations` / `installation_repositories` tables are a **cache**,
not the authorization authority — GitHub is checked LIVE at intake/publish. These handlers
maintain cache / retention / display state via simple, idempotent upserts; they NEVER make a
security decision, so there is no ordering clock, no reconcile fence, no per-event fail-closed
state machine (all dissolved by `#065` — spec `2026-07-06-b2-installation-lifecycle.md` lines
149-151). If a hint drifts, it doesn't matter: the authority is the live check.

Contract: each handler takes an already-parsed payload + an `AsyncSession` and issues its
UPDATE/INSERT statements. The CALLER (webhook router) owns the transaction (`session.begin()`)
and the commit — mirroring the `pull_request` review-insert path. Signature verification has
already run in the router before dispatch, so these run only on authentic deliveries.

`action` is allowlisted here (unknown -> `{"status": "ignored", "reason": "action"}`, a 2xx
no-op so GitHub doesn't retry) rather than pinned in the schema `Literal`, per the spec's
round-12 finding (only `created` / `installation_repositories.added` are doc-pinned).

Deferred to the reconcile janitor (not this slice): the `all->selected` transition reconcile
(non-load-bearing under `#065`'s live check — FUP-225 note) and live-confirmed restore. On
`installation.created` the tombstone is cleared OPTIMISTICALLY (a reinstall reactivates); the
janitor is the `#012` restore backstop — it re-tombstones any install GitHub no longer lists in
`GET /app/installations`, so a redelivered stale `created` cannot permanently cancel a purge.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from outrider.db.models.installations import Installation, InstallationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from outrider.api.webhooks.schemas import (
        InstallationEventPayload,
        InstallationRepositoriesEventPayload,
        InstallationRepositoryRef,
        WebhookInstallationDetail,
    )

logger = logging.getLogger(__name__)

# `#012` grace window: how long after an uninstall (`installation.deleted`) before the
# hard-delete purge runs. `#012` makes this an operator-tunable config default (captured in
# ITERATION_LOG, not the decision text); 30 days is an accidental-uninstall recovery window,
# privacy-favoring (shorter than the 90-day content retention). FUP-226 tracks moving it to
# `pydantic-settings` so operators can override without a code change (per `#012`).
_TOMBSTONE_GRACE: Final[timedelta] = timedelta(days=30)

# Handler-side action allowlists (unknown -> 2xx no-op; see module docstring).
_INSTALLATION_ACTIONS: Final[frozenset[str]] = frozenset(
    {"created", "deleted", "suspend", "unsuspend"}
)
_INSTALLATION_REPO_ACTIONS: Final[frozenset[str]] = frozenset({"added", "removed"})

_IGNORED_ACTION: Final[dict[str, str]] = {"status": "ignored", "reason": "action"}


async def handle_installation_event(
    payload: InstallationEventPayload, session: AsyncSession
) -> dict[str, str]:
    """Dispatch an `installation` event (created / deleted / suspend / unsuspend).

    Cache-hint upserts only (`#065`). Returns a 2xx-shaped status dict; an unknown action
    no-ops. The caller owns the transaction.
    """
    action = payload.action
    if action not in _INSTALLATION_ACTIONS:
        return _IGNORED_ACTION
    inst = payload.installation
    if action == "created":
        await _upsert_installation(session, inst, payload.repositories)
    elif action == "deleted":
        await _tombstone_installation(session, inst.id)
    elif action == "suspend":
        await _set_suspended(session, inst.id, suspended=True)
    else:  # unsuspend
        await _set_suspended(session, inst.id, suspended=False)
    return {"status": "ok", "action": action}


async def handle_installation_repositories_event(
    payload: InstallationRepositoriesEventPayload, session: AsyncSession
) -> dict[str, str]:
    """Dispatch an `installation_repositories` event (added / removed).

    Re-persists `repository_selection`, then maintains the per-repo membership set with re-add
    semantics (`removed_at = NULL` on add, `now()` on remove). Cache-hint only (`#065`); the
    `all->selected` reconcile is deferred to the janitor. The caller owns the transaction.
    """
    action = payload.action
    if action not in _INSTALLATION_REPO_ACTIONS:
        return _IGNORED_ACTION
    inst_id = payload.installation.id
    # Every event re-persists repository_selection (spec line 96) — a no-op if unchanged.
    await session.execute(
        update(Installation)
        .where(Installation.installation_id == inst_id)
        .values(repository_selection=payload.repository_selection)
    )
    if action == "added":
        for repo in payload.repositories_added:
            await _add_repo_membership(session, inst_id, repo)
    else:  # removed
        for repo in payload.repositories_removed:
            await _remove_repo_membership(session, inst_id, repo.id)
    return {"status": "ok", "action": action}


async def _upsert_installation(
    session: AsyncSession,
    inst: WebhookInstallationDetail,
    repositories: tuple[InstallationRepositoryRef, ...],
) -> None:
    """Upsert the `installations` cache row on `created` and OPTIMISTICALLY clear any tombstone
    (a reinstall reactivates; the janitor re-tombstones a still-gone install — see module doc).

    For a `selected` install, seed `installation_repositories` from the payload's granted repos
    (`all` never enumerates the unbounded list — the gate reads the install itself)."""
    values: dict[str, Any] = {
        "installation_id": inst.id,
        "app_slug": inst.app_slug,
        "account_id": inst.account.id,
        "account_login": inst.account.login,
        "account_type": inst.account.type,
        "permissions_at_install": inst.permissions,
        "repository_selection": inst.repository_selection,
        "suspended_at": inst.suspended_at,
        "tombstoned_at": None,
        "purge_after_at": None,
    }
    stmt = pg_insert(Installation).values(**values)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Installation.installation_id],
        set_={
            "app_slug": stmt.excluded.app_slug,
            "account_id": stmt.excluded.account_id,
            "account_login": stmt.excluded.account_login,
            "account_type": stmt.excluded.account_type,
            "permissions_at_install": stmt.excluded.permissions_at_install,
            "repository_selection": stmt.excluded.repository_selection,
            "suspended_at": stmt.excluded.suspended_at,
            # Optimistic restore: a reinstall clears the tombstone. The janitor is the #012
            # backstop (re-tombstones an install GitHub no longer lists), so a redelivered
            # stale `created` cannot permanently cancel a legitimate retention purge.
            "tombstoned_at": None,
            "purge_after_at": None,
        },
    )
    await session.execute(stmt)
    if inst.repository_selection == "selected":
        for repo in repositories:
            await _add_repo_membership(session, inst.id, repo)


async def _tombstone_installation(session: AsyncSession, installation_id: int) -> None:
    """Tombstone on `deleted` — set `tombstoned_at` + a grace `purge_after_at`. Idempotent: the
    `tombstoned_at IS NULL` guard makes a duplicate `deleted` a no-op that preserves the ORIGINAL
    grace deadline (a re-fired delete must not extend the retention window)."""
    now = datetime.now(UTC)
    await session.execute(
        update(Installation)
        .where(
            Installation.installation_id == installation_id,
            Installation.tombstoned_at.is_(None),
        )
        .values(tombstoned_at=now, purge_after_at=now + _TOMBSTONE_GRACE)
    )


async def _set_suspended(session: AsyncSession, installation_id: int, *, suspended: bool) -> None:
    """`suspend` -> set `suspended_at` (guarded on `IS NULL` so a re-fired suspend preserves the
    original suspension time); `unsuspend` -> clear it unconditionally."""
    if suspended:
        await session.execute(
            update(Installation)
            .where(
                Installation.installation_id == installation_id,
                Installation.suspended_at.is_(None),
            )
            .values(suspended_at=datetime.now(UTC))
        )
    else:
        await session.execute(
            update(Installation)
            .where(Installation.installation_id == installation_id)
            .values(suspended_at=None)
        )


async def _add_repo_membership(
    session: AsyncSession, installation_id: int, repo: InstallationRepositoryRef
) -> None:
    """Upsert an `installation_repositories` row with re-add semantics: `ON CONFLICT
    (installation_id, repo_id) DO UPDATE SET removed_at = NULL` restores a prior soft-remove and
    is idempotent under redelivery. A plain INSERT would UNIQUE-violate on re-add."""
    now = datetime.now(UTC)
    stmt = pg_insert(InstallationRepository).values(
        installation_id=installation_id,
        repo_id=repo.id,
        repo_full_name=repo.full_name,
        added_at=now,
        removed_at=None,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_installation_repo",
        set_={
            "repo_full_name": stmt.excluded.repo_full_name,
            "added_at": stmt.excluded.added_at,
            "removed_at": None,
        },
    )
    await session.execute(stmt)


async def _remove_repo_membership(
    session: AsyncSession, installation_id: int, repo_id: int
) -> None:
    """Soft-remove: set `removed_at = now()` for the `(installation_id, repo_id)` row. A remove
    of an absent/already-removed row is a harmless no-op (the WHERE simply matches nothing)."""
    await session.execute(
        update(InstallationRepository)
        .where(
            InstallationRepository.installation_id == installation_id,
            InstallationRepository.repo_id == repo_id,
        )
        .values(removed_at=datetime.now(UTC))
    )
