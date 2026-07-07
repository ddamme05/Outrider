"""Install-event dispatch handlers against a real DB (Arc B2, DECISIONS.md#012 + #065).

Verifies the cache-hint upsert semantics: `installation` created/deleted/suspend/unsuspend and
`installation_repositories` added/removed, plus idempotency under webhook redelivery and the
re-add (`removed_at -> NULL`) semantics. The handlers take a session; these tests own the
transaction (as the router does) and query on a fresh connection to assert.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.webhooks.installation_events import (
    handle_installation_event,
    handle_installation_repositories_event,
)
from outrider.api.webhooks.schemas import (
    InstallationEventPayload,
    InstallationRepositoriesEventPayload,
    InstallationRepositoryRef,
    WebhookAccount,
    WebhookInstallationDetail,
)

_INSTALLATION_ID = 12345
_REPO_ID = 100
_REPO_ID_2 = 200


def _installation(*, repository_selection: str = "selected") -> WebhookInstallationDetail:
    return WebhookInstallationDetail(
        id=_INSTALLATION_ID,
        account=WebhookAccount(id=1, login="octocat", type="User"),
        app_slug="test-app",
        repository_selection=repository_selection,
        permissions={"contents": "read", "pull_requests": "write"},
        suspended_at=None,
    )


def _installation_event(
    action: str,
    *,
    repository_selection: str = "selected",
    repositories: tuple[InstallationRepositoryRef, ...] = (),
) -> InstallationEventPayload:
    return InstallationEventPayload(
        action=action,
        installation=_installation(repository_selection=repository_selection),
        repositories=repositories,
    )


def _repos_event(
    action: str,
    *,
    added: tuple[InstallationRepositoryRef, ...] = (),
    removed: tuple[InstallationRepositoryRef, ...] = (),
    repository_selection: str = "selected",
) -> InstallationRepositoriesEventPayload:
    return InstallationRepositoriesEventPayload(
        action=action,
        installation=_installation(),
        repository_selection=repository_selection,
        repositories_added=added,
        repositories_removed=removed,
    )


_REPO = InstallationRepositoryRef(id=_REPO_ID, full_name="octocat/test-repo")
_REPO_2 = InstallationRepositoryRef(id=_REPO_ID_2, full_name="octocat/other-repo")


async def test_created_upserts_install_and_seeds_selected_repos(migrated_db: str) -> None:
    """`installation.created` (selected) upserts the install row and seeds the granted repos."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            result = await handle_installation_event(
                _installation_event("created", repositories=(_REPO,)), session
            )
        assert result == {"status": "ok", "action": "created"}

        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT app_slug, account_login, account_type, repository_selection, "
                        "tombstoned_at, suspended_at FROM installations "
                        "WHERE installation_id = :id"
                    ),
                    {"id": _INSTALLATION_ID},
                )
            ).one()
            assert row.app_slug == "test-app"
            assert row.account_login == "octocat"
            assert row.repository_selection == "selected"
            assert row.tombstoned_at is None
            assert row.suspended_at is None
            member_count = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) FROM installation_repositories "
                        "WHERE installation_id = :id AND removed_at IS NULL"
                    ),
                    {"id": _INSTALLATION_ID},
                )
            ).scalar_one()
            assert member_count == 1
    finally:
        await engine.dispose()


async def test_created_is_idempotent_under_redelivery(migrated_db: str) -> None:
    """A redelivered `created` produces the same single install row + no duplicate membership."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        for _ in range(2):
            async with sessionmaker() as session, session.begin():
                await handle_installation_event(
                    _installation_event("created", repositories=(_REPO,)), session
                )
        async with engine.connect() as conn:
            installs = (await conn.execute(text("SELECT COUNT(*) FROM installations"))).scalar_one()
            members = (
                await conn.execute(text("SELECT COUNT(*) FROM installation_repositories"))
            ).scalar_one()
        assert installs == 1
        assert members == 1
    finally:
        await engine.dispose()


async def test_created_clears_tombstone_optimistic_restore(migrated_db: str) -> None:
    """A `created` on a currently-tombstoned install clears the tombstone (optimistic restore;
    the janitor re-tombstones a still-gone install — DECISIONS.md#012/#065)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO installations (installation_id, app_slug, account_id, "
                    "account_login, account_type, permissions_at_install, tombstoned_at, "
                    "purge_after_at) VALUES (:id, 'old', 1, 'octocat', 'User', '{}'::jsonb, "
                    "NOW(), NOW() + INTERVAL '30 days')"
                ),
                {"id": _INSTALLATION_ID},
            )
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("created"), session)
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT tombstoned_at, purge_after_at, app_slug FROM installations "
                        "WHERE installation_id = :id"
                    ),
                    {"id": _INSTALLATION_ID},
                )
            ).one()
        assert row.tombstoned_at is None
        assert row.purge_after_at is None
        assert row.app_slug == "test-app"  # non-lifecycle fields refreshed too
    finally:
        await engine.dispose()


async def test_deleted_tombstones_and_is_idempotent(migrated_db: str) -> None:
    """`deleted` tombstones + sets a grace deadline; a duplicate `deleted` no-ops and PRESERVES
    the original deadline (the `tombstoned_at IS NULL` guard)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("created"), session)
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("deleted"), session)
        async with engine.connect() as conn:
            first = (
                await conn.execute(
                    text(
                        "SELECT tombstoned_at, purge_after_at FROM installations "
                        "WHERE installation_id = :id"
                    ),
                    {"id": _INSTALLATION_ID},
                )
            ).one()
        assert first.tombstoned_at is not None
        assert first.purge_after_at is not None
        # Second delete must not move the deadline.
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("deleted"), session)
        async with engine.connect() as conn:
            second = (
                await conn.execute(
                    text("SELECT purge_after_at FROM installations WHERE installation_id = :id"),
                    {"id": _INSTALLATION_ID},
                )
            ).one()
        assert second.purge_after_at == first.purge_after_at
    finally:
        await engine.dispose()


async def test_suspend_then_unsuspend(migrated_db: str) -> None:
    """`suspend` sets `suspended_at` (preserved under re-suspend); `unsuspend` clears it."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("created"), session)
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("suspend"), session)
        async with engine.connect() as conn:
            first_suspend = (
                await conn.execute(
                    text("SELECT suspended_at FROM installations WHERE installation_id = :id"),
                    {"id": _INSTALLATION_ID},
                )
            ).scalar_one()
        assert first_suspend is not None
        # Re-suspend preserves the original timestamp.
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("suspend"), session)
        async with engine.connect() as conn:
            re_suspend = (
                await conn.execute(
                    text("SELECT suspended_at FROM installations WHERE installation_id = :id"),
                    {"id": _INSTALLATION_ID},
                )
            ).scalar_one()
        assert re_suspend == first_suspend
        # Unsuspend clears it.
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("unsuspend"), session)
        async with engine.connect() as conn:
            cleared = (
                await conn.execute(
                    text("SELECT suspended_at FROM installations WHERE installation_id = :id"),
                    {"id": _INSTALLATION_ID},
                )
            ).scalar_one()
        assert cleared is None
    finally:
        await engine.dispose()


async def test_repositories_removed_then_readded_clears_removed_at(migrated_db: str) -> None:
    """`installation_repositories`: a repo removed (soft-delete `removed_at`) then re-added has
    its `removed_at` cleared back to NULL (re-add semantics), still one row."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(
                _installation_event("created", repositories=(_REPO,)), session
            )
        # Remove the repo.
        async with sessionmaker() as session, session.begin():
            await handle_installation_repositories_event(
                _repos_event("removed", removed=(_REPO,)), session
            )
        async with engine.connect() as conn:
            removed_at = (
                await conn.execute(
                    text(
                        "SELECT removed_at FROM installation_repositories "
                        "WHERE installation_id = :id AND repo_id = :repo"
                    ),
                    {"id": _INSTALLATION_ID, "repo": _REPO_ID},
                )
            ).scalar_one()
        assert removed_at is not None
        # Re-add the same repo.
        async with sessionmaker() as session, session.begin():
            await handle_installation_repositories_event(
                _repos_event("added", added=(_REPO,)), session
            )
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT removed_at, COUNT(*) OVER () AS n FROM installation_repositories "
                        "WHERE installation_id = :id AND repo_id = :repo"
                    ),
                    {"id": _INSTALLATION_ID, "repo": _REPO_ID},
                )
            ).one()
        assert row.removed_at is None  # re-add cleared the soft-delete
        assert row.n == 1  # still one row, not a duplicate
    finally:
        await engine.dispose()


async def test_repositories_added_new_repo(migrated_db: str) -> None:
    """`installation_repositories.added` upserts a brand-new membership row."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(_installation_event("created"), session)
        async with sessionmaker() as session, session.begin():
            result = await handle_installation_repositories_event(
                _repos_event("added", added=(_REPO_2,)), session
            )
        assert result == {"status": "ok", "action": "added"}
        async with engine.connect() as conn:
            active = (
                await conn.execute(
                    text(
                        "SELECT COUNT(*) FROM installation_repositories "
                        "WHERE installation_id = :id AND repo_id = :repo AND removed_at IS NULL"
                    ),
                    {"id": _INSTALLATION_ID, "repo": _REPO_ID_2},
                )
            ).scalar_one()
        assert active == 1
    finally:
        await engine.dispose()


async def test_unknown_action_is_ignored_no_write(migrated_db: str) -> None:
    """An unknown `installation` action no-ops (2xx `ignored`) and writes nothing."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            result = await handle_installation_event(
                _installation_event("new_permissions_accepted"), session
            )
        assert result == {"status": "ignored", "reason": "action"}
        async with engine.connect() as conn:
            count = (await conn.execute(text("SELECT COUNT(*) FROM installations"))).scalar_one()
        assert count == 0
    finally:
        await engine.dispose()


async def test_created_all_selection_does_not_enumerate_repos(migrated_db: str) -> None:
    """An `all`-selection install does NOT seed per-repo rows (the gate reads the install)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with sessionmaker() as session, session.begin():
            await handle_installation_event(
                _installation_event("created", repository_selection="all", repositories=(_REPO,)),
                session,
            )
        async with engine.connect() as conn:
            members = (
                await conn.execute(text("SELECT COUNT(*) FROM installation_repositories"))
            ).scalar_one()
            selection = (
                await conn.execute(
                    text(
                        "SELECT repository_selection FROM installations WHERE installation_id = :id"
                    ),
                    {"id": _INSTALLATION_ID},
                )
            ).scalar_one()
        assert members == 0  # all-selection never enumerates
        assert selection == "all"
    finally:
        await engine.dispose()
