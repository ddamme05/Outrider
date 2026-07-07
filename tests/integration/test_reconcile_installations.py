"""Reconcile janitor against a real DB (Arc B2, DECISIONS.md#065 / #012 / #067).

Verifies: tombstone a local install GitHub no longer lists (missed delete), live-confirmed restore
of a tombstoned install GitHub DOES list, the #067 session-scoped advisory-lock single-runner skip,
idempotent grace-deadline preservation, empty-list mass-tombstone, and list-failure abort.
The GitHub list is monkeypatched (no network); the reconcile SQL runs against Postgres.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import outrider.sweep.reconcile_installations as reconcile_mod
from outrider.sweep.reconcile_installations import RECONCILE_LOCK_ID, reconcile_installations

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

_ID_A = 111
_ID_B = 222
_SETTINGS = SimpleNamespace()  # ignored — list_installation_ids is monkeypatched


def _fake_list(ids: set[int]) -> Callable[..., Coroutine[object, object, set[int]]]:
    async def _list(_settings: object) -> set[int]:
        return ids

    return _list


def _raising_list() -> Callable[..., Coroutine[object, object, set[int]]]:
    async def _list(_settings: object) -> set[int]:
        msg = "simulated GET /app/installations failure"
        raise RuntimeError(msg)

    return _list


async def _seed_install(engine, installation_id: int, *, tombstoned: bool) -> None:
    tomb = (
        ", tombstoned_at, purge_after_at) VALUES (:id, 'app', 1, 'octocat', 'User', '{}'::jsonb, "
        "NOW(), NOW() + INTERVAL '30 days')"
        if tombstoned
        else ") VALUES (:id, 'app', 1, 'octocat', 'User', '{}'::jsonb)"
    )
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, account_login, "
                "account_type, permissions_at_install" + tomb
            ),
            {"id": installation_id},
        )


async def _tombstone_state(engine, installation_id: int) -> tuple[object, object]:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT tombstoned_at, purge_after_at FROM installations "
                    "WHERE installation_id = :id"
                ),
                {"id": installation_id},
            )
        ).one()


async def test_reconcile_tombstones_missing_install(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local install GitHub no longer lists is tombstoned (missed installation.deleted)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _seed_install(engine, _ID_A, tombstoned=False)
        # GitHub lists a DIFFERENT install, not _ID_A.
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _fake_list({_ID_B}))
        result = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        assert result.tombstoned == 1
        assert result.restored == 0
        row = await _tombstone_state(engine, _ID_A)
        assert row.tombstoned_at is not None
        assert row.purge_after_at is not None
    finally:
        await engine.dispose()


async def test_reconcile_restores_confirmed_live_install(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A TOMBSTONED local install GitHub DOES list gets its tombstone cleared (live-confirmed
    restore — the counterpart to intake `created` not clearing tombstones blindly)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _seed_install(engine, _ID_A, tombstoned=True)
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _fake_list({_ID_A}))
        result = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        assert result.restored == 1
        assert result.tombstoned == 0
        row = await _tombstone_state(engine, _ID_A)
        assert row.tombstoned_at is None
        assert row.purge_after_at is None
    finally:
        await engine.dispose()


async def test_reconcile_skips_when_lock_held(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#067 single-runner: if another connection holds the session-scoped advisory lock, the tick
    no-ops (no reconcile, no GitHub call)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    called = {"list": False}

    async def _tracking_list(_settings: object) -> set[int]:
        called["list"] = True
        return set()

    try:
        await _seed_install(engine, _ID_A, tombstoned=False)
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _tracking_list)
        # Hold the lock on a separate persistent connection for the duration of the call.
        holder = await engine.connect()
        try:
            got = (
                await holder.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": RECONCILE_LOCK_ID}
                )
            ).scalar_one()
            assert got is True  # holder acquired it
            result = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
            assert result.skipped_lock_held is True
            assert called["list"] is False  # never reached the GitHub call
            row = await _tombstone_state(engine, _ID_A)
            assert row.tombstoned_at is None  # no reconcile happened
        finally:
            await holder.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": RECONCILE_LOCK_ID}
            )
            await holder.close()
    finally:
        await engine.dispose()


async def test_reconcile_list_failure_aborts_without_writes(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the GitHub list raises, the tick aborts and writes NOTHING — never reconciling against
    partial data (which would wrongly tombstone). The lock is still released (finally)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _seed_install(engine, _ID_A, tombstoned=False)
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _raising_list())
        with pytest.raises(RuntimeError, match="simulated GET"):
            await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        row = await _tombstone_state(engine, _ID_A)
        assert row.tombstoned_at is None  # NOT tombstoned — no reconcile against a failed list
        # Lock was released (finally): a subsequent tick can acquire it.
        async with engine.connect() as conn:
            reacquire = (
                await conn.execute(
                    text("SELECT pg_try_advisory_lock(:lock_id)"), {"lock_id": RECONCILE_LOCK_ID}
                )
            ).scalar_one()
            assert reacquire is True
            await conn.execute(
                text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": RECONCILE_LOCK_ID}
            )
    finally:
        await engine.dispose()


async def test_reconcile_empty_github_list_tombstones_all(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty GitHub list (every install uninstalled) tombstones every non-tombstoned local
    install — `NOT IN ()` matches all. The list raises on ERROR, so an empty set is trusted."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _seed_install(engine, _ID_A, tombstoned=False)
        await _seed_install(engine, _ID_B, tombstoned=False)
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _fake_list(set()))
        result = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        assert result.tombstoned == 2
        for iid in (_ID_A, _ID_B):
            row = await _tombstone_state(engine, iid)
            assert row.tombstoned_at is not None
    finally:
        await engine.dispose()


async def test_reconcile_idempotent_preserves_grace_deadline(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the janitor twice on a missing install preserves the ORIGINAL grace deadline (the
    `tombstoned_at IS NULL` guard means the second pass no-ops it)."""
    engine = create_async_engine(migrated_db)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    try:
        await _seed_install(engine, _ID_A, tombstoned=False)
        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _fake_list(set()))
        first = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        assert first.tombstoned == 1
        deadline_1 = (await _tombstone_state(engine, _ID_A)).purge_after_at
        second = await reconcile_installations(sessionmaker, _SETTINGS)  # type: ignore[arg-type]
        assert second.tombstoned == 0  # already tombstoned — no re-tombstone
        deadline_2 = (await _tombstone_state(engine, _ID_A)).purge_after_at
        assert deadline_2 == deadline_1  # deadline unchanged
    finally:
        await engine.dispose()
