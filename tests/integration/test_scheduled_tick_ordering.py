"""Check that reconciliation restores live installations before deletion runs.

Run the installation lifecycle against PostgreSQL with unrelated sweep jobs
stubbed. A reinstalled installation must survive the complete tick even if its
old deletion deadline has passed. See DECISIONS.md#012 and #065."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

import outrider.sweep.hitl_expiry as hitl_mod
import outrider.sweep.reconcile_installations as reconcile_mod
import outrider.sweep.replay_verdict as replay_mod
from outrider.sweep.runner import run_scheduled_tick

if TYPE_CHECKING:
    import pytest

_INSTALL_ID = 111


class _ConfiguredProvider:
    """A credential provider reporting `is_configured() is True` so the tick proceeds to reconcile
    (`DECISIONS.md#070`). `reconcile_mod.list_installation_ids` is monkeypatched below, so the
    provider's `current()` is never reached — `is_configured()` is the only method the tick
    calls."""

    async def is_configured(self) -> bool:
        return True

    async def current(self) -> Any:  # pragma: no cover — list_installation_ids is monkeypatched
        return SimpleNamespace()


async def _seed_expired_tombstone(engine: Any) -> None:
    """An install tombstoned 40 days ago, grace deadline 10 days PAST — due for hard-delete."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                " account_login, account_type, permissions_at_install, tombstoned_at, "
                " purge_after_at) VALUES "
                "(:id, 'a', 1, 'x', 'User', '{}'::jsonb, NOW() - INTERVAL '40 days', "
                "   NOW() - INTERVAL '10 days')"
            ),
            {"id": _INSTALL_ID},
        )


async def test_github_listed_expired_tombstone_survives_full_tick(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A GitHub-listed install whose tombstone grace has EXPIRED survives one complete tick:
    reconcile restores it before the hard-delete can purge it."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_tombstone(engine)

        # GitHub DOES list the install — the user reinstalled during the grace window.
        async def _list(_settings: object) -> set[int]:
            return {_INSTALL_ID}

        async def _noop_hitl(**_kwargs: Any) -> dict[str, Any]:
            return {}

        async def _noop_replay(**_kwargs: Any) -> dict[str, Any]:
            return {}

        monkeypatch.setattr(reconcile_mod, "list_installation_ids", _list)
        monkeypatch.setattr(hitl_mod, "run_once", _noop_hitl)
        monkeypatch.setattr(replay_mod, "project_replay_verdicts", _noop_replay)

        result = await run_scheduled_tick(
            engine=engine,
            session_factory=None,  # unused — hitl + replay sub-jobs are stubbed
            anomaly_sink=None,
            review_status_sink=None,
            audit_persister=None,
            checkpointer=None,
            compiled_graph=None,
            provider=_ConfiguredProvider(),
        )

        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT tombstoned_at, purge_after_at FROM installations "
                        "WHERE installation_id = :id"
                    ),
                    {"id": _INSTALL_ID},
                )
            ).one_or_none()

        assert row is not None, (
            "the GitHub-listed install must SURVIVE the tick — reconcile clears its tombstone "
            "BEFORE the #012 hard-delete runs; purging it would be the data-loss bug"
        )
        assert row.tombstoned_at is None, "reconcile restored the live install (tombstone cleared)"
        assert row.purge_after_at is None
        # The install-purge DID run this tick (reconcile confirmed liveness); it found nothing due
        # because the restore beat it.
        assert result["reconcile"]["restored"] == 1
        assert result["install_purge"] == {}
    finally:
        await engine.dispose()
