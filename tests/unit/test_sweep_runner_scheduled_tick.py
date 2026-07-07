"""`run_scheduled_tick` orchestration invariants (Arc B2, DECISIONS.md#065/#012/#067).

The production tick MUST reconcile the install cache BEFORE running the sweep family, and MUST gate
the #012 install hard-delete (`run_all_sweeps(include_install_purge=...)`) on that reconcile having
CONFIRMED liveness this tick. A reconcile that fails, is lock-contended, or is skipped (no App
settings) leaves the hard-delete OFF for the tick — but the unrelated sweeps still run. These unit
tests pin that contract against monkeypatched reconcile + run_all_sweeps stand-ins; the real
end-to-end survival guarantee is in tests/integration/test_scheduled_tick_ordering.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import outrider.sweep.runner as runner_mod
from outrider.sweep.runner import run_scheduled_tick

if TYPE_CHECKING:
    import pytest


class _FakeTxn:
    async def __aenter__(self) -> _FakeTxn:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _FakeConn:
    async def __aenter__(self) -> _FakeConn:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def begin(self) -> _FakeTxn:
        return _FakeTxn()


class _FakeEngine:
    """Minimal engine: `connect()` yields a no-op async-CM connection with a `begin()` txn CM.
    `run_all_sweeps` is monkeypatched to a no-op, so nothing actually touches the connection."""

    def connect(self) -> _FakeConn:
        return _FakeConn()


def _tick_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "engine": _FakeEngine(),
        "session_factory": None,
        "anomaly_sink": None,
        "review_status_sink": None,
        "audit_persister": None,
        "checkpointer": None,
        "compiled_graph": None,
        "github_app_settings": SimpleNamespace(),
    }
    kwargs.update(overrides)
    return kwargs


async def test_reconcile_runs_before_sweeps_and_permits_purge_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The janitor runs FIRST and, when it confirms liveness, the #012 hard-delete is permitted."""
    order: list[str] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _settings: Any) -> Any:
        order.append("reconcile")
        return SimpleNamespace(skipped_lock_held=False, tombstoned=1, restored=2)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        order.append("sweeps")
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs())

    assert order == ["reconcile", "sweeps"]  # reconcile FIRST, always
    assert seen["include_install_purge"] is True  # confirmed liveness → hard-delete permitted
    assert result["reconcile"] == {
        "ran": True,
        "skipped_lock_held": False,
        "tombstoned": 1,
        "restored": 2,
    }


async def test_lock_contended_reconcile_skips_install_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lock-contended tick (another runner reconciling) is NOT this tick's confirmation → the
    #012 hard-delete is skipped, but the unrelated sweeps still run."""
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _settings: Any) -> Any:
        return SimpleNamespace(skipped_lock_held=True, tombstoned=0, restored=0)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs())

    assert seen["include_install_purge"] is False  # not confirmed this tick → skip hard-delete
    assert result["reconcile"]["skipped_lock_held"] is True


async def test_reconcile_failure_skips_install_purge_but_sweeps_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconcile FAILURE (GitHub unreachable) skips ONLY the install hard-delete — the unrelated
    sweeps run regardless (independent failure handling)."""
    ran_sweeps: list[bool] = []
    seen: dict[str, Any] = {}

    async def _failing_reconcile(_engine: Any, _settings: Any) -> Any:
        msg = "simulated GitHub unreachable"
        raise RuntimeError(msg)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        ran_sweeps.append(True)
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _failing_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs())

    assert ran_sweeps == [True]  # sweeps run REGARDLESS of the reconcile failure
    assert seen["include_install_purge"] is False  # unconfirmed → hard-delete skipped
    assert result["reconcile"] == {"ran": True, "failed": True}


async def test_no_app_settings_skips_reconcile_and_purge(monkeypatch: pytest.MonkeyPatch) -> None:
    """`github_app_settings=None` (demo / App not configured) → no reconcile authority → the #012
    hard-delete is skipped for the tick."""
    reconcile_calls: list[int] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _settings: Any) -> Any:
        reconcile_calls.append(1)
        return SimpleNamespace(skipped_lock_held=False, tombstoned=0, restored=0)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs(github_app_settings=None))

    assert reconcile_calls == []  # no App → reconcile never called
    assert seen["include_install_purge"] is False  # no liveness authority → skip hard-delete
    assert result["reconcile"] == {"ran": False, "reason": "no_app_settings"}
