"""`run_scheduled_tick` orchestration invariants (Arc B2, DECISIONS.md#065/#012/#067).

The production tick MUST reconcile the install cache BEFORE running the sweep family, and MUST gate
the #012 install hard-delete (`run_all_sweeps(include_install_purge=...)`) on that reconcile having
CONFIRMED liveness this tick. A reconcile that fails, is lock-contended, or is skipped (no
configured credentials) leaves the hard-delete OFF for the tick — but the unrelated sweeps still
run. These unit
tests pin that contract against monkeypatched reconcile + run_all_sweeps stand-ins, plus the
FAIL-SAFE default of `run_all_sweeps` itself (a bare call must NOT hard-delete installs); the real
end-to-end survival guarantee is in tests/integration/test_scheduled_tick_ordering.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import outrider.sweep.runner as runner_mod
from outrider.sweep.runner import run_all_sweeps, run_scheduled_tick

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


class _FakeTxnConn:
    """A connection stand-in that reports it IS in a transaction (run_all_sweeps' precondition)."""

    def in_transaction(self) -> bool:
        return True


class _FakeProvider:
    """A credential provider (`DECISIONS.md#070`): `run_scheduled_tick` skips reconcile when the
    provider is absent (`None`) OR reports `is_configured() is False` (a `database`-mode instance
    still onboarding). `reconcile_installations` is monkeypatched, so `current()` is never
    reached."""

    def __init__(self, *, configured: bool = True, raises: bool = False) -> None:
        self._configured = configured
        self._raises = raises

    async def is_configured(self) -> bool:
        if self._raises:
            # A `database`-mode `is_configured()` is a DB read: a missing singleton or a transient
            # connection error raises rather than returning False.
            msg = "simulated setup_state read failure"
            raise RuntimeError(msg)
        return self._configured

    async def current(self) -> Any:  # pragma: no cover — reconcile is monkeypatched
        return SimpleNamespace()


def _tick_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "engine": _FakeEngine(),
        "session_factory": None,
        "anomaly_sink": None,
        "review_status_sink": None,
        "audit_persister": None,
        "checkpointer": None,
        "compiled_graph": None,
        "provider": _FakeProvider(configured=True),
    }
    kwargs.update(overrides)
    return kwargs


async def test_reconcile_runs_before_sweeps_and_permits_purge_when_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The janitor runs FIRST and, when it confirms liveness, the #012 hard-delete is permitted."""
    order: list[str] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _provider: Any) -> Any:
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

    async def _fake_reconcile(_engine: Any, _provider: Any) -> Any:
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

    async def _failing_reconcile(_engine: Any, _provider: Any) -> Any:
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


async def test_no_provider_skips_reconcile_and_purge(monkeypatch: pytest.MonkeyPatch) -> None:
    """`provider=None` (demo / App not configured) → no reconcile authority → the #012
    hard-delete is skipped for the tick."""
    reconcile_calls: list[int] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _provider: Any) -> Any:
        reconcile_calls.append(1)
        return SimpleNamespace(skipped_lock_held=False, tombstoned=0, restored=0)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs(provider=None))

    assert reconcile_calls == []  # no provider → reconcile never called
    assert seen["include_install_purge"] is False  # no liveness authority → skip hard-delete
    assert result["reconcile"] == {"ran": False, "reason": "no_credentials_configured"}


async def test_unconfigured_provider_skips_reconcile_and_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `database`-mode provider that is not yet `CONFIGURED` (`is_configured() is False`, mid-
    onboarding) is treated exactly like `provider=None` (`DECISIONS.md#070`): no reconcile authority
    → the #012 hard-delete is skipped, reconcile never runs. Distinct from `provider=None`: the
    provider EXISTS but reports itself unconfigured, so the guard must check `is_configured()`, not
    only `is None`."""
    reconcile_calls: list[int] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _provider: Any) -> Any:
        reconcile_calls.append(1)
        return SimpleNamespace(skipped_lock_held=False, tombstoned=0, restored=0)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs(provider=_FakeProvider(configured=False)))

    assert reconcile_calls == []  # unconfigured → reconcile never called
    assert seen["include_install_purge"] is False  # no liveness authority → skip hard-delete
    assert result["reconcile"] == {"ran": False, "reason": "no_credentials_configured"}


async def test_config_check_error_skips_purge_but_unrelated_sweeps_still_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RAISE from `provider.is_configured()` (a `database`-mode DB read — missing singleton or a
    transient connection error) must degrade to skip-reconcile, NOT abort the whole tick: the
    unrelated sweeps (hitl-expiry, TTL purge, replay-verdict) still run. Regression guard — the
    config check must live INSIDE the reconcile try/except, not the if-condition. Move it back to
    the if-condition → is_configured()'s raise escapes `run_scheduled_tick` and this fails (no
    sweeps)."""
    reconcile_calls: list[int] = []
    ran_sweeps: list[bool] = []
    seen: dict[str, Any] = {}

    async def _fake_reconcile(_engine: Any, _provider: Any) -> Any:
        reconcile_calls.append(1)
        return SimpleNamespace(skipped_lock_held=False, tombstoned=0, restored=0)

    async def _fake_sweeps(**kwargs: Any) -> dict[str, Any]:
        ran_sweeps.append(True)
        seen["include_install_purge"] = kwargs["include_install_purge"]
        return {}

    monkeypatch.setattr(runner_mod, "reconcile_installations", _fake_reconcile)
    monkeypatch.setattr(runner_mod, "run_all_sweeps", _fake_sweeps)

    result = await run_scheduled_tick(**_tick_kwargs(provider=_FakeProvider(raises=True)))

    assert reconcile_calls == []  # is_configured() raised BEFORE reconcile could run
    assert ran_sweeps == [True]  # the unrelated sweeps run REGARDLESS of the config-check failure
    assert seen["include_install_purge"] is False  # unconfirmed → hard-delete skipped
    assert result["reconcile"] == {"ran": True, "failed": True}


async def test_run_all_sweeps_default_is_fail_safe_no_install_purge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FAIL-SAFE default: a bare `run_all_sweeps` (no `include_install_purge`) must NOT run the
    #012 install hard-delete. Only a caller that reconciled FIRST (`run_scheduled_tick`) opts in by
    passing True. Revert the default to True → this fails, re-exposing the tick-ordering data-loss
    path for any direct/legacy caller that omits the argument."""
    install_purge_calls: list[int] = []

    async def _noop(**_kwargs: Any) -> dict[str, Any]:
        return {}

    async def _track_install_purge(**_kwargs: Any) -> dict[str, Any]:
        install_purge_calls.append(1)
        return {}

    monkeypatch.setattr("outrider.sweep.hitl_expiry.run_once", _noop)
    monkeypatch.setattr("outrider.sweep.purge_expired.purge_expired", _noop)
    monkeypatch.setattr(
        "outrider.sweep.purge_expired.purge_expired_installations", _track_install_purge
    )
    monkeypatch.setattr("outrider.sweep.replay_verdict.project_replay_verdicts", _noop)

    result = await run_all_sweeps(
        conn=_FakeTxnConn(),  # type: ignore[arg-type]
        session_factory=None,  # type: ignore[arg-type]
        anomaly_sink=None,  # type: ignore[arg-type]
        review_status_sink=None,  # type: ignore[arg-type]
        audit_persister=None,  # type: ignore[arg-type]
        checkpointer=None,  # type: ignore[arg-type]
        compiled_graph=None,  # type: ignore[arg-type]
    )

    assert install_purge_calls == []  # hard-delete OFF by default (fail-safe)
    assert result["install_purge"] == {"skipped": "reconcile_unconfirmed"}
