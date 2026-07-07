"""The periodic sweep loop wires the reconcile janitor (Arc B2, DECISIONS.md#065/#012/#067).

Blocker regression: `reconcile_installations` previously had NO production caller — the loop only
ran `run_all_sweeps`, so missed-delete detection + live-confirmed restore never fired automatically.
These tests pin that the loop now invokes the janitor each tick (when the App is configured),
OUTSIDE the `run_all_sweeps` transaction, and skips it when there are no App settings (demo mode).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import outrider.api.lifespan_sweep_loop as loop_mod
from outrider.api.lifespan_sweep_loop import _sweep_loop


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


def _dummy_loop_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "engine": _FakeEngine(),
        "session_factory": None,  # unused — run_all_sweeps is mocked
        "anomaly_sink": None,
        "review_status_sink": None,
        "audit_persister": None,
        "checkpointer": None,
        "compiled_graph": None,
        "github_app_settings": SimpleNamespace(),
        "interval_seconds": 60.0,  # long sleep — the test cancels after the first tick
    }
    kwargs.update(overrides)
    return kwargs


async def _run_one_tick_then_cancel(task: asyncio.Task[None], *, until: list[Any] | None) -> None:
    """Wait for the first tick's side effect (`until` non-empty) or a brief window, then cancel."""
    if until is not None:
        for _ in range(200):
            if until:
                break
            await asyncio.sleep(0.005)
    else:
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_sweep_loop_invokes_reconcile_janitor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each tick runs `reconcile_installations` (the #065 janitor) alongside `run_all_sweeps`, with
    the App settings — proving the janitor now has a production caller."""
    reconcile_calls: list[tuple[Any, Any]] = []

    async def _fake_sweeps(**_kwargs: Any) -> dict[str, Any]:
        return {}

    async def _fake_reconcile(engine: Any, settings: Any) -> Any:
        reconcile_calls.append((engine, settings))
        return SimpleNamespace(tombstoned=0, restored=0)

    monkeypatch.setattr(loop_mod, "run_all_sweeps", _fake_sweeps)
    monkeypatch.setattr(loop_mod, "reconcile_installations", _fake_reconcile)

    settings = SimpleNamespace()
    task = asyncio.create_task(_sweep_loop(**_dummy_loop_kwargs(github_app_settings=settings)))
    await _run_one_tick_then_cancel(task, until=reconcile_calls)

    assert len(reconcile_calls) >= 1
    assert reconcile_calls[0][1] is settings  # the janitor received the App settings


async def test_sweep_loop_skips_reconcile_when_no_app_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With `github_app_settings=None` (demo / App not configured) the janitor is NOT invoked."""
    reconcile_calls: list[int] = []

    async def _fake_sweeps(**_kwargs: Any) -> dict[str, Any]:
        return {}

    async def _fake_reconcile(_engine: Any, _settings: Any) -> Any:
        reconcile_calls.append(1)
        return SimpleNamespace()

    monkeypatch.setattr(loop_mod, "run_all_sweeps", _fake_sweeps)
    monkeypatch.setattr(loop_mod, "reconcile_installations", _fake_reconcile)

    task = asyncio.create_task(_sweep_loop(**_dummy_loop_kwargs(github_app_settings=None)))
    await _run_one_tick_then_cancel(task, until=None)

    assert reconcile_calls == []


async def test_sweep_loop_reconcile_failure_does_not_kill_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A janitor failure (e.g. GitHub unreachable) is caught + logged; the loop keeps ticking
    (run_all_sweeps still runs on the next tick)."""
    sweep_calls: list[int] = []

    async def _fake_sweeps(**_kwargs: Any) -> dict[str, Any]:
        sweep_calls.append(1)
        return {}

    async def _failing_reconcile(_engine: Any, _settings: Any) -> Any:
        msg = "simulated GitHub unreachable"
        raise RuntimeError(msg)

    monkeypatch.setattr(loop_mod, "run_all_sweeps", _fake_sweeps)
    monkeypatch.setattr(loop_mod, "reconcile_installations", _failing_reconcile)

    task = asyncio.create_task(_sweep_loop(**_dummy_loop_kwargs(interval_seconds=0.01)))
    # Let several ticks run — a reconcile RuntimeError each time must not stop the loop.
    for _ in range(200):
        if len(sweep_calls) >= 2:
            break
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(sweep_calls) >= 2  # loop survived the reconcile failures and kept sweeping
