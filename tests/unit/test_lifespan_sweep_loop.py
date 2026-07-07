"""The periodic sweep loop drives one `run_scheduled_tick` per interval (Arc B2,
DECISIONS.md#065/#012/#067).

`run_scheduled_tick` owns the reconcile-first-then-sweep ordering + the #012 install-hard-delete
liveness gating; these loop-level tests pin only that the loop (a) invokes it each tick, (b)
forwards `github_app_settings` verbatim, and (c) survives a tick failure and keeps ticking. The
ordering / gating guarantee itself is pinned in tests/unit/test_sweep_runner_scheduled_tick.py and
the tests/integration/test_scheduled_tick_ordering.py end-to-end survival test.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import outrider.api.lifespan_sweep_loop as loop_mod
from outrider.api.lifespan_sweep_loop import _sweep_loop


def _dummy_loop_kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "engine": SimpleNamespace(),  # unused — run_scheduled_tick is mocked
        "session_factory": None,
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


async def test_sweep_loop_invokes_run_scheduled_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each tick runs `run_scheduled_tick`, forwarding the engine + App settings — proving the
    production-tick orchestrator (reconcile-first + install-purge gating) has a caller."""
    tick_calls: list[dict[str, Any]] = []

    async def _fake_tick(**kwargs: Any) -> dict[str, Any]:
        tick_calls.append(kwargs)
        return {}

    monkeypatch.setattr(loop_mod, "run_scheduled_tick", _fake_tick)

    settings = SimpleNamespace()
    engine = SimpleNamespace()
    task = asyncio.create_task(
        _sweep_loop(**_dummy_loop_kwargs(engine=engine, github_app_settings=settings))
    )
    await _run_one_tick_then_cancel(task, until=tick_calls)

    assert len(tick_calls) >= 1
    assert tick_calls[0]["engine"] is engine
    assert tick_calls[0]["github_app_settings"] is settings  # the tick received the App settings


async def test_sweep_loop_forwards_none_app_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """`github_app_settings=None` (demo / App not configured) is forwarded verbatim — the tick
    orchestrator (not the loop) decides to skip reconcile + the install hard-delete."""
    tick_calls: list[dict[str, Any]] = []

    async def _fake_tick(**kwargs: Any) -> dict[str, Any]:
        tick_calls.append(kwargs)
        return {}

    monkeypatch.setattr(loop_mod, "run_scheduled_tick", _fake_tick)

    task = asyncio.create_task(_sweep_loop(**_dummy_loop_kwargs(github_app_settings=None)))
    await _run_one_tick_then_cancel(task, until=tick_calls)

    assert len(tick_calls) >= 1
    assert tick_calls[0]["github_app_settings"] is None


async def test_sweep_loop_tick_failure_does_not_kill_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `run_scheduled_tick` failure is caught + logged; the loop keeps ticking."""
    tick_calls: list[int] = []

    async def _failing_tick(**_kwargs: Any) -> dict[str, Any]:
        tick_calls.append(1)
        msg = "simulated tick failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(loop_mod, "run_scheduled_tick", _failing_tick)

    task = asyncio.create_task(_sweep_loop(**_dummy_loop_kwargs(interval_seconds=0.01)))
    # Let several ticks run — a tick RuntimeError each time must not stop the loop.
    for _ in range(200):
        if len(tick_calls) >= 2:
            break
        await asyncio.sleep(0.005)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(tick_calls) >= 2  # loop survived the tick failures and kept going
