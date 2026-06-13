# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""Concurrency ceiling for background review execution.

Pins the bound `concurrency_limited` enforces: with a semaphore of N, at
most N wrapped calls run concurrently; the N+1th awaits a free slot and
enters only after a running call exits. This is the dispatch-level
ceiling (wired around the lifespan `run_graph` closure) that keeps a
webhook flood from saturating the shared Anthropic connection pool.

Also pins `DispatchConfig`: default 8, and the `ge=1` floor that rejects
a deadlocking `Semaphore(0)` ceiling at startup.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from outrider.dispatcher import DispatchConfig, concurrency_limited


@pytest.mark.asyncio
async def test_at_most_n_run_concurrently_n_plus_one_waits_for_a_slot() -> None:
    """N+1 submitted under Semaphore(N): exactly N enter; the N+1th is
    parked until one exits, and peak concurrency never exceeds N."""
    n = 3
    semaphore = asyncio.Semaphore(n)
    running = 0
    peak = 0
    n_entered = asyncio.Event()  # set once N have entered the runner
    release = asyncio.Event()  # held; runner bodies block until set

    async def runner(arg: int) -> int:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running >= n:
            n_entered.set()
        await release.wait()
        running -= 1
        return arg

    limited = concurrency_limited(runner, semaphore)
    tasks = [asyncio.create_task(limited(i)) for i in range(n + 1)]

    # N acquire the semaphore immediately; the N+1th blocks on acquire.
    await asyncio.wait_for(n_entered.wait(), timeout=1.0)
    # Spin the loop a few turns — the N+1th must still NOT have entered.
    for _ in range(5):
        await asyncio.sleep(0)
    assert running == n  # exactly N inside, not N+1
    assert peak == n
    assert not any(t.done() for t in tasks)  # none finished while held

    # Release: all N exit, freeing slots; the N+1th finally enters and runs.
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert sorted(results) == list(range(n + 1))
    assert peak == n  # the N+1th waited; peak never crossed the ceiling


@pytest.mark.asyncio
async def test_semaphore_released_on_runner_exception() -> None:
    """A runner that raises still releases its slot — a failing review
    must not permanently consume a connection-pool budget unit."""
    semaphore = asyncio.Semaphore(1)

    async def boom(_: int) -> int:
        raise RuntimeError("review blew up")

    limited = concurrency_limited(boom, semaphore)
    with pytest.raises(RuntimeError, match="review blew up"):
        await limited(0)
    # Slot freed: a fresh acquire succeeds without blocking.
    assert semaphore.locked() is False
    await asyncio.wait_for(semaphore.acquire(), timeout=1.0)
    semaphore.release()


def test_dispatch_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ceiling is 8 when the env var is unset."""
    monkeypatch.delenv("OUTRIDER_MAX_CONCURRENT_REVIEWS", raising=False)
    assert DispatchConfig().max_concurrent_reviews == 8


def test_dispatch_config_rejects_zero_ceiling() -> None:
    """ge=1: a zero ceiling (Semaphore(0) never admits) fails loud at
    construction rather than deadlocking every review."""
    with pytest.raises(ValidationError):
        DispatchConfig(max_concurrent_reviews=0)
