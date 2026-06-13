# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""Concurrency ceiling for background review execution.

`concurrency_limited(runner, semaphore)` wraps an async single-argument
runner so at most `semaphore`-many calls execute concurrently; excess
calls await a free slot. The lifespan composes it around the `run_graph`
closure so the V1 in-process dispatcher cannot saturate the shared
Anthropic connection pool with unbounded concurrent reviews (FUP-164,
DECISIONS.md#045).

Kept as a tiny standalone helper rather than inlined in the lifespan
closure so the bound is unit-testable in isolation (`build_lifespan`
does not expose a `build_graph` seam, so the closure is otherwise only
reachable through full lifespan wiring).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable

__all__ = ["concurrency_limited"]


def concurrency_limited[T, R](
    runner: Callable[[T], Awaitable[R]],
    semaphore: asyncio.Semaphore,
) -> Callable[[T], Awaitable[R]]:
    """Return an async callable that runs `runner` under `semaphore`.

    Each invocation acquires `semaphore` before awaiting `runner` and
    releases it on exit (including on exception — `async with` releases
    in its `__aexit__`). At most `semaphore`-many `runner` calls execute
    concurrently; the rest await a free slot as parked coroutines. The
    returned callable has the same single-argument signature as `runner`.
    """

    async def _limited(arg: T) -> R:
        async with semaphore:
            return await runner(arg)

    return _limited
