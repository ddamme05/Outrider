# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""Wrap the shared graph runner in a per-process semaphore.

The lifespan uses this limit across all webhook requests (DECISIONS.md#045,
FUP-164). Keeping the wrapper here lets tests exercise the bound independently.
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
    """Acquire a slot, await the runner, and release on return or exception."""

    async def _limited(arg: T) -> R:
        async with semaphore:
            return await runner(arg)

    return _limited
