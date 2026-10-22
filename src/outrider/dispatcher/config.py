# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""Configure the per-process limit on concurrent graph executions.

The lifespan shares one semaphore across request-scoped dispatchers. Excess reviews
wait for a slot; queued reviews are not bounded by this setting. With multiple
workers the total ceiling is workers × max_concurrent_reviews, not a global limit.
See DECISIONS.md#045 and FUP-164.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DispatchConfig(BaseSettings):
    """Read OUTRIDER_MAX_CONCURRENT_REVIEWS once at startup (default 8).

    Frozen settings prevent mutation after wiring; ge=1 rejects a semaphore that
    would never admit a review.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_",
        extra="forbid",
        frozen=True,
    )

    # ge=1: a zero/negative ceiling would deadlock every review (an
    # `asyncio.Semaphore(0)` never admits). Reject it loudly at startup.
    max_concurrent_reviews: int = Field(default=8, ge=1)
