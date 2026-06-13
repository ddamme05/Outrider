# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""DispatchConfig — bound on concurrently-executing reviews.

The V1 in-process dispatcher (`BackgroundTasksDispatcher`) enqueues each
review onto FastAPI BackgroundTasks with no upper bound on how many run
at once. An actor who can trigger webhooks (open PRs, force-push) could
enqueue unbounded concurrent reviews; each review's LLM calls hold
connections from the shared Anthropic pool (`anthropic_provider.py`:
`max_connections=50`, `read=300s`), so a flood saturates the pool and
degrades legitimate reviews (FUP-164).

`max_concurrent_reviews` caps how many graph executions run at once. The
ceiling is enforced at the lifespan-bound `run_graph` closure via a
process-level `asyncio.Semaphore` (see `dispatcher/concurrency.py`), NOT
in `BackgroundTasksDispatcher` — the dispatcher is request-scoped (one
instance per webhook), so a semaphore held there would not bound across
requests. Excess reviews await a free slot as cheap coroutines instead
of all entering analyze simultaneously.

Per-process bound: the semaphore is one per worker process, so a
multi-worker / multi-replica deployment's real ceiling is
`workers x max_concurrent_reviews`. A global/distributed bound (and a
per-installation cost rate-limit) is V2 scope — the Celery worker count
is the natural ceiling there. See DECISIONS.md#045.

Mirrors `PatchConfig` (bare `OUTRIDER_` env prefix, `frozen`, `extra="forbid"`):
a frozen `BaseSettings` read once at lifespan startup. Unlike `CacheConfig`, which
uses the narrower `OUTRIDER_CACHE_` prefix + `env_file=None`.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DispatchConfig(BaseSettings):
    """Reads `OUTRIDER_MAX_CONCURRENT_REVIEWS` (default 8).

    Tests construct `DispatchConfig(max_concurrent_reviews=2)` directly.
    `frozen=True`: construction-time-only.

    Default 8: comfortably above realistic single-tenant concurrent review
    load, comfortably below the 50-connection Anthropic pool (each V1 review
    holds ~1 in-flight connection during its sequential analyze pass), so a
    flood is bounded without throttling legitimate bursts. Raise it for a
    deployment with genuinely higher legitimate concurrency.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_",
        extra="forbid",
        frozen=True,
    )

    # ge=1: a zero/negative ceiling would deadlock every review (an
    # `asyncio.Semaphore(0)` never admits). Reject it loudly at startup.
    max_concurrent_reviews: int = Field(default=8, ge=1)
