# Stage B serve flip — analyze-cache read mode (specs/2026-06-13-analyze-cache-serve-flip.md).
"""CacheConfig — analyze-cache read mode (shadow vs serve).

Closure-injected at `build_graph(...)` per `nodes-receive-deps-via-closure`; the
analyze node reads `cache_mode` to decide whether a live cache hit is SERVED
(reconstruct the cached findings, skip the LLM call) or merely RECORDED (shadow
telemetry — the model always runs). `shadow` is the default so a deploy is
behavior-neutral until the serve flip is justified by the measured would-hit rate.

Orthogonal to the store-or-`None` enable switch: `analyze_cache_store=None`
disables the cache entirely; `cache_mode` governs behavior only when a store IS
wired. Mirrors `PatchConfig`/`HITLConfig`: a frozen `BaseSettings` read at startup.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheMode(StrEnum):
    """Analyze-cache read behavior.

    `SHADOW`: record `CacheLookupEvent(would_hit|miss)`, always call the model,
    never serve — the V1 default and the only behavior until the telemetry-gated
    flip. `SERVE`: on a live hit, reconstruct the cached findings and skip the LLM
    call (the served findings still flow through every deterministic downstream
    gate — reducers, synthesize, HITL, publish — unchanged).
    """

    SHADOW = "shadow"
    SERVE = "serve"


class CacheConfig(BaseSettings):
    """Reads `OUTRIDER_CACHE_MODE` (default `shadow`).

    Tests construct `CacheConfig(mode=CacheMode.SERVE)` directly and inject through
    `build_graph(...)` to exercise the serve path without an env var. `frozen=True`:
    construction-time-only. Default `SHADOW` keeps a merge/deploy behavior-neutral —
    the serve flip is a deliberate, telemetry-gated config change, not a default.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_CACHE_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    mode: CacheMode = CacheMode.SHADOW


__all__ = ["CacheConfig", "CacheMode"]
