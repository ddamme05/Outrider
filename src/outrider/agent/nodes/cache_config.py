# Stage B serve flip — analyze-cache read mode (specs/2026-06-13-analyze-cache-serve-flip.md).
"""Choose whether an analyze-cache hit supplies the review result.

Startup reads these settings and passes mode through build_graph's closures.
Shadow records completed lookups without serving them; normal analysis continues.
Serve can reconstruct a usable live hit and skip the model call.

Mode is separate from enablement: analyze_cache_store=None disables the cache.
The default remains shadow; serving requires a deliberate, telemetry-gated change.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class CacheMode(StrEnum):
    """Analyze-cache read behavior.

    SHADOW records CacheLookupEvent(would_hit|miss) for completed lookups without
    serving. Analysis continues; deterministic coverage can still skip the model.
    SERVE reconstructs a usable live hit and skips the model call. Served findings
    still pass through reducers, synthesis, HITL, and publication gates.
    """

    SHADOW = "shadow"
    SERVE = "serve"


class CacheConfig(BaseSettings):
    """Reads `OUTRIDER_CACHE_MODE` (default `shadow`).

    Mode can also be supplied at construction; settings are frozen afterward.
    Serving is a deliberate config change gated by measured would-hit telemetry.
    """

    model_config = SettingsConfigDict(
        env_prefix="OUTRIDER_CACHE_",
        env_file=None,
        extra="forbid",
        frozen=True,
    )

    mode: CacheMode = CacheMode.SHADOW


__all__ = ["CacheConfig", "CacheMode"]
