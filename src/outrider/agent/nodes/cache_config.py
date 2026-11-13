"""Choose whether cached analysis can replace a model call.

Shadow mode records usable cache lookups and continues normal review. Serve
mode can reuse a valid result. Both need a configured cache store; no store
means caching is disabled. Startup passes the mode into the review graph.

See specs/2026-06-13-analyze-cache-serve-flip.md.
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
