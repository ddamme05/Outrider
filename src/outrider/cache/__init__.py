# Per specs/2026-06-11-file-hash-analyze-cache.md — the file-hash analyze cache.
"""Analyze-result cache (cost lever #8).

V1 surface: `compute_analyze_cache_key` (the key recipe — canonical
prompt digest plus eight explicit scope/version components), the
DB-backed `AnalyzeCacheStore` with its `CacheScope` / `CacheEntry`
shapes, the contained `CacheStoreError`, and the `CACHE_TTL_DAYS`
bound. The analyze node consumes the store in shadow mode (lookup +
`CacheLookupEvent` telemetry + write-on-miss; the model is always
called); the serve flip is a later arc.
"""

from outrider.cache.key import compute_analyze_cache_key
from outrider.cache.store import (
    CACHE_TTL_DAYS,
    AnalyzeCacheStore,
    CacheEntry,
    CacheScope,
    CacheStoreError,
)

__all__ = [
    "CACHE_TTL_DAYS",
    "AnalyzeCacheStore",
    "CacheEntry",
    "CacheScope",
    "CacheStoreError",
    "compute_analyze_cache_key",
]
