# Per specs/2026-06-11-file-hash-analyze-cache.md — the file-hash analyze cache.
"""Expose analyze-cache identity, storage, and retention types (cost lever #8).

compute_analyze_cache_key owns the versioned recipe, including host identity
(DECISIONS.md#056). AnalyzeCacheStore supplies scoped live entries and bounded
writes. The analyze node chooses shadow telemetry or serving through CacheMode;
shadow does not itself skip the model, while deterministic coverage still can.
"""

from outrider.cache.key import ANALYZE_CACHE_KEY_VERSION, compute_analyze_cache_key
from outrider.cache.store import (
    CACHE_TTL_DAYS,
    AnalyzeCacheStore,
    CacheEntry,
    CacheScope,
    CacheStoreError,
)

__all__ = [
    "ANALYZE_CACHE_KEY_VERSION",
    "CACHE_TTL_DAYS",
    "AnalyzeCacheStore",
    "CacheEntry",
    "CacheScope",
    "CacheStoreError",
    "compute_analyze_cache_key",
]
