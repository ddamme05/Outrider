"""Expose the file-analysis cache key and database store.

The cache can record a potential hit without skipping analysis, or serve a
validated result when configured to do so. See DECISIONS.md#056 and
specs/2026-06-11-file-hash-analyze-cache.md.
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
