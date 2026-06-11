# Per specs/2026-06-11-file-hash-analyze-cache.md — the file-hash analyze cache.
"""Analyze-result cache (cost lever #8).

V1 surface: `compute_analyze_cache_key` (the eight-component key recipe)
plus the DB-backed store — `AnalyzeCacheStore` with its `CacheScope` /
`CacheEntry` shapes and the `CACHE_TTL_DAYS` bound. The lookup audit
event and the analyze wiring land with the same spec's later chunks.
"""

from outrider.cache.key import compute_analyze_cache_key
from outrider.cache.store import (
    CACHE_TTL_DAYS,
    AnalyzeCacheStore,
    CacheEntry,
    CacheScope,
)

__all__ = [
    "CACHE_TTL_DAYS",
    "AnalyzeCacheStore",
    "CacheEntry",
    "CacheScope",
    "compute_analyze_cache_key",
]
