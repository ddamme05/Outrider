# Per specs/2026-06-11-file-hash-analyze-cache.md — the file-hash analyze cache.
"""Analyze-result cache (cost lever #8).

V1 surface: `compute_analyze_cache_key` (the eight-component key recipe).
The store, lookup events, and serve path land with the same spec's later
chunks; this package deliberately starts key-first so the recipe is
testable before any storage exists.
"""

from outrider.cache.key import compute_analyze_cache_key

__all__ = [
    "compute_analyze_cache_key",
]
