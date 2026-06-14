# FUP-170: post-cost-gate parse bundle + the derived unsafe set.
"""`extract_triviality_and_scan` consolidates two head parses into one.

Pins the FUP-170 acceptance bar: the bundle is behavior-equivalent to the
separate `build_triviality_context` + `scan_parameterized_calls` calls, and it
proves the consolidation (the scan reuses the bundle's head tree — no separate
parse). Also pins `ParameterizedCallScan.unsafe_parameterized_calls` (the derived
multiset `all − safe` the coordinates veto now consumes).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from outrider.ast_facts.analyze_bundle import extract_triviality_and_scan
from outrider.ast_facts.parameterized_calls import (
    ExecuteCallSite,
    ParameterizedCallScan,
    scan_digest,
    scan_parameterized_calls,
)
from outrider.ast_facts.triviality import build_triviality_context

if TYPE_CHECKING:
    import pytest

# Head: a safe parameterized call (cursor.execute with a literal + params), a
# bare execute (unsafe), and a comment-only line — exercises triviality + scan.
_HEAD = b"""\
def f(cursor):
    cursor.execute("SELECT 1", (x,))
    # a note
    return execute("SELECT 2")
"""
_BASE = b"""\
def f(cursor):
    cursor.execute("SELECT 1", (x,))
    return execute("SELECT 2")
"""


def test_bundle_equivalent_to_separate_calls() -> None:
    """The acceptance bar: one shared head parse yields the same triviality
    context AND the same scan as the two separate public calls."""
    triv, scan = extract_triviality_and_scan(_HEAD, _BASE, compute_triviality=True, degraded=False)
    assert triv == build_triviality_context(_HEAD, _BASE)
    assert scan == scan_parameterized_calls(_HEAD)


def test_compute_triviality_false_still_scans() -> None:
    """When triviality isn't wanted (no patch / no scopes), the scan still
    rides — and no triviality (or base parse) is built."""
    triv, scan = extract_triviality_and_scan(_HEAD, _BASE, compute_triviality=False, degraded=False)
    assert triv is None
    assert scan == scan_parameterized_calls(_HEAD)


def test_degraded_suppresses_both() -> None:
    """Degraded mode has no trustworthy tree: no scan, no triviality, no parse."""
    triv, scan = extract_triviality_and_scan(_HEAD, _BASE, compute_triviality=True, degraded=True)
    assert triv is None
    assert scan is None


def test_bundle_parses_head_once_and_scan_reuses_the_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consolidation proof: the head is parsed exactly ONCE (by the bundle),
    the base once (triviality's separate file), and the parameterized scan
    performs ZERO parses — it reuses the bundle's head tree."""
    import outrider.ast_facts.analyze_bundle as ab
    import outrider.ast_facts.parameterized_calls as pc
    import outrider.ast_facts.triviality as tv

    counts = {"head": 0, "base": 0, "scan": 0}

    class _CountingParser:
        def __init__(self, real: object, key: str) -> None:
            self._real = real
            self._key = key

        def parse(self, source: bytes) -> object:
            counts[self._key] += 1
            return self._real.parse(source)  # type: ignore[attr-defined]

    monkeypatch.setattr(ab, "_PARSER", _CountingParser(ab._PARSER, "head"))
    monkeypatch.setattr(tv, "_PARSER", _CountingParser(tv._PARSER, "base"))
    monkeypatch.setattr(pc, "_PARSER", _CountingParser(pc._PARSER, "scan"))

    extract_triviality_and_scan(_HEAD, _BASE, compute_triviality=True, degraded=False)

    assert counts["head"] == 1  # the bundle's single head parse
    assert counts["base"] == 1  # triviality base side (a different file)
    assert counts["scan"] == 0  # scan reused the head tree — no separate parse


def test_unsafe_parameterized_calls_is_multiset_all_minus_safe() -> None:
    """The derived unsafe set cancels exactly one all-sites occurrence per safe
    site — two calls on one line (one safe, one not) leave the unsafe twin."""
    twin_a = ExecuteCallSite(line_start=5, line_end=5)
    twin_b = ExecuteCallSite(line_start=5, line_end=5)  # same line, the unsafe twin
    other = ExecuteCallSite(line_start=9, line_end=9)
    scan = ParameterizedCallScan(
        safe_parameterized_calls=(twin_a,),
        all_execute_like_calls=(twin_a, twin_b, other),
    )
    unsafe = Counter((s.line_start, s.line_end) for s in scan.unsafe_parameterized_calls)
    assert unsafe == Counter({(5, 5): 1, (9, 9): 1})


def test_unsafe_empty_when_all_calls_safe() -> None:
    site = ExecuteCallSite(line_start=3, line_end=3)
    scan = ParameterizedCallScan(
        safe_parameterized_calls=(site,),
        all_execute_like_calls=(site,),
    )
    assert scan.unsafe_parameterized_calls == ()


def test_unsafe_field_excluded_from_cache_digest() -> None:
    """`unsafe_parameterized_calls` is a derived `@property`, not a stored field —
    so it stays out of `scan_digest` (the cache key, FUP-171); the digest is a
    pure function of (safe, all)."""
    safe = ExecuteCallSite(line_start=2, line_end=2)
    unsafe = ExecuteCallSite(line_start=4, line_end=4)
    scan = ParameterizedCallScan(
        safe_parameterized_calls=(safe,),
        all_execute_like_calls=(safe, unsafe),
    )
    # The digest depends only on the two stored sets; a second scan with the same
    # (safe, all) digests identically regardless of the derived unsafe property.
    twin = ParameterizedCallScan(
        safe_parameterized_calls=(ExecuteCallSite(line_start=2, line_end=2),),
        all_execute_like_calls=(
            ExecuteCallSite(line_start=2, line_end=2),
            ExecuteCallSite(line_start=4, line_end=4),
        ),
    )
    assert scan_digest(scan) == scan_digest(twin)
    assert scan.unsafe_parameterized_calls  # non-empty, but absent from the digest
