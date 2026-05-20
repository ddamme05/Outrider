# See specs/2026-05-19-analyze-foundation.md §2.
"""`append_with_dedup_by` reducer factory tests.

Pins the §2 contract: (a) plain append when no duplicates; (b) dedup
on replay (same key appearing twice → kept once); (c) different-content-
same-key resolution to first-seen; (d) empty existing + empty incoming
→ empty; (e) the returned reducer is referentially stable for repeated
calls with the same key_fn (sanity).
"""

from __future__ import annotations

from dataclasses import dataclass

from outrider.agent.reducers import append_with_dedup_by


@dataclass(frozen=True)
class _Item:
    """Minimal frozen record for reducer tests; matches the shape of
    `AnalysisRound` / `TraceCandidate` (key field + content)."""

    key: str
    payload: str


def test_append_no_duplicates_plain_append() -> None:
    reducer = append_with_dedup_by(lambda i: i.key)
    existing = [_Item("a", "x"), _Item("b", "y")]
    incoming = [_Item("c", "z")]
    merged = reducer(existing, incoming)
    assert [i.key for i in merged] == ["a", "b", "c"]


def test_append_with_dedup_on_replay() -> None:
    """LangGraph checkpoint replay re-applies the same delta; second
    application of an item already in `existing` must be a no-op."""
    reducer = append_with_dedup_by(lambda i: i.key)
    existing = [_Item("a", "x")]
    incoming = [_Item("a", "x")]  # same as existing — replay shape
    merged = reducer(existing, incoming)
    assert len(merged) == 1
    assert merged[0] == _Item("a", "x")


def test_collision_resolves_to_first_seen() -> None:
    """Two items with the same key but different content: existing wins.

    Real-world callers use content-derived hashes so this case only
    fires on legitimate re-emission of the same logical item. The
    contract is to never silently overwrite — different-content-same-key
    is treated as a re-emission, not a state change.
    """
    reducer = append_with_dedup_by(lambda i: i.key)
    existing = [_Item("a", "original")]
    incoming = [_Item("a", "different")]
    merged = reducer(existing, incoming)
    assert len(merged) == 1
    assert merged[0].payload == "original"


def test_empty_existing_admits_incoming() -> None:
    reducer = append_with_dedup_by(lambda i: i.key)
    incoming = [_Item("a", "x"), _Item("b", "y")]
    merged = reducer([], incoming)
    assert [i.key for i in merged] == ["a", "b"]


def test_empty_incoming_returns_existing_copy() -> None:
    """Reducer must not mutate `existing` in place — return a fresh list."""
    reducer = append_with_dedup_by(lambda i: i.key)
    existing = [_Item("a", "x")]
    merged = reducer(existing, [])
    assert merged == existing
    assert merged is not existing  # fresh list, not the input ref


def test_empty_both_returns_empty() -> None:
    reducer = append_with_dedup_by(lambda i: i.key)
    assert reducer([], []) == []


def test_incoming_internal_duplicates_collapse() -> None:
    """If `incoming` itself carries duplicate keys, only the first is kept.

    Defends against a sister-spec producer that legitimately emits the
    same content twice within one delta (e.g., a retry that fires both
    on first attempt and retry); both arrive in `incoming` and only one
    should land in merged.
    """
    reducer = append_with_dedup_by(lambda i: i.key)
    incoming = [_Item("a", "x"), _Item("a", "y"), _Item("b", "z")]
    merged = reducer([], incoming)
    assert [i.key for i in merged] == ["a", "b"]
    # First-seen wins within incoming too.
    assert merged[0].payload == "x"


def test_reducer_is_stable_across_calls() -> None:
    """Same factory invocation returns one reducer; repeated calls with
    independent state lists do not leak state across calls."""
    reducer = append_with_dedup_by(lambda i: i.key)
    merged_a = reducer([], [_Item("a", "x")])
    merged_b = reducer([], [_Item("a", "y")])
    assert len(merged_a) == 1
    assert len(merged_b) == 1
    # Each call sees its own (empty) existing — the reducer is a
    # stateless function over its two args.
    assert merged_a[0].payload == "x"
    assert merged_b[0].payload == "y"
