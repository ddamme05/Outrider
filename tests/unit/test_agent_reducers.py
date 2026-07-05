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
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, Field

from outrider.agent.reducers import (
    SlotDivergenceError,
    append_with_dedup_by,
    append_with_slot_guard,
    semantic_digest,
)


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


# ---------------------------------------------------------------------------
# Slot-guard reducer + semantic digest (DECISIONS.md#063 amendment).
# ---------------------------------------------------------------------------


class _StandInFinding(BaseModel):
    finding_id: UUID = Field(default_factory=uuid4)
    title: str = "t"
    severity: str = "low"


class _SlotOutcome(BaseModel):
    """Stand-in worker outcome: a positional slot + semantic content + a
    generated identity nested at depth (the ReviewFinding pattern)."""

    path: str
    pass_index: int
    findings: tuple[_StandInFinding, ...] = ()
    description: str = ""


_EXCLUDED: frozenset[str] = frozenset({"finding_id"})


def _slot(o: _SlotOutcome) -> tuple[str, int]:
    return (o.path, o.pass_index)


def _digest(o: _SlotOutcome) -> str:
    return semantic_digest(o, exclude_fields=_EXCLUDED)


def test_same_semantics_different_uuids_dedup_as_noop() -> None:
    """THE #063 pin: an identical retry carries fresh generated UUIDs
    (ReviewFinding.finding_id is default_factory=uuid4); raw equality would
    falsely reject it, but the semantic digest excludes generated identities,
    so replay re-application is an idempotent no-op."""
    first = _SlotOutcome(path="a.py", pass_index=0, findings=(_StandInFinding(title="x"),))
    retry = _SlotOutcome(path="a.py", pass_index=0, findings=(_StandInFinding(title="x"),))
    assert first.findings[0].finding_id != retry.findings[0].finding_id  # fresh UUIDs
    reducer = append_with_slot_guard(_slot, _digest)
    merged = reducer([first], [retry])
    assert merged == [first]  # no-op; first occupant retained


def test_divergent_same_slot_fails_loud_never_first_wins() -> None:
    """Same slot, different semantic content → SlotDivergenceError. Silently
    keeping the first would fork state from the audit stream's record of the
    retry."""
    first = _SlotOutcome(path="a.py", pass_index=0, description="found nothing")
    diverged = _SlotOutcome(path="a.py", pass_index=0, description="found a bug")
    reducer = append_with_slot_guard(_slot, _digest)
    with pytest.raises(SlotDivergenceError, match="a.py"):
        reducer([first], [diverged])


def test_distinct_slots_append_and_replay_is_idempotent() -> None:
    reducer = append_with_slot_guard(_slot, _digest)
    a = _SlotOutcome(path="a.py", pass_index=0)
    b = _SlotOutcome(path="b.py", pass_index=0)
    a_pass1 = _SlotOutcome(path="a.py", pass_index=1)  # same file, later pass
    merged = reducer([a], [b, a_pass1])
    assert merged == [a, b, a_pass1]
    # Full-delta re-application (checkpoint replay) is a no-op.
    assert reducer(merged, [a, b, a_pass1]) == merged


def test_semantic_digest_excludes_named_fields_at_any_depth() -> None:
    """Exclusion is by-name-anywhere: the generated identity nests inside the
    findings list, not at the top level."""
    one = _SlotOutcome(path="a.py", pass_index=0, findings=(_StandInFinding(title="x"),))
    two = _SlotOutcome(path="a.py", pass_index=0, findings=(_StandInFinding(title="x"),))
    three = _SlotOutcome(path="a.py", pass_index=0, findings=(_StandInFinding(title="y"),))
    assert _digest(one) == _digest(two)  # UUIDs differ, semantics equal
    assert _digest(one) != _digest(three)  # semantic change moves the digest
