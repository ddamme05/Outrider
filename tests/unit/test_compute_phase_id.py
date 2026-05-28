"""Canonical-recipe pinning for `compute_phase_id`.

`ReviewPhaseEvent` uses `phase_id: str`. `PhaseEventSink` idempotency
keys on `(review_id, phase_id, marker)`; LangGraph checkpoint replay
re-runs the body and re-mints, so the helper produces a deterministic
SHA-256 hex digest over `(review_id, node_id, attempt_key)`.

Tests pin: determinism, cross-node uniqueness, cross-attempt_key
uniqueness, cross-review uniqueness, and recipe-equivalence against
the canonical-encoding chokepoint.
"""

from __future__ import annotations

import hashlib
import re
from uuid import uuid4

import pytest

from outrider.policy.canonical import (
    SHA256_HEX_PATTERN,
    canonicalize_for_hash,
    compute_phase_id,
)


def test_returns_sha256_hex_string() -> None:
    """Shape: 64-char lowercase hex matching the canonical pattern."""
    result = compute_phase_id(
        review_id="11111111-1111-1111-1111-111111111111",
        node_id="hitl",
        attempt_key="hitl",
    )
    assert isinstance(result, str)
    assert re.fullmatch(SHA256_HEX_PATTERN, result), result


def test_determinism_same_inputs_same_digest() -> None:
    """Same inputs MUST produce same digest across calls."""
    review_id = "22222222-2222-2222-2222-222222222222"
    a = compute_phase_id(review_id=review_id, node_id="trace", attempt_key="trace-pass-0")
    b = compute_phase_id(review_id=review_id, node_id="trace", attempt_key="trace-pass-0")
    assert a == b


def test_cross_node_uniqueness() -> None:
    """Different `node_id` with same other inputs produces different digest.

    Guards against the copy-paste failure mode where two nodes pass the
    same attempt_key (e.g., both use `"intake"`) and collide on the
    PhaseEventSink idempotency key.
    """
    review_id = "33333333-3333-3333-3333-333333333333"
    attempt_key = "shared"
    intake = compute_phase_id(review_id=review_id, node_id="intake", attempt_key=attempt_key)
    triage = compute_phase_id(review_id=review_id, node_id="triage", attempt_key=attempt_key)
    analyze = compute_phase_id(review_id=review_id, node_id="analyze", attempt_key=attempt_key)
    trace = compute_phase_id(review_id=review_id, node_id="trace", attempt_key=attempt_key)
    synthesize = compute_phase_id(
        review_id=review_id, node_id="synthesize", attempt_key=attempt_key
    )
    publish = compute_phase_id(review_id=review_id, node_id="publish", attempt_key=attempt_key)
    hitl = compute_phase_id(review_id=review_id, node_id="hitl", attempt_key=attempt_key)
    digests = {intake, triage, analyze, trace, synthesize, publish, hitl}
    assert len(digests) == 7, "all seven node_ids must produce distinct phase_ids"


def test_cross_attempt_key_uniqueness() -> None:
    """Different `attempt_key` (e.g., analyze-pass-0 vs analyze-pass-1) → different digest."""
    review_id = "44444444-4444-4444-4444-444444444444"
    pass_0 = compute_phase_id(review_id=review_id, node_id="analyze", attempt_key="analyze-pass-0")
    pass_1 = compute_phase_id(review_id=review_id, node_id="analyze", attempt_key="analyze-pass-1")
    pass_2 = compute_phase_id(review_id=review_id, node_id="analyze", attempt_key="analyze-pass-2")
    assert pass_0 != pass_1
    assert pass_1 != pass_2
    assert pass_0 != pass_2


def test_cross_review_uniqueness() -> None:
    """Different `review_id` → different digest (cross-review isolation)."""
    a = compute_phase_id(
        review_id="55555555-5555-5555-5555-555555555555",
        node_id="hitl",
        attempt_key="hitl",
    )
    b = compute_phase_id(
        review_id="66666666-6666-6666-6666-666666666666",
        node_id="hitl",
        attempt_key="hitl",
    )
    assert a != b


def test_routes_through_canonicalize_for_hash() -> None:
    """Recipe-equivalence: helper output equals SHA-256 over the canonical
    UTF-8 bytes of `{review_id, node_id, attempt_key}`.

    Pins the encoding so a future drift in `compute_phase_id`'s payload
    construction surfaces against a known recipe.
    """
    review_id = "77777777-7777-7777-7777-777777777777"
    node_id = "hitl"
    attempt_key = "hitl"
    expected = hashlib.sha256(
        canonicalize_for_hash(
            {"review_id": review_id, "node_id": node_id, "attempt_key": attempt_key}
        )
    ).hexdigest()
    actual = compute_phase_id(review_id=review_id, node_id=node_id, attempt_key=attempt_key)
    assert actual == expected


def test_uuid4_review_id_string_form_works() -> None:
    """Production call sites pass `str(state.review_id)` (UUID → str)."""
    review_id_uuid = uuid4()
    result = compute_phase_id(
        review_id=str(review_id_uuid),
        node_id="hitl",
        attempt_key="hitl",
    )
    assert re.fullmatch(SHA256_HEX_PATTERN, result)


@pytest.mark.parametrize(
    "kwarg",
    ["review_id", "node_id", "attempt_key"],
)
def test_all_kwargs_required(kwarg: str) -> None:
    """All three params are keyword-only and required."""
    all_kwargs = {
        "review_id": "88888888-8888-8888-8888-888888888888",
        "node_id": "hitl",
        "attempt_key": "hitl",
    }
    del all_kwargs[kwarg]
    with pytest.raises(TypeError):
        compute_phase_id(**all_kwargs)  # type: ignore[arg-type]


def test_known_golden() -> None:
    """Pin a known-input → known-output digest as a golden fixture.

    Dual-anchor: the hardcoded byte sequence catches accidental drift
    in `canonicalize_for_hash` (contributor updates canonicalize without
    realizing it shifts the wire-shape), AND we cross-check that
    `canonicalize_for_hash` over the same input produces the same bytes
    (contributor updates the hardcoded bytes without updating
    canonicalize, or vice versa).

    A change to the recipe (key ordering, encoding, key names) breaks
    this loudly. Update the golden ONLY when the recipe change is
    intentional and reflected in `DECISIONS.md` or the spec.
    """
    review_id = "00000000-0000-0000-0000-000000000000"
    node_id = "hitl"
    attempt_key = "hitl"
    result = compute_phase_id(
        review_id=review_id,
        node_id=node_id,
        attempt_key=attempt_key,
    )
    # Golden bytes per canonical recipe over
    # {"attempt_key":"hitl","node_id":"hitl","review_id":"00000000-..."}
    # (sort_keys=True, separators=(",", ":"), ensure_ascii=False).
    expected_payload = (
        b'{"attempt_key":"hitl","node_id":"hitl",'
        b'"review_id":"00000000-0000-0000-0000-000000000000"}'
    )
    # Cross-check: canonicalize_for_hash produces the same bytes.
    # If a contributor updates one anchor without the other this
    # comparison fails loudly with both byte sequences visible.
    canonicalized = canonicalize_for_hash(
        {"review_id": review_id, "node_id": node_id, "attempt_key": attempt_key}
    )
    assert canonicalized == expected_payload, (
        f"canonicalize_for_hash drifted from the golden bytes:\n"
        f"  expected: {expected_payload!r}\n"
        f"  actual:   {canonicalized!r}"
    )
    expected = hashlib.sha256(expected_payload).hexdigest()
    assert result == expected
