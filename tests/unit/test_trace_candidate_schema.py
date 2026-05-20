# See specs/2026-05-19-analyze-foundation.md §1.
"""`TraceCandidate` schema tests.

Pins the §1 schema discipline: frozen + `extra="forbid"`, SHA-256 hex
`candidate_id` and `source_proposal_hash`, bounded string lengths on
`reason` and `candidate_path`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.policy.canonical import compute_identity_hash
from outrider.schemas import TraceCandidate


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "candidate_id": compute_identity_hash({"cand": 1}),
        "source_proposal_hash": compute_identity_hash({"prop": 1}),
        "reason": "auth middleware referenced by the finding",
        "candidate_path": "src/middleware/auth.py",
    }
    base.update(overrides)
    return base


def test_trace_candidate_admits_well_formed() -> None:
    c = TraceCandidate(**_kwargs())  # type: ignore[arg-type]
    assert c.candidate_path == "src/middleware/auth.py"


def test_trace_candidate_candidate_id_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(candidate_id="not-a-hash"))  # type: ignore[arg-type]


def test_trace_candidate_source_proposal_hash_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(source_proposal_hash="bad"))  # type: ignore[arg-type]


def test_trace_candidate_reason_max_length() -> None:
    """500-char cap defends against unbounded model output."""
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(reason="x" * 501))  # type: ignore[arg-type]


def test_trace_candidate_candidate_path_max_length() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(candidate_path="src/" + ("a" * 1025) + ".py"))  # type: ignore[arg-type]


def test_trace_candidate_frozen_rejects_mutation() -> None:
    c = TraceCandidate(**_kwargs())  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        c.candidate_path = "src/other.py"  # type: ignore[misc]


def test_trace_candidate_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TraceCandidate(**_kwargs(unexpected="bad"))  # type: ignore[arg-type]


def test_trace_candidate_source_proposal_hash_distinct_from_candidate_id() -> None:
    """Sanity: the two hash fields are distinct; mixing them up is a
    common content-derived-id mistake."""
    cand = compute_identity_hash({"a": 1})
    prop = compute_identity_hash({"b": 2})
    c = TraceCandidate(**_kwargs(candidate_id=cand, source_proposal_hash=prop))  # type: ignore[arg-type]
    assert c.candidate_id == cand
    assert c.source_proposal_hash == prop
    assert c.candidate_id != c.source_proposal_hash
