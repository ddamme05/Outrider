# See specs/2026-05-19-analyze-foundation.md §1.
"""`TraceCandidate` schema tests.

Pins the §1 schema discipline: frozen + `extra="forbid"`, SHA-256 hex
`candidate_id` and `source_proposal_hash`, bounded string lengths on
`reason` and `candidate_path`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.policy.canonical import compute_candidate_id, compute_identity_hash
from outrider.schemas import TraceCandidate


def _kwargs(**overrides: object) -> dict[str, object]:
    """Fixture kwargs with a canonical `candidate_id` derived from the
    candidate's payload. Tests that deliberately exercise drift override
    `candidate_id` directly; tests that exercise other-field validators
    (path normalization, length, hex-shape on `source_proposal_hash`,
    etc.) override those fields and the helper recomputes the
    `candidate_id` to match — otherwise a canonical-ID-mismatch fires
    first and the field the test is actually exercising is never
    reached.
    """
    source_proposal_hash = compute_identity_hash({"prop": 1})
    candidate_path = "src/middleware/auth.py"
    reason = "auth middleware referenced by the finding"
    base: dict[str, object] = {
        "source_proposal_hash": source_proposal_hash,
        "reason": reason,
        "candidate_path": candidate_path,
    }
    base.update(overrides)
    if "candidate_id" not in overrides:
        source_value = base["source_proposal_hash"]
        path_value = base["candidate_path"]
        reason_value = base["reason"]
        assert isinstance(source_value, str)
        assert isinstance(path_value, str)
        assert isinstance(reason_value, str)
        base["candidate_id"] = compute_candidate_id(
            source_proposal_hash=source_value,
            candidate_path=path_value,
            reason=reason_value,
        )
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
    """Sanity: the two hash fields are distinct.

    Construct via the canonical recipe (post-PR review fold:
    `candidate_id` is now bound to payload). The two hex strings come
    from independent derivations — `source_proposal_hash` from the raw
    proposal payload, `candidate_id` from `compute_candidate_id` — and
    are not coincidentally equal.
    """
    prop = compute_identity_hash({"b": 2})
    path = "src/whatever.py"
    reason = "r"
    cand = compute_candidate_id(
        source_proposal_hash=prop,
        candidate_path=path,
        reason=reason,
    )
    c = TraceCandidate(
        candidate_id=cand,
        source_proposal_hash=prop,
        reason=reason,
        candidate_path=path,
    )
    assert c.candidate_id == cand
    assert c.source_proposal_hash == prop
    assert c.candidate_id != c.source_proposal_hash
