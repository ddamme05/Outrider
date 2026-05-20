# See specs/2026-05-19-analyze-foundation.md §7.
"""Raw + admitted analyze proposal schema tests.

Pins:
- Raw layer admits bounded INVALID enum strings (so the parser can
  emit rejection events before construction fails).
- Admitted layer rejects non-enum values (the type system gates
  finding_type + evidence_tier).
- Materially distinct field names between raw and admitted layers
  for TraceCandidate proposals (post-split S4 pit-of-success).
- Byte-for-byte span invariant: admitted == raw (post-split S6).
- Length caps on all bounded strings.
- Max-50 findings per response.
- Max-20 trace candidates per proposal.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from outrider.ast_facts.models import Span
from outrider.policy import EvidenceTier, FindingType
from outrider.schemas.llm import (
    AnalyzeFindingProposal,
    AnalyzeFindingProposalRaw,
    AnalyzeResponseRaw,
    TraceCandidateProposal,
    TraceCandidateProposalRaw,
)

# ---------------------------------------------------------------------------
# AnalyzeFindingProposalRaw — accepts bounded strings, not enums.
# ---------------------------------------------------------------------------


def _raw_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "finding_type": "sql_injection",
        "evidence_tier": "JUDGED",
        "title": "SQL injection",
        "description": "Concatenated user input.",
        "evidence": "src/foo.py:11 — raw SQL concat",
        "span": Span(byte_start=100, byte_end=120),
    }
    base.update(overrides)
    return base


def test_raw_admits_well_formed() -> None:
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs())
    assert raw.finding_type == "sql_injection"


def test_raw_admits_off_enum_finding_type() -> None:
    """The whole point of the raw layer: an off-list `finding_type`
    survives construction long enough for the parser to emit
    `FindingProposalRejectedEvent(rejection_reason='finding_type_not_in_enum')`.
    """
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs(finding_type="not_a_real_type"))
    assert raw.finding_type == "not_a_real_type"


def test_raw_admits_off_enum_evidence_tier() -> None:
    """Same shape for evidence_tier — admit invalid strings so the
    parser can emit `evidence_tier_not_in_enum` rejections."""
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs(evidence_tier="not_a_tier"))
    assert raw.evidence_tier == "not_a_tier"


def test_raw_rejects_finding_type_over_max() -> None:
    """128-char cap on finding_type — prevents unbounded hostile strings."""
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(finding_type="x" * 129))


def test_raw_rejects_title_over_max() -> None:
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(title="x" * 121))


def test_raw_rejects_description_over_max() -> None:
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(description="x" * 1001))


def test_raw_rejects_evidence_over_max() -> None:
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(evidence="x" * 2001))


def test_raw_rejects_trace_path_with_empty_step() -> None:
    """trace_path step `min_length=1` — empty strings rejected."""
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(trace_path=("",)))


def test_raw_admits_optional_trace_path() -> None:
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs(trace_path=("step1", "step2")))
    assert raw.trace_path == ("step1", "step2")


def test_raw_rejects_more_than_20_trace_candidates() -> None:
    """max_length=20 on trace_candidates."""
    candidates = tuple(
        TraceCandidateProposalRaw(candidate_path_raw=f"src/f{i}.py", reason="x") for i in range(21)
    )
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(trace_candidates=candidates))


def test_raw_frozen() -> None:
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs())
    with pytest.raises(ValidationError):
        raw.title = "other"  # type: ignore[misc]


def test_raw_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AnalyzeFindingProposalRaw(**_raw_kwargs(unexpected="bad"))


# ---------------------------------------------------------------------------
# AnalyzeFindingProposal (admitted) — enum-constrained.
# ---------------------------------------------------------------------------


def _admitted_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "finding_type": FindingType.SQL_INJECTION,
        "evidence_tier": EvidenceTier.JUDGED,
        "title": "SQL injection",
        "description": "Concatenated user input.",
        "evidence": "src/foo.py:11 — raw SQL concat",
        "span": Span(byte_start=100, byte_end=120),
    }
    base.update(overrides)
    return base


def test_admitted_admits_well_formed() -> None:
    admitted = AnalyzeFindingProposal(**_admitted_kwargs())
    assert admitted.finding_type == FindingType.SQL_INJECTION


def test_admitted_rejects_off_enum_finding_type() -> None:
    """Admitted layer enforces FindingType enum — different from raw."""
    with pytest.raises(ValidationError):
        AnalyzeFindingProposal(**_admitted_kwargs(finding_type="not_a_real_type"))


def test_admitted_rejects_off_enum_evidence_tier() -> None:
    """Admitted layer enforces EvidenceTier enum."""
    with pytest.raises(ValidationError):
        AnalyzeFindingProposal(**_admitted_kwargs(evidence_tier="not_a_tier"))


def test_admitted_frozen() -> None:
    admitted = AnalyzeFindingProposal(**_admitted_kwargs())
    with pytest.raises(ValidationError):
        admitted.title = "other"  # type: ignore[misc]


def test_admitted_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AnalyzeFindingProposal(**_admitted_kwargs(unexpected="bad"))


# ---------------------------------------------------------------------------
# Span byte-for-byte invariant (post-split S6).
# ---------------------------------------------------------------------------


def test_admitted_span_must_equal_raw_span_byte_for_byte() -> None:
    """Round-trip a raw proposal through admission; admitted.span equals
    raw.span exactly (no normalization/clipping).

    The parser MAY reject on span containment failure; it MUST NOT
    normalize the span between layers. proposal_hash on the rejection
    event canonicalizes raw.span values — if admitted normalized them,
    downstream consumers of the admitted finding would describe
    different bytes from the same hash, breaking replay.
    """
    raw_span = Span(byte_start=100, byte_end=120)
    raw = AnalyzeFindingProposalRaw(**_raw_kwargs(span=raw_span))
    # Simulate parser admission: same span, enum-coerced enum fields.
    admitted = AnalyzeFindingProposal(
        finding_type=FindingType(raw.finding_type),
        evidence_tier=EvidenceTier(raw.evidence_tier.lower()),
        title=raw.title,
        description=raw.description,
        evidence=raw.evidence,
        span=raw.span,  # Byte-for-byte: same instance, same values.
    )
    assert admitted.span == raw.span
    assert admitted.span.byte_start == raw_span.byte_start
    assert admitted.span.byte_end == raw_span.byte_end


# ---------------------------------------------------------------------------
# TraceCandidate proposals: raw vs admitted layer structural distinction.
# ---------------------------------------------------------------------------


def test_trace_candidate_raw_uses_candidate_path_raw() -> None:
    raw = TraceCandidateProposalRaw(candidate_path_raw="src/middleware/auth.py", reason="x")
    assert raw.candidate_path_raw == "src/middleware/auth.py"


def test_trace_candidate_admitted_uses_candidate_path() -> None:
    """Admitted layer uses `candidate_path` (post-validate_diff_path)."""
    admitted = TraceCandidateProposal(candidate_path="src/middleware/auth.py", reason="x")
    assert admitted.candidate_path == "src/middleware/auth.py"


@pytest.mark.parametrize(
    "bad_path",
    [
        "../escape.py",
        "src/../../escape.py",
        "/etc/passwd",
        "src/foo.py\x00",
    ],
)
def test_trace_candidate_admitted_rejects_invalid_path(bad_path: str) -> None:
    """The admitted-layer `candidate_path` validator re-runs
    `validate_diff_path` so traversal / absolute / NUL-bearing paths
    are refused at the schema boundary, NOT just at the raw→admitted
    translator. Pins the schema-layer guarantee against silent
    construction of an admitted proposal with a path the API surface
    would reject.

    Pydantic V2's `field_validator` re-raises non-`ValueError` /
    non-`AssertionError` exceptions directly, so a `CoordinateError`
    from `validate_diff_path` surfaces as itself, not wrapped in a
    `ValidationError`. The test accepts either.
    """
    from outrider.coordinates import CoordinateError

    with pytest.raises((ValidationError, CoordinateError)):
        TraceCandidateProposal(candidate_path=bad_path, reason="x")


def test_trace_candidate_admitted_rejects_candidate_path_raw_kwarg() -> None:
    """The raw layer's field name is NOT a valid admitted field — post-
    split S4 pit-of-success: a raw→admitted swap fails Pydantic
    construction under `extra='forbid'`."""
    with pytest.raises(ValidationError):
        TraceCandidateProposal(candidate_path_raw="src/x.py", reason="x")  # type: ignore[call-arg]


def test_trace_candidate_raw_rejects_candidate_path_kwarg() -> None:
    """Symmetric: the admitted field name is not a raw field."""
    with pytest.raises(ValidationError):
        TraceCandidateProposalRaw(candidate_path="src/x.py", reason="x")  # type: ignore[call-arg]


def test_trace_candidate_reason_max_length() -> None:
    """500-char cap shared between raw and admitted reason fields."""
    with pytest.raises(ValidationError):
        TraceCandidateProposalRaw(candidate_path_raw="src/x.py", reason="r" * 501)


# ---------------------------------------------------------------------------
# AnalyzeResponseRaw — top-level wrapper.
# ---------------------------------------------------------------------------


def test_response_raw_admits_well_formed() -> None:
    response = AnalyzeResponseRaw(findings=(AnalyzeFindingProposalRaw(**_raw_kwargs()),))
    assert len(response.findings) == 1


def test_response_raw_admits_empty_findings() -> None:
    """A 0-findings analyze pass is valid (clean review, nothing flagged)."""
    response = AnalyzeResponseRaw(findings=())
    assert response.findings == ()


def test_response_raw_rejects_over_50_findings() -> None:
    """Per-call output ceiling defends against runaway emission."""
    one = AnalyzeFindingProposalRaw(**_raw_kwargs())
    with pytest.raises(ValidationError):
        AnalyzeResponseRaw(findings=tuple([one] * 51))


def test_response_raw_frozen() -> None:
    response = AnalyzeResponseRaw(findings=())
    with pytest.raises(ValidationError):
        response.findings = (AnalyzeFindingProposalRaw(**_raw_kwargs()),)  # type: ignore[misc]


def test_response_raw_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        AnalyzeResponseRaw(findings=(), unexpected="bad")  # type: ignore[call-arg]
