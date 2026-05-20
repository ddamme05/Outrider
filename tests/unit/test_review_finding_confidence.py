"""ReviewFinding.confidence is a deterministic computed_field.

Backs `confidence-is-computed-not-assigned`. Confidence is derived from
evidence_tier (OBSERVED=0.9, INFERRED=0.75, JUDGED=0.5 per spec §7.3),
not persisted by the model and not settable. The schema-layer migration
deliberately omits a `confidence` column; this test verifies the
Pydantic side mirrors that omission as a read-only computed field.
"""

from typing import Any
from uuid import uuid4

import pytest

from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension, ReviewFinding


def _build_finding(**overrides: Any) -> ReviewFinding:
    """Construct a valid finding; overrides replace defaults.

    `content_hash` is computed from the canonical recipe over the
    post-override payload so the `_verify_content_hash` validator
    doesn't fire on tests that override identity-tuple fields.
    """
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "policy_version": "1.0.0",
        "finding_type": FindingType.SQL_INJECTION,
        "dimension": ReviewDimension.SECURITY,
        "severity": FindingSeverity.CRITICAL,
        "evidence_tier": EvidenceTier.JUDGED,
        "file_path": "src/foo.py",
        "line_start": 1,
        "line_end": 1,
        "title": "t",
        "description": "d",
        "evidence": "e",
    }
    fields.update(overrides)
    if "content_hash" not in overrides:
        fields["content_hash"] = compute_finding_content_hash(
            file_path=fields["file_path"],
            line_start=fields["line_start"],
            line_end=fields["line_end"],
            finding_type=fields["finding_type"],
        )
    return ReviewFinding(**fields)


def test_confidence_is_observed_0_9() -> None:
    """OBSERVED → 0.9 per spec §7.3."""
    finding = _build_finding(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    assert finding.confidence == 0.9


def test_confidence_is_inferred_0_75() -> None:
    """INFERRED → 0.75 per spec §7.3."""
    finding = _build_finding(
        evidence_tier=EvidenceTier.INFERRED,
        trace_path=["scope_a", "scope_b"],
    )
    assert finding.confidence == 0.75


def test_confidence_is_judged_0_5() -> None:
    """JUDGED → 0.5 per spec §7.3."""
    finding = _build_finding(evidence_tier=EvidenceTier.JUDGED)
    assert finding.confidence == 0.5


def test_confidence_assignment_raises() -> None:
    """confidence is a computed_field; assigning to it raises AttributeError.

    Pydantic computed_field is always read-only at the descriptor level,
    regardless of model frozen-ness. Backs `confidence-is-computed-not-assigned`.
    """
    finding = _build_finding(evidence_tier=EvidenceTier.JUDGED)
    with pytest.raises(AttributeError):
        finding.confidence = 0.99  # type: ignore[misc]


def test_confidence_appears_in_model_dump() -> None:
    """model_dump includes the computed field — audit/dashboard surfaces see it."""
    finding = _build_finding(evidence_tier=EvidenceTier.JUDGED)
    dumped = finding.model_dump()
    assert "confidence" in dumped
    assert dumped["confidence"] == 0.5
