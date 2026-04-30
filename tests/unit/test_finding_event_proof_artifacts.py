"""FindingEvent proof + severity + computed-field surfaces.

Backs `evidence-tier-schema-enforced` (validator wired at the audit-event
layer per the spec, not just at ReviewFinding), `severity-set-by-policy`,
`finding-type-enum-constrained`, and `confidence-is-computed-not-assigned`
(by absence — confidence is NOT a field on the event; replay re-derives it
from evidence_tier).
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import FindingEvent
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension


def _build_event(**overrides: Any) -> FindingEvent:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "finding_type": FindingType.SQL_INJECTION,
        "severity": FindingSeverity.CRITICAL,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": "sha256-h",
        "evidence_tier": EvidenceTier.JUDGED,
        "policy_version": "1.0.0",
    }
    fields.update(overrides)
    return FindingEvent(**fields)


def test_finding_event_carries_evidence_tier() -> None:
    """evidence_tier is required; missing raises."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "finding_id": uuid4(),
        "finding_type": FindingType.SQL_INJECTION,
        "severity": FindingSeverity.CRITICAL,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": "sha256-h",
        "policy_version": "1.0.0",
    }
    with pytest.raises(ValidationError):
        FindingEvent(**fields)


def test_finding_event_severity_is_finding_severity_enum() -> None:
    """Bare invalid string raises (severity-set-by-policy gate)."""
    with pytest.raises(ValidationError):
        _build_event(severity="catastrophic")


def test_finding_event_finding_type_is_constrained_enum() -> None:
    """Bare invalid string raises (finding-type-enum-constrained)."""
    with pytest.raises(ValidationError):
        _build_event(finding_type="not_a_real_finding_type")


def test_finding_event_has_no_confidence_field() -> None:
    """confidence is NOT a field on the event (confidence-is-computed-not-assigned).

    Replay derives confidence at read time from evidence_tier, mirroring
    the ReviewFinding rule. Storing it here would duplicate state.
    """
    event = _build_event()
    assert "confidence" not in event.model_dump()
    with pytest.raises(ValidationError):
        _build_event(confidence=0.9)


def test_finding_event_observed_admits_with_query_match_id() -> None:
    """Happy path: OBSERVED + non-empty query_match_id constructs cleanly."""
    event = _build_event(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    assert event.evidence_tier == EvidenceTier.OBSERVED


def test_finding_event_observed_rejects_without_query_match_id() -> None:
    """OBSERVED + None query_match_id raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty str query_match_id"):
        _build_event(
            evidence_tier=EvidenceTier.OBSERVED,
            query_match_id=None,
        )


def test_finding_event_inferred_rejects_without_trace_path() -> None:
    """INFERRED + None trace_path raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty list"):
        _build_event(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=None,
        )


def test_finding_event_judged_admits_without_artifacts() -> None:
    """JUDGED admits without query_match_id or trace_path (no-structural-claim path)."""
    event = _build_event(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
    )
    assert event.evidence_tier == EvidenceTier.JUDGED
