"""ReviewFinding wires enforce_proof_boundary as a Pydantic model_validator.

Backs `evidence-tier-schema-enforced`. The validator's own correctness
is covered by the policy-module proof-boundary tests
(test_proof_boundary_observed.py / _inferred.py / _judged.py /
_invalid_tier.py); these tests verify the wiring — that constructing
a ReviewFinding triggers the validator at model-construction time and
surfaces ProofBoundaryViolationError as a Pydantic ValidationError.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension, ReviewFinding


def _build_finding(**overrides: Any) -> ReviewFinding:
    """Construct a valid OBSERVED ReviewFinding; overrides replace defaults."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "policy_version": "1.0.0",
        "finding_type": FindingType.SQL_INJECTION,
        "dimension": ReviewDimension.SECURITY,
        "severity": FindingSeverity.CRITICAL,
        "evidence_tier": EvidenceTier.OBSERVED,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "title": "SQL injection via string concatenation",
        "description": "User input is concatenated into the SQL string.",
        "evidence": 'cursor.execute("SELECT * FROM users WHERE id = " + user_id)',
        "query_match_id": "py.security.sql_injection.string_concat",
        "trace_path": None,
        "content_hash": "sha256-abc123",
    }
    fields.update(overrides)
    return ReviewFinding(**fields)


def test_review_finding_admits_observed_with_query_match_id() -> None:
    """Happy path: OBSERVED + query_match_id constructs cleanly."""
    finding = _build_finding(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    assert finding.evidence_tier == EvidenceTier.OBSERVED


def test_review_finding_rejects_observed_without_query_match_id() -> None:
    """OBSERVED + None query_match_id raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty str query_match_id"):
        _build_finding(
            evidence_tier=EvidenceTier.OBSERVED,
            query_match_id=None,
        )


def test_review_finding_rejects_inferred_without_trace_path() -> None:
    """INFERRED + None trace_path raises ValidationError via the validator."""
    with pytest.raises(ValidationError, match="non-empty list"):
        _build_finding(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=None,
        )


def test_review_finding_admits_judged_without_artifacts() -> None:
    """JUDGED admits without query_match_id or trace_path."""
    finding = _build_finding(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
    )
    assert finding.evidence_tier == EvidenceTier.JUDGED
