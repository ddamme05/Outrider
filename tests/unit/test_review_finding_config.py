"""ReviewFinding model configuration: extra='forbid', NOT frozen, validate_assignment.

Four rule families covered here:
  - Pydantic config: extra='forbid' rejects unknown fields; frozen is
    deliberately OFF (multi-stage lifecycle — see review_finding.py
    module docstring); validate_assignment=True so post-construction
    writes re-run model_validators + Field constraints + enum typing.
  - Enum gates: invalid string values raise (at construction AND on
    assignment). Pydantic V2 coerces VALID string values to enum members
    (that's fine; the resulting field is still an enum instance, so
    `severity-set-by-policy` and `finding-type-enum-constrained` still
    hold). The gate that matters is rejection of invalid values.
  - Line constraints: line_start ≥ 1, line_end ≥ line_start.
  - Validate-on-assignment: lifecycle setters cannot bypass the proof
    boundary, the line constraint, or the enum gates.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import PublishDestination, ReviewDimension, ReviewFinding


def _build_finding(**overrides: Any) -> ReviewFinding:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "policy_version": "1.0.0",
        "finding_type": FindingType.SQL_INJECTION,
        "dimension": ReviewDimension.SECURITY,
        "severity": FindingSeverity.CRITICAL,
        "evidence_tier": EvidenceTier.JUDGED,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "content_hash": "h",
    }
    fields.update(overrides)
    return ReviewFinding(**fields)


def test_review_finding_extra_forbid() -> None:
    """Unknown fields raise ValidationError per docs/conventions.md."""
    with pytest.raises(ValidationError, match="extra"):
        _build_finding(unknown_field="oops")  # type: ignore[call-arg]


def test_review_finding_is_mutable_for_lifecycle() -> None:
    """ReviewFinding is NOT frozen: lifecycle stages set fields after construction.

    Regression guard against a future PR that adds frozen=True without
    recognizing the lifecycle implication. `coordinates/` sets
    publish_destination; HITL flow sets the override fields. Both rely
    on mutation. See review_finding.py module docstring + the
    schemas-module spec's Compliance notes for the full rationale.
    """
    finding = _build_finding()
    assert finding.publish_destination is None
    finding.publish_destination = PublishDestination.INLINE_COMMENT
    assert finding.publish_destination == PublishDestination.INLINE_COMMENT


def test_review_finding_severity_rejects_invalid_string() -> None:
    """A string value not in FindingSeverity raises."""
    with pytest.raises(ValidationError):
        _build_finding(severity="catastrophic")


def test_review_finding_severity_accepts_enum_member() -> None:
    """FindingSeverity enum member admits cleanly."""
    finding = _build_finding(severity=FindingSeverity.HIGH)
    assert finding.severity == FindingSeverity.HIGH


def test_review_finding_finding_type_rejects_invalid_string() -> None:
    """A string value not in FindingType raises (backs finding-type-enum-constrained)."""
    with pytest.raises(ValidationError):
        _build_finding(finding_type="not_a_real_finding_type")


def test_review_finding_line_start_ge_1() -> None:
    """line_start = 0 raises (1-indexed per coordinates/)."""
    with pytest.raises(ValidationError):
        _build_finding(line_start=0, line_end=5)


def test_review_finding_line_end_ge_line_start() -> None:
    """line_end < line_start raises via the model_validator."""
    with pytest.raises(ValidationError, match="line_end"):
        _build_finding(line_start=10, line_end=5)


def test_review_finding_line_end_equal_line_start_admits() -> None:
    """Single-line findings (line_start == line_end) admit."""
    finding = _build_finding(line_start=42, line_end=42)
    assert finding.line_start == 42
    assert finding.line_end == 42


def test_review_finding_validate_assignment_blocks_invalid_severity_string() -> None:
    """Post-construction, assigning an invalid string to .severity raises.

    Without validate_assignment=True, `finding.severity = "garbage"` would
    silently admit because Pydantic does not revalidate by default. With
    it, the assignment runs the same enum-coercion check as construction.
    """
    finding = _build_finding()
    with pytest.raises(ValidationError):
        finding.severity = "catastrophic"  # type: ignore[assignment]


def test_review_finding_validate_assignment_blocks_invalid_publish_destination() -> None:
    """Lifecycle assignment of publish_destination revalidates the enum."""
    finding = _build_finding()
    with pytest.raises(ValidationError):
        finding.publish_destination = "broadcast"  # type: ignore[assignment]


def test_review_finding_validate_assignment_runs_proof_boundary() -> None:
    """Stripping query_match_id from an OBSERVED finding post-construction raises.

    The proof-boundary model_validator runs on assignment, so the lifecycle
    cannot wash out the OBSERVED → query_match_id requirement after the fact.
    """
    finding = _build_finding(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.placeholder",
    )
    with pytest.raises(ValidationError, match="non-empty str query_match_id"):
        finding.query_match_id = None


def test_review_finding_validate_assignment_runs_line_constraint() -> None:
    """Setting line_end below line_start post-construction raises."""
    finding = _build_finding(line_start=10, line_end=20)
    with pytest.raises(ValidationError, match="line_end"):
        finding.line_end = 5
