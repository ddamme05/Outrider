"""PerFindingDecision: enforce_override_fields covers three spec §7.4 rules.

Rule 1: SEVERITY_OVERRIDE requires both override_severity and original_severity.
Rule 2: APPROVE / REJECT / SUPPRESS must NOT carry override_severity or
        original_severity — those fields are SEVERITY_OVERRIDE-specific per
        the field docstrings.
Rule 3: any non-APPROVE outcome requires a non-empty reason.

APPROVE is the only outcome where reason="" admits — the field itself is
required (no default), so APPROVE callers pass reason="" to keep the
decision-record shape uniform across outcomes.

Also includes the frozen=True regression guard: PerFindingDecision IS frozen
(decision artifact, final at construction), in contrast to ReviewFinding which
is NOT frozen due to its multi-stage lifecycle.
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.policy import FindingSeverity
from outrider.schemas import PerFindingDecision, PerFindingOutcome


def test_severity_override_requires_both_severities() -> None:
    """SEVERITY_OVERRIDE without override_severity OR original_severity raises."""
    with pytest.raises(ValidationError, match="severity_override requires"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
            reason="bumped to high after manual review",
            override_severity=FindingSeverity.HIGH,
            original_severity=None,
        )
    with pytest.raises(ValidationError, match="severity_override requires"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
            reason="bumped to high after manual review",
            override_severity=None,
            original_severity=FindingSeverity.MEDIUM,
        )


def test_severity_override_admits_with_both_severities() -> None:
    """Happy path: SEVERITY_OVERRIDE with both severities + reason admits."""
    decision = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason="this is a false-positive in this context",
        override_severity=FindingSeverity.LOW,
        original_severity=FindingSeverity.HIGH,
    )
    assert decision.override_severity == FindingSeverity.LOW
    assert decision.original_severity == FindingSeverity.HIGH


def test_reject_requires_reason() -> None:
    """REJECT with empty reason raises (non-APPROVE rule)."""
    with pytest.raises(ValidationError, match="requires a reason"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.REJECT,
            reason="",
        )


def test_suppress_requires_reason() -> None:
    """SUPPRESS with empty reason raises (non-APPROVE rule)."""
    with pytest.raises(ValidationError, match="requires a reason"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SUPPRESS,
            reason="",
        )


def test_severity_override_requires_reason() -> None:
    """SEVERITY_OVERRIDE with empty reason raises (third leg of the non-APPROVE rule)."""
    with pytest.raises(ValidationError, match="requires a reason"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
            reason="",
            override_severity=FindingSeverity.LOW,
            original_severity=FindingSeverity.HIGH,
        )


def test_approve_allows_empty_reason() -> None:
    """APPROVE is the only outcome where reason="" admits.

    The field itself is required at construction (no default per spec §7.4
    line 269); APPROVE callers pass reason="" explicitly. A non-empty reason
    on APPROVE is also allowed (positive note).
    """
    silent_approve = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="",
    )
    assert silent_approve.reason == ""

    annotated_approve = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="looks good, nice work on the validation",
    )
    assert annotated_approve.reason == "looks good, nice work on the validation"


def test_approve_does_not_require_override_severity_fields() -> None:
    """APPROVE / REJECT / SUPPRESS admit without override_severity / original_severity."""
    PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="",
    )
    PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.REJECT,
        reason="duplicate of finding XYZ",
    )
    PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.SUPPRESS,
        reason="known false positive on this codebase",
    )


def test_approve_with_override_severity_raises() -> None:
    """APPROVE must NOT carry override_severity (field is SEVERITY_OVERRIDE-specific)."""
    with pytest.raises(ValidationError, match="must not carry override_severity"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.APPROVE,
            reason="",
            override_severity=FindingSeverity.HIGH,
            original_severity=None,
        )


def test_approve_with_original_severity_raises() -> None:
    """APPROVE must NOT carry original_severity either."""
    with pytest.raises(ValidationError, match="must not carry override_severity"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.APPROVE,
            reason="",
            override_severity=None,
            original_severity=FindingSeverity.HIGH,
        )


def test_reject_with_override_severity_raises() -> None:
    """REJECT must NOT carry override_severity (field is SEVERITY_OVERRIDE-specific)."""
    with pytest.raises(ValidationError, match="must not carry override_severity"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.REJECT,
            reason="duplicate of finding XYZ",
            override_severity=FindingSeverity.LOW,
            original_severity=FindingSeverity.HIGH,
        )


def test_suppress_with_original_severity_raises() -> None:
    """SUPPRESS must NOT carry original_severity (field is SEVERITY_OVERRIDE-specific)."""
    with pytest.raises(ValidationError, match="must not carry override_severity"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SUPPRESS,
            reason="known false positive on this codebase",
            override_severity=None,
            original_severity=FindingSeverity.MEDIUM,
        )


def test_per_finding_decision_is_frozen() -> None:
    """PerFindingDecision IS frozen — assigning to a field after construction raises.

    Decision artifact, final at construction. Contrast ReviewFinding (NOT
    frozen) which has a multi-stage lifecycle.
    """
    decision = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="",
    )
    with pytest.raises(ValidationError):
        decision.reason = "changed my mind"  # type: ignore[misc]


def test_per_finding_decision_extra_forbid() -> None:
    """Unknown fields raise per docs/conventions.md."""
    with pytest.raises(ValidationError, match="extra"):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.APPROVE,
            reason="",
            unknown_field="oops",  # type: ignore[call-arg]
        )
