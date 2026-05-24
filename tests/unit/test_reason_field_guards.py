"""Reason-field length bound on TraceDecisionEvent + nested PerFindingDecision.reason.

Per `docs/schema.md` line 213 + `DECISIONS.md#014` point 5 (Amended same-day
clauses): two distinct properties — only one is mechanically enforced.

  - Length bound (mechanical): Pydantic Field(max_length=500) on each
    reason field; the validator rejects any string longer than 500 chars.
  - Structural-description-not-code-snippet (author/reviewer discipline):
    no automated heuristic for "looks like code" vs "is a structural
    description"; the rule is documented and reviewed at PR time, not
    gated by code.

These tests cover the mechanical gate. The discipline rule is out of
scope for validators; the happy-path test below admits short structural
text but does NOT attempt to verify the content rule.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import HITLDecisionEvent, TraceDecisionEvent
from outrider.policy import FindingSeverity
from outrider.schemas import PerFindingDecision, PerFindingOutcome


def _build_trace_event(**overrides: Any) -> TraceDecisionEvent:
    """Per #024 amendment to #017: trace decisions carry parallel
    proposed_import_strings + resolved_candidate_paths tuples."""
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "source_finding_id": uuid4(),
        "target_file": None,
        "reason": "x",
        "resolution_status": "unresolved",
        "proposed_import_strings": (),
        "resolved_candidate_paths": (),
    }
    fields.update(overrides)
    return TraceDecisionEvent(**fields)


def test_trace_decision_event_reason_max_length_500() -> None:
    """501-char reason raises; 500-char reason admits."""
    with pytest.raises(ValidationError):
        _build_trace_event(reason="x" * 501)

    event = _build_trace_event(reason="x" * 500)
    assert len(event.reason) == 500


def test_hitl_decision_event_decisions_reason_max_length_500() -> None:
    """Nested PerFindingDecision.reason length bound survives HITLDecisionEvent wrapping.

    PerFindingDecision.reason carries Field(max_length=500) at the
    schemas-layer; the wrapping audit event inherits the constraint.
    """
    with pytest.raises(ValidationError):
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
            reason="x" * 501,
            override_severity=FindingSeverity.LOW,
            original_severity=FindingSeverity.HIGH,
        )

    decision_at_bound = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.REJECT,
        reason="x" * 500,
    )
    event = HITLDecisionEvent(
        review_id=uuid4(),
        reviewer_id="reviewer@example.com",
        decisions=(decision_at_bound,),
        decision_latency_seconds=42.5,
    )
    assert len(event.decisions[0].reason) == 500


def test_reason_fields_admit_short_structural_descriptions() -> None:
    """Happy-path: short structural text admits.

    This test does NOT verify the structural-description-not-code-snippet
    content rule — that rule is author/reviewer discipline, not validator
    gate. It admits both structural and code-snippet text equally as long
    as they're under the length bound.
    """
    structural = _build_trace_event(reason="called from middleware/auth.py:42")
    assert structural.reason.startswith("called from")
