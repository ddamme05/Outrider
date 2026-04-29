# HITL gate envelope per docs/spec.md §6.4 + §7.4 (severity override)
"""HITL gate envelope: PerFindingOutcome / PerFindingDecision / HITLRequest / HITLDecision.

These models are the typed surface the publish gate consumes. The HITL node
(agent/nodes/hitl.py, separate spec) interrupts the LangGraph state machine
when any finding has severity CRITICAL or HIGH per spec §6.4; the dashboard's
POST /reviews/{id}/decide endpoint constructs a HITLDecision from reviewer
input and resumes the graph with Command(resume=hitl_decision).

All three of PerFindingDecision / HITLRequest / HITLDecision use frozen=True:
they are decision artifacts that are final at construction. A reviewer's
per-finding decision doesn't change after submission; HITLDecision is the
full set of decisions, final at submit; HITLRequest is the agent's gate
envelope at the moment it interrupts the graph (reviewer state mutates
HITLDecision, not HITLRequest). Contrast ReviewFinding (NOT frozen) — see
schemas/review_finding.py module docstring for the lifecycle rationale.

PerFindingDecision.enforce_override_fields covers BOTH spec §7.4 rules:
(1) SEVERITY_OVERRIDE requires both override_severity and original_severity
    (lines 277-283), and
(2) any non-APPROVE outcome requires a non-empty reason (lines 284-285).
APPROVE callers pass reason="" to keep the decision-record shape uniform.
"""

from enum import StrEnum
from typing import Self
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from outrider.policy import FindingSeverity


class PerFindingOutcome(StrEnum):
    """Per-finding HITL decision outcomes per spec §7.4."""

    APPROVE = "approve"
    REJECT = "reject"
    SUPPRESS = "suppress"
    SEVERITY_OVERRIDE = "severity_override"


class PerFindingDecision(BaseModel):
    """One reviewer's decision on one finding.

    Frozen: a per-finding decision is final at construction. Reviewer-state
    revisions produce a new decision, not a mutation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: UUID
    outcome: PerFindingOutcome
    reason: str = Field(max_length=500)
    override_severity: FindingSeverity | None = None
    original_severity: FindingSeverity | None = None

    @model_validator(mode="after")
    def enforce_override_fields(self) -> Self:
        """Spec §7.4: SEVERITY_OVERRIDE needs both severities; non-APPROVE needs reason."""
        if self.outcome == PerFindingOutcome.SEVERITY_OVERRIDE and (
            not self.override_severity or not self.original_severity
        ):
            raise ValueError("severity_override requires override_severity and original_severity")
        if self.outcome != PerFindingOutcome.APPROVE and not self.reason:
            raise ValueError(f"{self.outcome.value} requires a reason")
        return self


class HITLRequest(BaseModel):
    """Agent's gate envelope when it interrupts the graph for HITL approval.

    Frozen: the request is the snapshot of the gate context at interrupt time.
    A new request is constructed if the gate is re-entered; the old one is
    not mutated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings_requiring_approval: list[UUID]
    auto_post_findings: list[UUID]
    created_at: AwareDatetime
    expires_at: AwareDatetime


class HITLDecision(BaseModel):
    """Reviewer's full decision set for a HITL gate, final at submission.

    Frozen: the decision is the audit record at submit time. Per spec §7.4
    line 290 the field is `decisions: list[PerFindingDecision]` (NOT
    `per_finding_decisions`); the dashboard endpoint constructs this from
    reviewer input and the graph resumes with Command(resume=hitl_decision).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_id: str
    decisions: list[PerFindingDecision]
    annotation: str | None = None
    decided_at: AwareDatetime


__all__ = [
    "HITLDecision",
    "HITLRequest",
    "PerFindingDecision",
    "PerFindingOutcome",
]
