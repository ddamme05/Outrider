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

PerFindingDecision.enforce_override_fields covers THREE spec §7.4 rules:
(1) SEVERITY_OVERRIDE requires both override_severity and original_severity
    (lines 277-283),
(2) APPROVE / REJECT / SUPPRESS must NOT carry override_severity or
    original_severity — those fields are SEVERITY_OVERRIDE-specific per the
    field docstrings ("Only set when outcome == SEVERITY_OVERRIDE"), and
(3) any non-APPROVE outcome requires a non-empty reason (lines 284-285).
APPROVE callers pass reason="" to keep the decision-record shape uniform.

HITL artifact container fields use tuple[..., ...] for true immutability:
Pydantic frozen=True only blocks attribute reassignment, not in-place
container mutation, so a list field can still be .append()'d after
construction. tuple delivers what frozen=True is meant to deliver. The
spec.md §6.4 / §7.4 sketches now also use tuple[..., ...]; field names
and roles match the spec verbatim.
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
        """Spec §7.4: bidirectional override-fields gate + non-APPROVE needs reason.

        `is None` / `is not None` rather than truthiness — `FindingSeverity`
        is a `StrEnum` where members today are all truthy strings, but
        any future member with value `""` (e.g., a `NONE = ""` placeholder)
        would silently round-trip past a `not self.override_severity`
        truthy-check. Identity comparison is the documented intent.
        """
        if self.outcome == PerFindingOutcome.SEVERITY_OVERRIDE and (
            self.override_severity is None or self.original_severity is None
        ):
            raise ValueError("severity_override requires override_severity and original_severity")
        if self.outcome != PerFindingOutcome.SEVERITY_OVERRIDE and (
            self.override_severity is not None or self.original_severity is not None
        ):
            raise ValueError(
                f"{self.outcome.value} must not carry override_severity or original_severity "
                "(those fields are severity_override-specific)"
            )
        # No-op override defense: a SEVERITY_OVERRIDE decision whose
        # `override_severity` equals the policy `original_severity`
        # claims an override that doesn't actually change severity.
        # Without this guard, the no-op override passes endpoint
        # validation, the `HITLDecisionEvent` lands in audit (append-
        # only, can't undo), `mark_running` flips `reviews.status` to
        # `running`, and the publish-time
        # `PublishEligibilityEvent._enforce_override_legitimacy`
        # validator rejects the row — wedging the review outside both
        # `/decide` retry (preflight sees `hitl_decision != NULL` →
        # 409) AND `reclaim_stuck_hitl_states` (status is `running`,
        # not `awaiting_approval` / `awaiting_approval_expired`).
        # Reject at the decision-construction site so the audit row
        # never lands. The endpoint's pre-construction check produces
        # the 422; this validator is defense-in-depth for any path
        # constructing PerFindingDecision (replay code, sweep code,
        # tests).
        if (
            self.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
            and self.override_severity == self.original_severity
        ):
            both = self.override_severity.value if self.override_severity else None
            raise ValueError(
                f"severity_override requires override_severity != original_severity "
                f"(got both = {both!r}); a real override implies a baseline-to-applied "
                f"transition. If no override is intended, use outcome=APPROVE."
            )
        if self.outcome != PerFindingOutcome.APPROVE and not self.reason.strip():
            raise ValueError(f"{self.outcome.value} requires a non-blank reason")
        return self


class HITLRequest(BaseModel):
    """Agent's gate envelope when it interrupts the graph for HITL approval.

    Frozen: the request is the snapshot of the gate context at interrupt time.
    A new request is constructed if the gate is re-entered; the old one is
    not mutated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    findings_requiring_approval: tuple[UUID, ...]
    auto_post_findings: tuple[UUID, ...]
    created_at: AwareDatetime
    expires_at: AwareDatetime

    @model_validator(mode="after")
    def _enforce_finding_partition(self) -> Self:
        """Set-semantic: each finding appears at most once across the two
        tuples. A finding is either approval-gated or auto-postable, never
        both, and never listed twice in the same tuple.
        """
        if len(self.findings_requiring_approval) != len(set(self.findings_requiring_approval)):
            raise ValueError(
                f"HITLRequest.findings_requiring_approval contains duplicate ids: "
                f"{sorted(str(u) for u in self.findings_requiring_approval)!r}"
            )
        if len(self.auto_post_findings) != len(set(self.auto_post_findings)):
            raise ValueError(
                f"HITLRequest.auto_post_findings contains duplicate ids: "
                f"{sorted(str(u) for u in self.auto_post_findings)!r}"
            )
        overlap = set(self.findings_requiring_approval) & set(self.auto_post_findings)
        if overlap:
            raise ValueError(
                f"HITLRequest: a finding cannot be in both "
                f"findings_requiring_approval and auto_post_findings; "
                f"overlap: {sorted(str(u) for u in overlap)!r}"
            )
        return self


class HITLDecision(BaseModel):
    """Reviewer's full decision set for a HITL gate, final at submission.

    Frozen: the decision is the audit record at submit time. Per spec §7.4
    line 290 the field is `decisions: tuple[PerFindingDecision, ...]` (NOT
    `per_finding_decisions`; tuple-not-list for true immutability — see
    module docstring); the dashboard endpoint constructs this from reviewer
    input and the graph resumes with Command(resume=hitl_decision).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    reviewer_id: str
    decisions: tuple[PerFindingDecision, ...]
    # Mirror of `HITLDecisionEvent.annotation` cap. Without parity, the
    # state-layer could hold an annotation that fails the audit-event
    # mirror's `max_length=2000` validation — breaking the state→audit
    # shadow contract.
    annotation: str | None = Field(default=None, max_length=2000)
    decided_at: AwareDatetime

    @model_validator(mode="after")
    def _enforce_one_decision_per_finding(self) -> Self:
        """A reviewer renders one decision per finding. Two
        PerFindingDecisions targeting the same `finding_id` would be a
        contradiction — and the downstream consumer (publish) can only
        act on one verdict per finding.
        """
        finding_ids = [d.finding_id for d in self.decisions]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError(
                f"HITLDecision.decisions contains multiple decisions for the "
                f"same finding_id: {sorted(str(fid) for fid in finding_ids)!r}"
            )
        return self


__all__ = [
    "HITLDecision",
    "HITLRequest",
    "PerFindingDecision",
    "PerFindingOutcome",
]
