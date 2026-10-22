"""Accept human approval and resume a paused review in a background task.

Read the approval request from `reviews.hitl_request`, saved atomically by
the HITL node before it pauses. The node returns its state delta only after
resume, so the pending request is not yet in the graph's `hitl_request` field.

Validation order is auth -> review state -> finding decisions. It prevents
unauthenticated callers from probing review state, and callers whose review
is not awaiting approval from probing the gated finding set.

The resume wrapper logs conflicting submissions; it logs and re-raises other
failures. It leaves status changes to the node's sinks so a saved decision
can still be retried. The `reclaim_stuck_hitl_states` sweep owns recovery of
stuck approvals; the wrapper does not consult the graph checkpointer.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID  # noqa: TC003  (runtime: Pydantic field type)

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, model_validator

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.audit.persister import (
    AuditPersisterHITLDecisionNaturalKeyConflict,
)
from outrider.policy.severity import FindingSeverity  # noqa: TC001  (runtime: Pydantic field type)
from outrider.schemas.hitl import (
    HITLDecision,
    PerFindingDecision,
    PerFindingOutcome,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langgraph.graph.state import CompiledStateGraph

    from outrider.db.sinks import ReviewDecidePreflight, ReviewStatusReader
    from outrider.schemas.hitl import HITLRequest


_LOGGER = logging.getLogger("outrider.api.dashboard.hitl")


_REVIEWER_ID = "admin"


router = APIRouter(prefix="/reviews", tags=["hitl"])


class PerFindingDecisionPayload(BaseModel):
    """Reviewer-submitted decision shape.

    Distinct from `PerFindingDecision`: `original_severity` is NOT
    accepted here (the endpoint derives it server-side from the
    persisted `FindingEvent.severity` via `ReviewDecidePreflight`)
    AND `reviewer_id` is NOT in this payload (set server-side from
    auth context to the literal `admin` under V1 ADMIN_API_KEY
    scope).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    finding_id: UUID
    outcome: PerFindingOutcome
    reason: str = Field(max_length=500)
    override_severity: FindingSeverity | None = None

    @model_validator(mode="after")
    def _enforce_payload_invariants(self) -> PerFindingDecisionPayload:
        """Reject invalid outcome/field combinations during request parsing.

        Mirror the domain model's rules here so bad input returns 422 rather
        than raising a domain `ValueError` (500) inside the handler.
        """
        if self.outcome == PerFindingOutcome.SEVERITY_OVERRIDE and self.override_severity is None:
            msg = "severity_override outcome requires override_severity"
            raise ValueError(msg)
        if (
            self.outcome != PerFindingOutcome.SEVERITY_OVERRIDE
            and self.override_severity is not None
        ):
            msg = f"{self.outcome.value} outcome must not carry override_severity"
            raise ValueError(msg)
        if self.outcome != PerFindingOutcome.APPROVE and not self.reason.strip():
            msg = f"{self.outcome.value} outcome requires a non-blank reason"
            raise ValueError(msg)
        return self


class HITLDecisionPayload(BaseModel):
    """Body shape for `POST /reviews/{review_id}/decide`.

    `decisions` is a non-empty tuple bounded at 256 entries (matches
    `HITLRequest.findings_requiring_approval` producer-side cap);
    `annotation` is optional forensic prose, capped at 2000 chars
    matching `HITLDecisionEvent.annotation`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    decisions: tuple[PerFindingDecisionPayload, ...] = Field(min_length=1, max_length=256)
    annotation: str | None = Field(default=None, max_length=2000)


class _DecideResponse(BaseModel):
    """202 Accepted body."""

    review_id: str
    status: str


def _build_domain_decisions(
    *,
    payload: HITLDecisionPayload,
    gated_finding_severities: dict[UUID, FindingSeverity] | Any,
) -> tuple[PerFindingDecision, ...]:
    """Build domain decisions using each finding's saved severity baseline.

    `original_severity` comes from the persisted FindingEvent, so a later
    policy change cannot alter the baseline of the reviewer's override.
    """
    decisions: list[PerFindingDecision] = []
    for submitted_decision in payload.decisions:
        original_severity: FindingSeverity | None = None
        if submitted_decision.outcome == PerFindingOutcome.SEVERITY_OVERRIDE:
            # The endpoint checks for missing persisted baselines before this call.
            original_severity = gated_finding_severities[submitted_decision.finding_id]
        decisions.append(
            PerFindingDecision(
                finding_id=submitted_decision.finding_id,
                outcome=submitted_decision.outcome,
                reason=submitted_decision.reason,
                override_severity=submitted_decision.override_severity,
                original_severity=original_severity,
            )
        )
    return tuple(decisions)


async def _run_resume_under_failure_wrapper(
    *,
    review_id: UUID,
    hitl_decision: HITLDecision,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Resume the graph; leave status transitions and recovery to their owners."""
    try:
        await graph.ainvoke(
            Command(resume=hitl_decision.model_dump(mode="json")),
            config={"configurable": {"thread_id": str(review_id)}},
        )
    except AuditPersisterHITLDecisionNaturalKeyConflict:
        # A different submission either lost a concurrent race, or retried after
        # a crash between decision audit and status write. The winner advances
        # the first case; the sweep recovers the second. Log both as a warning
        # because distinguishing them requires the checkpointer, owned by recovery.
        _LOGGER.warning(
            "hitl_resume_natural_key_conflict",
            extra={
                "review_id": str(review_id),
                "note": (
                    "Existing HITLDecisionEvent rejected the incoming submission as "
                    "divergent content. If a concurrent task is still in flight, the "
                    "lifecycle advances via that task. Otherwise the row is stuck in "
                    "awaiting_approval pending sweep job reclaim_stuck_hitl_states."
                ),
            },
        )
        return
    except Exception:
        # Marking the review failed here would block retry of a saved decision.
        # Leave recovery of stuck rows to the sweep.
        _LOGGER.exception(
            "hitl_resume_failed",
            extra={"review_id": str(review_id)},
        )
        raise


async def _decidable_review(
    *,
    reader: ReviewStatusReader,
    review_id: UUID,
) -> tuple[ReviewDecidePreflight, HITLRequest]:
    """Return the review's preflight snapshot and its gate, or raise.

    Raises 409 unless the review is still awaiting a decision, and 500 when
    the persister reports that the gated set diverged from the FindingEvent
    rows. Returning the `HITLRequest` separately is what narrows it from the
    snapshot's optional field for the caller.
    """
    # A mismatch between the saved request and FindingEvent rows is corruption.
    # Match the structured 500 returned by the endpoint's completeness check.
    try:
        preflight = await reader.fetch_for_decide(review_id=review_id)
    except ValueError as exc:
        _LOGGER.exception(
            "preflight_gated_severities_incomplete_persister",
            extra={"review_id": str(review_id)},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "preflight_gated_severities_incomplete",
                "note": (
                    "State corruption surfaced at the persister: "
                    "reviews.hitl_request.findings_requiring_approval "
                    "diverged from audit_events FindingEvent rows. "
                    "Operator triage via audit log + reviews.hitl_request "
                    "JSONB; structured detail in the persister exception."
                ),
            },
        ) from exc

    if preflight is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.hitl_request is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.status not in ("awaiting_approval", "awaiting_approval_expired"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.hitl_decision is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")

    return preflight, preflight.hitl_request


def _assert_decisions_match_gate(
    *,
    payload: HITLDecisionPayload,
    expected_ids: set[UUID],
) -> None:
    """Raise 422 unless the submitted finding_ids are exactly the gated set.

    Duplicates need their own check because set-equality does not catch them:
    `[fid_a, fid_a, fid_b]` against gate `{fid_a, fid_b}` compares equal while
    carrying two decisions for fid_a. Admitting it makes `HITLDecision`
    construction raise mid-handler, so the caller sees 500 instead of a
    controlled 422.
    """
    submitted_id_list = [d.finding_id for d in payload.decisions]
    submitted_ids = set(submitted_id_list)
    if len(submitted_id_list) != len(submitted_ids):
        seen: set[UUID] = set()
        duplicates: list[str] = []
        for fid in submitted_id_list:
            if fid in seen and str(fid) not in duplicates:
                duplicates.append(str(fid))
            seen.add(fid)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"duplicate_finding_ids": sorted(duplicates)},
        )

    if submitted_ids != expected_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "missing": sorted(str(u) for u in expected_ids - submitted_ids),
                "extras": sorted(str(u) for u in submitted_ids - expected_ids),
            },
        )


def _assert_override_baselines_known(
    *,
    payload: HITLDecisionPayload,
    gated_finding_severities: Mapping[UUID, FindingSeverity],
) -> None:
    """Raise 500 if any SEVERITY_OVERRIDE finding lacks a persisted baseline.

    The persister raises on incomplete FindingEvent rows; this is
    defense-in-depth so a missing key surfaces as a labelled 500 instead of a
    `KeyError` inside `_build_domain_decisions` (opaque 500, stack-trace-only).
    """
    override_ids = {
        d.finding_id for d in payload.decisions if d.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
    }
    missing_severities = sorted(
        str(fid) for fid in override_ids if fid not in gated_finding_severities
    )
    if missing_severities:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "preflight_gated_severities_incomplete",
                "missing_finding_ids": missing_severities,
                "note": (
                    "State corruption: reviews.hitl_request.findings_requiring_approval "
                    "diverged from audit_events FindingEvent rows. Operator triage "
                    "via audit log + reviews.hitl_request JSONB."
                ),
            },
        )


def _assert_overrides_change_severity(
    *,
    payload: HITLDecisionPayload,
    gated_finding_severities: Mapping[UUID, FindingSeverity],
) -> None:
    """Raise 422 for a SEVERITY_OVERRIDE equal to the persisted baseline.

    `PerFindingDecision.enforce_override_fields` rejects the same no-op during
    domain-model construction, but that validation failure would surface as 500.
    This endpoint gate returns a client-facing 422 before construction.
    """
    no_op_overrides = sorted(
        str(d.finding_id)
        for d in payload.decisions
        if d.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
        and d.finding_id in gated_finding_severities
        and d.override_severity == gated_finding_severities[d.finding_id]
    )
    if no_op_overrides:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "noop_severity_override",
                "finding_ids": no_op_overrides,
                "note": (
                    "severity_override requires override_severity != "
                    "original_severity (a real override implies a "
                    "baseline-to-applied transition). If no severity "
                    "change is intended, submit outcome=approve instead."
                ),
            },
        )


@router.post(
    "/{review_id}/decide",
    response_model=_DecideResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_admin_api_key)],
)
async def decide(
    review_id: UUID,
    payload: HITLDecisionPayload,
    request: Request,
    background_tasks: BackgroundTasks,
) -> _DecideResponse:
    """Accept a HITL approval submission for `review_id`.

    Auth runs first as a `require_admin_api_key` dependency (401), then the
    state preflight (409 unless the review is still awaiting a decision), then
    the payload gates: the submitted `finding_id` set must equal the gated set
    and carry no duplicates (422), and a `severity_override` must actually
    change the severity (422). That order is load-bearing — it keeps an
    unauthenticated caller, or one whose review is not awaiting a decision,
    from probing the gated finding set.

    On success the typed `HITLDecision` is constructed server-side —
    `reviewer_id`, `original_severity` from the preflight map, `decided_at` —
    and the resume is enqueued via FastAPI `BackgroundTasks`, returning 202
    without waiting for it.
    """
    preflight, hitl_request = await _decidable_review(
        reader=request.app.state.review_status_reader,
        review_id=review_id,
    )

    _assert_decisions_match_gate(
        payload=payload,
        expected_ids=set(hitl_request.findings_requiring_approval),
    )
    _assert_override_baselines_known(
        payload=payload,
        gated_finding_severities=preflight.gated_finding_severities,
    )
    _assert_overrides_change_severity(
        payload=payload,
        gated_finding_severities=preflight.gated_finding_severities,
    )

    decisions = _build_domain_decisions(
        payload=payload,
        gated_finding_severities=preflight.gated_finding_severities,
    )
    hitl_decision = HITLDecision(
        reviewer_id=_REVIEWER_ID,
        decisions=decisions,
        annotation=payload.annotation,
        decided_at=datetime.now(UTC),
    )

    background_tasks.add_task(
        _run_resume_under_failure_wrapper,
        review_id=review_id,
        hitl_decision=hitl_decision,
        graph=request.app.state.compiled_graph,
    )

    return _DecideResponse(review_id=str(review_id), status="resuming")


__all__ = [
    "HITLDecisionPayload",
    "PerFindingDecisionPayload",
    "decide",
    "router",
]
