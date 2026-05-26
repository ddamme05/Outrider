"""POST /reviews/{review_id}/decide — the HITL approval endpoint.

The endpoint reads the canonical HITLRequest snapshot from
`reviews.hitl_request` JSONB (written atomically by the HITL node
body's `mark_awaiting_approval` BEFORE `interrupt()` per the
single-transaction contract). It does NOT call `graph.aget_state(...)`
— the state delta `{"hitl_request": ..., "hitl_decision": ...}` only
returns AFTER resume completes, so at interrupt time the
state-snapshot view of `hitl_request` is empty.

M12 step-order is load-bearing: auth -> state -> mismatch. Auth fires
FIRST so an unauthenticated caller cannot enumerate review state by
observing 409/422 response patterns. State fires BEFORE mismatch so a
caller hitting a non-HITL review can't probe the gated finding set.

The failure wrapper `_run_resume_under_failure_wrapper` is the
async-task boundary that bounds the FastAPI BackgroundTasks dispatch.
Two catch arms:

1. `AuditPersisterHITLDecisionNaturalKeyConflict` — divergent-content
   conflict. Two sub-cases:
   (a) DIVERGENT CONCURRENT race — two reviewers submitted with
       different content; one task wins the natural-key insert, this
       is the loser. Lifecycle advances via the winning task.
   (b) WINDOW-(f) DIVERGENT RETRY — a first task crashed between
       `emit_hitl_decision` and `mark_running`, leaving an audit row +
       NULL `reviews.hitl_decision`. A later retry with different
       content admits past the JSONB-cache preflight but hits the
       audit-row natural-key check. Lifecycle is STUCK in
       `awaiting_approval` pending Group 8's
       `reclaim_stuck_hitl_states` sweep recovery.
   The wrapper cannot reliably distinguish (a) from (b) without
   consulting the LangGraph checkpointer (pending-interrupt presence
   is what tells them apart). It logs at WARNING level with a
   diagnostic note so operators see (b) in alerting; the sweep is
   the canonical recovery. Identical-content races and crash-replays
   of the SAME submission are absorbed by the idempotent cascade
   INSIDE the node body (status-predicate discrimination on
   mark_awaiting_approval + natural-key no-ops on both events +
   publish-side `query_prior_publish_event`) and never reach this
   branch.

2. `Exception` — log + re-raise, NO status flip. Marking
   `reviews.status='failed'` would orphan a persisted
   `HITLDecisionEvent` from the lifecycle state, blocking
   operator-driven retry at the endpoint's state-gate predicate. The
   sweep's `reclaim_stuck_hitl_states` sub-job is the canonical
   cleanup for genuinely-stuck rows after a grace period.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID  # noqa: TC003  (runtime: Pydantic field type)

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

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
    from langgraph.graph.state import CompiledStateGraph


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
    """Map reviewer-submitted payload to typed `PerFindingDecision`s,
    populating `original_severity` server-side from the preflight map.

    The preflight map carries the persisted-at-admit-time severity
    from `FindingEvent.severity` (with its policy version), so a
    SEVERITY_OVERRIDE decision's `original_severity` ALWAYS reflects
    the policy baseline at the time the finding was admitted — even
    if the live `SEVERITY_POLICY` mapping has since changed.
    """
    out: list[PerFindingDecision] = []
    for d in payload.decisions:
        original_severity: FindingSeverity | None = None
        if d.outcome == PerFindingOutcome.SEVERITY_OVERRIDE:
            # Lookup from the preflight map; missing key indicates
            # state-corruption (the gated set on `hitl_request`
            # diverged from the FindingEvent rows in audit_events) —
            # surfaces here as KeyError, the endpoint translates to
            # 500 via FastAPI's default error path.
            original_severity = gated_finding_severities[d.finding_id]
        out.append(
            PerFindingDecision(
                finding_id=d.finding_id,
                outcome=d.outcome,
                reason=d.reason,
                override_severity=d.override_severity,
                original_severity=original_severity,
            )
        )
    return tuple(out)


async def _run_resume_under_failure_wrapper(
    *,
    review_id: UUID,
    hitl_decision: HITLDecision,
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> None:
    """Background-task entry point. Resumes the suspended graph with
    the reviewer's decision under explicit failure semantics.

    Catch arms documented in the module docstring. Status transitions
    are owned by deterministic sink methods (`mark_awaiting_approval`,
    `mark_running`, `mark_awaiting_approval_expired`); this wrapper
    never mutates `reviews.status`.
    """
    try:
        await graph.ainvoke(
            Command(resume=hitl_decision.model_dump(mode="json")),
            config={"configurable": {"thread_id": str(review_id)}},
        )
    except AuditPersisterHITLDecisionNaturalKeyConflict:
        # Two distinct cases land here, both characterized by "an
        # existing HITLDecisionEvent's `decisions_content_hash` differs
        # from the incoming submission's":
        #
        #   (a) DIVERGENT CONCURRENT race — two reviewers submitted
        #       near-simultaneously with different content. One task
        #       won the natural-key insert; this is the loser. The
        #       lifecycle WILL advance via the winning task's
        #       `mark_running`. The right outcome: silent absorb.
        #
        #   (b) WINDOW-(f) DIVERGENT RETRY — a first task crashed
        #       after `emit_hitl_decision` but before `mark_running`,
        #       leaving an audit row + NULL `reviews.hitl_decision`.
        #       A later retry with different content (e.g., the
        #       reviewer edited the annotation) admits past the
        #       endpoint's preflight (which only checks the JSONB
        #       cache) and reaches this catch. The audit row is
        #       canonical; the lifecycle is STUCK at
        #       `awaiting_approval` and won't advance without
        #       operator intervention.
        #
        # The wrapper cannot reliably distinguish (a) from (b) without
        # consulting the LangGraph checkpointer (pending interrupt
        # presence) — which is owned by Group 8's
        # `reclaim_stuck_hitl_states` sub-job. The wrapper logs at
        # WARNING level (not INFO) so the case surfaces in operator
        # alerts; the sweep job is the canonical recovery path.
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
        # Log + re-raise. Status is NOT flipped to `failed` — the row
        # stays in `awaiting_approval` / `awaiting_approval_expired`
        # so the operator can re-issue the resume. The sweep's
        # `reclaim_stuck_hitl_states` is the canonical cleanup for
        # genuinely-stuck rows.
        _LOGGER.exception(
            "hitl_resume_failed",
            extra={"review_id": str(review_id)},
        )
        raise


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

    Flow (M12 step-order is load-bearing):
      1. Auth (via `require_admin_api_key` Depends). 401 on failure.
      2. State preflight via `ReviewStatusReader.fetch_for_decide`.
         409 if: hitl_request is None / status not in the HITL set /
         hitl_decision already landed.
      3. Mismatch check: payload `finding_id` set must equal the
         gated set on `hitl_request.findings_requiring_approval`.
         422 with `{"missing": [...], "extras": [...]}` on mismatch.
      4. Construct typed `HITLDecision` server-side (server-set
         `reviewer_id`, server-derived `original_severity` from the
         preflight map for SEVERITY_OVERRIDE outcomes, server-set
         `decided_at=datetime.now(UTC)`).
      5. Enqueue `_run_resume_under_failure_wrapper` via FastAPI
         `BackgroundTasks`; return 202 immediately.
    """
    reader = request.app.state.review_status_reader
    preflight = await reader.fetch_for_decide(review_id=review_id)

    # State gate.
    if preflight is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.hitl_request is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.status not in ("awaiting_approval", "awaiting_approval_expired"):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")
    if preflight.hitl_decision is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="conflict")

    # Mismatch gate (auth -> state -> mismatch; this is "mismatch").
    #
    # Duplicate-finding-id check FIRST: a payload like
    # `[{fid_a}, {fid_a}, {fid_b}]` against gate `{fid_a, fid_b}`
    # would satisfy set-equality (`{fid_a, fid_b} == {fid_a, fid_b}`)
    # but carries two decisions for fid_a. Without this check, the
    # endpoint admits the duplicate and downstream HITLDecision
    # construction raises mid-handler — the caller sees 500 instead
    # of a controlled 422.
    submitted_id_list = [d.finding_id for d in payload.decisions]
    submitted_ids = set(submitted_id_list)
    if len(submitted_id_list) != len(submitted_ids):
        # Identify the duplicates for the operator response.
        seen: set = set()
        duplicates: list[str] = []
        for fid in submitted_id_list:
            if fid in seen and str(fid) not in duplicates:
                duplicates.append(str(fid))
            seen.add(fid)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"duplicate_finding_ids": sorted(duplicates)},
        )

    expected_ids = set(preflight.hitl_request.findings_requiring_approval)
    if submitted_ids != expected_ids:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "missing": sorted(str(u) for u in expected_ids - submitted_ids),
                "extras": sorted(str(u) for u in submitted_ids - expected_ids),
            },
        )

    # Construct typed HITLDecision with server-derived original_severity.
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

    # Enqueue resume.
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
