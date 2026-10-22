# See specs/2026-05-26-hitl-node.md
"""Pause for human approval (HITL) when findings have CRITICAL/HIGH severity.

Read the deduplicated review report, using policy to split gated findings
from those that can publish automatically. An empty gate passes through.
Otherwise, save the request to audit, write the awaiting-approval snapshot
atomically, optionally notify Slack, then interrupt the graph. On resume,
validate and audit the decision before marking the review running again.

LangGraph re-runs this body from the top on resume. Audit writes deduplicate
by review identity (and decision content); status writes are conditional,
and phase IDs are deterministic. Replaying an identical submission is safe.
A conflicting decision raises into the dashboard endpoint's resume wrapper;
`sweep/hitl_expiry.py::reclaim_stuck_hitl_states` owns stuck-state recovery.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from langgraph.types import interrupt

from outrider.audit.events import (
    HITLDecisionEvent,
    HITLRequestEvent,
    ReviewPhaseEvent,
)
from outrider.policy.canonical import (
    compute_hitl_decision_content_hash,
    compute_phase_id,
)
from outrider.policy.publish_eligibility import is_hitl_gated_severity
from outrider.schemas.hitl import HITL_MAX_GATED_FINDINGS, HITLDecision, HITLRequest

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from outrider.agent.nodes.hitl_config import HITLConfig
    from outrider.audit.sinks import HITLEventSink, PhaseEventSink
    from outrider.db.sinks import ReviewStatusSink
    from outrider.notify.orchestrator import SlackTargetResolver
    from outrider.schemas import ReviewState


logger = logging.getLogger(__name__)


def _partition_findings(
    state: ReviewState,
) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
    """Split the synthesized report into approval and automatic-publication IDs.

    Synthesize has already deduplicated findings and rejected conflicting
    severities. Use policy's gated-severity rule and return sorted ID tuples.
    """
    if state.review_report is None:
        msg = (
            "HITL requires state.review_report to be set "
            "(synthesize node must have run before hitl — graph wiring "
            "or test fixture bug). Fail-closed: a miswired path that "
            "bypasses synthesize would otherwise silently lose the "
            "content-hash dedup + cross-round severity-divergence "
            "detection contracts."
        )
        raise RuntimeError(msg)
    gated: set[UUID] = set()
    autopost: set[UUID] = set()
    for finding in state.review_report.findings:
        if is_hitl_gated_severity(finding.severity):
            gated.add(finding.finding_id)
        else:
            autopost.add(finding.finding_id)
    # Match the schema's cap, with a useful error before request construction.
    if len(gated) > HITL_MAX_GATED_FINDINGS:
        raise ValueError(
            f"HITL gate exceeds the {HITL_MAX_GATED_FINDINGS}-finding cap: "
            f"{len(gated)} findings require approval. A PR producing this many "
            f"CRITICAL/HIGH findings is past the point a human reviewer can "
            f"meaningfully approve per-finding; investigate before retrying."
        )
    return tuple(sorted(gated)), tuple(sorted(autopost))


def _build_request(
    *,
    review_id: UUID,
    findings_requiring_approval: tuple[UUID, ...],
    auto_post_findings: tuple[UUID, ...],
    created_at: datetime,
    expires_at: datetime,
    is_eval: bool,
) -> tuple[HITLRequest, HITLRequestEvent]:
    """Build matching request and audit payloads.

    The shared finding sets and timestamps let the audit sink deduplicate replays.
    """
    request = HITLRequest(
        findings_requiring_approval=findings_requiring_approval,
        auto_post_findings=auto_post_findings,
        created_at=created_at,
        expires_at=expires_at,
    )
    event = HITLRequestEvent(
        event_id=uuid4(),
        review_id=review_id,
        is_eval=is_eval,
        findings_requiring_approval=findings_requiring_approval,
        auto_post_findings=auto_post_findings,
        created_at=created_at,
        expires_at=expires_at,
    )
    return request, event


def _build_decision(
    *,
    review_id: UUID,
    resume_value: dict[str, object],
    is_eval: bool,
    request_created_at: datetime,
) -> tuple[HITLDecision, HITLDecisionEvent]:
    """Validate the endpoint's resume payload and build its decision audit event.

    The endpoint supplies reviewer identity, saved severity baselines and
    decision time; Pydantic checks their shape again at this boundary.
    """
    decision = HITLDecision.model_validate(resume_value)
    content_hash = compute_hitl_decision_content_hash(
        decisions=decision.decisions,
        annotation=decision.annotation,
    )
    # Save latency with the event so readers need no request/decision join.
    # Clamp clock skew to zero.
    latency = max(0.0, (decision.decided_at - request_created_at).total_seconds())
    event = HITLDecisionEvent(
        event_id=uuid4(),
        review_id=review_id,
        is_eval=is_eval,
        reviewer_id=decision.reviewer_id,
        decisions=decision.decisions,
        annotation=decision.annotation,
        decided_at=decision.decided_at,
        decisions_content_hash=content_hash,
        decision_latency_seconds=latency,
    )
    return decision, event


def _validate_resume_against_request(
    *,
    request: HITLRequest,
    decision: HITLDecision,
) -> None:
    """Check the finding set again before the graph accepts the decision.

    The endpoint checks the same gate; a mismatch here must stop the graph.
    """
    submitted = {d.finding_id for d in decision.decisions}
    expected = set(request.findings_requiring_approval)
    if submitted != expected:
        raise ValueError(
            "HITL resume value finding set diverges from re-derived "
            "request set "
            f"(missing={sorted(str(u) for u in (expected - submitted))!r}; "
            f"extras={sorted(str(u) for u in (submitted - expected))!r}). "
            "The dashboard endpoint should have rejected this before "
            "calling Command(resume=...). Defense-in-depth raise; the "
            "graph cannot continue with a divergent decision set."
        )


async def hitl(
    state: ReviewState,
    *,
    phase_event_sink: PhaseEventSink,
    hitl_event_sink: HITLEventSink,
    review_status_sink: ReviewStatusSink,
    hitl_config: HITLConfig,
    resolve_slack_target: SlackTargetResolver | None = None,
) -> dict[str, object]:
    """Pass through, or pause for approval and return the canonical decision.

    Return `{}` when no findings need approval; otherwise return
    `{"hitl_request": HITLRequest, "hitl_decision": HITLDecision}` after resume.
    These single-value state slots overwrite safely on checkpoint replay.
    """
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="hitl",
        attempt_key="hitl",
    )

    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="hitl",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    findings_requiring_approval, auto_post_findings = _partition_findings(state)

    if not findings_requiring_approval:
        await phase_event_sink.emit_phase(
            ReviewPhaseEvent(
                review_id=state.review_id,
                phase_id=phase_id,
                node_id="hitl",
                marker="end",
                is_eval=state.is_eval,
                phase_key=None,
            )
        )
        return {}

    # Anchor expiry to the received time so body re-runs use the same request.
    created_at = state.received_at
    expires_at = created_at + timedelta(minutes=hitl_config.timeout_minutes)
    request, request_event = _build_request(
        review_id=state.review_id,
        findings_requiring_approval=findings_requiring_approval,
        auto_post_findings=auto_post_findings,
        created_at=created_at,
        expires_at=expires_at,
        is_eval=state.is_eval,
    )

    # Use the persisted event: a replay may have returned the original request.
    persisted_request = await hitl_event_sink.emit_hitl_request(request_event)
    canonical_request = HITLRequest(
        findings_requiring_approval=persisted_request.findings_requiring_approval,
        auto_post_findings=persisted_request.auto_post_findings,
        created_at=persisted_request.created_at,
        expires_at=persisted_request.expires_at,
    )

    # Save status, expiry and request together before interrupt. The sink's
    # `hitl_request IS NULL` condition makes this a no-op on resume.
    await review_status_sink.mark_awaiting_approval(
        review_id=state.review_id,
        expires_at=canonical_request.expires_at,
        hitl_request_payload=canonical_request.model_dump(mode="json"),
    )

    # Slack is optional. The orchestrator deduplicates resume notifications and
    # bounds transport waits (FUP-188); resolver failures must also allow the pause.
    report = state.review_report
    if resolve_slack_target is not None and report is not None:
        # _partition_findings requires a report; this guard also narrows its type.
        # Catch resolve+notify failures so an awaiting row reaches the checkpoint.
        try:
            slack_target = await resolve_slack_target(state.pr_context.installation_id)
            if slack_target is not None:
                gated_ids = set(findings_requiring_approval)
                await slack_target.orchestrator.notify_hitl_pending(
                    review_id=state.review_id,
                    is_eval=state.is_eval,
                    channel_id=slack_target.channel_id,
                    repo=f"{state.pr_context.owner}/{state.pr_context.repo}",
                    pr_number=state.pr_context.pr_number,
                    pr_title=state.pr_context.pr_title,
                    findings=[f for f in report.findings if f.finding_id in gated_ids],
                )
        except Exception:
            logger.exception(
                "slack hitl-pending notification failed; gate proceeds",
                extra={"review_id": str(state.review_id)},
            )

    # LangGraph saves the checkpoint and pauses. On resume the body runs again,
    # and this call returns the submitted decision.
    resume_value: dict[str, object] = interrupt(
        canonical_request.model_dump(mode="json"),
    )

    decision, decision_event = _build_decision(
        review_id=state.review_id,
        resume_value=resume_value,
        is_eval=state.is_eval,
        request_created_at=canonical_request.created_at,
    )
    _validate_resume_against_request(request=canonical_request, decision=decision)

    # Audit before changing status; use the saved decision on an identical replay.
    persisted_decision = await hitl_event_sink.emit_hitl_decision(decision_event)
    canonical_decision = HITLDecision(
        reviewer_id=persisted_decision.reviewer_id,
        decisions=persisted_decision.decisions,
        annotation=persisted_decision.annotation,
        decided_at=persisted_decision.decided_at,
    )

    await review_status_sink.mark_running(
        review_id=state.review_id,
        hitl_decision_payload=canonical_decision.model_dump(mode="json"),
    )

    # Publish alone writes the terminal Slack update. An interim update could
    # arrive later and overwrite it: chat.update has no compare-and-set. Restoring
    # interim/expiry updates needs monotonic lifecycle ordering (FUP-252), not a
    # PublishEvent guard; some terminal outcomes emit no PublishEvent.

    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="hitl",
            marker="end",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    return {
        "hitl_request": canonical_request,
        "hitl_decision": canonical_decision,
    }


__all__ = ["hitl"]
