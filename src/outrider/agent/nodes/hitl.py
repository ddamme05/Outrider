# See specs/2026-05-26-hitl-node.md
"""HITL node — partitions findings by severity, optionally interrupts the
graph for human approval, and emits the canonical request + decision
audit events.

13-step body (audit-first emit -> status-write -> interrupt ordering):

  1. Phase event start (deterministic phase_id via compute_phase_id).
  2. Partition state.review_report.findings by severity (already
     deduplicated by synthesize): CRITICAL/HIGH ->
     findings_requiring_approval; MEDIUM/LOW/INFO -> auto_post_findings.
     Uses `is_hitl_gated_severity` from policy/publish_eligibility.py
     as the single source of truth for the gated set.
  3. Empty gate set -> emit phase end + return {} (no state delta;
     LangGraph proceeds to publish).
  4. Else build HITLRequest with deterministic sorted tuples +
     deterministic `expires_at = state.received_at + timedelta(...)`.
  5. emit_hitl_request (audit-first; returned canonical event becomes
     the state-layer HITLRequest).
  6. mark_awaiting_approval (status flip + expires_at + JSONB write
     in ONE atomic UPDATE; predicate `status='running' AND
     hitl_request IS NULL` makes the call first-write-only).
  7. interrupt(hitl_request.model_dump(mode="json")) — LangGraph
     checkpoints state to Postgres.
  8. Resume: body re-runs from the top, then `interrupt()` returns the
     resume value. Deserialize via HITLDecision.model_validate.
  9. Validate resume value against re-derived HITLRequest (defense-
     in-depth — endpoint should have already rejected mismatches).
 10. emit_hitl_decision (audit-first; returned canonical event becomes
     the state-layer HITLDecision).
 11. mark_running (status flip back + hitl_decision JSONB write).
 12. Phase event end (same phase_id as step 1).
 13. Return {"hitl_request": ..., "hitl_decision": ...} state delta.

Idempotency cascade: every step from 5 onward is idempotent against
post-completion state. A concurrent identical-content resume is
absorbed by:
  - emit_hitl_request natural-key no-op on (review_id)
  - mark_awaiting_approval predicate-filter no-op on hitl_request != NULL
  - emit_hitl_decision natural-key no-op on (review_id) +
    identity-subset match via decisions_content_hash
  - mark_running predicate filters on `hitl_decision IS NULL` AND
    admits only `awaiting_approval` / `awaiting_approval_expired` as
    source states — re-fire after first successful flip is
    rowcount=0 (no-op) because `hitl_decision IS NOT NULL`
  - phase events idempotent on phase_id (deterministic compute_phase_id)

Concurrent divergent-content resume raises
`AuditPersisterHITLDecisionNaturalKeyConflict` from step 10, caught by
the endpoint's failure wrapper and logged at WARNING with the
diagnostic message naming both the concurrent-loser case and the
window-(f) crash-retry case (recovery for the latter is owned by
`sweep/hitl_expiry.py::reclaim_stuck_hitl_states`).
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
    """Split synthesize-deduplicated findings by severity.

    Returns `(findings_requiring_approval, auto_post_findings)` both
    sorted by `finding_id` for deterministic body output.

    Consumes `state.review_report.findings` (the deduplicated +
    severity-sorted tuple produced by synthesize). The earlier shape
    of this function walked `state.analysis_rounds[*].findings`
    directly and carried a two-pass "gated-takes-precedence" rule for
    cross-round severity disagreement; per the synthesize-node spec
    (pre-spec gate #7) that disagreement is corruption per
    `severity-set-by-policy` and synthesize raises
    SynthesizeAggregationError before HITL ever sees the state. So the
    partition is a simple single-pass classification on the
    pre-deduplicated tuple — gated if severity is in the V1 gated set,
    autopost otherwise.

    Uses `is_hitl_gated_severity` from `policy/publish_eligibility.py`
    as the single source of truth for "what counts as gated severity"
    (V1: CRITICAL + HIGH). The earlier `_GATED_SEVERITIES` literal at
    this site was a sibling copy; consolidating to the policy helper
    keeps every consumer of "gated severity" pointed at one canonical
    definition. See `policy/publish_eligibility.py:138`.
    """
    # Direct attribute access (not getattr-with-default) so a future
    # schema rename of `review_report` surfaces as `AttributeError`
    # rather than silently triggering the "synthesize must have run"
    # RuntimeError. Test doubles must include the attribute (set to
    # None or to a stub); the `_make_state` helper in
    # `tests/unit/test_hitl_node.py` does.
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
    # Fail loud one step before HITLRequest construction (same bound as the
    # schema's max_length): a gate with more than HITL_MAX_GATED_FINDINGS
    # gated findings is unreviewable; refuse rather than ship a wall of
    # findings to the dashboard. Clearer message than the schema's generic
    # "tuple too long". See schemas/hitl.py HITL_MAX_GATED_FINDINGS.
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
    """Construct the state-layer HITLRequest and its audit-shadow event.

    Both share `findings_requiring_approval`, `auto_post_findings`,
    `created_at`, `expires_at` exactly so the audit-first emission's
    natural-key identity-subset match collapses re-emits cleanly.
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
    """Construct the state-layer HITLDecision + audit-shadow event from
    the resume value the endpoint enqueued via `Command(resume=...)`.

    `resume_value` is the JSON-shaped payload the endpoint serialized
    from its server-constructed HITLDecision (server-set reviewer_id,
    server-derived per-finding original_severity, server-set
    decided_at). The HITL node validates the shape via Pydantic
    `model_validate`; downstream divergence from the endpoint's
    construction (missing fields, non-canonical timestamps) raises
    ValidationError at deserialization time.
    """
    decision = HITLDecision.model_validate(resume_value)
    content_hash = compute_hitl_decision_content_hash(
        decisions=decision.decisions,
        annotation=decision.annotation,
    )
    # `decision_latency_seconds`: the elapsed wall-clock between the
    # canonical HITLRequest's `created_at` and the reviewer's
    # `decided_at`. Computed here at audit-emit time so the field is
    # canonical on the persisted row (alternative: leave at 0.0 and
    # let dashboard compute from the audit-row pair at query time;
    # rejected because storing the derived metric at emit time keeps
    # consumers simpler + the value is available right here without
    # the dashboard needing to join HITLRequestEvent ↔ HITLDecisionEvent
    # for every read). `max(0.0, ...)` guards against a clock-skew
    # case where decided_at < created_at (shouldn't happen in V1's
    # in-process clock, but defensive).
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
    """Defense-in-depth check on the resume value's finding set.

    The endpoint should have rejected this mismatch via its
    auth -> state -> mismatch ordered preflight check using
    `ReviewStatusReader.fetch_for_decide(...)`. Reaching this raise
    means the endpoint's check missed something — fail loud so the
    failure is observable in graph runtime logs.
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
    """Run the HITL gate node. See module docstring for the 13-step
    contract.

    Returns `{}` on the pass-through path (no gated findings) or
    `{"hitl_request": HITLRequest, "hitl_decision": HITLDecision}` on
    the gated path. Both keys use Pydantic's default merge reducer
    (overwrite-on-set), correct for single-value slots; deterministic
    body output makes set-twice safe under checkpoint replay.
    """
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="hitl",
        attempt_key="hitl",
    )

    # Step 1: phase start.
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

    # Step 2: partition by severity, deduped by finding_id, sorted.
    findings_requiring_approval, auto_post_findings = _partition_findings(state)

    # Step 3: pass-through if no gated findings.
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

    # Step 4: deterministic request + expires_at.
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

    # Step 5: audit-first emit. Returned canonical event drives state.
    persisted_request = await hitl_event_sink.emit_hitl_request(request_event)
    canonical_request = HITLRequest(
        findings_requiring_approval=persisted_request.findings_requiring_approval,
        auto_post_findings=persisted_request.auto_post_findings,
        created_at=persisted_request.created_at,
        expires_at=persisted_request.expires_at,
    )

    # Step 6: status flip + expires_at + JSONB write atomically. The
    # mark_awaiting_approval predicate's `hitl_request IS NULL`
    # discriminator makes this a no-op on resume body re-run.
    await review_status_sink.mark_awaiting_approval(
        review_id=state.review_id,
        expires_at=canonical_request.expires_at,
        hitl_request_payload=canonical_request.model_dump(mode="json"),
    )

    # Step 6b: best-effort Slack HITL-pending notification. Awaited inline but
    # never gate-breaking — the orchestrator swallows every transport/audit
    # failure (degrades-gracefully), so the gate always proceeds. Placed before
    # interrupt so it fires on entry to awaiting_approval; the body re-runs on
    # resume and the orchestrator's pre-post dedup makes the re-post a no-op.
    # A degraded Slack endpoint adds at most the notifier's bounded per-call
    # timeout (FUP-188) to the pre-interrupt checkpoint, not slack_sdk's ~30s.
    report = state.review_report
    if resolve_slack_target is not None and report is not None:
        # `report is not None` narrows the Optional for mypy across the await; it
        # is a guaranteed invariant here — Step 2 (_partition_findings) raises if
        # review_report is None, so this branch is always taken on the gated path.
        # The whole resolve+notify is wrapped: Slack is optional and NEVER
        # gate-breaking, so a resolver failure (DB read / decrypt / notifier build)
        # degrades to no notification rather than propagating between
        # mark_awaiting_approval and interrupt() — which would leave an `awaiting`
        # row with no checkpoint handoff. The orchestrator's notify_* is itself
        # no-raise; this wrapper extends that envelope to cover the resolver.
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
            # Never gate-breaking: the gate must reach interrupt() + checkpoint
            # regardless of any Slack-path failure.
            logger.exception(
                "slack hitl-pending notification failed; gate proceeds",
                extra={"review_id": str(state.review_id)},
            )

    # Step 7: interrupt. LangGraph checkpoints state to Postgres + the
    # call yields control back to the resume endpoint.
    resume_value: dict[str, object] = interrupt(
        canonical_request.model_dump(mode="json"),
    )

    # Step 8-9: deserialize + defense-in-depth check.
    decision, decision_event = _build_decision(
        review_id=state.review_id,
        resume_value=resume_value,
        is_eval=state.is_eval,
        request_created_at=canonical_request.created_at,
    )
    _validate_resume_against_request(request=canonical_request, decision=decision)

    # Step 10: audit-first emit for decision.
    persisted_decision = await hitl_event_sink.emit_hitl_decision(decision_event)
    canonical_decision = HITLDecision(
        reviewer_id=persisted_decision.reviewer_id,
        decisions=persisted_decision.decisions,
        annotation=persisted_decision.annotation,
        decided_at=persisted_decision.decided_at,
    )

    # Step 11: status flip back to running + hitl_decision JSONB write.
    await review_status_sink.mark_running(
        review_id=state.review_id,
        hitl_decision_payload=canonical_decision.model_dump(mode="json"),
    )

    # Step 12: phase end.
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

    # Step 13: state delta.
    return {
        "hitl_request": canonical_request,
        "hitl_decision": canonical_decision,
    }


__all__ = ["hitl"]
