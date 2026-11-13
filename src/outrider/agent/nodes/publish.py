# Publish node per specs/2026-05-21-publish-node.md §V + DECISIONS.md #023.
# See DECISIONS.md#065-authorization-is-a-live-github-check-the-local-install-db-is-a-cache
"""Post review findings to GitHub after checking policy and approvals.

`publish` records where each finding belongs and whether it can be posted.
`publish_render.py` builds the text. `_publish_body` checks saved results and
GitHub before posting so retries can recover an earlier successful post.

The node records the outcome and updates review status. `publish` also handles
optional Slack updates. See `_publish_body` for posting order and failures.
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from uuid import UUID

from outrider.agent.nodes.publish_render import (
    _AGENT_MARKER_TEMPLATE as _AGENT_MARKER_TEMPLATE,
)
from outrider.agent.nodes.publish_render import (
    _AGENT_PROMPT_SUMMARY_MAX_BYTES as _AGENT_PROMPT_SUMMARY_MAX_BYTES,
)
from outrider.agent.nodes.publish_render import (
    _build_agent_markers as _build_agent_markers,
)
from outrider.agent.nodes.publish_render import (
    _build_agent_prompt_block as _build_agent_prompt_block,
)
from outrider.agent.nodes.publish_render import (
    _build_finding_comment_body as _build_finding_comment_body,
)
from outrider.agent.nodes.publish_render import (
    _escape_angle_brackets as _escape_angle_brackets,
)
from outrider.agent.nodes.publish_render import (
    _marker_safe as _marker_safe,
)
from outrider.agent.nodes.publish_render import (
    _render_proof_line as _render_proof_line,
)
from outrider.agent.nodes.publish_render import (
    _render_related_concern_entry as _render_related_concern_entry,
)
from outrider.agent.nodes.publish_render import (
    _render_review_body as _render_review_body,
)
from outrider.agent.nodes.publish_render import (
    _render_suggestion_block as _render_suggestion_block,
)
from outrider.agent.nodes.publish_render import (
    _review_deep_link as _review_deep_link,
)
from outrider.audit.events import (
    PublishAttemptEvent,
    PublishAttemptOutcome,
    PublishEligibility,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    PublishRoutingReason,
    ReviewPhaseEvent,
    compute_finding_content_hash,
    compute_publish_attempt_content_hash,
    compute_publish_eligibility_decision_hash,
    compute_publish_routing_decision_hash,
)
from outrider.coordinates import (
    CoordinateError,
    GitHubCommentLocation,
    removed_source_to_review_body,
    source_line_to_github,
)
from outrider.coordinates.errors import CoordinateErrorKind
from outrider.policy.canonical import compute_phase_id
from outrider.policy.publish_eligibility import (
    count_gated_approvals,
    is_eligible_for_v1_publish,
    is_hitl_gated_severity,
    verification_hold_finding_ids,
)
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    InlineComment,
    PublishDestination,
    PublishResult,
)
from outrider.schemas.hitl import PerFindingOutcome

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from outrider.audit.sinks import PhaseEventSink, PublishEventSink
    from outrider.db.sinks import ReviewStatusSink
    from outrider.github import InstallationGitHubClient
    from outrider.github.publisher import GitHubPublisher
    from outrider.notify.orchestrator import SlackTargetResolver
    from outrider.policy import FindingSeverity
    from outrider.schemas import ReviewFinding, ReviewState
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision

__all__ = ["publish"]


# The review-body marker lets a retry find a successful POST on the same head_sha
# when its local PublishEvent is missing (spec §V, DECISIONS.md#023).
logger = logging.getLogger(__name__)

_BODY_MARKER_TEMPLATE = "<!-- outrider-review-id:{review_id} -->"


# Both GET and POST treat these as authorization revocation: retain and complete
# the review without re-raising (DECISIONS.md#065). Other failures propagate.
_AUTH_REVOKED_STATUS_CODES: frozenset[int] = frozenset({401, 403, 404})


# A review-posted FYI requires a created or previously confirmed GitHub review.
# Replays can repair a failed notification; empty and revoked outcomes cannot
# truthfully claim that a review posted.
_FYI_TRUTHFUL_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"success", "idempotently_skipped", "idempotently_skipped_external_record"}
)


async def _finalize_slack_notifications(
    state: ReviewState,
    *,
    resolve_slack_target: SlackTargetResolver | None,
    publish_event_sink: PublishEventSink,
    delta: dict[str, object],
) -> None:
    """Finalize Slack on every non-failed exit, including idempotent replay.

    Reviews without a HITL request use the FYI for `_FYI_TRUTHFUL_OUTCOMES`.
    Gated reviews use the terminal mirror, including an empty result after all
    findings were rejected, except on revocation (see its ordering guard below).
    Keeping this outside the success branch lets replay repair a failed post or
    update. Raised publish failures leave the existing card unchanged.
    """
    if resolve_slack_target is None:
        return
    report = state.review_report
    result = delta.get("publish_result")
    if report is None or not isinstance(result, PublishResult):
        return
    posted_count = result.comments_posted + result.review_body_findings_posted
    dashboard_only_count = result.dashboard_only_findings_surfaced
    gated_request = state.hitl_request
    try:
        slack_target = await resolve_slack_target(state.pr_context.installation_id)
        if slack_target is None:
            return
        repo = f"{state.pr_context.owner}/{state.pr_context.repo}"
        if gated_request is None:
            # FYI dedup is query → post → record, so concurrent retries must share
            # the per-review lock (DECISIONS.md#027). The body already released it;
            # re-acquire here to prevent two misses from posting twice. The accepted
            # post-before-record crash window remains. The gated mirror needs no
            # lock because duplicate chat.update calls use the same rendering.
            if result.outcome in _FYI_TRUTHFUL_OUTCOMES:
                async with publish_event_sink.acquire_publish_lock(review_id=state.review_id):
                    await slack_target.orchestrator.notify_review_posted(
                        review_id=state.review_id,
                        is_eval=state.is_eval,
                        channel_id=slack_target.channel_id,
                        repo=repo,
                        pr_number=state.pr_context.pr_number,
                        posted_count=posted_count,
                        dashboard_only_count=dashboard_only_count,
                    )
            return
        if result.outcome == "not_published_auth_revoked":
            # A delayed revoked edit could overwrite a later successful rendering:
            # chat.update has no compare-and-set, and a lock cannot impose lifecycle
            # order. A prior-PublishEvent check also misses external-record recovery,
            # which emits no such event. Leave the pending card until monotonic
            # status updates exist (FUP-252); all other terminal renderings converge.
            return
        approved_count, gated_count = count_gated_approvals(
            gated_finding_ids=gated_request.findings_requiring_approval,
            hitl_decision=state.hitl_decision,
        )
        mirror_ids = set(gated_request.findings_requiring_approval)
        await slack_target.orchestrator.mirror_status(
            review_id=state.review_id,
            channel_id=slack_target.channel_id,
            repo=repo,
            pr_number=state.pr_context.pr_number,
            pr_title=state.pr_context.pr_title,
            findings=[f for f in report.findings if f.finding_id in mirror_ids],
            reviewer_id=(
                state.hitl_decision.reviewer_id if state.hitl_decision is not None else None
            ),
            approved_count=approved_count,
            gated_count=gated_count,
            posted_count=posted_count,
            dashboard_only_count=dashboard_only_count,
        )
    except Exception:
        # Never gate-breaking: the publish outcome is already recorded; a Slack-path
        # failure must not lose the PublishResult.
        logger.exception(
            "slack terminal notification failed; publish result unaffected",
            extra={"review_id": str(state.review_id)},
        )


async def publish(
    state: ReviewState,
    *,
    publisher: GitHubPublisher,
    publish_event_sink: PublishEventSink,
    phase_event_sink: PhaseEventSink,
    review_status_sink: ReviewStatusSink,
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    dashboard_base_url: str | None = None,
    resolve_slack_target: SlackTargetResolver | None = None,
) -> dict[str, object]:
    """Run publish, finalize Slack on every non-failed outcome, then close the phase."""
    delta = await _publish_body(
        state,
        publisher=publisher,
        publish_event_sink=publish_event_sink,
        phase_event_sink=phase_event_sink,
        review_status_sink=review_status_sink,
        github_factory=github_factory,
        dashboard_base_url=dashboard_base_url,
        resolve_slack_target=resolve_slack_target,
    )
    await _finalize_slack_notifications(
        state,
        resolve_slack_target=resolve_slack_target,
        publish_event_sink=publish_event_sink,
        delta=delta,
    )
    # Close the body's deterministic phase only after Slack work completes
    # (`phase-events-bound-work`). Publish or completion-write failures never
    # reach this point, preserving the dangling-start interruption signal.
    await _emit_phase_end(
        phase_event_sink=phase_event_sink,
        review_id=state.review_id,
        phase_id=compute_phase_id(
            review_id=str(state.review_id), node_id="publish", attempt_key="publish"
        ),
        is_eval=state.is_eval,
    )
    return delta


async def _publish_body(
    state: ReviewState,
    *,
    publisher: GitHubPublisher,
    publish_event_sink: PublishEventSink,
    phase_event_sink: PhaseEventSink,
    review_status_sink: ReviewStatusSink,
    # TYPE_CHECKING keeps the vendor-wrapper type out of runtime node imports.
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    dashboard_base_url: str | None = None,
    resolve_slack_target: SlackTargetResolver | None = None,
) -> dict[str, object]:
    """Publish the synthesized report using injected publisher, sinks, and client factory.

    Returns `{"publish_result": PublishResult}` for the state overwrite reducer.
    Terminal paths call `mark_completed`, which no-ops for an already completed
    review. A missing or malformed dashboard URL renders concerns without links
    (DECISIONS.md#050). Slack finalization and phase-end belong to `publish`.

    The per-review lock covers local idempotency, empty-tier handling, the external
    recovery check, and POST. Each wrapped failure records a failed attempt before
    propagating; GitHub authorization revocation uses the retained outcome instead.
    Completion-write failures leave phase-start dangling and can retry on replay.
    """
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="publish",
        attempt_key="publish",
    )
    started_at = datetime.now(UTC)

    # Step 1: start phase event. If this raises (audit infra outage),
    # the node fails before any work — no dangling start.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="publish",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Step 2: use the synthesized report, with a duplicate-ID defense for reports
    # constructed outside synthesize's content-hash dedup path.
    admitted_findings = _collect_admitted_findings(state)
    _assert_no_duplicate_finding_ids(admitted_findings)
    verification_holds = set(
        verification_hold_finding_ids(
            admitted_findings,
            state.native_candidates,
            state.candidate_assessments,
            policy_version=(
                state.triage_result.policy_version if state.triage_result else ACTIVE_POLICY_VERSION
            ),
        )
    )

    # Recovery matches literal startswith(body_marker). Require UUID formatting
    # and stringify explicitly so producer and matcher retain the pinned shape
    # (`test_body_marker_shape_pinned`).
    if not isinstance(state.review_id, UUID):
        raise TypeError(
            f"state.review_id must be UUID (got {type(state.review_id).__name__}); "
            f"the body marker's literal shape is load-bearing for crash-after-"
            f"success recovery and depends on UUID.__str__ formatting."
        )
    body_marker = _BODY_MARKER_TEMPLATE.format(review_id=str(state.review_id))

    # Registry membership avoids coordinate translation for files outside the diff.
    changed_paths: set[str] = {cf.path for cf in state.pr_context.changed_files}

    # Step 3: interleaved per-finding routing + eligibility loop. Three
    # eligibility-gated accumulators, one per publish tier (DECISIONS.md#050):
    # a WITHHELD CRITICAL/HIGH finding reaches NONE of them.
    eligible_inline_comments: list[InlineComment] = []
    eligible_review_body_findings: list[tuple[ReviewFinding, FindingSeverity]] = []
    surfaced_dashboard_only_findings: list[ReviewFinding] = []
    for finding in admitted_findings:
        await _route_and_gate_one_finding(
            finding=finding,
            state=state,
            changed_paths=changed_paths,
            publish_event_sink=publish_event_sink,
            eligible_inline_comments=eligible_inline_comments,
            eligible_review_body_findings=eligible_review_body_findings,
            surfaced_dashboard_only_findings=surfaced_dashboard_only_findings,
            verification_hold=finding.finding_id in verification_holds,
        )

    sorted_finding_ids = tuple(sorted(f.finding_id for f in admitted_findings))

    # Step 4: local idempotency precedes empty-tier handling, so a replay cannot
    # hide prior success behind NO_OP_EMPTY. A committed PublishEvent skips all
    # GitHub calls; its absence still requires Step 6's external recovery check.
    #
    # Serialize Steps 4–7 under the per-review lock (DECISIONS.md#027): concurrent
    # human and sweep resumes must not both miss the local event and POST. If the
    # winner crashes after POST, the lock releases and the body marker recovers
    # its success. Lock/backoff rationale lives on PublishEventSink.acquire_publish_lock.
    #
    # Catch acquisition separately: its failure needs a FAILED attempt, while a
    # failure inside the critical section must retain its own single failure emit.
    lock_stack = AsyncExitStack()
    try:
        await lock_stack.enter_async_context(
            publish_event_sink.acquire_publish_lock(review_id=state.review_id),
        )
    except Exception as exc:
        await _emit_attempt(
            publish_event_sink=publish_event_sink,
            review_id=state.review_id,
            attempt_index=1,
            outcome=PublishAttemptOutcome.FAILED,
            sorted_finding_ids=sorted_finding_ids,
            comments_attempted=len(eligible_inline_comments),
            failure_class=type(exc).__name__,
            is_eval=state.is_eval,
        )
        raise

    try:
        try:
            prior_publish_event = await publish_event_sink.query_prior_publish_event(
                review_id=state.review_id,
            )
        except Exception as exc:
            await _emit_attempt(
                publish_event_sink=publish_event_sink,
                review_id=state.review_id,
                attempt_index=1,
                outcome=PublishAttemptOutcome.FAILED,
                sorted_finding_ids=sorted_finding_ids,
                comments_attempted=len(eligible_inline_comments),
                failure_class=type(exc).__name__,
                is_eval=state.is_eval,
            )
            raise
        if prior_publish_event is not None:
            await _emit_attempt(
                publish_event_sink=publish_event_sink,
                review_id=state.review_id,
                attempt_index=1,
                outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED,
                sorted_finding_ids=sorted_finding_ids,
                comments_attempted=len(eligible_inline_comments),
                is_eval=state.is_eval,
            )
            # Completion precedes the wrapper's phase-end. If it fails, replay
            # finds the committed PublishEvent and retries this write.
            await review_status_sink.mark_completed(review_id=state.review_id)
            return {
                "publish_result": PublishResult.skipped(
                    comments_posted=prior_publish_event.comments_posted,
                    review_body_findings_posted=prior_publish_event.review_body_findings_posted,
                    dashboard_only_findings_surfaced=(
                        prior_publish_event.dashboard_only_findings_surfaced
                    ),
                )
            }

        # Step 5: truly-empty short-circuit — no GitHub call ONLY when all three
        # eligibility-gated tiers are empty (DECISIONS.md#050). A review-body-only
        # or dashboard-only-only review still posts.
        if (
            not eligible_inline_comments
            and not eligible_review_body_findings
            and not surfaced_dashboard_only_findings
        ):
            await _emit_attempt(
                publish_event_sink=publish_event_sink,
                review_id=state.review_id,
                attempt_index=1,
                outcome=PublishAttemptOutcome.NO_OP_EMPTY,
                sorted_finding_ids=sorted_finding_ids,
                comments_attempted=0,
                is_eval=state.is_eval,
            )
            await review_status_sink.mark_completed(review_id=state.review_id)
            return {"publish_result": PublishResult.empty()}

        async def _handle_github_failure(exc: Exception) -> dict[str, object] | None:
            """Retain findings on revoked access; audit other failures before the caller raises."""
            status_code = _extract_status_code(exc)
            if status_code in _AUTH_REVOKED_STATUS_CODES:
                # GitHub is the final authorization gate. Its rejection is a
                # completed outcome without publication, rather than a failure.
                return await _emit_auth_revoked_and_retain(
                    publish_event_sink=publish_event_sink,
                    review_status_sink=review_status_sink,
                    phase_event_sink=phase_event_sink,
                    review_id=state.review_id,
                    phase_id=phase_id,
                    is_eval=state.is_eval,
                    sorted_finding_ids=sorted_finding_ids,
                    comments_attempted=len(eligible_inline_comments),
                    status_code=status_code,
                )
            await _emit_attempt(
                publish_event_sink=publish_event_sink,
                review_id=state.review_id,
                attempt_index=1,
                outcome=PublishAttemptOutcome.FAILED,
                sorted_finding_ids=sorted_finding_ids,
                comments_attempted=len(eligible_inline_comments),
                failure_class=type(exc).__name__,
                status_code=status_code,
                is_eval=state.is_eval,
            )
            return None

        # Recover a POST whose local PublishEvent never committed.
        try:
            # Token-minting failures belong to the same audited failure path.
            gh = await github_factory(state.pr_context.installation_id)
            existing_review_id = await publisher.find_existing_review_on_head_sha(
                gh=gh,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                pull_number=state.pr_context.pr_number,
                head_sha=state.pr_context.head_sha,
                body_marker=body_marker,
            )
        except Exception as exc:
            failure_result = await _handle_github_failure(exc)
            if failure_result is not None:
                return failure_result
            raise
        if existing_review_id is not None:
            await _emit_attempt(
                publish_event_sink=publish_event_sink,
                review_id=state.review_id,
                attempt_index=1,
                outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD,
                sorted_finding_ids=sorted_finding_ids,
                comments_attempted=len(eligible_inline_comments),
                is_eval=state.is_eval,
                # Required for this outcome — audit-only replay needs
                # the github_review_id binding (no paired PublishEvent
                # lands on the recovery path).
                recovered_github_review_id=existing_review_id,
            )
            await review_status_sink.mark_completed(review_id=state.review_id)
            return {
                "publish_result": PublishResult.skipped_external(
                    existing_review_id=existing_review_id,
                    # No prior PublishEvent on the recovery path — report what the
                    # CURRENT routing pass would have posted (DECISIONS.md#050).
                    comments_posted=len(eligible_inline_comments),
                    review_body_findings_posted=len(eligible_review_body_findings),
                    dashboard_only_findings_surfaced=len(surfaced_dashboard_only_findings),
                )
            }

        # Failure handling leaves phase-start dangling when an error propagates.
        github_review_event = "COMMENT"  # V1: every published review is a comment.
        # The renderer preserves the offset-zero recovery marker and caps display
        # text (DECISIONS.md#050). Counts describe materialized routing tiers; an
        # extreme body can be tail-truncated while the dashboard retains all findings.
        review_body = _render_review_body(
            body_marker=body_marker,
            review_body_findings=tuple(eligible_review_body_findings),
            dashboard_only_findings=tuple(surfaced_dashboard_only_findings),
            review_id=state.review_id,
            dashboard_base_url=dashboard_base_url,
        )
        try:
            review_created = await publisher.create_review(
                gh=gh,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                pull_number=state.pr_context.pr_number,
                head_sha=state.pr_context.head_sha,
                review_status=github_review_event,
                body_marker=body_marker,
                body=review_body,
                comments=tuple(eligible_inline_comments),
            )
        except Exception as exc:
            failure_result = await _handle_github_failure(exc)
            if failure_result is not None:
                return failure_result
            raise

        # Step 8: persist success before completion; the wrapper closes the phase.
        await _emit_attempt(
            publish_event_sink=publish_event_sink,
            review_id=state.review_id,
            attempt_index=1,
            outcome=PublishAttemptOutcome.SUCCESS,
            sorted_finding_ids=sorted_finding_ids,
            comments_attempted=len(eligible_inline_comments),
            is_eval=state.is_eval,
        )
        await publish_event_sink.emit_publish_result(
            PublishEvent(
                review_id=state.review_id,
                is_eval=state.is_eval,
                github_review_id=review_created.github_review_id,
                comments_posted=review_created.comments_posted,
                review_body_findings_posted=len(eligible_review_body_findings),
                dashboard_only_findings_surfaced=len(surfaced_dashboard_only_findings),
                review_status=github_review_event,
            )
        )
        # The committed PublishEvent lets Step 4 retry a failed completion write
        # without posting again. Failure still leaves phase-start dangling.
        await review_status_sink.mark_completed(review_id=state.review_id)

        # Slack runs at the wrapper's shared exit so idempotent returns can repair it.
        # Timing is represented by phase events; started_at is not part of the result.
        _ = started_at
        return {
            "publish_result": PublishResult.success(
                github_review_id=review_created.github_review_id,
                comments_posted=review_created.comments_posted,
                review_body_findings_posted=len(eligible_review_body_findings),
                dashboard_only_findings_surfaced=len(surfaced_dashboard_only_findings),
            )
        }
    finally:
        # Close the lock-holding transaction on every return or raised failure.
        await lock_stack.aclose()


# ---------------------------------------------------------------------------
# Per-finding orchestration helpers
# ---------------------------------------------------------------------------


def _extract_status_code(exc: BaseException) -> int | None:
    """Read HTTP status from the wrapper, raw SDK response, or chained cause.

    Prefer the wrapper's direct `status_code`, then `response.status_code`.
    GitHubPublishError can carry either shape on its original `__cause__`;
    ignoring that cause would lose status on GET and non-422 POST failures.
    Exceptions without HTTP context return None.
    """
    direct = getattr(exc, "status_code", None)
    if isinstance(direct, int):
        return direct
    nested = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(nested, int):
        return nested
    cause = exc.__cause__
    if cause is not None and cause is not exc:
        cause_direct = getattr(cause, "status_code", None)
        if isinstance(cause_direct, int):
            return cause_direct
        cause_nested = getattr(getattr(cause, "response", None), "status_code", None)
        if isinstance(cause_nested, int):
            return cause_nested
    return None


async def _emit_auth_revoked_and_retain(
    *,
    publish_event_sink: PublishEventSink,
    review_status_sink: ReviewStatusSink,
    phase_event_sink: PhaseEventSink,
    review_id: UUID,
    phase_id: str,
    is_eval: bool,
    sorted_finding_ids: tuple[UUID, ...],
    comments_attempted: int,
    status_code: int,
) -> dict[str, object]:
    """Record GitHub authorization revocation and retain the completed review (#065).

    GET and POST use this same outcome. Its attempt has a status code and no
    failure_class, as required for a non-FAILED event. No PublishEvent or posted
    FYI is produced; the wrapper handles Slack finalization and phase-end.
    """
    await _emit_attempt(
        publish_event_sink=publish_event_sink,
        review_id=review_id,
        attempt_index=1,
        outcome=PublishAttemptOutcome.NOT_PUBLISHED_AUTH_REVOKED,
        sorted_finding_ids=sorted_finding_ids,
        comments_attempted=comments_attempted,
        status_code=status_code,
        is_eval=is_eval,
    )
    await review_status_sink.mark_completed(review_id=review_id)
    return {"publish_result": PublishResult.not_published_auth_revoked()}


def _collect_admitted_findings(state: ReviewState) -> list[ReviewFinding]:
    """Clone admitted findings from the required synthesized report.

    Missing review_report fails closed: bypassing synthesize would lose its
    content-hash dedup and cross-round severity-divergence checks. Findings are
    mutable children of a shallow-frozen report, so copies prevent downstream
    publish_destination assignment from mutating checkpointed report state.
    `_assert_no_duplicate_finding_ids` also guards directly constructed reports.
    """
    # Direct access distinguishes a renamed schema field from a missing report.
    if state.review_report is None:
        msg = (
            "publish requires state.review_report to be set "
            "(synthesize node must have run before publish — graph wiring "
            "or test fixture bug). Fail-closed: a miswired path that "
            "bypasses synthesize would otherwise silently lose the "
            "content-hash dedup + cross-round severity-divergence "
            "detection contracts."
        )
        raise RuntimeError(msg)
    # model_copy is shallow; adding update= would bypass ReviewFinding validators.
    # Rebuild changed copies with model_validate to re-run those validators.
    return [f.model_copy() for f in state.review_report.findings]


def _assert_no_duplicate_finding_ids(findings: list[ReviewFinding]) -> None:
    """Reject duplicate finding IDs; round-level dedup cannot guard producer drift."""
    seen: set[UUID] = set()
    for f in findings:
        if f.finding_id in seen:
            raise ValueError(
                f"publish node received duplicate finding_id "
                f"{f.finding_id!r} in admitted_findings — analyze layer "
                f"reducer or producer bug. Aborting publish to defend "
                f"against double-posting on GitHub."
            )
        seen.add(f.finding_id)


async def _route_and_gate_one_finding(
    *,
    finding: ReviewFinding,
    state: ReviewState,
    changed_paths: set[str],
    publish_event_sink: PublishEventSink,
    eligible_inline_comments: list[InlineComment],
    eligible_review_body_findings: list[tuple[ReviewFinding, FindingSeverity]],
    surfaced_dashboard_only_findings: list[ReviewFinding],
    verification_hold: bool = False,
) -> None:
    """Route, gate, and audit a finding before materializing its destination (#050).

    Only ELIGIBLE findings enter one of the three accumulators; a WITHHELD
    critical/high finding reaches none. Routing-sink failure instead emits
    withheld eligibility with reason=routing_emission_failed.
    """
    # Check diff membership before attempting coordinate translation.
    destination: PublishDestination
    routing_reason: PublishRoutingReason
    coord_kind: CoordinateErrorKind | None
    routing_emission_failed = False
    inline_path: str | None = None
    inline_line: int | None = None
    inline_side: str | None = None

    if finding.file_path not in changed_paths:
        # Registry miss — coordinates not called, kind=None.
        destination = PublishDestination.DASHBOARD_ONLY
        routing_reason = PublishRoutingReason.NON_DIFFED_FILE
        coord_kind = None
    elif finding.source_side == "base":
        try:
            _validate_removed_source(finding=finding, state=state)
        except CoordinateError as coord_exc:
            destination, routing_reason, coord_kind = _classify_coordinate_error(coord_exc)
        else:
            destination = PublishDestination.REVIEW_BODY
            routing_reason = PublishRoutingReason.REMOVED_SOURCE
            coord_kind = None
    else:
        try:
            location = _resolve_inline_location(finding=finding, state=state)
        except CoordinateError as coord_exc:
            # Map CoordinateError.kind → reason per the spec table.
            destination, routing_reason, coord_kind = _classify_coordinate_error(coord_exc)
        else:
            destination = PublishDestination.INLINE_COMMENT
            routing_reason = PublishRoutingReason.REVIEWABLE_DIFF_LINE
            coord_kind = None
            inline_path = location.file_path
            inline_line = location.line
            inline_side = location.side

    # Overwrite any pre-set destination: routing is never model-assigned.
    finding.publish_destination = destination

    # Build outside sink-I/O recovery: hash or schema bugs must fail, not become
    # routing_emission_failed and silently withhold a finding.
    coord_kind_value = coord_kind.value if coord_kind is not None else None
    finding_content_hash = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
        source_side=finding.source_side,
    )
    decision_content_hash = compute_publish_routing_decision_hash(
        destination=destination,
        reason=routing_reason,
        coordinate_error_kind=coord_kind,
    )
    routing_event = PublishRoutingEvent(
        review_id=state.review_id,
        is_eval=state.is_eval,
        finding_id=finding.finding_id,
        destination=destination,
        reason=routing_reason,
        coordinate_error_kind=coord_kind_value,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
        finding_content_hash=finding_content_hash,
        source_side=finding.source_side,
        decision_content_hash=decision_content_hash,
    )
    # Sink I/O only — recovery path falls through to the eligibility
    # emit with `withheld + routing_emission_failed` per spec.
    try:
        await publish_event_sink.emit_publish_routing(routing_event)
    except Exception:
        # Recovery continues with one fewer eligible finding; warn so the reduced
        # comments_posted count is explained in logs and the demo's error tee.
        logger.warning(
            "publish: routing-event emit failed for finding %s; withholding from GitHub "
            "(ROUTING_EMISSION_FAILED)",
            finding.finding_id,
            exc_info=True,
        )
        routing_emission_failed = True

    # Eligibility decision. If routing emission failed, use the
    # withheld recovery path (override the policy's would-be answer).
    from outrider.audit.events import (
        PublishEligibilityReason,  # local import to keep top-level clean
    )

    if routing_emission_failed:
        eligibility = PublishEligibility.WITHHELD
        eligibility_reason: PublishEligibilityReason | None = (
            PublishEligibilityReason.ROUTING_EMISSION_FAILED
        )
    else:
        # The pure gate receives HITL context explicitly, including a missing gate.
        eligibility, eligibility_reason = is_eligible_for_v1_publish(
            finding,
            hitl_request=state.hitl_request,
            hitl_decision=state.hitl_decision,
            verification_hold=verification_hold,
        )

    eligibility_decision_hash = compute_publish_eligibility_decision_hash(
        eligibility=eligibility,
        reason=eligibility_reason,
    )
    # Routing and eligibility share the finding identity hash. Rendering and this
    # event also share effective severity; an override retains its policy baseline.
    effective_severity, original_severity_for_audit = _resolve_effective_severity(
        finding=finding, hitl_decision=state.hitl_decision
    )

    await publish_event_sink.emit_publish_eligibility(
        PublishEligibilityEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            finding_id=finding.finding_id,
            file_path=finding.file_path,
            line_start=finding.line_start,
            line_end=finding.line_end,
            finding_type=finding.finding_type,
            source_side=finding.source_side,
            severity=effective_severity,
            original_severity=original_severity_for_audit,
            finding_content_hash=finding_content_hash,
            decision_content_hash=eligibility_decision_hash,
            eligibility=eligibility,
            reason=eligibility_reason,
            # See DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult.
            # HITL can span a policy deploy. Use the finding's captured version so
            # replay binds this severity to the policy that classified it.
            policy_version=finding.policy_version,
        )
    )

    # Collect into the materializable list only if both:
    #   (a) routing landed on INLINE_COMMENT, AND
    #   (b) eligibility is ELIGIBLE.
    if (
        destination is PublishDestination.INLINE_COMMENT
        and eligibility is PublishEligibility.ELIGIBLE
        and inline_path is not None
        and inline_line is not None
        and inline_side is not None
    ):
        # hitl-gated describes the baseline that triggered approval, even when a
        # stored override has lowered finding.severity. It must agree with the
        # reviewer markers, `_resolve_effective_severity`, and publish_eligibility.
        baseline_severity = (
            finding.original_severity if finding.original_severity is not None else finding.severity
        )
        hitl_gated = state.hitl_request is not None and (
            is_hitl_gated_severity(baseline_severity)
            or finding.finding_id in state.hitl_request.verification_hold_finding_ids
        )
        agent_markers = _build_agent_markers(
            finding,
            effective_severity=effective_severity,
            hitl_gated=hitl_gated,
            hitl_decision=state.hitl_decision,
        )
        # The copy-paste agent prompt fences and bounds model prose without an LLM call.
        agent_prompt = _build_agent_prompt_block(finding, effective_severity=effective_severity)
        # Fixes are generated before routing but render only for eligible inline
        # findings (#040). The renderer omits absent or unsafe fixes without mutation.
        suggestion = _render_suggestion_block(finding.suggested_fix)
        body = _build_finding_comment_body(
            finding,
            effective_severity=effective_severity,
            suggestion=suggestion,
            agent_prompt=agent_prompt,
            markers=agent_markers,
        )
        eligible_inline_comments.append(
            InlineComment.from_finding(
                finding=finding,
                path=inline_path,
                line=inline_line,
                side=inline_side,
                body=body,
            )
        )
    elif (
        destination is PublishDestination.REVIEW_BODY and eligibility is PublishEligibility.ELIGIBLE
    ):
        # Body concerns use the same eligibility gate and effective severity as inline.
        eligible_review_body_findings.append((finding, effective_severity))
    elif (
        destination is PublishDestination.DASHBOARD_ONLY
        and eligibility is PublishEligibility.ELIGIBLE
    ):
        # Eligible trace-discovered finding outside the diff → counted in the
        # aggregate dashboard-only note (count + link only, never per-finding).
        surfaced_dashboard_only_findings.append(finding)


def _validate_removed_source(*, finding: ReviewFinding, state: ReviewState) -> None:
    matching = [file for file in state.pr_context.changed_files if file.path == finding.file_path]
    if len(matching) != 1:
        raise CoordinateError(
            "removed-source finding lacks one matching changed file",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        )
    changed_file = matching[0]
    removed_source_to_review_body(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        status=changed_file.status,
        base_content=changed_file.content_base,
        head_content=changed_file.content_head,
        patch=changed_file.patch,
    )


def _resolve_inline_location(
    *, finding: ReviewFinding, state: ReviewState
) -> GitHubCommentLocation:
    """Translate head-source lines into the canonical GitHub comment location.

    The caller maps CoordinateError kinds to routing reasons. Location.side is
    passed through unchanged; current translation uses head content and yields
    RIGHT. LEFT-side support belongs in coordinates, not this resolver
    (`coordinates-module-is-sole-translator`).
    """
    matching = []
    for candidate_file in state.pr_context.changed_files:
        if candidate_file.path == finding.file_path:
            matching.append(candidate_file)
    if not matching:
        # Defensive: caller already filtered on `changed_paths`. If
        # the registry says yes but pr_context disagrees, that's a
        # producer-side drift bug.
        raise CoordinateError(
            f"file {finding.file_path!r} in changed_paths registry but "
            f"absent from pr_context.changed_files — drift bug.",
            kind=CoordinateErrorKind.FILE_NOT_IN_PATCH,
        )
    changed_file = matching[0]
    if changed_file.content_head is None or changed_file.patch is None:
        # A deleted file is in the diff but has no head anchor. Preserve that
        # reason instead of collapsing it into the FILE_NOT_IN_PATCH bucket.
        raise CoordinateError(
            f"file {finding.file_path!r} has no content_head or patch "
            f"(status={changed_file.status!r}); cannot anchor inline comment.",
            kind=CoordinateErrorKind.HEAD_CONTENT_UNAVAILABLE,
        )
    # Findings carry head-source line ranges; coordinates owns line-to-byte translation.
    return source_line_to_github(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        head_content=changed_file.content_head,
        patch=changed_file.patch,
    )


def _classify_coordinate_error(
    exc: CoordinateError,
) -> tuple[PublishDestination, PublishRoutingReason, CoordinateErrorKind]:
    """Map `CoordinateError.kind` → `(destination, reason, kind)` per spec §V.

    Per `PublishRoutingEvent._enforce_coordinate_error_kind_required_iff_coordinate_error`:
      - UNCHANGED_REGION → REVIEW_BODY / unchanged_region
      - FILE_NOT_IN_PATCH → DASHBOARD_ONLY / non_diffed_file
      - everything else → DASHBOARD_ONLY / coordinate_error
    """
    if exc.kind is CoordinateErrorKind.UNCHANGED_REGION:
        return (
            PublishDestination.REVIEW_BODY,
            PublishRoutingReason.UNCHANGED_REGION,
            CoordinateErrorKind.UNCHANGED_REGION,
        )
    if exc.kind is CoordinateErrorKind.FILE_NOT_IN_PATCH:
        return (
            PublishDestination.DASHBOARD_ONLY,
            PublishRoutingReason.NON_DIFFED_FILE,
            CoordinateErrorKind.FILE_NOT_IN_PATCH,
        )
    return (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
        exc.kind,
    )


def _resolve_effective_severity(
    *,
    finding: ReviewFinding,
    hitl_decision: HITLDecision | None,
) -> tuple[FindingSeverity, FindingSeverity | None]:
    """Return `(effective_severity, original_severity_for_audit)` for rendering.

    A matching HITL SEVERITY_OVERRIDE supplies the effective value and preserves
    the policy baseline in original_severity. Without it, return baseline and None;
    forged override fields are withheld by publish_eligibility, never legitimized
    here. Baseline is finding.original_severity when present, otherwise severity.

    The GitHub header and eligibility event use this same pair. Override reason
    and reviewer identity remain on the HITLDecisionEvent, joined by finding_id.
    """
    # Find a matching SEVERITY_OVERRIDE decision (if any).
    matching: PerFindingDecision | None = None
    if hitl_decision is not None:
        for d in hitl_decision.decisions:
            if d.finding_id == finding.finding_id:
                matching = d
                break

    # Compute baseline from finding state (mirror of ReviewFinding):
    # `original_severity` IS the baseline when set; else `severity` IS.
    baseline = (
        finding.original_severity if finding.original_severity is not None else finding.severity
    )

    if (
        matching is not None
        and matching.outcome == PerFindingOutcome.SEVERITY_OVERRIDE
        and matching.override_severity is not None
    ):
        return matching.override_severity, baseline
    return baseline, None


# ---------------------------------------------------------------------------
# Audit-event emission helpers
# ---------------------------------------------------------------------------


async def _emit_attempt(
    *,
    publish_event_sink: PublishEventSink,
    review_id: UUID,
    attempt_index: int,
    outcome: PublishAttemptOutcome,
    sorted_finding_ids: tuple[UUID, ...],
    comments_attempted: int,
    is_eval: bool,
    failure_class: str | None = None,
    status_code: int | None = None,
    recovered_github_review_id: int | None = None,
) -> None:
    """Build and emit a `PublishAttemptEvent`.

    Single emission per attempt, per Q2 (no in_flight pre-call).
    `recovered_github_review_id` is required (not None) iff
    `outcome == IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD` and forbidden
    (None) otherwise — the event's model validator raises on misuse.
    """
    attempt_content_hash = compute_publish_attempt_content_hash(
        review_id=review_id,
        attempt_index=attempt_index,
        sorted_finding_ids=sorted_finding_ids,
        outcome=outcome,
        status_code=status_code,
        failure_class=failure_class,
        comments_attempted=comments_attempted,
        recovered_github_review_id=recovered_github_review_id,
    )
    await publish_event_sink.emit_publish_attempt(
        PublishAttemptEvent(
            review_id=review_id,
            is_eval=is_eval,
            attempt_index=attempt_index,
            outcome=outcome,
            status_code=status_code,
            failure_class=failure_class,
            comments_attempted=comments_attempted,
            sorted_finding_ids=sorted_finding_ids,
            attempt_content_hash=attempt_content_hash,
            recovered_github_review_id=recovered_github_review_id,
        )
    )


async def _emit_phase_end(
    *,
    phase_event_sink: PhaseEventSink,
    review_id: UUID,
    phase_id: str,
    is_eval: bool,
) -> None:
    """Emit the phase-end event matching this node's phase-start.

    Only called on successful exit paths; mid-execution failures
    propagate without emitting end (the dangling-start convention).
    """
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=review_id,
            phase_id=phase_id,
            node_id="publish",
            marker="end",
            is_eval=is_eval,
            phase_key=None,
        )
    )
