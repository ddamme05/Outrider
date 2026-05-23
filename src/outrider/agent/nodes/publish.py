# Publish node per specs/2026-05-21-publish-node.md §V + DECISIONS.md #023.
"""Publish node: post inline-review comments to GitHub via githubkit.

Per DECISIONS.md #023 (publish routing and eligibility are separate
decisions, not one combined gate): this node interleaves routing +
eligibility per finding in a single loop, emits the four publish event
types per the spec, and orchestrates the GitHub POST through a
`GitHubPublisher` injected at `build_graph` time.

**No LLM calls.** Verified by absence of `LLMCallEvent` emission AND
an import-graph unit test pinning that `agent.nodes.publish` doesn't
transitively import `outrider.llm`.

Failure discipline matches analyze: phase-start emits at entry,
phase-end emits at successful exit, mid-execution failures propagate
WITHOUT emitting end (the dangling-start is the audit signal for
"publish interrupted"). The per-finding `try/except` around routing
emission catches into `PublishEligibilityEvent(withheld,
routing_emission_failed)` so the per-finding audit contract holds
even when routing emit fails — but a failure of the GitHub call
itself propagates, after emitting `PublishAttemptEvent(failed)`.

Pre-flight order (intra-Outrider + external):
  1. Per-finding routing + eligibility loop (always).
  2. Intra-Outrider idempotency: prior `PublishEvent` for this
     `review_id` → emit `PublishAttemptEvent(idempotently_skipped)`,
     return `PublishResult.skipped()`.
  3. Empty-eligible-inline-comments → emit
     `PublishAttemptEvent(no_op_empty)`, return `PublishResult.empty()`.
  4. External-record check: `find_existing_review_on_head_sha` via
     body marker → emit
     `PublishAttemptEvent(idempotently_skipped_external_record)`,
     return `PublishResult.skipped_external(...)`.
  5. POST review → on success, emit `PublishAttemptEvent(success)` +
     `PublishEvent(...)`, return `PublishResult.success(...)`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

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
    tree_sitter_to_github,
)
from outrider.coordinates.errors import CoordinateErrorKind
from outrider.policy.publish_eligibility import is_eligible_for_v1_publish
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    InlineComment,
    PublishDestination,
    PublishResult,
)

if TYPE_CHECKING:
    from outrider.audit.sinks import PhaseEventSink, PublishEventSink
    from outrider.github.publisher import GitHubPublisher
    from outrider.schemas import ReviewFinding, ReviewState

__all__ = ["publish"]


# Body-marker template per spec §V + DECISIONS.md #023's crash-after-
# success defense. The marker rides on the review body so a retry can
# query GitHub for an existing review carrying this exact marker on
# the same head_sha. Per Q6 + 4d sandbox verification, GitHub preserves
# the body text verbatim under apiVersion 2026-03-10.
_BODY_MARKER_TEMPLATE = "<!-- outrider-review-id:{review_id} -->"


async def publish(
    state: ReviewState,
    *,
    publisher: GitHubPublisher,
    publish_event_sink: PublishEventSink,
    phase_event_sink: PhaseEventSink,
    github_factory,  # Callable[[int], InstallationGitHubClient] — typed
    # loosely here to avoid the github wrapper import at module level
    # (nodes consume githubkit ONLY through the factory closure per
    # `vendor-sdks-only-in-wrappers`).
    active_policy_version: str = ACTIVE_POLICY_VERSION,
) -> dict[str, object]:
    """Run the V1 publish flow over admitted findings.

    Args:
        state: `ReviewState` with `analysis_rounds` populated.
        publisher: `GitHubPublisher` Protocol implementation
            (production: `GitHubKitPublisher`; test:
            hand-rolled stub).
        publish_event_sink: `PublishEventSink` for the four publish
            event types (production: `AuditPersister`).
        phase_event_sink: `PhaseEventSink` for the start/end phase
            event bracket.
        github_factory: per-installation githubkit client factory
            per `nodes-receive-deps-via-closure`.
        active_policy_version: V1 default is `ACTIVE_POLICY_VERSION`;
            tests override to pin replay equivalence under historical
            policies.

    Returns:
        `{"publish_result": PublishResult}` for LangGraph's default
        overwrite reducer to merge into `state.publish_result`.

    Raises:
        Propagates any uncaught exception (GitHub HTTP failure,
        publisher contract violation, persister conflict) AFTER
        emitting `PublishAttemptEvent(outcome=failed,
        failure_class=type(exc).__name__)`. The phase-start event
        remains dangling — that's the audit signal for "publish
        interrupted" per the analyze convention.
    """
    phase_id = str(uuid4())
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

    # Step 2: collect admitted findings from analysis_rounds. Per the
    # spec's "intra-execution drift detection" test, defend against
    # producer regression that emits duplicate finding_ids.
    admitted_findings = _collect_admitted_findings(state)
    _assert_no_duplicate_finding_ids(admitted_findings)

    # Build the body marker once — embedded in the review body for
    # crash-after-success recovery.
    body_marker = _BODY_MARKER_TEMPLATE.format(review_id=state.review_id)

    # Build a quick-lookup registry of file paths in the diff so
    # routing's "non_diffed_file" short-circuit can decide WITHOUT
    # calling tree_sitter_to_github (per FUP-057 resolution: V1
    # publish does file-membership via the in-memory ChangedFile
    # registry, not file_in_patch).
    changed_paths: set[str] = {cf.path for cf in state.pr_context.changed_files}

    # Step 3: interleaved per-finding routing + eligibility loop.
    eligible_inline_comments: list[InlineComment] = []
    for finding in admitted_findings:
        await _route_and_gate_one_finding(
            finding=finding,
            state=state,
            changed_paths=changed_paths,
            publish_event_sink=publish_event_sink,
            active_policy_version=active_policy_version,
            eligible_inline_comments=eligible_inline_comments,
        )

    sorted_finding_ids = tuple(sorted(f.finding_id for f in admitted_findings))

    # Step 4: empty-eligible-inline short-circuit. No GitHub call.
    if not eligible_inline_comments:
        await _emit_attempt(
            publish_event_sink=publish_event_sink,
            review_id=state.review_id,
            attempt_index=1,
            outcome=PublishAttemptOutcome.NO_OP_EMPTY,
            sorted_finding_ids=sorted_finding_ids,
            comments_attempted=0,
            is_eval=state.is_eval,
        )
        await _emit_phase_end(
            phase_event_sink=phase_event_sink,
            review_id=state.review_id,
            phase_id=phase_id,
            is_eval=state.is_eval,
        )
        return {"publish_result": PublishResult.empty()}

    # Step 5: external-record check (crash-after-success defense).
    # V1 minimally exercises intra-Outrider idempotency via a future
    # `_query_prior_publish_event` (not shipped in this PR — depends on
    # a DB-side query helper the persister doesn't expose yet); the
    # external-record path is the load-bearing defense today.
    gh = github_factory(state.pr_context.installation_id)
    existing_review_id = await publisher.find_existing_review_on_head_sha(
        gh=gh,
        owner=state.pr_context.owner,
        repo=state.pr_context.repo,
        pull_number=state.pr_context.pr_number,
        head_sha=state.pr_context.head_sha,
        body_marker=body_marker,
    )
    if existing_review_id is not None:
        await _emit_attempt(
            publish_event_sink=publish_event_sink,
            review_id=state.review_id,
            attempt_index=1,
            outcome=PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD,
            sorted_finding_ids=sorted_finding_ids,
            comments_attempted=len(eligible_inline_comments),
            is_eval=state.is_eval,
        )
        await _emit_phase_end(
            phase_event_sink=phase_event_sink,
            review_id=state.review_id,
            phase_id=phase_id,
            is_eval=state.is_eval,
        )
        return {
            "publish_result": PublishResult.skipped_external(
                existing_review_id=existing_review_id,
            )
        }

    # Step 6: POST the review. Failures emit attempt(failed) BEFORE
    # re-raising so the audit trail has the failure_class on record.
    # The phase-start remains dangling on failure (analyze convention).
    review_status = "COMMENT"  # V1: every published review is a comment.
    # When synthesize ships, status is derived from the highest-severity
    # finding that actually posted (per docs/spec.md §V).
    try:
        review_created = await publisher.create_review(
            gh=gh,
            owner=state.pr_context.owner,
            repo=state.pr_context.repo,
            pull_number=state.pr_context.pr_number,
            head_sha=state.pr_context.head_sha,
            review_status=review_status,
            body_marker=body_marker,
            comments=tuple(eligible_inline_comments),
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
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
            is_eval=state.is_eval,
        )
        raise

    # Step 7: success path — emit attempt + canonical PublishEvent +
    # phase end + return success result.
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
            review_status=review_status,
        )
    )
    await _emit_phase_end(
        phase_event_sink=phase_event_sink,
        review_id=state.review_id,
        phase_id=phase_id,
        is_eval=state.is_eval,
    )

    # Started_at is not part of the result shape — kept as a local
    # marker for future eval-timing metrics; PublishEvent doesn't carry
    # it because the phase event bracket is the canonical timing source.
    _ = started_at
    return {
        "publish_result": PublishResult.success(
            github_review_id=review_created.github_review_id,
            comments_posted=review_created.comments_posted,
        )
    }


# ---------------------------------------------------------------------------
# Per-finding orchestration helpers
# ---------------------------------------------------------------------------


def _collect_admitted_findings(state: ReviewState) -> list[ReviewFinding]:
    """Flatten admitted findings across all analysis_rounds.

    V1 analyze is single-pass per `pass_index=0` so this is one round
    in practice; the loop is shaped for V1.5 parallel-analyze where
    multiple rounds accumulate.
    """
    out: list[ReviewFinding] = []
    for round_ in state.analysis_rounds:
        out.extend(round_.findings)
    return out


def _assert_no_duplicate_finding_ids(findings: list[ReviewFinding]) -> None:
    """Pin the contract that admitted findings have distinct finding_ids.

    Analyze's `append_with_dedup_by(lambda r: r.round_id)` reducer
    dedups at the round layer; per the spec's "intra-execution drift
    detection" the publish node ALSO defends at the finding layer
    against producer regression.
    """
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
    active_policy_version: str,
    eligible_inline_comments: list[InlineComment],
) -> None:
    """Route + gate one finding, emit both per-finding events, optionally
    collect into `eligible_inline_comments` if the finding is materializable.

    Per the spec's "routing-emission-failed recovery": if routing
    emission raises, the per-finding `try/except` falls back to
    eligibility=withheld + reason=routing_emission_failed.
    """
    # Routing: branch on registry membership FIRST (cheap), then call
    # tree_sitter_to_github for the slow path. Per the spec's reason
    # mapping table at §V.
    destination: PublishDestination
    routing_reason: PublishRoutingReason
    coord_kind: CoordinateErrorKind | None
    routing_emission_failed = False
    inline_path: str | None = None
    inline_line: int | None = None

    if finding.file_path not in changed_paths:
        # Registry miss — coordinates not called, kind=None.
        destination = PublishDestination.DASHBOARD_ONLY
        routing_reason = PublishRoutingReason.NON_DIFFED_FILE
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
            inline_path = location["path"]
            inline_line = location["line"]

    # Per the spec's "publish_destination pre-set overwrite" test:
    # routing ALWAYS overwrites the finding's publish_destination
    # regardless of any pre-set value (model can't pick destination).
    finding.publish_destination = destination

    # Build + emit the routing event. Wrapped in try/except so the
    # per-finding eligibility emission still fires (routing-emission-
    # failed recovery).
    try:
        coord_kind_value = coord_kind.value if coord_kind is not None else None
        finding_content_hash = compute_finding_content_hash(
            file_path=finding.file_path,
            line_start=finding.line_start,
            line_end=finding.line_end,
            finding_type=finding.finding_type,
        )
        decision_content_hash = compute_publish_routing_decision_hash(
            destination=destination,
            reason=routing_reason,
            coordinate_error_kind=coord_kind,
        )
        await publish_event_sink.emit_publish_routing(
            PublishRoutingEvent(
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
                decision_content_hash=decision_content_hash,
            )
        )
    except Exception:
        # Per spec: routing-emission-failed recovery. Fall through to
        # eligibility emission with withheld + routing_emission_failed.
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
        eligibility, eligibility_reason = is_eligible_for_v1_publish(finding)

    eligibility_decision_hash = compute_publish_eligibility_decision_hash(
        eligibility=eligibility,
        reason=eligibility_reason,
    )
    finding_content_hash_for_eligibility = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
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
            severity=finding.severity,
            original_severity=None,  # V1 always None; gate already rejected non-None
            finding_content_hash=finding_content_hash_for_eligibility,
            decision_content_hash=eligibility_decision_hash,
            eligibility=eligibility,
            reason=eligibility_reason,
            policy_version=active_policy_version,
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
    ):
        # Build the inline comment via the canonical factory. Body
        # construction is V1-minimal: severity + finding type + title
        # + description. The full sanitizer pipeline applies — caller
        # never sees raw model output.
        body = _build_finding_comment_body(finding)
        eligible_inline_comments.append(
            InlineComment.from_finding(
                finding=finding,
                path=inline_path,
                line=inline_line,
                body=body,
            )
        )


def _resolve_inline_location(*, finding: ReviewFinding, state: ReviewState) -> dict[str, int | str]:
    """Resolve a `ReviewFinding` to (path, line) via coordinates.

    Returns `{"path": str, "line": int}` on success; raises
    `CoordinateError` on unchanged-region / past-EOF / etc. The
    publisher's caller catches and maps the kind to a routing reason.

    The finding's `byte_start`/`byte_end` are tree-sitter byte spans
    over the head version of the file; coordinates translates these
    to source-line + side per the 4d sandbox-verified shape (line +
    side=RIGHT under 2026-03-10).
    """
    # Find the ChangedFile for this finding.file_path so we have the
    # head content + patch. _collect_admitted_findings's caller has
    # already ensured the file is in `changed_paths`, so this lookup
    # is guaranteed to find a match for the inline path. The
    # comprehension expresses this intent.
    matching = [cf for cf in state.pr_context.changed_files if cf.path == finding.file_path]
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
    if changed_file.head_content is None or changed_file.patch is None:
        # `removed` files have head_content=None; trying to publish
        # against a removed file can't anchor inline. Distinct from
        # FILE_NOT_IN_PATCH (which means "absent from patch entirely")
        # — the file IS in the patch, just deleted. Per audit-stream
        # replay equivalence: a finding on a deleted file should
        # surface with a discriminating reason, not collapse into the
        # registry-miss bucket.
        raise CoordinateError(
            f"file {finding.file_path!r} has no head_content or patch "
            f"(status={changed_file.status!r}); cannot anchor inline comment.",
            kind=CoordinateErrorKind.HEAD_CONTENT_UNAVAILABLE,
        )
    location = tree_sitter_to_github(
        file_path=finding.file_path,
        byte_start=finding.byte_start,
        byte_end=finding.byte_end,
        head_content=changed_file.head_content.decode("utf-8", errors="replace"),
        patch=changed_file.patch,
    )
    return {"path": location.file_path, "line": location.line}


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


def _build_finding_comment_body(finding: ReviewFinding) -> str:
    """V1-minimal comment body. Full sanitizer pipeline applies.

    Builds `**severity** · **finding_type** — title\n\ndescription` and
    runs it through `sanitize_display_string` + `apply_size_cap`. The
    sanitizer enforces the byte cap including the truncation marker
    and any fencing overhead the body composes.
    """
    # Local import to keep the module top-level clean and to avoid
    # pulling the sanitizer's HMAC-secret env-read at import time.
    from outrider.policy.output_sanitizer import (  # noqa: PLC0415
        apply_size_cap,
        sanitize_display_string,
    )

    title_sanitized = sanitize_display_string(finding.title)
    description_sanitized = sanitize_display_string(finding.description)
    header = (
        f"**{finding.severity.value.upper()}** · "
        f"**{finding.finding_type.value}** — {title_sanitized}"
    )
    body = f"{header}\n\n{description_sanitized}"
    return apply_size_cap(body)


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
) -> None:
    """Build and emit a `PublishAttemptEvent`.

    Single emission per attempt, per Q2 (no in_flight pre-call).
    """
    attempt_content_hash = compute_publish_attempt_content_hash(
        review_id=review_id,
        attempt_index=attempt_index,
        sorted_finding_ids=sorted_finding_ids,
        outcome=outcome,
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
