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

from contextlib import AsyncExitStack
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

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
    source_line_to_github,
)
from outrider.coordinates.errors import CoordinateErrorKind
from outrider.policy.canonical import compute_phase_id
from outrider.policy.publish_eligibility import is_eligible_for_v1_publish
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    InlineComment,
    PublishDestination,
    PublishResult,
)
from outrider.schemas.hitl import PerFindingOutcome

if TYPE_CHECKING:
    from collections.abc import Callable

    from outrider.audit.sinks import PhaseEventSink, PublishEventSink
    from outrider.db.sinks import ReviewStatusSink
    from outrider.github import InstallationGitHubClient
    from outrider.github.publisher import GitHubPublisher
    from outrider.policy import FindingSeverity
    from outrider.schemas import ReviewFinding, ReviewState
    from outrider.schemas.hitl import HITLDecision, PerFindingDecision

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
    review_status_sink: ReviewStatusSink,
    # `InstallationGitHubClient` is the typed `GitHub[AppInstallationAuthStrategy]`
    # alias from `outrider.github.auth`; the TYPE_CHECKING import keeps
    # the runtime free of the githubkit wrapper-module dependency (the
    # node never references the type at runtime — it just calls the
    # factory and passes the result to the publisher).
    github_factory: Callable[[int], InstallationGitHubClient],
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
        review_status_sink: `ReviewStatusSink` used at terminal-success
            paths to write `reviews.status='completed'` +
            `completed_at=NOW()` per canonical lifecycle (`docs/spec.md`
            §3.3 step 10; `docs/architecture.md` step 10). The write
            is predicate-gated on `status='running'`; a re-run sees
            `completed` already and no-ops.
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

    # Step 2: collect admitted findings from analysis_rounds. Per the
    # spec's "intra-execution drift detection" test, defend against
    # producer regression that emits duplicate finding_ids.
    admitted_findings = _collect_admitted_findings(state)
    _assert_no_duplicate_finding_ids(admitted_findings)

    # Build the body marker once — embedded in the review body for
    # crash-after-success recovery. Explicit `str(...)` (rather than
    # implicit f-string `__str__`) defends against silent format drift
    # if `ReviewState.review_id` is ever retyped from UUID to a
    # different identity type: the matcher at
    # `find_existing_review_on_head_sha` does a literal `startswith`
    # so the marker shape MUST be deterministic across producer +
    # consumer. `UUID.__str__` is the canonical 8-4-4-4-12 hex form;
    # any other identity type would land here with a different shape
    # and break crash-recovery silently. Cast via `str(...)` makes the
    # stringification explicit so a future type change surfaces as a
    # test failure (test_body_marker_shape_pinned) rather than silently
    # at runtime.
    if not isinstance(state.review_id, UUID):
        raise TypeError(
            f"state.review_id must be UUID (got {type(state.review_id).__name__}); "
            f"the body marker's literal shape is load-bearing for crash-after-"
            f"success recovery and depends on UUID.__str__ formatting."
        )
    body_marker = _BODY_MARKER_TEMPLATE.format(review_id=str(state.review_id))

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

    # Step 4: intra-Outrider idempotency pre-flight (FUP-064 closed).
    # The publish_event_sink's `query_prior_publish_event`
    # method (shipped on AuditPersister) returns the most-recent prior
    # `PublishEvent` for this review_id if one exists. Same-review_id
    # redispatch (e.g., dispatcher re-fires the webhook after agent
    # crash + restart) short-circuits here — no GitHub round-trip
    # burned. Distinct from the Step 6 external-record check:
    #   - Step 4 (here): the prior process succeeded AND persisted
    #     PublishEvent. Local audit log proves the publish happened.
    #   - Step 6: the prior process succeeded at the GitHub POST but
    #     died BEFORE persisting PublishEvent. Local audit log has no
    #     prior; the external-record body-marker query on GitHub is
    #     the only signal.
    # Spec ordering (§V lines 314-326): intra-Outrider BEFORE empty-
    # eligible BEFORE external-record. Reasoning: if we already
    # published, even an empty-eligible re-run should report skipped
    # rather than producing a no_op_empty result that would mask the
    # prior success on the dashboard.
    # Symmetric with Step 7's POST-failure handling: if the read-side
    # query raises (e.g., corrupted JSONB payload fails
    # `PublishEvent.model_validate`, DB connection drops mid-SELECT),
    # emit `PublishAttemptEvent(FAILED, failure_class=type(exc).__name__)`
    # BEFORE re-raising so the audit trail records the failure class.
    # Without this wrap, the dangling phase-start would be the only
    # signal — operators diagnosing the failure couldn't distinguish
    # "intra-Outrider idempotency query crashed" from "node hung
    # mid-execution".
    #
    # Concurrent-invocation race defense — per-review advisory lock
    # (try-lock with bounded backoff, serialize-then-observe).
    # See DECISIONS.md#027 — V1 per-review publish-side advisory lock.
    # The `query_prior_publish_event` → POST → `emit_publish_result`
    # sequence is read-before-write. Two concurrent
    # `ainvoke(Command(resume=...))` on the same `thread_id` (e.g., a
    # human-issued resume racing with a `reclaim_stuck_hitl_states`
    # graph-driven resume) could both observe `prior_publish_event=None`
    # and both POST. Defense: `acquire_publish_lock(review_id)` runs
    # `pg_try_advisory_xact_lock(hashtext('publish:<uuid>'))` in a
    # bounded backoff loop. On NOT-acquired, the probe session
    # releases its connection back to the pool and sleeps with
    # exponential backoff (50ms doubling to 1s cap) before retrying;
    # on acquired, holds the session+transaction for the lifetime of
    # the critical section. The eventual acquire puts the second task
    # BEHIND the first task's transaction boundary — Step 4's
    # `query_prior_publish_event` then observes the first task's
    # committed `PublishEvent` (success path → authentic
    # `IDEMPOTENTLY_SKIPPED`) OR its absence (first task crashed
    # before emit → second task POSTs through Step 7).
    #
    # Why try-lock + bounded backoff (NOT plain blocking
    # `pg_advisory_xact_lock`): blocking holds a connection for the
    # entire wait. With N same-review contenders, blocking pins N
    # connections — the winner's `emit_publish_*` calls (each opens
    # a fresh session per the per-emit discipline) could be starved
    # by the held waiters. Backoff releases the connection between
    # probes; pool pressure drops from N held to ~1 held + occasional
    # probes.
    #
    # Why NOT single-shot try-lock with immediate loser-skip: the
    # immediate loser cannot observe whether the winner actually
    # committed the POST. Skipping → false `IDEMPOTENTLY_SKIPPED`
    # when winner crashes mid-POST → publish lost.
    #
    # Timeout: default 120s from first probe. On exhaustion,
    # `AuditPersisterPublishLockAcquisitionTimeoutError` raises out
    # of `enter_async_context` and the outer try/except below emits
    # `PublishAttemptEvent(FAILED, failure_class="...PublishLock
    # AcquisitionTimeoutError")` before re-raising.
    # `find_existing_review_on_head_sha` at Step 6 remains the
    # defense for the cross-process crash-after-success-before-emit
    # case (matches by body marker on a process restart, when no
    # in-process lock can apply). The structural split — lock-
    # acquire in its own try, critical-section in a separate
    # try/finally — means inner-step failures (Step 4/6/7) reach
    # only their own existing FAILED emits, never the outer catch,
    # so no double-emit.
    # Acquire the lock BEFORE entering the critical section, in its
    # own try/except so a failure during `__aenter__` (DB outage,
    # connection drop) emits `PublishAttemptEvent(FAILED)` BEFORE
    # re-raising — honoring the node's raises contract. The
    # `AsyncExitStack` holds the lock for the lifetime of the
    # critical section that follows; `stack.aclose()` releases on
    # both success and exception paths.
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
            # Terminal-success lifecycle write per canonical
            # `docs/spec.md` §3.3 step 10. Placed BEFORE
            # `_emit_phase_end` so a lifecycle-write failure leaves
            # phase-start dangling (the canonical "publish interrupted"
            # signal); a sweep-driven retry hits the prior PublishEvent
            # short-circuit again and re-attempts the completion write.
            await review_status_sink.mark_completed(review_id=state.review_id)
            await _emit_phase_end(
                phase_event_sink=phase_event_sink,
                review_id=state.review_id,
                phase_id=phase_id,
                is_eval=state.is_eval,
            )
            return {"publish_result": PublishResult.skipped()}

        # Step 5: empty-eligible-inline short-circuit. No GitHub call.
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
            # Terminal-success lifecycle write per canonical
            # `docs/spec.md` §3.3 step 10. See the equivalent comment
            # at the Step-4 short-circuit above for ordering rationale.
            await review_status_sink.mark_completed(review_id=state.review_id)
            await _emit_phase_end(
                phase_event_sink=phase_event_sink,
                review_id=state.review_id,
                phase_id=phase_id,
                is_eval=state.is_eval,
            )
            return {"publish_result": PublishResult.empty()}

        # Step 6: external-record check (crash-after-success defense).
        # The intra-Outrider check at Step 4 returns None for this
        # scenario (the prior process died BEFORE persisting
        # PublishEvent), so the external-record body-marker query on
        # GitHub is the load-bearing signal here.
        #
        # Symmetric with Steps 4 + 7 failure handling: if the GitHub
        # GET raises (network drop, 403 App-uninstalled mid-run, 5xx
        # upstream, pagination cap exhausted), emit
        # `PublishAttemptEvent(FAILED, failure_class=type(exc).__name__)`
        # BEFORE re-raising so the audit trail records the failure
        # class — otherwise the dangling phase-start is the only signal
        # and operators can't distinguish "external-record query
        # crashed" from "node hung mid-execution".
        try:
            # `github_factory(...)` is inside the try because
            # installation-token minting can raise (App uninstalled,
            # JWT clock skew, GitHub identity-API outage) — its failure
            # must land in the audit chain as `PublishAttemptEvent(FAILED)`,
            # not as a dangling phase-start.
            gh = github_factory(state.pr_context.installation_id)
            existing_review_id = await publisher.find_existing_review_on_head_sha(
                gh=gh,
                owner=state.pr_context.owner,
                repo=state.pr_context.repo,
                pull_number=state.pr_context.pr_number,
                head_sha=state.pr_context.head_sha,
                body_marker=body_marker,
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
                status_code=_extract_status_code(exc),
                is_eval=state.is_eval,
            )
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
            # Terminal-success lifecycle write per canonical
            # `docs/spec.md` §3.3 step 10. See the equivalent comment
            # at the Step-4 short-circuit above for ordering rationale.
            await review_status_sink.mark_completed(review_id=state.review_id)
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

        # Step 7: POST the review. Failures emit attempt(failed) BEFORE
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
                status_code=_extract_status_code(exc),
                is_eval=state.is_eval,
            )
            raise

        # Step 8: success path — emit attempt + canonical PublishEvent
        # + phase end + return success result.
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
        # Terminal-success lifecycle write per canonical `docs/spec.md`
        # §3.3 step 10. Placed BEFORE `_emit_phase_end` so a lifecycle-
        # write failure leaves phase-start dangling (the canonical
        # "publish interrupted" signal); the `PublishEvent` is already
        # committed so a sweep-driven retry sees prior-publish-event at
        # Step 4 and short-circuits to IDEMPOTENTLY_SKIPPED, where
        # `mark_completed` retries.
        await review_status_sink.mark_completed(review_id=state.review_id)
        await _emit_phase_end(
            phase_event_sink=phase_event_sink,
            review_id=state.review_id,
            phase_id=phase_id,
            is_eval=state.is_eval,
        )

        # Started_at is not part of the result shape — kept as a local
        # marker for future eval-timing metrics; PublishEvent doesn't
        # carry it because the phase event bracket is the canonical
        # timing source.
        _ = started_at
        return {
            "publish_result": PublishResult.success(
                github_review_id=review_created.github_review_id,
                comments_posted=review_created.comments_posted,
            )
        }
    finally:
        # Release the advisory lock on every exit path (success,
        # short-circuit return, raised exception). The acquired lock
        # is transaction-scoped, so `aclose()` commits the lock-
        # holding transaction and releases.
        await lock_stack.aclose()


# ---------------------------------------------------------------------------
# Per-finding orchestration helpers
# ---------------------------------------------------------------------------


def _extract_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status code from a publish-path exception, if present.

    Three exception shapes carry status; checked in order:
      1. Wrapper exceptions (`GitHubReviewValidationError`,
         `GitHubSecondaryRateLimitError`) set `exc.status_code` directly
         at construction. Prefer this — it's the wrapper's contract.
      2. Raw githubkit `RequestFailed` carries `exc.response.status_code`.
         Falls through to this for raw passes-from-the-SDK paths.
      3. Bare `GitHubPublishError` wraps the original SDK exception via
         `raise ... from exc`; the original lives at `exc.__cause__`,
         which carries `.response.status_code` for the SDK shape and
         (in the wrapper subclasses' raise sites) `.status_code` directly.
         Without this hop, non-422 POST failures + GET-reviews-list
         failures would record `status_code=None` on `PublishAttemptEvent`.

    Returns `None` for exceptions with no HTTP context (network errors
    pre-response, programmer-error exceptions like `ValueError`).
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
    inline_side: str | None = None

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
            inline_path = location.file_path
            inline_line = location.line
            inline_side = location.side

    # Per the spec's "publish_destination pre-set overwrite" test:
    # routing ALWAYS overwrites the finding's publish_destination
    # regardless of any pre-set value (model can't pick destination).
    finding.publish_destination = destination

    # Build the routing event OUTSIDE the try/except: hash computation
    # and `PublishRoutingEvent(...)` construction (Pydantic validation)
    # are producer/schema concerns, not sink-I/O concerns. Wrapping
    # them in the recovery path would silently convert producer bugs
    # (drift between event schema + emitter, hash recipe regression)
    # into `routing_emission_failed` — masking the bug as a withheld
    # finding instead of failing fast.
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
        decision_content_hash=decision_content_hash,
    )
    # Sink I/O only — recovery path falls through to the eligibility
    # emit with `withheld + routing_emission_failed` per spec.
    try:
        await publish_event_sink.emit_publish_routing(routing_event)
    except Exception:
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
        # HITL context flows through explicit kwargs per the gate's
        # pure-function contract; state.hitl_request / state.hitl_decision
        # are populated by the HITL node body (None for the pass-through
        # path or for graph-wiring bypasses).
        eligibility, eligibility_reason = is_eligible_for_v1_publish(
            finding,
            hitl_request=state.hitl_request,
            hitl_decision=state.hitl_decision,
        )

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

    # Look up the matching HITL decision (if any) to honor a
    # SEVERITY_OVERRIDE outcome. Per the post-HITL audit convention
    # (mirror of ReviewFinding): when override is in effect, the audit
    # row's `severity` carries the OVERRIDE value and
    # `original_severity` carries the POLICY BASELINE; the rendered
    # GitHub comment header uses the override value too. When no
    # override, severity = baseline (= SEVERITY_POLICY[finding_type]),
    # original_severity = None.
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
            severity=effective_severity,
            original_severity=original_severity_for_audit,
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
        and inline_side is not None
    ):
        # Build the inline comment via the canonical factory. Body
        # construction is V1-minimal: severity + finding type + title
        # + description. The full sanitizer pipeline applies — caller
        # never sees raw model output.
        body = _build_finding_comment_body(finding, effective_severity=effective_severity)
        eligible_inline_comments.append(
            InlineComment.from_finding(
                finding=finding,
                path=inline_path,
                line=inline_line,
                side=inline_side,
                body=body,
            )
        )


def _resolve_inline_location(
    *, finding: ReviewFinding, state: ReviewState
) -> GitHubCommentLocation:
    """Resolve a `ReviewFinding` to a `GitHubCommentLocation` via coordinates.

    Returns the `GitHubCommentLocation` (file_path + line + side) on
    success; raises `CoordinateError` on unchanged-region / past-EOF /
    etc. The publisher's caller catches and maps the kind to a routing
    reason. Returning the canonical `coordinates` type directly (rather
    than a dict) lets the caller type-narrow each field at attribute
    access without an `int | str` union dance.

    `side` ("LEFT" | "RIGHT") comes from `GitHubCommentLocation` and is
    passed through to `InlineComment` unchanged — the publisher does
    not independently decide head-vs-base. V1 always sees "RIGHT"
    because `tree_sitter_to_github` accepts only `head_content` today;
    a future spec adding LEFT-side commenting extends the translator,
    not this resolver (per `coordinates-module-is-sole-translator` +
    spec §V publisher-input-contract sub-rule).
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
    if changed_file.content_head is None or changed_file.patch is None:
        # `removed` files have content_head=None; trying to publish
        # against a removed file can't anchor inline. Distinct from
        # FILE_NOT_IN_PATCH (which means "absent from patch entirely")
        # — the file IS in the patch, just deleted. Per audit-stream
        # replay equivalence: a finding on a deleted file should
        # surface with a discriminating reason, not collapse into the
        # registry-miss bucket.
        raise CoordinateError(
            f"file {finding.file_path!r} has no content_head or patch "
            f"(status={changed_file.status!r}); cannot anchor inline comment.",
            kind=CoordinateErrorKind.HEAD_CONTENT_UNAVAILABLE,
        )
    # `source_line_to_github` is the line-based entry point in
    # coordinates: `ReviewFinding` carries `line_start` / `line_end`
    # (source-line coords on head), not the byte-span coords
    # `tree_sitter_to_github` consumes directly. The line→byte
    # translation lives in `coordinates/translator.py` per
    # `coordinates-module-is-sole-translator`.
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
    """Apply a matching HITL `SEVERITY_OVERRIDE` decision to the
    finding's severity for publish-time rendering.

    Returns `(effective_severity, original_severity_for_audit)`:

      - LEGITIMATE OVERRIDE — a matching
        `PerFindingDecision(outcome=SEVERITY_OVERRIDE)` is present in
        `hitl_decision`:
          effective = decision.override_severity (reviewer's choice)
          original_severity_for_audit = baseline (policy mapping)
        The audit event records the override on `severity` + the
        baseline on `original_severity` — replay reconstructs "what
        severity did the GitHub comment show" from this pair.

      - NO MATCHING OVERRIDE — either `hitl_decision is None` OR the
        matching decision is not SEVERITY_OVERRIDE OR the finding's
        forged `original_severity` lacks a legitimating decision:
          effective = baseline (policy mapping)
          original_severity_for_audit = None
        The audit event records the baseline on `severity` + None on
        `original_severity` — the gate's WITHHELD outcome
        (UNEXPECTED_OVERRIDE_FIELDS_PRESENT) records that the forged
        finding never reached GitHub.

    The baseline is computed mirror-of-ReviewFinding: when the finding
    itself carries a non-None `original_severity`, that field IS the
    baseline (and `finding.severity` carries the would-be-override).
    Otherwise `finding.severity` IS the baseline (LLM-produced under
    policy, no override path involved).

    Per `severity-set-by-policy`, the override REASON + reviewer
    identity live on the paired `HITLDecisionEvent.decisions[i]`
    joined by `finding_id` — the publish event records only the
    override SIGNAL + the resolved effective severity.
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


def _build_finding_comment_body(
    finding: ReviewFinding, *, effective_severity: FindingSeverity
) -> str:
    """V1-minimal comment body. Full sanitizer pipeline applies.

    `effective_severity` is the post-HITL-override severity to render
    in the header. Resolved by `_resolve_effective_severity(...)` —
    matches what publish.py emits on `PublishEligibilityEvent.severity`
    so the user-visible GitHub comment and the audit shadow agree on
    the effective severity.

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
        f"**{effective_severity.value.upper()}** · "
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
