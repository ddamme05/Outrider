# Per spec §V lines 233-241: publish-idempotency coverage (FUP-066 + FUP-064).
"""Pin the publish node's idempotency contracts.

Per the publish-node spec at §V lines 233-241 the publish node MUST:

1. Content-hash dedup: two executions emit two routing rows with the
   SAME `finding_content_hash` AND same `decision_content_hash` for
   the same logical decision (consumer-side dedup identity).
2. Decision-drift surfacing: same finding, different coordinate
   outcome → SAME `finding_content_hash`, DIFFERENT
   `decision_content_hash` (both rows present; V1.5 anomaly rule
   per FUP-063 surfaces the drift).
3. Intra-execution drift detection: `_assert_no_duplicate_finding_ids`
   rejects duplicate finding_ids in admitted_findings (producer-bug
   defense).
4. **Same `review_id` dispatched twice → `idempotently_skipped`** —
   the FUP-064 closure path (`query_prior_publish_event` returns a
   prior `PublishEvent`; node short-circuits without GitHub call).
5. Crash-after-success → `idempotently_skipped_external_record` —
   covered in `test_publish_node_end_to_end.py`; sanity-check here
   for the per-spec scenario list.
6. External-record filter correctness — deleted bot review, different-
   installation review, human review all → publisher proceeds (no
   false short-circuit).
7. `PublishEvent` divergence signal: two distinct `github_review_id`
   for same `review_id` surface as two rows (consumer-side anomaly).

Helpers (`_make_*`, `_Recording*`, `_StubPublisher`) inlined per the
existing repo convention — `tests/unit/` has no `__init__.py` so
cross-file imports aren't first-class.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes import publish as publish_module
from outrider.audit.events import (
    PublishAttemptEvent,
    PublishAttemptOutcome,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    ChangedFile,
    GitHubReviewCreated,
    PRContext,
    PublishResult,
    ReviewFinding,
    ReviewState,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Autouse env fixture — scopes the truncation-HMAC env var to test execution
# (the sanitizer reads it per call; module-level os.environ.setdefault would
# leak into sibling test modules and mask missing-env failures).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_truncation_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-for-unit-tests-012345")


# ---------------------------------------------------------------------------
# Recording stubs (inlined)
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSink:
    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingPublishEventSink:
    def __init__(self) -> None:
        self.routing: list[PublishRoutingEvent] = []
        self.eligibility: list[PublishEligibilityEvent] = []
        self.attempts: list[PublishAttemptEvent] = []
        self.results: list[PublishEvent] = []
        self.prior_publish_event: PublishEvent | None = None
        self.query_calls: list[UUID] = []

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        self.routing.append(event)

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        self.eligibility.append(event)

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        self.attempts.append(event)

    async def emit_publish_result(self, event: PublishEvent) -> None:
        self.results.append(event)

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:
        self.query_calls.append(review_id)
        return self.prior_publish_event

    @asynccontextmanager
    async def acquire_publish_lock(
        self,
        *,
        review_id: UUID,  # noqa: ARG002
    ) -> AsyncIterator[None]:
        yield


class _StubPublisher:
    def __init__(
        self,
        *,
        existing_review_id: int | None = None,
        github_review_id: int = 42,
    ) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.find_calls: list[dict[str, Any]] = []
        self._existing_review_id = existing_review_id
        self._github_review_id = github_review_id

    async def create_review(self, **kwargs: Any) -> GitHubReviewCreated:
        self.create_calls.append(kwargs)
        return GitHubReviewCreated(
            github_review_id=self._github_review_id,
            comments_posted=len(kwargs["comments"]),
        )

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:
        self.find_calls.append(kwargs)
        return self._existing_review_id


class _StubReviewStatusSink:
    """No-op ReviewStatusSink stub — publish only calls mark_completed
    at terminal-success paths in these tests; the other three methods
    exist solely to satisfy Protocol membership."""

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_running(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_awaiting_approval_expired(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    async def mark_completed(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None


def _stub_github_factory(installation_id: int) -> Any:  # noqa: ARG001
    return object()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_changed_file(*, path: str = "src/foo.py") -> ChangedFile:
    return ChangedFile(
        path=path,
        status="added",
        additions=2,
        deletions=0,
        patch="@@ -0,0 +1,2 @@\n+def foo():\n+    return 1\n",
        content_base=None,
        content_head="def foo():\n    return 1\n",
        previous_path=None,
    )


def _make_finding(
    *,
    file_path: str = "src/foo.py",
    line_start: int = 2,
    line_end: int = 2,
    finding_id: UUID | None = None,
) -> ReviewFinding:
    finding_type = FindingType.MISSING_INPUT_VALIDATION
    return ReviewFinding(
        finding_id=finding_id if finding_id is not None else uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=FindingSeverity.MEDIUM,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        proposal_hash="a" * 64,  # Per DECISIONS.md#025; dummy SHA-256 hex.
    )


def _make_state(
    *,
    findings: tuple[ReviewFinding, ...],
    changed_files: tuple[ChangedFile, ...],
    review_id: UUID | None = None,
    analysis_round_findings: tuple[ReviewFinding, ...] | None = None,
) -> ReviewState:
    """Sibling of `test_hitl_node.py::_make_state` /
    `test_publish_routing.py::_make_state` /
    `test_publish_node_end_to_end.py::_make_state`. `findings`
    populates `review_report.findings` (publish's consumer surface);
    `analysis_round_findings` defaults to mirror but can be overridden
    so a regression to reading `state.analysis_rounds` instead of
    `state.review_report.findings` surfaces loudly. Per CodeRabbit /
    Pass-1 multi-lens audit sibling sweep 2026-05-28.
    """
    from outrider.policy.canonical import compute_round_id
    from outrider.schemas import ReviewMetrics, ReviewReport
    from outrider.schemas.analysis_round import AnalysisRound
    from outrider.schemas.triage_result import RiskLevel

    files_examined = tuple(cf.path for cf in changed_files)
    rounds_findings = analysis_round_findings if analysis_round_findings is not None else findings
    round_id = compute_round_id(
        pass_index=0,
        files_examined=files_examined,
        files_skipped=(),
        finding_content_hashes=tuple(f.content_hash for f in rounds_findings),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=0,
        findings=rounds_findings,
        files_examined=files_examined,
        files_skipped=(),
        started_at=datetime.now(UTC),
        ended_at=datetime.now(UTC),
    )
    # Synthesize-canonical state for publish's fail-loud check.
    review_report = ReviewReport(
        summary="test summary",
        overall_risk=RiskLevel.LOW,
        findings=findings,
        metrics=ReviewMetrics(
            files_examined=len(files_examined),
            files_traced_beyond_diff=0,
            wall_clock_seconds=0.0,
        ),
    )
    return ReviewState(
        review_id=review_id if review_id is not None else uuid4(),
        pr_context=PRContext(
            installation_id=42,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="t",
            base_sha="1" * 40,
            head_sha="0" * 40,
            author="a",
            total_additions=2,
            total_deletions=0,
            changed_files=changed_files,
        ),
        received_at=datetime.now(UTC),
        is_eval=False,
        analysis_rounds=[analysis_round],
        review_report=review_report,
    )


# ===========================================================================
# FUP-064 — intra-Outrider idempotency (query_prior_publish_event)
# ===========================================================================


@pytest.mark.asyncio
async def test_same_review_id_dispatched_twice_emits_idempotently_skipped() -> None:
    """Per spec §V line 238: same `review_id` dispatched twice → the
    second dispatch finds a prior `PublishEvent` via
    `query_prior_publish_event`, emits
    `PublishAttemptEvent(outcome=idempotently_skipped)`, makes NO GitHub call,
    and returns `PublishResult.skipped()`. FUP-064 closure path.
    """
    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()
    # Simulate the prior dispatch having succeeded — seed the sink's
    # query to return a real PublishEvent.
    sink.prior_publish_event = PublishEvent(
        review_id=state.review_id,
        is_eval=False,
        github_review_id=999,
        comments_posted=3,
        review_status="COMMENT",
    )
    publisher = _StubPublisher()

    result = await publish_module.publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert sink.query_calls == [state.review_id]  # the pre-flight check ran
    assert len(sink.attempts) == 1
    assert sink.attempts[0].outcome is PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED
    # NO GitHub round-trip burned — the intra-Outrider check short-circuits
    # BEFORE the external-record query AND BEFORE create_review.
    assert len(publisher.find_calls) == 0
    assert len(publisher.create_calls) == 0
    # No new PublishEvent — the prior one is canonical.
    assert len(sink.results) == 0
    assert isinstance(result["publish_result"], PublishResult)
    assert result["publish_result"].outcome == "idempotently_skipped"


@pytest.mark.asyncio
async def test_query_prior_publish_event_fires_before_empty_eligible_check() -> None:
    """Per spec §V pre-flight ordering (lines 314-326): intra-Outrider
    idempotency check fires BEFORE the empty-eligible short-circuit.

    Scenario: state has CRITICAL findings (all withheld → zero eligible
    inline comments) AND a prior PublishEvent exists. The node MUST
    return `idempotently_skipped`, NOT `no_op_empty` — otherwise the
    dashboard would erase the prior successful publish behind a
    no_op_empty result.
    """
    finding = _make_finding()
    # Make it CRITICAL so eligibility withholds it → eligible_inline_comments is empty.
    crit_type = FindingType.SQL_INJECTION
    finding = ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=crit_type,
        severity=FindingSeverity.CRITICAL,
        file_path="src/foo.py",
        line_start=2,
        line_end=2,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(crit_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=2,
            line_end=2,
            finding_type=crit_type,
        ),
        proposal_hash="b" * 64,  # Per DECISIONS.md#025; distinct from default fixture.
    )
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()
    sink.prior_publish_event = PublishEvent(
        review_id=state.review_id,
        is_eval=False,
        github_review_id=999,
        comments_posted=3,
        review_status="COMMENT",
    )

    result = await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    # MUST be idempotently_skipped (intra-Outrider), NOT no_op_empty.
    assert sink.attempts[0].outcome is PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED
    assert result["publish_result"].outcome == "idempotently_skipped"


@pytest.mark.asyncio
async def test_query_prior_publish_event_failure_emits_attempt_failed_before_raising() -> None:
    """If `query_prior_publish_event` raises (corrupted JSONB payload,
    DB connection drop), the publish node MUST emit
    `PublishAttemptEvent(outcome=FAILED, failure_class=type(exc).__name__)`
    BEFORE re-raising — symmetric with Step 7's GitHub-POST failure
    handling.

    Without this guard, the Step 4 failure path would leave only the
    dangling phase-start as an audit signal, with no `failure_class`
    breadcrumb for operators diagnosing the crash.
    """

    class _RaisingSink(_RecordingPublishEventSink):
        async def query_prior_publish_event(
            self,
            review_id: UUID,  # noqa: ARG002
        ) -> PublishEvent | None:
            raise RuntimeError("simulated DB-side query failure")

    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RaisingSink()
    publisher = _StubPublisher()

    with pytest.raises(RuntimeError, match="simulated DB-side query failure"):
        await publish_module.publish(
            state,
            publisher=publisher,
            publish_event_sink=sink,
            phase_event_sink=_RecordingPhaseEventSink(),
            review_status_sink=_StubReviewStatusSink(),
            github_factory=_stub_github_factory,
        )

    # Attempt event was emitted BEFORE re-raise, carrying failure_class.
    assert len(sink.attempts) == 1
    assert sink.attempts[0].outcome is PublishAttemptOutcome.FAILED
    assert sink.attempts[0].failure_class == "RuntimeError"
    # No GitHub call was made — failure preceded Step 6/7.
    assert len(publisher.find_calls) == 0
    assert len(publisher.create_calls) == 0


@pytest.mark.asyncio
async def test_failed_attempt_records_wrapper_exception_status_code() -> None:
    """`PublishAttemptEvent.status_code` MUST reflect the wrapper
    exception's `.status_code` (e.g., 422 on
    `GitHubReviewValidationError`), NOT None.

    The `getattr(getattr(exc, "response", None), "status_code", None)`
    pattern (response-only) misses wrapper exceptions that set
    `.status_code` directly. `_extract_status_code(exc)` prefers
    `exc.status_code` over `exc.response.status_code`. This test pins
    that contract end-to-end:
    a publisher that raises `GitHubReviewValidationError(status_code=422,
    ...)` MUST land `status_code=422` on the failed-attempt audit row.
    """
    from outrider.github.publisher import GitHubReviewValidationError

    class _RaisingPublisher(_StubPublisher):
        async def create_review(self, **kwargs: Any) -> GitHubReviewCreated:  # noqa: ARG002
            raise GitHubReviewValidationError(
                "atomic 422 with validation body",
                status_code=422,
                body_text='{"message":"Unprocessable Entity","errors":["..."]}',
            )

    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    publisher = _RaisingPublisher()
    sink = _RecordingPublishEventSink()

    with pytest.raises(GitHubReviewValidationError):
        await publish_module.publish(
            state,
            publisher=publisher,
            publish_event_sink=sink,
            phase_event_sink=_RecordingPhaseEventSink(),
            review_status_sink=_StubReviewStatusSink(),
            github_factory=_stub_github_factory,
        )

    # Failed-attempt row was emitted BEFORE re-raise.
    assert len(sink.attempts) == 1
    assert sink.attempts[0].outcome is PublishAttemptOutcome.FAILED
    assert sink.attempts[0].failure_class == "GitHubReviewValidationError"
    # The fix: status_code lands as 422, NOT None.
    assert sink.attempts[0].status_code == 422


@pytest.mark.asyncio
async def test_no_prior_publish_event_proceeds_normally() -> None:
    """No prior PublishEvent → query returns None → normal flow continues
    (empty-eligible check, external-record check, POST). Pins the
    no-op-on-missing-prior contract.
    """
    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()
    # sink.prior_publish_event remains None (default).
    publisher = _StubPublisher()

    result = await publish_module.publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert sink.query_calls == [state.review_id]
    # Normal success path: GitHub POST happens, success attempt emits.
    assert len(publisher.create_calls) == 1
    assert sink.attempts[0].outcome is PublishAttemptOutcome.SUCCESS
    assert result["publish_result"].outcome == "success"


# ===========================================================================
# Content-hash dedup contracts
# ===========================================================================


@pytest.mark.asyncio
async def test_content_hash_dedup_identical_decision_two_executions() -> None:
    """Two executions over the same finding → same finding_content_hash
    AND same decision_content_hash. Pins the consumer-side dedup
    identity: `(review_id, finding_id, finding_content_hash,
    decision_content_hash)` collapses to one logical row.

    NOTE: each execution still emits a row (Q5 withdrawal: no PK-no-op
    promise); consumer-side replay-equivalence dedup hashes the tuple.
    """
    # Build two distinct ReviewStates representing two dispatches of the
    # SAME logical finding. Per FUP-064 the second dispatch's prior-publish
    # query needs to return None for this scenario (no PublishEvent landed
    # in the first run yet — we're testing the routing hash, not the
    # publish-level dedup).
    finding_a = _make_finding()
    finding_b = _make_finding(finding_id=finding_a.finding_id)  # Same logical finding
    state_a = _make_state(findings=(finding_a,), changed_files=(_make_changed_file(),))
    state_b = _make_state(findings=(finding_b,), changed_files=(_make_changed_file(),))

    sink_a = _RecordingPublishEventSink()
    await publish_module.publish(
        state_a,
        publisher=_StubPublisher(),
        publish_event_sink=sink_a,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )
    sink_b = _RecordingPublishEventSink()
    await publish_module.publish(
        state_b,
        publisher=_StubPublisher(),
        publish_event_sink=sink_b,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    # Both finding_content_hash AND decision_content_hash MUST collide
    # — same logical decision produces the same dedup tuple.
    assert sink_a.routing[0].finding_content_hash == sink_b.routing[0].finding_content_hash
    assert sink_a.routing[0].decision_content_hash == sink_b.routing[0].decision_content_hash


@pytest.mark.asyncio
async def test_decision_drift_surfaces_as_distinct_decision_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same finding, DIFFERENT coordinate outcome → same finding_content_hash,
    DIFFERENT decision_content_hash. The drift signal is the
    decision-hash divergence; V1.5 anomaly rule (FUP-063) surfaces it.
    """
    finding_a = _make_finding()
    finding_b = _make_finding(finding_id=finding_a.finding_id)
    state_a = _make_state(findings=(finding_a,), changed_files=(_make_changed_file(),))
    state_b = _make_state(findings=(finding_b,), changed_files=(_make_changed_file(),))

    # Execution A: coordinates returns success (INLINE_COMMENT route).
    sink_a = _RecordingPublishEventSink()
    await publish_module.publish(
        state_a,
        publisher=_StubPublisher(),
        publish_event_sink=sink_a,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    # Execution B: monkeypatch source_line_to_github to raise UNCHANGED_REGION.
    def _raise(**_kwargs: object) -> object:
        raise CoordinateError("drift", kind=CoordinateErrorKind.UNCHANGED_REGION)

    monkeypatch.setattr(publish_module, "source_line_to_github", _raise)
    sink_b = _RecordingPublishEventSink()
    await publish_module.publish(
        state_b,
        publisher=_StubPublisher(),
        publish_event_sink=sink_b,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    # finding_content_hash agrees (same identity tuple).
    assert sink_a.routing[0].finding_content_hash == sink_b.routing[0].finding_content_hash
    # decision_content_hash DIFFERS (the drift signal).
    assert sink_a.routing[0].decision_content_hash != sink_b.routing[0].decision_content_hash


# ===========================================================================
# Intra-execution drift detection
# ===========================================================================


@pytest.mark.asyncio
async def test_duplicate_finding_ids_across_rounds_rejected_by_publish() -> None:
    """`_assert_no_duplicate_finding_ids` rejects duplicate finding_ids
    on the canonical post-synthesize ReviewReport path.

    Post-synthesize, the primary defense against cross-round duplicate
    findings is synthesize's content_hash dedup + cross-round severity-
    divergence detection (`_detect_and_report_divergence`). The publish
    node's `_assert_no_duplicate_finding_ids` is the belt-and-suspenders
    second layer: a forged or test-fixture-constructed ReviewReport
    that bypassed synthesize (different content_hashes but same
    finding_id — which the ReviewReport schema validator does NOT
    catch, since it only checks content_hash uniqueness) is still
    rejected by publish before any GitHub call.

    This test constructs such a forged state: two findings with
    different content_hashes (line_start differs) but the same
    finding_id. ReviewReport schema admits the construction; publish's
    finding-id-uniqueness check fires.
    """
    from outrider.schemas import ReviewMetrics, ReviewReport
    from outrider.schemas.triage_result import RiskLevel

    finding_id = uuid4()
    cf = _make_changed_file()
    # Distinct content_hashes (line_start differs) so ReviewReport
    # schema validator admits the tuple — only finding_id collides.
    f1 = _make_finding(finding_id=finding_id, line_start=2, line_end=2)
    f2 = _make_finding(finding_id=finding_id, line_start=3, line_end=3)

    review_report = ReviewReport(
        summary="test summary",
        overall_risk=RiskLevel.LOW,
        findings=(f1, f2),
        metrics=ReviewMetrics(
            files_examined=1,
            files_traced_beyond_diff=0,
            wall_clock_seconds=0.0,
        ),
    )
    state = ReviewState(
        review_id=uuid4(),
        pr_context=PRContext(
            installation_id=42,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="t",
            base_sha="1" * 40,
            head_sha="0" * 40,
            author="a",
            total_additions=2,
            total_deletions=0,
            changed_files=(cf,),
        ),
        received_at=datetime.now(UTC),
        is_eval=False,
        analysis_rounds=[],
        review_report=review_report,
    )

    with pytest.raises(ValueError, match="duplicate finding_id"):
        await publish_module.publish(
            state,
            publisher=_StubPublisher(),
            publish_event_sink=_RecordingPublishEventSink(),
            phase_event_sink=_RecordingPhaseEventSink(),
            review_status_sink=_StubReviewStatusSink(),
            github_factory=_stub_github_factory,
        )


# ===========================================================================
# External-record filter correctness
# ===========================================================================


@pytest.mark.asyncio
async def test_external_record_returns_none_proceeds_to_post() -> None:
    """When `find_existing_review_on_head_sha` returns None (no prior
    matching review), the publish node proceeds to the GitHub POST.

    This is the most common case (no prior publish, no crash-recovery
    needed) AND covers the three external-record filter sub-cases the
    spec names (deleted bot review, different-installation review,
    human review) — each of which causes the publisher's matcher to
    return None.
    """
    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    publisher = _StubPublisher(existing_review_id=None)
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(publisher.find_calls) == 1  # external-record check ran
    assert len(publisher.create_calls) == 1  # POST proceeded
    assert sink.attempts[0].outcome is PublishAttemptOutcome.SUCCESS


@pytest.mark.asyncio
async def test_external_record_match_short_circuits_no_post() -> None:
    """When the body-marker matcher in `find_existing_review_on_head_sha`
    hits, the publish node emits `idempotently_skipped_external_record`
    and skips the POST. The matcher itself has 3 checks (body-marker
    startswith + commit_id + Bot author) — tested in publisher unit
    tests; this test pins the node's behavior on a hit.
    """
    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    publisher = _StubPublisher(existing_review_id=777)
    sink = _RecordingPublishEventSink()

    result = await publish_module.publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(publisher.create_calls) == 0  # POST skipped
    assert sink.attempts[0].outcome is (PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD)
    assert result["publish_result"].github_review_id == 777


@pytest.mark.asyncio
async def test_external_record_query_failure_emits_attempt_failed_before_raising() -> None:
    """When `find_existing_review_on_head_sha` raises (network drop,
    App-uninstalled mid-run, 5xx upstream, pagination cap exhaustion),
    the publish node emits `PublishAttemptEvent(FAILED,
    failure_class=type(exc).__name__, status_code=...)` BEFORE
    re-raising. Symmetric with Step 4 (intra-Outrider query) + Step 7
    (POST) failure handling. Without this guard, the dangling
    phase-start would be the only signal; operators couldn't
    distinguish "external-record query crashed" from "node hung
    mid-execution".
    """
    from outrider.github.publisher import GitHubPublishError

    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))

    class _RaisingPublisher(_StubPublisher):
        async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:
            self.find_calls.append(kwargs)
            raise GitHubPublishError(
                "simulated: GitHub GET reviews failed (503 upstream)",
            )

    publisher = _RaisingPublisher(existing_review_id=None)
    sink = _RecordingPublishEventSink()

    # Pin the re-raise contract: the publish node MUST propagate the
    # exact `GitHubPublishError` (not a wrapped or transformed type)
    # after emitting the FAILED attempt. A broader `pytest.raises
    # (Exception)` would silently pass if a future refactor wrapped
    # the Step 6 exception as something else.
    with pytest.raises(GitHubPublishError, match="simulated: GitHub GET reviews failed"):
        await publish_module.publish(
            state,
            publisher=publisher,
            publish_event_sink=sink,
            phase_event_sink=_RecordingPhaseEventSink(),
            review_status_sink=_StubReviewStatusSink(),
            github_factory=_stub_github_factory,
        )

    # Exactly one PublishAttemptEvent emitted, with outcome=FAILED and
    # failure_class set to the raised exception's class name. Step 4's
    # `prior_publish_event` short-circuit didn't fire (sink returns None
    # from query_prior_publish_event), so we reached Step 6 and emitted
    # FAILED there.
    assert len(sink.attempts) == 1
    assert sink.attempts[0].outcome is PublishAttemptOutcome.FAILED
    assert sink.attempts[0].failure_class == "GitHubPublishError"
    # The publisher's create_review must NOT have been called (we
    # raised before reaching Step 7).
    assert len(publisher.create_calls) == 0
    # find_existing_review_on_head_sha WAS called and raised (Step 6).
    assert len(publisher.find_calls) == 1


# ===========================================================================
# PublishEvent divergence signal
# ===========================================================================


@pytest.mark.asyncio
async def test_publish_event_emits_with_canonical_dedup_identity() -> None:
    """The success path emits exactly one `PublishEvent` carrying
    `(review_id, github_review_id)` — the consumer-side dedup identity
    per spec §V "Consumer dedup contract for PublishEvent." Two distinct
    `github_review_id` for the same `review_id` would surface as
    divergent rows (anomaly signal).

    This test verifies the single-emission contract; the divergence
    test would require two successful publishes for the same review_id,
    which the intra-Outrider check (FUP-064, tested above) prevents in
    V1. The divergence-signal test is V1.5 dashboard scope.
    """
    finding = _make_finding()
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    publisher = _StubPublisher(github_review_id=555)
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=publisher,
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(sink.results) == 1  # exactly one PublishEvent
    assert sink.results[0].review_id == state.review_id
    assert sink.results[0].github_review_id == 555
