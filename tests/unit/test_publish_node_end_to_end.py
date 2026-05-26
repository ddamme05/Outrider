# End-to-end publish-node tests with real `ChangedFile` + `ReviewFinding`.
"""Run the publish node body against real schemas + stub sinks/publisher.

This file exists because the prior unit-test suite (`test_publish_*.py`)
covered every helper in isolation but never invoked `publish(...)` with
a real `ReviewState` carrying real `ChangedFile` / `ReviewFinding`
instances. That gap let two HIGH-confidence cross-file attribute bugs
escape (`changed_file.head_content` instead of `content_head`;
`finding.byte_start` / `byte_end` instead of the actual `line_start` /
`line_end` fields) until Codex caught them at PR review.

The lesson: AST-based unit tests + isolated-helper tests are necessary
but not sufficient for cross-file integration; one end-to-end node-body
test catches the class of bugs static audits miss.

Sister of the FUP-066 tracking (`test_publish_routing.py` +
`test_publish_idempotency.py`); this file covers the regression floor
that the spec didn't explicitly name but should have.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes.publish import publish
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
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    ChangedFile,
    GitHubReviewCreated,
    PRContext,
    PublishDestination,
    PublishResult,
    ReviewFinding,
    ReviewState,
)

# ---------------------------------------------------------------------------
# Autouse env fixture — scopes the truncation-HMAC env var to test execution.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_truncation_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-for-unit-tests")


# ---------------------------------------------------------------------------
# Recording stubs
# ---------------------------------------------------------------------------


class _RecordingPhaseEventSink:
    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


class _RecordingPublishEventSink:
    """Recording PublishEventSink — captures all emit_* calls + serves a
    configurable `prior_publish_event` from `query_prior_publish_event`.

    Tests configure `prior_publish_event` before invoking publish to
    simulate the intra-Outrider idempotency hit path (FUP-064 closed).
    Default `None` = "no prior publish" = continues to empty-eligible
    + external-record + POST paths.
    """

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

    async def query_prior_publish_event(self, review_id: UUID) -> PublishEvent | None:
        self.query_calls.append(review_id)
        return self.prior_publish_event


class _StubPublisher:
    """Hand-rolled GitHubPublisher stub.

    Records every call so tests can assert "publisher was/wasn't called"
    against the eligibility-gate contract. `create_review` returns a
    canned `GitHubReviewCreated` unless `should_raise` is set;
    `find_existing_review_on_head_sha` returns `existing_review_id`
    (default None = "no prior matching review found").
    """

    def __init__(
        self,
        *,
        existing_review_id: int | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.find_calls: list[dict[str, Any]] = []
        self._existing_review_id = existing_review_id
        self._should_raise = should_raise

    async def create_review(self, **kwargs: Any) -> GitHubReviewCreated:
        self.create_calls.append(kwargs)
        if self._should_raise is not None:
            raise self._should_raise
        return GitHubReviewCreated(
            github_review_id=42,
            comments_posted=len(kwargs["comments"]),
        )

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:
        self.find_calls.append(kwargs)
        return self._existing_review_id


def _stub_github_factory(installation_id: int) -> Any:  # noqa: ARG001
    """Returns a sentinel — the stub publisher never actually uses it."""
    return object()


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_changed_file(
    *,
    path: str = "src/foo.py",
    content_head: str | None = "def foo():\n    return 1\n",
    content_base: str | None = None,
    patch: str | None = "@@ -0,0 +1,2 @@\n+def foo():\n+    return 1\n",
    status: str = "added",
    additions: int = 2,
    deletions: int = 0,
) -> ChangedFile:
    return ChangedFile(
        path=path,
        status=status,  # type: ignore[arg-type]
        additions=additions,
        deletions=deletions,
        patch=patch,
        content_base=content_base,
        content_head=content_head,
        previous_path=None,
    )


def _make_finding(
    *,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    file_path: str = "src/foo.py",
    line_start: int = 1,
    line_end: int = 1,
    original_severity: FindingSeverity | None = None,
) -> ReviewFinding:
    finding_type_by_severity = {
        FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
        FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
        FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
        FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
        FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
    }
    baseline = original_severity if original_severity is not None else severity
    finding_type = finding_type_by_severity[baseline]
    override_kwargs: dict[str, Any] = {}
    if original_severity is not None:
        override_kwargs = {
            "original_severity": original_severity,
            "override_reason": "test override",
            "overrider_id": uuid4(),
        }
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
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
        **override_kwargs,
    )


def _make_state(
    *,
    findings: tuple[ReviewFinding, ...] = (),
    changed_files: tuple[ChangedFile, ...] = (),
) -> ReviewState:
    """Build a ReviewState with one AnalysisRound carrying the findings."""
    from outrider.policy.canonical import compute_round_id
    from outrider.schemas.analysis_round import AnalysisRound

    review_id = uuid4()
    pr_context = PRContext(
        installation_id=42,
        owner="test-owner",
        repo="test-repo",
        pr_number=1,
        pr_title="test PR",
        base_sha="1" * 40,
        head_sha="0" * 40,
        author="test-user",
        total_additions=2,
        total_deletions=0,
        changed_files=changed_files,
    )
    if findings:
        files_examined = tuple(cf.path for cf in changed_files)
        round_id = compute_round_id(
            pass_index=0,
            files_examined=files_examined,
            files_skipped=(),
            finding_content_hashes=tuple(f.content_hash for f in findings),
        )
        analysis_round = AnalysisRound(
            round_id=round_id,
            pass_index=0,
            findings=findings,
            files_examined=files_examined,
            files_skipped=(),
            started_at=datetime.now(UTC),
            ended_at=datetime.now(UTC),
        )
        analysis_rounds = [analysis_round]
    else:
        analysis_rounds = []
    return ReviewState(
        review_id=review_id,
        pr_context=pr_context,
        received_at=datetime.now(UTC),
        is_eval=False,
        analysis_rounds=analysis_rounds,
    )


# ---------------------------------------------------------------------------
# Happy path: one MEDIUM finding → routing → eligible → publish → success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_node_happy_path_emits_all_four_event_types() -> None:
    """End-to-end: one MEDIUM finding routes INLINE, gates ELIGIBLE,
    publisher posts, all four event types land.

    THIS test would have caught the head_content vs content_head bug
    AND the byte_start vs line_start bug at first run.
    """
    finding = _make_finding(severity=FindingSeverity.MEDIUM, line_start=2, line_end=2)
    changed_file = _make_changed_file(path=finding.file_path)
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    result = await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    # Phase event bracket landed start + end.
    assert len(phase_sink.events) == 2
    assert phase_sink.events[0].marker == "start"
    assert phase_sink.events[1].marker == "end"
    # Per-finding routing + eligibility events landed.
    assert len(publish_sink.routing) == 1
    assert publish_sink.routing[0].destination is PublishDestination.INLINE_COMMENT
    assert publish_sink.routing[0].reason is PublishRoutingReason.REVIEWABLE_DIFF_LINE
    assert len(publish_sink.eligibility) == 1
    assert publish_sink.eligibility[0].eligibility is PublishEligibility.ELIGIBLE
    # Attempt + result events landed on success path.
    assert len(publish_sink.attempts) == 1
    assert publish_sink.attempts[0].outcome is PublishAttemptOutcome.SUCCESS
    assert len(publish_sink.results) == 1
    assert publish_sink.results[0].github_review_id == 42
    # State delta carries success result.
    assert isinstance(result["publish_result"], PublishResult)
    assert result["publish_result"].outcome == "success"
    # Publisher was called.
    assert len(publisher.create_calls) == 1


# ---------------------------------------------------------------------------
# Eligibility-gate-before-publisher contract (FUP-062 functional pin)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critical_finding_withheld_publisher_not_called() -> None:
    """CRITICAL finding → withheld at gate → publisher.create_review NOT called.

    FUP-062's functional exit condition: the gate fires BEFORE
    materialization, so a critical-severity finding never reaches GitHub.
    """
    finding = _make_finding(severity=FindingSeverity.CRITICAL, line_start=2, line_end=2)
    changed_file = _make_changed_file(path=finding.file_path)
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    result = await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    # Routing happened (always); eligibility=withheld.
    assert len(publish_sink.routing) == 1
    assert len(publish_sink.eligibility) == 1
    assert publish_sink.eligibility[0].eligibility is PublishEligibility.WITHHELD
    # Publisher was NOT called for the withheld finding.
    assert len(publisher.create_calls) == 0
    # Attempt event = no_op_empty (zero eligible+INLINE findings).
    assert len(publish_sink.attempts) == 1
    assert publish_sink.attempts[0].outcome is PublishAttemptOutcome.NO_OP_EMPTY
    assert result["publish_result"].outcome == "empty"


@pytest.mark.asyncio
async def test_fabricated_override_withheld_publisher_not_called() -> None:
    """Forged-override finding → withheld at gate → publisher NOT called.

    FUP-062's threat-model pin: a producer bug or replay-injected row
    forging `original_severity` does NOT reach GitHub.
    """
    finding = _make_finding(
        severity=FindingSeverity.LOW,
        original_severity=FindingSeverity.CRITICAL,
        line_start=2,
        line_end=2,
    )
    changed_file = _make_changed_file(path=finding.file_path)
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    from outrider.audit.events import PublishEligibilityReason

    assert publish_sink.eligibility[0].eligibility is PublishEligibility.WITHHELD
    assert (
        publish_sink.eligibility[0].reason
        is PublishEligibilityReason.UNEXPECTED_OVERRIDE_FIELDS_PRESENT
    )
    assert len(publisher.create_calls) == 0


# ---------------------------------------------------------------------------
# External-record short-circuit (crash-after-success recovery)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_review_on_head_sha_short_circuits() -> None:
    """`find_existing_review_on_head_sha` returns prior review_id →
    publish.create_review NOT called → idempotently_skipped_external_record."""
    finding = _make_finding(severity=FindingSeverity.MEDIUM, line_start=2, line_end=2)
    changed_file = _make_changed_file(path=finding.file_path)
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher(existing_review_id=999)

    result = await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    assert len(publisher.find_calls) == 1  # external-record query ran
    assert len(publisher.create_calls) == 0  # no new POST
    assert publish_sink.attempts[0].outcome is (
        PublishAttemptOutcome.IDEMPOTENTLY_SKIPPED_EXTERNAL_RECORD
    )
    assert result["publish_result"].github_review_id == 999


# ---------------------------------------------------------------------------
# Coordinate-error routing (removed-file → HEAD_CONTENT_UNAVAILABLE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_removed_file_routes_dashboard_only_with_head_content_unavailable() -> None:
    """A finding on a removed file (content_head=None) routes to
    DASHBOARD_ONLY with reason=COORDINATE_ERROR + kind=HEAD_CONTENT_UNAVAILABLE.

    Pins the routing distinction that prevents this case from being
    mis-classified as `non_diffed_file`.
    """
    finding = _make_finding(severity=FindingSeverity.MEDIUM, line_start=1, line_end=1)
    # Removed file: content_head=None, patch present but no head version.
    # `removed` status requires content_base set + content_head=None +
    # additions=0 per ChangedFile's enforce_status validator.
    changed_file = ChangedFile(
        path=finding.file_path,
        status="removed",
        additions=0,
        deletions=1,
        patch="@@ -1,1 +0,0 @@\n-old\n",
        content_base="old\n",
        content_head=None,
        previous_path=None,
    )
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    assert len(publish_sink.routing) == 1
    routing = publish_sink.routing[0]
    assert routing.destination is PublishDestination.DASHBOARD_ONLY
    assert routing.reason is PublishRoutingReason.COORDINATE_ERROR
    assert routing.coordinate_error_kind == "head_content_unavailable"
    assert len(publisher.create_calls) == 0


# ---------------------------------------------------------------------------
# Registry-miss short-circuit (finding on non-diffed file)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_diffed_file_routes_dashboard_only_registry_miss() -> None:
    """Finding on a file NOT in changed_files → DASHBOARD_ONLY +
    reason=non_diffed_file (registry-miss code path, NOT a CoordinateError)."""
    finding = _make_finding(
        severity=FindingSeverity.MEDIUM,
        file_path="src/other.py",  # NOT in changed_files
        line_start=1,
        line_end=1,
    )
    changed_file = _make_changed_file(path="src/foo.py")  # different file
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    assert publish_sink.routing[0].destination is PublishDestination.DASHBOARD_ONLY
    assert publish_sink.routing[0].reason is PublishRoutingReason.NON_DIFFED_FILE
    assert publish_sink.routing[0].coordinate_error_kind is None  # registry miss
    assert len(publisher.create_calls) == 0


# ---------------------------------------------------------------------------
# F1: SEVERITY_OVERRIDE renders the override in header + audit event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_severity_override_renders_override_severity_in_comment_and_audit() -> None:
    """When a HITL SEVERITY_OVERRIDE decision matches, publish:
      - Emits PublishEligibilityEvent with severity=override, original_severity=baseline
      - Renders the GitHub comment header using the OVERRIDE severity

    Without this fix (the F1 audit finding), publish silently dropped the
    override on the floor — audit + GitHub showed the original CRITICAL
    severity even though the reviewer downgraded to LOW.
    """
    from datetime import UTC, datetime, timedelta

    from outrider.schemas.hitl import (
        HITLDecision,
        HITLRequest,
        PerFindingDecision,
        PerFindingOutcome,
    )

    # CRITICAL finding admitted from analyze (no override on the finding).
    finding = _make_finding(
        severity=FindingSeverity.CRITICAL,
        original_severity=None,
        line_start=2,
        line_end=2,
    )
    now = datetime.now(UTC)
    hitl_request = HITLRequest(
        findings_requiring_approval=(finding.finding_id,),
        auto_post_findings=(),
        created_at=now,
        expires_at=now + timedelta(minutes=30),
    )
    # Reviewer downgrades to LOW via SEVERITY_OVERRIDE.
    decision = PerFindingDecision(
        finding_id=finding.finding_id,
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason="downgrade per project context",
        override_severity=FindingSeverity.LOW,
        original_severity=FindingSeverity.CRITICAL,
    )
    hitl_decision = HITLDecision(
        reviewer_id="admin",
        decisions=(decision,),
        decided_at=now,
    )

    changed_file = _make_changed_file(path=finding.file_path)
    state = _make_state(findings=(finding,), changed_files=(changed_file,))
    # Inject HITL state (both request + decision) — mirrors the hitl
    # node's state delta when reviewer authorizes the override.
    state = state.model_copy(update={"hitl_request": hitl_request, "hitl_decision": hitl_decision})

    phase_sink = _RecordingPhaseEventSink()
    publish_sink = _RecordingPublishEventSink()
    publisher = _StubPublisher()

    await publish(
        state,
        publisher=publisher,
        publish_event_sink=publish_sink,
        phase_event_sink=phase_sink,
        github_factory=_stub_github_factory,
    )

    # Audit event records override on severity + baseline on original_severity.
    elig = publish_sink.eligibility[0]
    assert elig.severity is FindingSeverity.LOW, (
        f"expected effective severity=LOW (override), got {elig.severity}"
    )
    assert elig.original_severity is FindingSeverity.CRITICAL, (
        f"expected original_severity=CRITICAL (baseline), got {elig.original_severity}"
    )
    assert elig.eligibility is PublishEligibility.ELIGIBLE

    # GitHub comment posted; header rendered with OVERRIDE severity.
    assert len(publisher.create_calls) == 1
    posted_comments = publisher.create_calls[0]["comments"]
    assert len(posted_comments) == 1
    body = posted_comments[0].body
    # Header begins with `**LOW**` (the override), NOT `**CRITICAL**` (baseline).
    assert "**LOW**" in body, f"expected override severity LOW in body, got: {body[:80]!r}"
    assert "**CRITICAL**" not in body, (
        f"baseline CRITICAL should not appear in body, got: {body[:80]!r}"
    )
