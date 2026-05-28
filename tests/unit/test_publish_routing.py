# Per spec §V lines 204-211: publish-routing branch coverage (FUP-066).
"""Pin the publish node's routing decision matrix.

Per the publish-node spec at §V lines 204-211, the publish node MUST:

1. Route reviewable findings to `INLINE_COMMENT` / `reviewable_diff_line`.
2. Route `CoordinateError(kind=UNCHANGED_REGION)` to `REVIEW_BODY` /
   `unchanged_region`.
3. Route registry-miss findings to `DASHBOARD_ONLY` / `non_diffed_file`
   WITHOUT calling `source_line_to_github` (FUP-057 short-circuit).
4. For each `CoordinateErrorKind`, route to the right reason + kind
   payload. Publisher branches on the TYPED `kind` discriminator,
   NEVER on `str(exc)`.
5. ALWAYS emit a routing event for every admitted finding.
6. OVERWRITE any pre-set `publish_destination` on the finding.
7. NEVER leak `CoordinateError.message` text into the audit row's
   `coordinate_error_kind` field (info-leak defense).

Helpers (`_make_*`, `_Recording*`, `_StubPublisher`) inlined per the
existing `test_publish_node_end_to_end.py` pattern — `tests/unit/` has no
`__init__.py` so cross-file imports aren't first-class in this repo.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes import publish as publish_module
from outrider.audit.events import (
    PublishAttemptEvent,
    PublishEligibility,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    PublishRoutingReason,
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
    PublishDestination,
    ReviewFinding,
    ReviewState,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# ---------------------------------------------------------------------------
# Autouse env fixture — scopes the truncation-HMAC env var to test execution.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_truncation_hmac_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "test-secret-for-unit-tests")


# ---------------------------------------------------------------------------
# Recording stubs (inlined per repo's no-cross-test-import convention)
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

    async def emit_publish_routing(self, event: PublishRoutingEvent) -> None:
        self.routing.append(event)

    async def emit_publish_eligibility(self, event: PublishEligibilityEvent) -> None:
        self.eligibility.append(event)

    async def emit_publish_attempt(self, event: PublishAttemptEvent) -> None:
        self.attempts.append(event)

    async def emit_publish_result(self, event: PublishEvent) -> None:
        self.results.append(event)

    async def query_prior_publish_event(self, *, review_id: UUID) -> PublishEvent | None:  # noqa: ARG002
        return self.prior_publish_event

    @asynccontextmanager
    async def acquire_publish_lock(
        self,
        *,
        review_id: UUID,  # noqa: ARG002
    ) -> AsyncIterator[None]:
        yield


class _StubPublisher:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.find_calls: list[dict[str, Any]] = []

    async def create_review(self, **kwargs: Any) -> GitHubReviewCreated:
        self.create_calls.append(kwargs)
        return GitHubReviewCreated(github_review_id=42, comments_posted=len(kwargs["comments"]))

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:
        self.find_calls.append(kwargs)
        return None


class _StubReviewStatusSink:
    """No-op ReviewStatusSink stub — publish only calls mark_completed
    at terminal-success paths; the other three methods exist solely to
    satisfy Protocol membership in case future test changes route into
    the HITL node body via the same fixture."""

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


_FINDING_TYPE_BY_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
    FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
    FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
}


def _make_changed_file(
    *,
    path: str = "src/foo.py",
    content_head: str | None = "def foo():\n    return 1\n",
) -> ChangedFile:
    return ChangedFile(
        path=path,
        status="added",
        additions=2,
        deletions=0,
        patch="@@ -0,0 +1,2 @@\n+def foo():\n+    return 1\n",
        content_base=None,
        content_head=content_head,
        previous_path=None,
    )


def _make_finding(
    *,
    severity: FindingSeverity = FindingSeverity.MEDIUM,
    file_path: str = "src/foo.py",
    line_start: int = 1,
    line_end: int = 1,
    policy_version: str = ACTIVE_POLICY_VERSION,
) -> ReviewFinding:
    finding_type = _FINDING_TYPE_BY_SEVERITY[severity]
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
        policy_version=policy_version,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        # Per DECISIONS.md#025: distinct per-call so multi-finding tests
        # don't trip the AnalysisRound._enforce_findings_proposal_hash_unique
        # validator on identical defaults. Fresh `uuid4().hex + uuid4().hex`
        # (two random 32-char hex strings concatenated to fit the SHA256
        # 64-hex shape) — non-deterministic per call but uniqueness is the
        # contract, not reproducibility. Per CodeRabbit round-9 N3: prior
        # comment claimed "Derives from finding_id" but the code didn't —
        # rewritten to match what the body actually does.
        proposal_hash=uuid4().hex + uuid4().hex,
    )


def _make_state(
    *,
    findings: tuple[ReviewFinding, ...],
    changed_files: tuple[ChangedFile, ...],
) -> ReviewState:
    from outrider.policy.canonical import compute_round_id
    from outrider.schemas import ReviewMetrics, ReviewReport
    from outrider.schemas.analysis_round import AnalysisRound
    from outrider.schemas.triage_result import RiskLevel

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
    # Synthesize-canonical state: post-Phase-5, publish reads findings
    # from `state.review_report.findings`. Build a minimal valid
    # ReviewReport so the publish node's fail-loud check passes.
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
            changed_files=changed_files,
        ),
        received_at=datetime.now(UTC),
        is_eval=False,
        analysis_rounds=[analysis_round],
        review_report=review_report,
    )


# ---------------------------------------------------------------------------
# (1) Happy path — reviewable diff line → INLINE_COMMENT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reviewable_diff_line_routes_inline_comment() -> None:
    """Finding on a line in the diff → INLINE_COMMENT / REVIEWABLE_DIFF_LINE."""
    finding = _make_finding(line_start=2, line_end=2)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(sink.routing) == 1
    assert sink.routing[0].destination is PublishDestination.INLINE_COMMENT
    assert sink.routing[0].reason is PublishRoutingReason.REVIEWABLE_DIFF_LINE
    assert sink.routing[0].coordinate_error_kind is None


# ---------------------------------------------------------------------------
# (2) UNCHANGED_REGION → REVIEW_BODY
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unchanged_region_routes_review_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """CoordinateError(kind=UNCHANGED_REGION) → REVIEW_BODY / unchanged_region."""

    def _raise(**_kwargs: object) -> object:
        raise CoordinateError("span in unchanged code", kind=CoordinateErrorKind.UNCHANGED_REGION)

    monkeypatch.setattr(publish_module, "source_line_to_github", _raise)
    finding = _make_finding(line_start=2, line_end=2)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert sink.routing[0].destination is PublishDestination.REVIEW_BODY
    assert sink.routing[0].reason is PublishRoutingReason.UNCHANGED_REGION
    assert sink.routing[0].coordinate_error_kind == "unchanged_region"


# ---------------------------------------------------------------------------
# (3) Registry miss → DASHBOARD_ONLY (and source_line_to_github NEVER called)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_diffed_file_routes_dashboard_only_never_calls_sitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding on a file NOT in changed_files → DASHBOARD_ONLY /
    non_diffed_file. `source_line_to_github` MUST NOT be called — the
    FUP-057 short-circuit avoids re-parsing the patch for files known
    not to be in the diff.
    """
    sitter_mock = MagicMock(
        side_effect=AssertionError(
            "source_line_to_github must NOT be called for registry-miss findings"
        )
    )
    monkeypatch.setattr(publish_module, "source_line_to_github", sitter_mock)
    finding = _make_finding(file_path="src/other.py", line_start=1)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(path="src/foo.py"),))

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=_RecordingPublishEventSink(),
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    sitter_mock.assert_not_called()


# ---------------------------------------------------------------------------
# (4) Parametrized: each CoordinateErrorKind → correct routing
# ---------------------------------------------------------------------------


_KIND_TO_ROUTING: dict[CoordinateErrorKind, tuple[PublishDestination, PublishRoutingReason]] = {
    CoordinateErrorKind.UNCHANGED_REGION: (
        PublishDestination.REVIEW_BODY,
        PublishRoutingReason.UNCHANGED_REGION,
    ),
    CoordinateErrorKind.FILE_NOT_IN_PATCH: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.NON_DIFFED_FILE,
    ),
    CoordinateErrorKind.BYTE_OFFSET_INVALID: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.MALFORMED_PATCH: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.DUPLICATE_FILE_ENTRY: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.INVALID_DIFF_LINE: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.PATH_VALIDATION_FAILED: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
    CoordinateErrorKind.HEAD_CONTENT_UNAVAILABLE: (
        PublishDestination.DASHBOARD_ONLY,
        PublishRoutingReason.COORDINATE_ERROR,
    ),
}


@pytest.mark.parametrize(
    ("kind", "expected_destination", "expected_reason"),
    [(k, d, r) for k, (d, r) in _KIND_TO_ROUTING.items()],
    ids=[k.name for k in _KIND_TO_ROUTING],
)
@pytest.mark.asyncio
async def test_each_coordinate_error_kind_routes_correctly(
    kind: CoordinateErrorKind,
    expected_destination: PublishDestination,
    expected_reason: PublishRoutingReason,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per spec §V line 208: each CoordinateErrorKind routes to the right
    (destination, reason, kind) triple. The publisher branches on TYPED
    `kind`, NOT `str(exc)` — verified by stuffing the message with
    cross-class wording.
    """

    def _raise(**_kwargs: object) -> object:
        raise CoordinateError(
            "unchanged_region path_validation_failed argument_validation_failed",
            kind=kind,
        )

    monkeypatch.setattr(publish_module, "source_line_to_github", _raise)
    finding = _make_finding(line_start=2, line_end=2)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert sink.routing[0].destination is expected_destination
    assert sink.routing[0].reason is expected_reason
    assert sink.routing[0].coordinate_error_kind == kind.value


# ---------------------------------------------------------------------------
# (5) Always-route assertion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_always_route_one_event_per_finding() -> None:
    """3 findings with varied outcomes → exactly 3 routing events."""
    findings = (
        _make_finding(severity=FindingSeverity.MEDIUM, line_start=1, line_end=1),
        _make_finding(severity=FindingSeverity.CRITICAL, line_start=2, line_end=2),
        _make_finding(file_path="src/other.py", line_start=1, line_end=1),
    )
    state = _make_state(findings=findings, changed_files=(_make_changed_file(path="src/foo.py"),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(sink.routing) == 3


# ---------------------------------------------------------------------------
# (6) publish_destination pre-set overwrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_destination_preset_is_overwritten_in_routing_event() -> None:
    """Pre-set `finding.publish_destination` is ignored — routing decisions
    are sourced from coordinates, not from the finding attribute.

    Post-synthesize: publish clones each finding via `model_copy()` at
    `_collect_admitted_findings` before any downstream handling (so the
    original `state.review_report.findings[i]` is NEVER mutated). The
    canonical contract is therefore: the emitted `PublishRoutingEvent`
    carries the coordinates-derived destination, regardless of any
    stale pre-set on the original finding. The original finding's
    `publish_destination` attribute is no longer modified by publish.
    """
    finding = _make_finding(line_start=2, line_end=2)
    finding.publish_destination = PublishDestination.DASHBOARD_ONLY  # stale pre-set
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    # Routing event carries the routing-derived destination.
    assert sink.routing[0].destination is PublishDestination.INLINE_COMMENT
    # Original finding's pre-set is UNCHANGED — publish clones before
    # mutating, so the state-layer object stays as constructed.
    assert finding.publish_destination is PublishDestination.DASHBOARD_ONLY


# ---------------------------------------------------------------------------
# (7) Information-leak defense
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_coordinate_error_message_never_leaks_into_routing_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`PublishRoutingEvent.coordinate_error_kind` carries ONLY the enum
    `.value` string; `CoordinateError.message` MUST NOT leak into the
    serialized audit row.
    """
    attacker_text = (  # noqa: S105  (test fixture text, not an actual secret)
        "validate_diff_path rejected: backslash trojan_source windows_drive "
        "git_internal shell_metacharacters"
    )

    def _raise(**_kwargs: object) -> object:
        raise CoordinateError(attacker_text, kind=CoordinateErrorKind.PATH_VALIDATION_FAILED)

    monkeypatch.setattr(publish_module, "source_line_to_github", _raise)
    finding = _make_finding(line_start=2, line_end=2)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    routing = sink.routing[0]
    assert routing.coordinate_error_kind == "path_validation_failed"
    serialized = routing.model_dump_json()
    for rule_name in (
        "backslash",
        "trojan_source",
        "windows_drive",
        "git_internal",
        "shell_metacharacters",
    ):
        assert rule_name not in serialized, (
            f"info-leak: '{rule_name}' (from CoordinateError.message) appears "
            f"in serialized PublishRoutingEvent payload"
        )


# ---------------------------------------------------------------------------
# (8) Bonus: routing fires for WITHHELD findings (decoupling pin)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_critical_finding_routes_inline_then_eligibility_withholds() -> None:
    """Per spec §V "Routing always fires regardless of eligibility":
    CRITICAL finding's routing records what coordinates WOULD have done
    (INLINE_COMMENT) even when eligibility withholds the actual publish.
    Pins routing-vs-eligibility decoupling per DECISIONS #023.
    """
    finding = _make_finding(severity=FindingSeverity.CRITICAL, line_start=2, line_end=2)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert sink.routing[0].destination is PublishDestination.INLINE_COMMENT
    assert sink.eligibility[0].eligibility is PublishEligibility.WITHHELD


# ---------------------------------------------------------------------------
# Fail-loud: publish must reject states without review_report
# ---------------------------------------------------------------------------


def test_collect_admitted_findings_raises_when_review_report_is_none() -> None:
    """Production fail-loud: publish must not silently fall back to
    flattening analysis_rounds when synthesize hasn't populated
    review_report. A miswired graph that reaches publish with
    `state.review_report=None` would otherwise bypass synthesize's
    content_hash dedup + cross-round severity-divergence detection
    contracts.
    """
    from outrider.agent.nodes.publish import _collect_admitted_findings

    state = _make_state(
        findings=(),
        changed_files=(_make_changed_file(),),
    )
    # The `_make_state` helper constructs a synthesize-canonical state
    # with `review_report` populated; null it to exercise the
    # fail-loud branch. `ReviewState` is not frozen so plain
    # attribute assignment works.
    state.review_report = None

    with pytest.raises(RuntimeError, match="synthesize node must have run"):
        _collect_admitted_findings(state)


# ---------------------------------------------------------------------------
# Eligibility event stamps the finding's policy_version snapshot, not live
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_eligibility_stamps_finding_policy_version_not_active() -> None:
    """`PublishEligibilityEvent.policy_version` MUST mirror
    `finding.policy_version` (the snapshot under which the finding was
    classified) — NOT the live `ACTIVE_POLICY_VERSION`.

    Scenario: a review classified findings under historical policy
    `0.0.1`. A HITL pause spans a deploy that bumps ACTIVE to `1.0.0`.
    When publish resumes from checkpoint and emits the eligibility event,
    the row MUST record `0.0.1` (the snapshot the finding's severity was
    computed under) — stamping live `1.0.0` would break
    `severity-policy-versioned-for-replay`.

    Pins Codex 2026-05-28 finding: eligibility was using process-current
    `active_policy_version` parameter, which can drift from the
    finding's snapshot across HITL pauses + deploy bumps.
    """
    # Construct a finding whose policy_version diverges from live ACTIVE
    # (simulating a finding produced under a prior deploy).
    historical_version = "0.0.1"
    assert historical_version != ACTIVE_POLICY_VERSION
    finding = _make_finding(line_start=2, line_end=2, policy_version=historical_version)
    state = _make_state(findings=(finding,), changed_files=(_make_changed_file(),))
    sink = _RecordingPublishEventSink()

    await publish_module.publish(
        state,
        publisher=_StubPublisher(),
        publish_event_sink=sink,
        phase_event_sink=_RecordingPhaseEventSink(),
        review_status_sink=_StubReviewStatusSink(),
        github_factory=_stub_github_factory,
    )

    assert len(sink.eligibility) == 1
    # The audit row carries the finding's snapshot — NOT the live ACTIVE.
    assert sink.eligibility[0].policy_version == historical_version
    assert sink.eligibility[0].policy_version != ACTIVE_POLICY_VERSION
