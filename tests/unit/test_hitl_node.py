"""Unit tests for the HITL node body.

Covers the spec's 13-step contract: partition logic, pass-through path
(empty gated set), gated-path interrupt firing, deterministic body
output under re-entrancy, and the audit-first emit ordering.

Resume-path tests live in integration (they require LangGraph
checkpoint replay); this file exercises the pure-body shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes.hitl import (
    _partition_findings,
    _validate_resume_against_request,
    hitl,
)
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.audit.events import (  # noqa: TC001  (used in test-double type annotations + recorder list element types)
    HITLDecisionEvent,
    HITLRequestEvent,
)
from outrider.notify.orchestrator import SlackNotifyTarget
from outrider.policy import FindingSeverity
from outrider.policy.canonical import compute_hitl_decision_content_hash
from outrider.policy.publish_eligibility import is_hitl_gated_severity
from outrider.schemas.hitl import (
    HITL_MAX_GATED_FINDINGS,
    HITLDecision,
    HITLRequest,
    PerFindingDecision,
    PerFindingOutcome,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingPhaseSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit_phase(self, event: Any) -> None:
        self.events.append(event)


class _RecordingHITLSink:
    def __init__(self) -> None:
        self.requests: list[HITLRequestEvent] = []
        self.decisions: list[HITLDecisionEvent] = []

    async def emit_hitl_request(self, event: HITLRequestEvent) -> HITLRequestEvent:
        self.requests.append(event)
        return event

    async def emit_hitl_decision(self, event: HITLDecisionEvent) -> HITLDecisionEvent:
        self.decisions.append(event)
        return event


class _RecordingStatusSink:
    def __init__(self) -> None:
        self.awaiting: list[dict[str, Any]] = []
        self.running: list[dict[str, Any]] = []
        self.expired: list[dict[str, Any]] = []
        self.completed: list[dict[str, Any]] = []

    async def mark_awaiting_approval(self, **kwargs: Any) -> None:
        self.awaiting.append(kwargs)

    async def mark_running(self, **kwargs: Any) -> None:
        self.running.append(kwargs)

    async def mark_awaiting_approval_expired(self, **kwargs: Any) -> None:
        self.expired.append(kwargs)

    async def mark_completed(self, **kwargs: Any) -> None:
        self.completed.append(kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FindingStub:
    """Duck-typed finding stub.

    `_partition_findings` only reads `finding.finding_id` and
    `finding.severity`. Constructing a full ReviewFinding here would
    couple this test to the entire finding schema (content_hash,
    proposal_hash, evidence tier, etc.) without exercising any of it.
    """

    def __init__(self, *, finding_id: UUID, severity: FindingSeverity) -> None:
        self.finding_id = finding_id
        self.severity = severity


def _make_finding(*, review_id: UUID, severity: FindingSeverity) -> _FindingStub:
    del review_id  # signature parity with the realistic factory; unused in stub
    return _FindingStub(finding_id=uuid4(), severity=severity)


def _make_state(
    *,
    findings: list[_FindingStub],
    review_id: UUID,
    received_at: datetime,
    use_review_report: bool = True,
    analysis_round_findings: list[_FindingStub] | None = None,
) -> Any:
    """Build a minimal ReviewState-like object the hitl body reads.

    The node body reads `state.review_id`, `state.is_eval`,
    `state.received_at`, and `state.review_report.findings`. Use a
    duck-typed stub rather than constructing a full ReviewState (which
    has many required fields and would couple this test to the
    review-state schema details unnecessarily).

    `findings` populates `review_report.findings` (the canonical
    post-synthesize consumer path).

    `analysis_round_findings` populates `analysis_rounds[0].findings`
    INDEPENDENTLY. Defaults to `findings` (mirror) for backward
    compatibility with existing tests; callers that want to pin
    `_partition_findings` actually reads `review_report.findings` and
    NOT `analysis_rounds` pass `analysis_round_findings=[]` or a
    distinct list so a regression to the analysis_rounds branch would
    surface as an assertion failure. Sibling pattern to the
    `use_review_report=False` flag (which exists for the missing-
    `review_report` fail-loud test).

    `use_review_report=True` (default) builds a stub with
    `review_report.findings` set (canonical post-synthesize path).
    `use_review_report=False` is reserved for the fail-loud test —
    the production helper raises RuntimeError when `review_report` is
    None to prevent miswired graphs from bypassing synthesize's
    content-hash dedup + cross-round severity-divergence detection.
    """

    class _Round:
        def __init__(self, findings_: tuple[_FindingStub, ...]) -> None:
            self.findings = findings_

    class _ReviewReport:
        def __init__(self, findings_: tuple[_FindingStub, ...]) -> None:
            self.findings = findings_

    class _PRCtx:
        owner = "acme"
        repo = "api"
        pr_number = 7
        pr_title = "Add login"
        installation_id = 12345  # read by the Slack resolver hook

    rounds_findings: tuple[_FindingStub, ...] = (
        tuple(analysis_round_findings) if analysis_round_findings is not None else tuple(findings)
    )

    class _State:
        def __init__(self) -> None:
            self.review_id = review_id
            self.is_eval = False
            self.received_at = received_at
            self.analysis_rounds = (_Round(rounds_findings),)
            self.review_report = _ReviewReport(tuple(findings)) if use_review_report else None
            self.pr_context = _PRCtx()

    return _State()


# ---------------------------------------------------------------------------
# Partition tests
# ---------------------------------------------------------------------------


def test_partition_separates_high_severity_from_low() -> None:
    review_id = uuid4()
    crit = _make_finding(review_id=review_id, severity=FindingSeverity.CRITICAL)
    high = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    med = _make_finding(review_id=review_id, severity=FindingSeverity.MEDIUM)
    low = _make_finding(review_id=review_id, severity=FindingSeverity.LOW)
    info = _make_finding(review_id=review_id, severity=FindingSeverity.INFO)

    state = _make_state(
        findings=[crit, high, med, low, info],
        review_id=review_id,
        received_at=datetime.now(UTC),
    )
    gated, autopost = _partition_findings(state)

    assert set(gated) == {crit.finding_id, high.finding_id}
    assert set(autopost) == {med.finding_id, low.finding_id, info.finding_id}


def test_partition_raises_when_review_report_is_none() -> None:
    """Production fail-loud: HITL must not silently fall back to
    analysis_rounds when synthesize hasn't populated review_report.
    A miswired graph that reaches HITL with `state.review_report=None`
    would otherwise bypass synthesize's content_hash dedup +
    cross-round severity-divergence detection contracts.
    """
    review_id = uuid4()
    finding = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    state = _make_state(
        findings=[finding],
        review_id=review_id,
        received_at=datetime.now(UTC),
        use_review_report=False,  # bypass canonical path
    )
    # Sanity: the fixture genuinely produced a None review_report.
    assert state.review_report is None

    with pytest.raises(RuntimeError, match="synthesize node must have run"):
        _partition_findings(state)


def test_partition_canonical_review_report_path() -> None:
    """Canonical path coverage: when `state.review_report.findings` is set
    (post-synthesize), `_partition_findings` reads it (NOT
    `state.analysis_rounds`).

    Addresses the audit gap that all other partition tests exercise the
    fallback branch only — leaving the production-canonical
    `state.review_report.findings` consumer untested. Mirrors the
    CRITICAL/HIGH gating + auto-post separation from
    `test_partition_separates_high_severity_from_low` but on the new
    canonical input shape.
    """
    review_id = uuid4()
    crit = _make_finding(review_id=review_id, severity=FindingSeverity.CRITICAL)
    high = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    med = _make_finding(review_id=review_id, severity=FindingSeverity.MEDIUM)
    low = _make_finding(review_id=review_id, severity=FindingSeverity.LOW)
    info = _make_finding(review_id=review_id, severity=FindingSeverity.INFO)

    state = _make_state(
        findings=[crit, high, med, low, info],
        review_id=review_id,
        received_at=datetime.now(UTC),
        use_review_report=True,  # canonical post-synthesize path
        # Empty analysis_rounds so a regression to the old fallback
        # (reading state.analysis_rounds[*].findings) yields ZERO
        # findings and the assertions below fail loudly. Per CodeRabbit
        # 2026-05-28 catch: mirror-default _make_state admitted a
        # silent regression to the analysis_rounds branch.
        analysis_round_findings=[],
    )
    # Verify the test setup actually populates review_report (not
    # accidentally falling through to the analysis_rounds branch).
    assert state.review_report is not None
    assert len(state.review_report.findings) == 5
    assert len(state.analysis_rounds[0].findings) == 0  # regression pin

    gated, autopost = _partition_findings(state)

    assert set(gated) == {crit.finding_id, high.finding_id}
    assert set(autopost) == {med.finding_id, low.finding_id, info.finding_id}


def test_partition_consumes_synthesize_deduplicated_findings() -> None:
    """Post-synthesize: HITL consumes the already-deduplicated tuple from
    `state.review_report.findings`. Multi-round dedup semantics moved
    to synthesize (where the content_hash dedup + cross-round severity-
    divergence detection live); HITL no longer walks raw
    analysis_rounds. The legacy
    `test_partition_dedupes_finding_admitted_across_passes` semantics
    are now covered at the synthesize layer; this test verifies HITL's
    contract on the new shape: each finding in `review_report.findings`
    is classified exactly once.
    """
    review_id = uuid4()
    high1 = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    high2 = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    med = _make_finding(review_id=review_id, severity=FindingSeverity.MEDIUM)

    # Canonical shape: review_report.findings already deduplicated by
    # synthesize. HITL classifies each entry exactly once.
    state = _make_state(
        findings=[high1, high2, med],
        review_id=review_id,
        received_at=datetime.now(UTC),
        # Empty analysis_rounds — same regression pin as the sibling
        # canonical-path test above. A future regression to
        # `state.analysis_rounds[*].findings` would yield zero entries
        # and the assertions below would fail loudly instead of
        # silently passing on the mirror.
        analysis_round_findings=[],
    )
    gated, autopost = _partition_findings(state)

    assert len(gated) == 2
    assert set(gated) == {high1.finding_id, high2.finding_id}
    assert len(autopost) == 1
    assert autopost[0] == med.finding_id
    # Negative pin: state.analysis_rounds was empty, so a regression
    # to the fallback branch would have returned (), () — the gated
    # assertion above already catches it; this is documentation.
    assert len(state.analysis_rounds[0].findings) == 0


def test_partition_returns_sorted_tuples_for_determinism() -> None:
    review_id = uuid4()
    findings = [_make_finding(review_id=review_id, severity=FindingSeverity.HIGH) for _ in range(5)]

    state = _make_state(
        findings=findings,
        review_id=review_id,
        received_at=datetime.now(UTC),
    )
    gated_1, autopost_1 = _partition_findings(state)
    gated_2, autopost_2 = _partition_findings(state)

    # Re-entrancy invariant: same state -> same partition.
    assert gated_1 == gated_2
    assert autopost_1 == autopost_2
    # Sortedness: every adjacent pair is in order.
    assert list(gated_1) == sorted(gated_1)


def test_partition_accepts_gated_findings_at_cap() -> None:
    """Exactly HITL_MAX_GATED_FINDINGS gated findings partition without error."""
    review_id = uuid4()
    findings = [
        _make_finding(review_id=review_id, severity=FindingSeverity.CRITICAL)
        for _ in range(HITL_MAX_GATED_FINDINGS)
    ]
    state = _make_state(
        findings=findings,
        review_id=review_id,
        received_at=datetime.now(UTC),
    )

    gated, autopost = _partition_findings(state)

    assert len(gated) == HITL_MAX_GATED_FINDINGS
    assert autopost == ()


def test_partition_raises_over_gated_findings_cap() -> None:
    """One finding past the cap fails loud at partition — before HITLRequest
    construction would hit the schema's max_length — with a clearer message."""
    review_id = uuid4()
    findings = [
        _make_finding(review_id=review_id, severity=FindingSeverity.CRITICAL)
        for _ in range(HITL_MAX_GATED_FINDINGS + 1)
    ]
    state = _make_state(
        findings=findings,
        review_id=review_id,
        received_at=datetime.now(UTC),
    )

    with pytest.raises(ValueError, match=r"exceeds the 256-finding cap"):
        _partition_findings(state)


def test_gated_severities_are_exactly_critical_and_high() -> None:
    """Pin the gated-severity set per the policy/publish_eligibility helper.

    Earlier shape pinned a `_GATED_SEVERITIES` literal local to hitl.py;
    after synthesize-node refactor, the gated-set is the responsibility
    of `policy/publish_eligibility.py::is_hitl_gated_severity` (single
    source of truth). Per the helper's docstring: V1 gates CRITICAL +
    HIGH only.
    """
    assert is_hitl_gated_severity(FindingSeverity.CRITICAL) is True
    assert is_hitl_gated_severity(FindingSeverity.HIGH) is True
    assert is_hitl_gated_severity(FindingSeverity.MEDIUM) is False
    assert is_hitl_gated_severity(FindingSeverity.LOW) is False
    assert is_hitl_gated_severity(FindingSeverity.INFO) is False


# ---------------------------------------------------------------------------
# Body pass-through (no gated findings)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_body_returns_empty_delta_when_no_gated_findings() -> None:
    review_id = uuid4()
    state = _make_state(
        findings=[_make_finding(review_id=review_id, severity=FindingSeverity.LOW)],
        review_id=review_id,
        received_at=datetime.now(UTC),
    )
    phase_sink = _RecordingPhaseSink()
    hitl_sink = _RecordingHITLSink()
    status_sink = _RecordingStatusSink()

    delta = await hitl(
        state,  # type: ignore[arg-type]
        phase_event_sink=phase_sink,  # type: ignore[arg-type]
        hitl_event_sink=hitl_sink,  # type: ignore[arg-type]
        review_status_sink=status_sink,  # type: ignore[arg-type]
        hitl_config=HITLConfig(),
    )

    assert delta == {}
    assert len(phase_sink.events) == 2  # start + end
    # No HITL events emitted, no status flip on pass-through.
    assert hitl_sink.requests == []
    assert hitl_sink.decisions == []
    assert status_sink.awaiting == []
    assert status_sink.running == []


# ---------------------------------------------------------------------------
# Defense-in-depth validate
# ---------------------------------------------------------------------------


def test_validate_resume_raises_on_missing_finding() -> None:
    f1, f2 = uuid4(), uuid4()
    request = HITLRequest(
        findings_requiring_approval=(f1, f2),
        auto_post_findings=(),
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(finding_id=f1, outcome=PerFindingOutcome.APPROVE, reason="ok"),
        ),
        decided_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="diverges"):
        _validate_resume_against_request(request=request, decision=decision)


def test_validate_resume_raises_on_extra_finding() -> None:
    f1, f2 = uuid4(), uuid4()
    request = HITLRequest(
        findings_requiring_approval=(f1,),
        auto_post_findings=(),
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(finding_id=f1, outcome=PerFindingOutcome.APPROVE, reason="ok"),
            PerFindingDecision(finding_id=f2, outcome=PerFindingOutcome.APPROVE, reason="extra"),
        ),
        decided_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="diverges"):
        _validate_resume_against_request(request=request, decision=decision)


def test_validate_resume_accepts_matching_set() -> None:
    f1 = uuid4()
    request = HITLRequest(
        findings_requiring_approval=(f1,),
        auto_post_findings=(),
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    decision = HITLDecision(
        reviewer_id="admin",
        decisions=(
            PerFindingDecision(finding_id=f1, outcome=PerFindingOutcome.APPROVE, reason="ok"),
        ),
        decided_at=datetime.now(UTC),
    )
    # No raise = pass.
    _validate_resume_against_request(request=request, decision=decision)


# ---------------------------------------------------------------------------
# Deterministic expires_at + phase_id (re-entrancy invariant)
# ---------------------------------------------------------------------------


def test_compute_hitl_decision_content_hash_is_order_invariant() -> None:
    """Two reviewers submitting same per-finding decisions in different
    orders produce the same hash."""
    f1, f2 = uuid4(), uuid4()
    d1 = PerFindingDecision(finding_id=f1, outcome=PerFindingOutcome.APPROVE, reason="ok")
    d2 = PerFindingDecision(finding_id=f2, outcome=PerFindingOutcome.REJECT, reason="no")

    h1 = compute_hitl_decision_content_hash(decisions=(d1, d2), annotation="note")
    h2 = compute_hitl_decision_content_hash(decisions=(d2, d1), annotation="note")
    assert h1 == h2


def test_compute_hitl_decision_content_hash_rejects_non_basemodel() -> None:
    """A dict-shaped caller should fail loud at the helper, not later
    at HITLDecisionEvent construction."""
    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        compute_hitl_decision_content_hash(
            decisions=({"finding_id": str(uuid4()), "outcome": "approve"},),  # type: ignore[arg-type]
            annotation=None,
        )


# ---------------------------------------------------------------------------
# Slack HITL-pending notification hook
# ---------------------------------------------------------------------------


class _FakeSlackOrchestrator:
    """Records notify_hitl_pending calls; the hitl node calls only this method."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def notify_hitl_pending(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _resolver_for(orch: _FakeSlackOrchestrator, channel_id: str = "C0123ABC") -> Any:
    """A per-install resolver that always resolves to `orch` on `channel_id` — the
    test seam replacing the retired slack_orchestrator/slack_channel_id params."""

    async def _resolve(_installation_id: int) -> SlackNotifyTarget:
        return SlackNotifyTarget(channel_id=channel_id, orchestrator=orch)  # type: ignore[arg-type]

    return _resolve


@pytest.mark.asyncio
async def test_hitl_notifies_slack_on_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gated path: the node posts a HITL-pending notification (before interrupt)
    carrying ONLY the gated findings + the PR identity. interrupt is patched to
    halt the body right after the notify so the call is observable in isolation."""
    review_id = uuid4()
    crit = _make_finding(review_id=review_id, severity=FindingSeverity.CRITICAL)
    high = _make_finding(review_id=review_id, severity=FindingSeverity.HIGH)
    low = _make_finding(review_id=review_id, severity=FindingSeverity.LOW)
    state = _make_state(
        findings=[crit, high, low], review_id=review_id, received_at=datetime.now(UTC)
    )
    fake = _FakeSlackOrchestrator()

    class _StopInterruptError(Exception):
        pass

    def _raise(*_a: Any, **_k: Any) -> None:
        raise _StopInterruptError

    monkeypatch.setattr("outrider.agent.nodes.hitl.interrupt", _raise)

    with pytest.raises(_StopInterruptError):
        await hitl(
            state,  # type: ignore[arg-type]
            phase_event_sink=_RecordingPhaseSink(),  # type: ignore[arg-type]
            hitl_event_sink=_RecordingHITLSink(),  # type: ignore[arg-type]
            review_status_sink=_RecordingStatusSink(),  # type: ignore[arg-type]
            hitl_config=HITLConfig(),
            resolve_slack_target=_resolver_for(fake),
        )

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["channel_id"] == "C0123ABC"
    assert call["review_id"] == review_id
    assert call["repo"] == "acme/api"
    assert call["pr_number"] == 7
    assert call["pr_title"] == "Add login"
    assert call["is_eval"] is False  # threaded from state.is_eval (eval isolation)
    # Only the gated (CRITICAL/HIGH) findings reach the card, not the LOW one.
    assert {f.finding_id for f in call["findings"]} == {crit.finding_id, high.finding_id}


@pytest.mark.asyncio
async def test_hitl_no_slack_on_passthrough() -> None:
    """No gated findings → pass-through → no Slack notification, even with Slack
    fully configured."""
    review_id = uuid4()
    state = _make_state(
        findings=[_make_finding(review_id=review_id, severity=FindingSeverity.LOW)],
        review_id=review_id,
        received_at=datetime.now(UTC),
    )
    fake = _FakeSlackOrchestrator()
    delta = await hitl(
        state,  # type: ignore[arg-type]
        phase_event_sink=_RecordingPhaseSink(),  # type: ignore[arg-type]
        hitl_event_sink=_RecordingHITLSink(),  # type: ignore[arg-type]
        review_status_sink=_RecordingStatusSink(),  # type: ignore[arg-type]
        hitl_config=HITLConfig(),
        resolve_slack_target=_resolver_for(fake),
    )
    assert delta == {}
    assert fake.calls == []
