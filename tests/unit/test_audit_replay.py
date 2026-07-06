"""Unit tests for the pure (DB-free) surface of `audit/replay.py`.

Covers phase grouping, mode classification, and the verify-only checklist
helpers over in-memory reconstructed objects. The async DB-backed pieces
(`reconstruct`, `assert_replay_equivalent`, historical-policy severity) are
covered in `tests/integration/test_audit_replay.py`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import pytest

from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEvent,
    FileExaminationEvent,
    FindingEvent,
    HITLDecisionEvent,
    LLMCallEvent,
    ReplayVerdictEvent,
    ReviewPhaseEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
)
from outrider.audit.replay import (
    AuditReplayer,
    FindingContent,
    ReconstructedFinding,
    ReconstructedLLMExchange,
    ReconstructedReview,
    ReconstructedReviewMetadata,
    ReplayEquivalenceError,
    ReplayError,
    ReplayMode,
    _classify_mode,
    _group_phases,
    _hitl_override_decisions,
    _verify_cross_event_refs,
    _verify_finding_override_projection,
    _verify_full_finding,
    _verify_is_eval_consistent,
    _verify_mode_consistency,
    _verify_phase_wellformed,
    _verify_proof_boundary,
    _verify_row_consistent,
    _verify_sequence_monotonic,
)
from outrider.db.models.findings import Finding
from outrider.db.models.llm_call_content import LLMCallContent
from outrider.db.models.reviews import Review
from outrider.policy.canonical import compute_hitl_decision_content_hash
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension
from outrider.schemas.hitl import PerFindingDecision, PerFindingOutcome

_REVIEW_ID = UUID("11111111-1111-1111-1111-111111111111")


# ---------------------------------------------------------------------------
# Builders (production-semantic; mirror tests/unit/test_audit_event_discriminator)
# ---------------------------------------------------------------------------


def _finding_event(
    *,
    evidence_tier: EvidenceTier = EvidenceTier.JUDGED,
    query_match_id: str | None = None,
    trace_path: tuple[str, ...] | None = None,
    finding_id: UUID | None = None,
    sequence_number: int | None = None,
    file_path: str = "src/app/models.py",
    line_start: int = 10,
    line_end: int = 20,
    phase_key: str | None = None,
) -> FindingEvent:
    finding_type = FindingType.SQL_INJECTION
    return FindingEvent(
        review_id=_REVIEW_ID,
        finding_id=finding_id or uuid4(),
        finding_type=finding_type,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        dimension=ReviewDimension.SECURITY,
        finding_content_hash=compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        evidence_tier=evidence_tier,
        query_match_id=query_match_id,
        trace_path=trace_path,
        policy_version="1.0.0",
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
        sequence_number=sequence_number,
        phase_key=phase_key,
    )


def _phase_event(
    *,
    node_id: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"],
    marker: Literal["start", "end"],
    phase_id: str | None = None,
    sequence_number: int | None = None,
    phase_key: str | None = None,
) -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=_REVIEW_ID,
        phase_id=phase_id or (f"{node_id}:{phase_key}" if phase_key else f"{node_id}:0"),
        node_id=node_id,
        marker=marker,
        phase_key=phase_key,
        sequence_number=sequence_number,
    )


def _llm_call_event(
    *, sequence_number: int | None = None, phase_key: str | None = None
) -> LLMCallEvent:
    return LLMCallEvent(
        review_id=_REVIEW_ID,
        phase_key=phase_key,
        model="claude-sonnet-4-5",
        node_id="analyze",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        cost_usd=0.01,
        pricing_version="v1",
        latency_ms=1200,
        prompt_hash=hashlib.sha256(b"prompt").hexdigest(),
        cache_hit=False,
        context_summary=(),
        prompt_template_version="analyze.v1",
        system_prompt_hash=hashlib.sha256(b"sys").hexdigest(),
        degraded_mode=False,
        sequence_number=sequence_number,
    )


def _transition_event(*, sequence_number: int | None = None) -> AgentTransitionEvent:
    return AgentTransitionEvent(
        review_id=_REVIEW_ID,
        from_node="webhook",
        to_node="intake",
        latency_ms=5,
        sequence_number=sequence_number,
    )


def _content_for(event: FindingEvent) -> FindingContent:
    """A FindingContent that agrees with the event (full-mode happy path)."""
    return FindingContent(
        finding_type=event.finding_type,
        severity=event.severity,
        evidence_tier=event.evidence_tier,
        file_path=event.file_path,
        line_start=event.line_start,
        line_end=event.line_end,
        title="t",
        description="d",
        evidence="e",
        suggested_fix=None,
        query_match_id=event.query_match_id,
        trace_path=event.trace_path,
        original_severity=None,
        override_reason=None,
        overrider_id=None,
        publish_destination=None,
        policy_version=event.policy_version,
        content_hash=event.finding_content_hash,
    )


def _review_metadata() -> ReconstructedReviewMetadata:
    ts = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    return ReconstructedReviewMetadata(
        review_id=_REVIEW_ID,
        installation_id=12345,
        status="completed",
        is_eval=False,
        repo_id=100,
        pr_number=1,
        head_sha="sha1",
        created_at=ts,
        updated_at=ts,
        completed_at=ts,
        expires_at=None,
    )


def _review(
    *,
    mode: ReplayMode,
    review: ReconstructedReviewMetadata | None,
    findings: tuple[ReconstructedFinding, ...] = (),
    llm_exchanges: tuple[ReconstructedLLMExchange, ...] = (),
    events: tuple[AuditEvent, ...] = (),
) -> ReconstructedReview:
    return ReconstructedReview(
        review_id=_REVIEW_ID,
        mode=mode,
        is_eval=False,
        review=review,
        events=events,
        phases=(),
        findings=findings,
        llm_exchanges=llm_exchanges,
    )


def _override_decision(
    finding_id: UUID,
    *,
    original: FindingSeverity,
    override: FindingSeverity,
    reason: str = "downgraded after manual review",
) -> PerFindingDecision:
    return PerFindingDecision(
        finding_id=finding_id,
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason=reason,
        original_severity=original,
        override_severity=override,
    )


def _hitl_decision_event(
    decisions: tuple[PerFindingDecision, ...],
    *,
    sequence_number: int | None = None,
) -> HITLDecisionEvent:
    return HITLDecisionEvent(
        review_id=_REVIEW_ID,
        reviewer_id="admin",
        decisions=decisions,
        annotation=None,
        decided_at=datetime(2026, 5, 1, 12, 30, 0, tzinfo=UTC),
        decision_latency_seconds=12.5,
        decisions_content_hash=compute_hitl_decision_content_hash(
            decisions=decisions, annotation=None
        ),
        sequence_number=sequence_number,
    )


def _forge_override(
    content: FindingContent,
    *,
    original_severity: FindingSeverity | None,
    override_reason: str | None,
) -> FindingContent:
    """Hand-populate the (V1-NULL) override projection columns on a content row.

    Real V1 `findings` rows leave these NULL (no post-HITL writer), so the
    cross-check is vacuous on production data — forging them is the only way to
    exercise the guard (FUP-122).
    """
    return content.model_copy(
        update={"original_severity": original_severity, "override_reason": override_reason}
    )


# ---------------------------------------------------------------------------
# Mode classification
# ---------------------------------------------------------------------------


def test_classify_mode_full_with_content() -> None:
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    finding = finding.model_copy(update={"content": _content_for(finding.event)})
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt="p", completion="c")
    assert (
        _classify_mode(review_present=True, findings=(finding,), llm_exchanges=(exchange,))
        == ReplayMode.FULL
    )


def test_classify_mode_full_when_empty_and_review_present() -> None:
    assert _classify_mode(review_present=True, findings=(), llm_exchanges=()) == ReplayMode.FULL


def test_classify_mode_metadata_only() -> None:
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt=None, completion=None)
    assert (
        _classify_mode(review_present=False, findings=(finding,), llm_exchanges=(exchange,))
        == ReplayMode.METADATA_ONLY
    )


def test_classify_mode_metadata_only_when_empty_and_review_absent() -> None:
    assert (
        _classify_mode(review_present=False, findings=(), llm_exchanges=())
        == ReplayMode.METADATA_ONLY
    )


def test_classify_mode_mixed_when_llm_content_purged() -> None:
    # The legitimate MIXED window under the retention ordering
    # (llm_content <= findings <= review): findings present, the shorter-or-equal
    # LLM content gone.
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    finding = finding.model_copy(update={"content": _content_for(finding.event)})
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt=None, completion=None)
    assert (
        _classify_mode(review_present=True, findings=(finding,), llm_exchanges=(exchange,))
        == ReplayMode.MIXED
    )


def test_classify_mode_review_absent_with_llm_content_raises() -> None:
    # Impossible under the retention ordering (llm_content <= findings <= review):
    # a purged review with a surviving LLM content row is corruption, not a
    # legitimate mixed window.
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt="p", completion="c")
    with pytest.raises(ReplayEquivalenceError, match="surviving content with no review row"):
        _classify_mode(review_present=False, findings=(), llm_exchanges=(exchange,))


def test_classify_mode_review_absent_with_finding_content_raises() -> None:
    # Same impossibility via a surviving finding content row.
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    finding = finding.model_copy(update={"content": _content_for(finding.event)})
    with pytest.raises(ReplayEquivalenceError, match="surviving content with no review row"):
        _classify_mode(review_present=False, findings=(finding,), llm_exchanges=())


def test_classify_mode_llm_present_finding_purged_raises() -> None:
    # Sibling of the review-absent guard: under the retention ordering
    # (llm_content <= findings), LLM content purges no later than finding content,
    # so surviving LLM content with a purged finding is an out-of-order purge —
    # corruption, not a legitimate MIXED window.
    stub_finding = ReconstructedFinding(event=_finding_event(), content=None)  # content purged
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt="p", completion="c")
    with pytest.raises(ReplayEquivalenceError, match="LLM content survives while finding content"):
        _classify_mode(review_present=True, findings=(stub_finding,), llm_exchanges=(exchange,))


def test_classify_mode_llm_present_no_findings_ok() -> None:
    # A review with LLM calls but zero findings is FULL, not corruption:
    # all_finding_content is vacuously true, so Guard 2 does not fire.
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt="p", completion="c")
    assert (
        _classify_mode(review_present=True, findings=(), llm_exchanges=(exchange,))
        == ReplayMode.FULL
    )


def test_classify_mode_half_present_llm_prompt_only_raises() -> None:
    # prompt + completion are NOT NULL + co-inserted, so they purge together;
    # a one-sided row is corruption, not a legitimate retention state.
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt="p", completion=None)
    with pytest.raises(ReplayEquivalenceError, match="half-present"):
        _classify_mode(review_present=True, findings=(), llm_exchanges=(exchange,))


def test_classify_mode_half_present_llm_completion_only_raises() -> None:
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt=None, completion="c")
    with pytest.raises(ReplayEquivalenceError, match="half-present"):
        _classify_mode(review_present=True, findings=(), llm_exchanges=(exchange,))


# ---------------------------------------------------------------------------
# is_eval coherence (stream + content tables)
# ---------------------------------------------------------------------------


def test_verify_is_eval_consistent_all_agree_ok() -> None:
    _verify_is_eval_consistent(
        stream_is_eval=False,
        events=(_finding_event(),),
        review_row=Review(is_eval=False),
        finding_rows=(Finding(is_eval=False),),
        content_rows=(LLMCallContent(is_eval=False),),
    )  # no raise


def test_verify_is_eval_consistent_rejects_review_row_drift() -> None:
    with pytest.raises(ReplayEquivalenceError, match="reviews row is_eval"):
        _verify_is_eval_consistent(
            stream_is_eval=False,
            events=(_finding_event(),),
            review_row=Review(is_eval=True),  # drifts from stream
            finding_rows=(),
            content_rows=(),
        )


def test_verify_is_eval_consistent_rejects_finding_row_drift() -> None:
    with pytest.raises(ReplayEquivalenceError, match="findings row"):
        _verify_is_eval_consistent(
            stream_is_eval=False,
            events=(_finding_event(),),
            review_row=None,
            finding_rows=(Finding(is_eval=True),),  # drifts
            content_rows=(),
        )


def test_verify_is_eval_consistent_rejects_llm_content_row_drift() -> None:
    with pytest.raises(ReplayEquivalenceError, match="llm_call_content row"):
        _verify_is_eval_consistent(
            stream_is_eval=False,
            events=(_finding_event(),),
            review_row=None,
            finding_rows=(),
            content_rows=(LLMCallContent(is_eval=True),),  # drifts
        )


def test_verify_is_eval_consistent_rejects_mixed_events() -> None:
    eval_event = _finding_event()
    eval_event = eval_event.model_copy(update={"is_eval": True})
    with pytest.raises(ReplayEquivalenceError, match="mixed is_eval"):
        _verify_is_eval_consistent(
            stream_is_eval=False,
            events=(_finding_event(), eval_event),  # second event drifts
            review_row=None,
            finding_rows=(),
            content_rows=(),
        )


# ---------------------------------------------------------------------------
# Phase grouping
# ---------------------------------------------------------------------------


def test_group_phases_pairs_start_end_and_nests_events() -> None:
    finding = _finding_event()
    events = (
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        finding,
        _phase_event(node_id="intake", marker="end", phase_id="intake:0"),
        _phase_event(node_id="triage", marker="start", phase_id="triage:0"),
        _phase_event(node_id="triage", marker="end", phase_id="triage:0"),
    )
    phases = _group_phases(events)
    assert [p.phase_id for p in phases] == ["intake:0", "triage:0"]
    assert phases[0].start is not None
    assert phases[0].end is not None
    assert phases[0].events == (finding,)
    assert phases[1].events == ()


def test_group_phases_unterminated_phase_has_no_end() -> None:
    finding = _finding_event()
    events = (
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        finding,
    )
    phases = _group_phases(events)
    assert len(phases) == 1
    assert phases[0].end is None
    assert phases[0].events == (finding,)


# ---------------------------------------------------------------------------
# Sequence monotonicity
# ---------------------------------------------------------------------------


def test_verify_sequence_monotonic_ok() -> None:
    events = (
        _finding_event(sequence_number=1),
        _finding_event(sequence_number=2),
        _finding_event(sequence_number=7),
    )
    _verify_sequence_monotonic(events)  # no raise


def test_verify_sequence_monotonic_rejects_non_ascending() -> None:
    events = (_finding_event(sequence_number=2), _finding_event(sequence_number=2))
    with pytest.raises(ReplayEquivalenceError, match="not strictly ascending"):
        _verify_sequence_monotonic(events)


def test_verify_sequence_monotonic_rejects_missing() -> None:
    events = (_finding_event(sequence_number=None),)
    with pytest.raises(ReplayEquivalenceError, match="missing sequence_number"):
        _verify_sequence_monotonic(events)


# ---------------------------------------------------------------------------
# Phase well-formedness
# ---------------------------------------------------------------------------


def test_verify_phase_wellformed_ok() -> None:
    events = (
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        _phase_event(node_id="intake", marker="end", phase_id="intake:0"),
    )
    _verify_phase_wellformed(events)  # no raise


def test_verify_phase_wellformed_tolerates_unterminated() -> None:
    events = (_phase_event(node_id="intake", marker="start", phase_id="intake:0"),)
    _verify_phase_wellformed(events)  # no raise — a crash mid-phase is a real state


def test_verify_phase_wellformed_rejects_duplicate_start() -> None:
    events = (
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="more than one start marker"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_rejects_overlapping_phases() -> None:
    # V1 phases are sequential/non-nested: a second (different) phase that starts
    # while another is still open is rejected. Distinct from duplicate-start,
    # which is the same phase_id reused ("more than one start marker").
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _phase_event(node_id="trace", marker="start", phase_id="trace:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="still open"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_rejects_end_without_start() -> None:
    events = (_phase_event(node_id="intake", marker="end", phase_id="intake:0"),)
    with pytest.raises(ReplayEquivalenceError, match="end marker with no preceding start"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_allows_work_within_phase() -> None:
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _llm_call_event(),
        _finding_event(),
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
    )
    _verify_phase_wellformed(events)  # no raise — work is bounded


def test_verify_phase_wellformed_rejects_work_outside_phase() -> None:
    # phase-events-bound-work: a finding with no enclosing phase is unbounded.
    events = (_finding_event(),)
    with pytest.raises(ReplayEquivalenceError, match="outside any open review phase"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_allows_transition_outside_phase() -> None:
    # AgentTransitionEvent legitimately occurs before/between phases.
    events = (
        _transition_event(),
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        _phase_event(node_id="intake", marker="end", phase_id="intake:0"),
    )
    _verify_phase_wellformed(events)  # no raise


def test_verify_phase_wellformed_rejects_node_mismatched_work() -> None:
    # Node containment: an analyze LLM call inside a triage phase is not
    # graph-faithful — the work's node_id must match an open phase's node.
    events = (
        _phase_event(node_id="triage", marker="start", phase_id="triage:0"),
        _llm_call_event(),  # node_id="analyze" (unit builder default)
        _phase_event(node_id="triage", marker="end", phase_id="triage:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="no open phase matches that node"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_allows_node_matched_work() -> None:
    # The same LLM call inside its own analyze phase is fine.
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _llm_call_event(),  # node_id="analyze"
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
    )
    _verify_phase_wellformed(events)  # no raise


def test_verify_phase_wellformed_intake_file_examination_in_intake_phase() -> None:
    # FileExaminationEvent carries node_id="intake"; the intake node DOES emit
    # intake phase markers (intake.py), so an intake file-examination inside an
    # intake phase is graph-faithful and accepted. Guards the false-positive
    # where node-containment would wrongly reject the most common real stream
    # (every production review intake-fetches files).
    fe = FileExaminationEvent(
        review_id=_REVIEW_ID,
        file_path="src/app/models.py",
        examination_type="intake_fetch",
        node_id="intake",
        parse_status="clean",
    )
    events = (
        _phase_event(node_id="intake", marker="start", phase_id="intake:0"),
        fe,
        _phase_event(node_id="intake", marker="end", phase_id="intake:0"),
    )
    _verify_phase_wellformed(events)  # no raise


def test_verify_phase_wellformed_rejects_file_examination_in_wrong_phase() -> None:
    # The same intake file-examination inside a triage phase is not graph-faithful.
    fe = FileExaminationEvent(
        review_id=_REVIEW_ID,
        file_path="src/app/models.py",
        examination_type="intake_fetch",
        node_id="intake",
        parse_status="clean",
    )
    events = (
        _phase_event(node_id="triage", marker="start", phase_id="triage:0"),
        fe,
        _phase_event(node_id="triage", marker="end", phase_id="triage:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="owned by node 'intake'"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_node_less_owned_work_matches_owner() -> None:
    # FindingEvent carries no node_id but is analyze-owned: in an analyze phase
    # it is accepted.
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _finding_event(),  # node-less, owned by analyze
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
    )
    _verify_phase_wellformed(events)  # no raise


def test_verify_phase_wellformed_rejects_node_less_owned_work_in_wrong_phase() -> None:
    # An analyze-owned FindingEvent inside a triage phase is not graph-faithful:
    # production emits admitted findings from the analyze node (analyze.py).
    events = (
        _phase_event(node_id="triage", marker="start", phase_id="triage:0"),
        _finding_event(),  # node-less, owned by analyze
        _phase_event(node_id="triage", marker="end", phase_id="triage:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="owned by node 'analyze'"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_accepts_replay_verdict_after_phases_close() -> None:
    # A ReplayVerdictEvent is appended post-completion, OUTSIDE any open phase. It
    # is phase-unbounded replay metadata (like AgentTransitionEvent), so it must
    # NOT raise "occurs outside any open review phase". This is the load-bearing
    # guard: without the runtime _PHASE_UNBOUNDED_EVENTS broadening, appending a
    # verdict would break every later replay of the review.
    verdict = ReplayVerdictEvent(
        review_id=_REVIEW_ID,
        replay_equivalent=True,
        mode="full",
        event_count=3,
        finding_count=1,
        orphan_finding_count=0,
        target_max_sequence_number=3,
    )
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _finding_event(),
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
        verdict,  # appended after all phases closed
    )
    _verify_phase_wellformed(events)  # no raise


def test_node_less_events_have_owner_or_exemption() -> None:
    # Completeness guard: every concrete AuditEvent subtype must either carry
    # its own `node_id`, be in the node-less owner map, or be explicitly
    # phase-unbounded — so a future node-less event type cannot silently skip
    # the node-containment check (the loophole this round closes).
    import typing

    from outrider.audit.events import AuditEvent as _AuditEventUnion
    from outrider.audit.replay import _NODE_LESS_EVENT_OWNER, _PHASE_UNBOUNDED_EVENTS

    members = typing.get_args(typing.get_args(_AuditEventUnion)[0])
    assert members, "AuditEvent union should have members"
    for member in members:
        has_node_id = "node_id" in member.model_fields
        classified = (
            has_node_id or member in _NODE_LESS_EVENT_OWNER or member in _PHASE_UNBOUNDED_EVENTS
        )
        assert classified, (
            f"{member.__name__} carries no node_id and is neither in "
            f"_NODE_LESS_EVENT_OWNER nor _PHASE_UNBOUNDED_EVENTS; it would silently "
            f"skip node-containment. Add it to the owner map (or the exempt set "
            f"with a reason)."
        )


def test_verify_phase_wellformed_rejects_end_before_start() -> None:
    # In sequence order, the end precedes the start → no preceding start.
    events = (
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="end marker with no preceding start"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_rejects_node_id_mismatch() -> None:
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _phase_event(node_id="triage", marker="end", phase_id="analyze:0"),
    )
    with pytest.raises(ReplayEquivalenceError, match="node_id"):
        _verify_phase_wellformed(events)


def test_verify_phase_wellformed_require_terminated_rejects_unterminated() -> None:
    # A completed review (require_all_terminated=True) must close every phase.
    events = (_phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),)
    with pytest.raises(ReplayEquivalenceError, match="requires a phase end event on success"):
        _verify_phase_wellformed(events, require_all_terminated=True)


def test_verify_phase_wellformed_require_terminated_allows_balanced() -> None:
    events = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:0"),
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:0"),
    )
    _verify_phase_wellformed(events, require_all_terminated=True)  # no raise


# ---------------------------------------------------------------------------
# Row-vs-payload base-field consistency
# ---------------------------------------------------------------------------


def _row_kwargs(event: object) -> dict[str, object]:
    """Build matching row-column kwargs from an event (the happy path)."""
    return {
        "event_id": event.event_id,
        "review_id": event.review_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "is_eval": event.is_eval,
        "phase_key": event.phase_key if isinstance(event, ReviewPhaseEvent) else None,
    }


def test_verify_row_consistent_matching_ok() -> None:
    event = _finding_event()
    _verify_row_consistent(event, **_row_kwargs(event))  # type: ignore[arg-type]  # no raise


def test_verify_row_consistent_rejects_is_eval_drift() -> None:
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["is_eval"] = not event.is_eval
    with pytest.raises(ReplayEquivalenceError, match="is_eval"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


def test_verify_row_consistent_rejects_event_type_drift() -> None:
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["event_type"] = "llm_call"
    with pytest.raises(ReplayEquivalenceError, match="event_type"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


def test_verify_row_consistent_rejects_phase_key_on_non_phase_event() -> None:
    # phase_key column must be NULL for a non-ReviewPhaseEvent row.
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["phase_key"] = "analyze:0"
    with pytest.raises(ReplayEquivalenceError, match="phase_key"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


def test_verify_row_consistent_rejects_event_id_drift() -> None:
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["event_id"] = uuid4()
    with pytest.raises(ReplayEquivalenceError, match="event_id"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


def test_verify_row_consistent_rejects_review_id_drift() -> None:
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["review_id"] = uuid4()
    with pytest.raises(ReplayEquivalenceError, match="review_id"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


def test_verify_row_consistent_rejects_timestamp_drift() -> None:
    event = _finding_event()
    kwargs = _row_kwargs(event)
    kwargs["timestamp"] = datetime(2000, 1, 1, tzinfo=UTC)
    with pytest.raises(ReplayEquivalenceError, match="timestamp"):
        _verify_row_consistent(event, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Proof boundary re-verification (verify-only)
# ---------------------------------------------------------------------------


def test_verify_proof_boundary_observed_real_registry_id_ok() -> None:
    events = (
        _finding_event(
            evidence_tier=EvidenceTier.OBSERVED, query_match_id="python.function_definition"
        ),
    )
    _verify_proof_boundary(events)  # no raise


def test_verify_proof_boundary_observed_fabricated_id_raises() -> None:
    # The schema-layer proof boundary admits any non-empty query_match_id;
    # replay's registry check is the stricter, verify-only gate.
    events = (
        _finding_event(evidence_tier=EvidenceTier.OBSERVED, query_match_id="python.nonexistent"),
    )
    with pytest.raises(ReplayEquivalenceError, match="not in the registry"):
        _verify_proof_boundary(events)


def test_verify_proof_boundary_judged_needs_no_artifact() -> None:
    _verify_proof_boundary((_finding_event(evidence_tier=EvidenceTier.JUDGED),))  # no raise


# ---------------------------------------------------------------------------
# Cross-event reference checks
# ---------------------------------------------------------------------------


def test_verify_cross_event_refs_resolves() -> None:
    finding = _finding_event()
    trace = TraceDecisionEvent(
        review_id=_REVIEW_ID,
        source_finding_id=finding.finding_id,
        target_file=None,
        reason="no candidates",
        resolution_status="unresolved",
        proposed_import_strings=(),
        resolved_candidate_paths=(),
    )
    _verify_cross_event_refs((finding, trace))  # no raise


def test_verify_cross_event_refs_rejects_dangling_reference() -> None:
    finding = _finding_event()
    trace = TraceDecisionEvent(
        review_id=_REVIEW_ID,
        source_finding_id=uuid4(),  # references a finding not in the stream
        target_file=None,
        reason="no candidates",
        resolution_status="unresolved",
        proposed_import_strings=(),
        resolved_candidate_paths=(),
    )
    with pytest.raises(ReplayEquivalenceError, match="no FindingEvent in the stream"):
        _verify_cross_event_refs((finding, trace))


# ---------------------------------------------------------------------------
# Full-finding content equality + mode consistency
# ---------------------------------------------------------------------------


def test_verify_full_finding_ok() -> None:
    event = _finding_event()
    # No override projection on the content row -> the HITL cross-check is
    # vacuous, so an empty override map is correct.
    _verify_full_finding(ReconstructedFinding(event=event, content=_content_for(event)), {})


def test_verify_full_finding_stub_raises() -> None:
    with pytest.raises(ReplayEquivalenceError, match="expected full content but is a stub"):
        _verify_full_finding(ReconstructedFinding(event=_finding_event(), content=None), {})


def test_verify_full_finding_content_mismatch_raises() -> None:
    event = _finding_event()
    content = _content_for(event).model_copy(update={"severity": FindingSeverity.LOW})
    with pytest.raises(ReplayEquivalenceError, match="disagrees with audit event"):
        _verify_full_finding(ReconstructedFinding(event=event, content=content), {})


def test_verify_full_finding_proof_artifact_mismatch_raises() -> None:
    # The content row's proof artifact (query_match_id) disagrees with the
    # canonical FindingEvent — full mode must catch it.
    event = _finding_event(
        evidence_tier=EvidenceTier.OBSERVED, query_match_id="python.function_definition"
    )
    content = _content_for(event).model_copy(update={"query_match_id": "python.class_definition"})
    with pytest.raises(ReplayEquivalenceError, match="query_match_id"):
        _verify_full_finding(ReconstructedFinding(event=event, content=content), {})


def test_verify_mode_consistency_full_ok() -> None:
    event = _finding_event()
    finding = ReconstructedFinding(event=event, content=_content_for(event))
    review = _review(mode=ReplayMode.FULL, review=_review_metadata(), findings=(finding,))
    _verify_mode_consistency(review)  # no raise


def test_verify_mode_consistency_metadata_only_ok() -> None:
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    review = _review(mode=ReplayMode.METADATA_ONLY, review=None, findings=(finding,))
    _verify_mode_consistency(review)  # no raise


def test_verify_mode_consistency_label_disagrees_raises() -> None:
    # Labeled METADATA_ONLY but the review row is present → recomputes FULL.
    review = _review(mode=ReplayMode.METADATA_ONLY, review=_review_metadata())
    with pytest.raises(ReplayEquivalenceError, match="disagrees with content presence"):
        _verify_mode_consistency(review)


# ---------------------------------------------------------------------------
# FUP-122 — override-projection cross-check against the HITL stream (#034)
# ---------------------------------------------------------------------------


def test_hitl_override_decisions_indexes_only_severity_override() -> None:
    # A HITLDecisionEvent with one SEVERITY_OVERRIDE + one REJECT decision: only
    # the override is indexed (REJECT carries no override projection to check).
    overridden, rejected = uuid4(), uuid4()
    event = _hitl_decision_event(
        (
            _override_decision(
                overridden, original=FindingSeverity.CRITICAL, override=FindingSeverity.LOW
            ),
            PerFindingDecision(
                finding_id=rejected,
                outcome=PerFindingOutcome.REJECT,
                reason="false positive",
            ),
        )
    )
    overrides = _hitl_override_decisions((event,))
    assert set(overrides) == {overridden}
    assert overrides[overridden].override_severity == FindingSeverity.LOW


def test_hitl_override_decisions_empty_without_hitl_event() -> None:
    assert _hitl_override_decisions((_finding_event(), _llm_call_event())) == {}


def test_null_override_projection_is_vacuous() -> None:
    # original_severity AND override_reason both NULL -> no claim -> always valid,
    # even with an empty override map (the V1 production shape; one-directional).
    event = _finding_event()
    content = _content_for(event)  # override fields are None
    _verify_finding_override_projection(event.finding_id, content, {})  # no raise


def test_override_projection_matches_stream_ok() -> None:
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason="downgraded after manual review",
    )
    overrides = {
        event.finding_id: _override_decision(
            event.finding_id,
            original=FindingSeverity.CRITICAL,
            override=FindingSeverity.LOW,
            reason="downgraded after manual review",
        )
    }
    _verify_finding_override_projection(event.finding_id, content, overrides)  # no raise


def test_override_claim_without_decision_raises() -> None:
    # The row claims an override but the HITL stream carries none — a forged row.
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason="forged",
    )
    with pytest.raises(ReplayEquivalenceError, match="no SEVERITY_OVERRIDE decision"):
        _verify_finding_override_projection(event.finding_id, content, {})


def test_override_original_severity_mismatch_raises() -> None:
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.HIGH,  # row claims HIGH baseline
        override_reason="downgraded after manual review",
    )
    overrides = {
        event.finding_id: _override_decision(
            event.finding_id,
            original=FindingSeverity.CRITICAL,  # stream says CRITICAL baseline
            override=FindingSeverity.LOW,
            reason="downgraded after manual review",
        )
    }
    with pytest.raises(ReplayEquivalenceError, match="original_severity"):
        _verify_finding_override_projection(event.finding_id, content, overrides)


def test_override_reason_mismatch_raises() -> None:
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason="reviewer wrote X",
    )
    overrides = {
        event.finding_id: _override_decision(
            event.finding_id,
            original=FindingSeverity.CRITICAL,
            override=FindingSeverity.LOW,
            reason="but the stream says Y",
        )
    }
    with pytest.raises(ReplayEquivalenceError, match="override_reason"):
        _verify_finding_override_projection(event.finding_id, content, overrides)


def test_override_partial_projection_severity_only_raises() -> None:
    # original_severity populated but override_reason NULL — a partial envelope.
    # Rejected even though a matching decision exists (the partial check fires
    # BEFORE corroboration): a faithful projection populates both.
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason=None,
    )
    overrides = {
        event.finding_id: _override_decision(
            event.finding_id,
            original=FindingSeverity.CRITICAL,
            override=FindingSeverity.LOW,
        )
    }
    with pytest.raises(ReplayEquivalenceError, match="partial override projection"):
        _verify_finding_override_projection(event.finding_id, content, overrides)


def test_override_partial_projection_reason_only_raises() -> None:
    # override_reason populated but original_severity NULL — the mirror partial.
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=None,
        override_reason="downgraded after manual review",
    )
    overrides = {
        event.finding_id: _override_decision(
            event.finding_id,
            original=FindingSeverity.CRITICAL,
            override=FindingSeverity.LOW,
        )
    }
    with pytest.raises(ReplayEquivalenceError, match="partial override projection"):
        _verify_finding_override_projection(event.finding_id, content, overrides)


def test_verify_mode_consistency_forged_override_projection_raises() -> None:
    # End-to-end through _verify_mode_consistency: a FULL review whose finding
    # content forges an override, but the event stream carries NO HITLDecision.
    # Proves the override map is built from review.events and threaded through.
    event = _finding_event()
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason="forged downgrade",
    )
    finding = ReconstructedFinding(event=event, content=content)
    review = _review(
        mode=ReplayMode.FULL,
        review=_review_metadata(),
        findings=(finding,),
        events=(event,),  # FindingEvent only — no HITLDecisionEvent
    )
    with pytest.raises(ReplayEquivalenceError, match="no SEVERITY_OVERRIDE decision"):
        _verify_mode_consistency(review)


def test_verify_mode_consistency_corroborated_override_projection_ok() -> None:
    # Same forged override, but now the stream carries the matching
    # SEVERITY_OVERRIDE decision — the row is a legitimate projection. Note the
    # content's `severity` stays the CRITICAL analyze-time snapshot (== event
    # severity); the LOW override_severity is NOT compared to it (#034).
    event = _finding_event()  # severity CRITICAL
    content = _forge_override(
        _content_for(event),
        original_severity=FindingSeverity.CRITICAL,
        override_reason="downgraded after manual review",
    )
    finding = ReconstructedFinding(event=event, content=content)
    decision_event = _hitl_decision_event(
        (
            _override_decision(
                event.finding_id,
                original=FindingSeverity.CRITICAL,
                override=FindingSeverity.LOW,
                reason="downgraded after manual review",
            ),
        )
    )
    review = _review(
        mode=ReplayMode.FULL,
        review=_review_metadata(),
        findings=(finding,),
        events=(event, decision_event),
    )
    _verify_mode_consistency(review)  # no raise
    # The snapshot is preserved: the row still shows CRITICAL, not the override.
    assert finding.content is not None
    assert finding.content.severity == FindingSeverity.CRITICAL


# ---------------------------------------------------------------------------
# Construction guard + structural invariant
# ---------------------------------------------------------------------------


def test_auditreplayer_requires_session_factory() -> None:
    with pytest.raises(ReplayError, match="session_factory is required"):
        AuditReplayer(session_factory=None)  # type: ignore[arg-type]


def test_finding_event_carries_no_stored_confidence() -> None:
    # confidence-is-computed-not-assigned: replay cannot read a stored
    # confidence because neither the event nor the findings table exposes one.
    assert "confidence" not in FindingEvent.model_fields
    assert "confidence" not in FindingContent.model_fields


# ---------------------------------------------------------------------------
# Strict derived-owner hybrid grouping (parallel-analyze increment 5;
# specs/2026-07-05-parallel-analyze.md + DECISIONS.md#064). The five pins.
# ---------------------------------------------------------------------------


def _keyed_pass_stream() -> tuple[object, ...]:
    """A faithful fan-out pass-0 stream with TWO CONCURRENT workers:
    plan pair, both worker starts, interleaved keyed work, both worker
    ends, then the aggregate pair with its keyed finding."""
    ka = "file:src/a.py#0"
    kb = "file:src/b.py#0"
    return (
        _phase_event(node_id="analyze", marker="start", phase_key="plan#0", sequence_number=1),
        _phase_event(node_id="analyze", marker="end", phase_key="plan#0", sequence_number=2),
        _phase_event(node_id="analyze", marker="start", phase_key=ka, sequence_number=3),
        _phase_event(node_id="analyze", marker="start", phase_key=kb, sequence_number=4),
        # Interleaved: b's call lands between a's start and a's call.
        _llm_call_event(sequence_number=5, phase_key=kb),
        _llm_call_event(sequence_number=6, phase_key=ka),
        _phase_event(node_id="analyze", marker="end", phase_key=ka, sequence_number=7),
        _phase_event(node_id="analyze", marker="end", phase_key=kb, sequence_number=8),
        _phase_event(node_id="analyze", marker="start", phase_key="aggregate#0", sequence_number=9),
        _finding_event(sequence_number=10, phase_key="aggregate#0"),
        _phase_event(node_id="analyze", marker="end", phase_key="aggregate#0", sequence_number=11),
    )


def test_keyed_events_group_by_identity_across_concurrent_workers() -> None:
    """Pin 1: interleaved keyed work groups into ITS OWN worker phase by
    (derived owner, phase_key) — adjacency would cross-attribute the two
    concurrent workers' calls. The same stream also verifies clean."""
    events = _keyed_pass_stream()
    _verify_phase_wellformed(events)  # concurrent KEYED phases are legal
    phases = _group_phases(events)
    by_key = {p.phase_key: p for p in phases}
    (a_call,) = by_key["file:src/a.py#0"].events
    (b_call,) = by_key["file:src/b.py#0"].events
    assert a_call.sequence_number == 6  # NOT the adjacent (5) one
    assert b_call.sequence_number == 5
    assert by_key["plan#0"].events == ()


def test_aggregate_keyed_finding_groups_under_analyze_aggregate() -> None:
    """Pin 2: FindingEvent carries no node_id — its owner derives from the
    logical owner map (→ analyze) and composes with the aggregate key, so
    it groups under (analyze, aggregate#0)."""
    phases = _group_phases(_keyed_pass_stream())
    agg = next(p for p in phases if p.phase_key == "aggregate#0")
    (finding,) = agg.events
    assert finding.phase_key == "aggregate#0"
    assert agg.node_id == "analyze"


def test_unkeyed_event_inside_open_keyed_phase_fails_loud() -> None:
    """Pin 3: the strict None-branch — a missing key inside an open keyed
    owner phase is a stamp-omission defect, never legacy data. Both the
    worker envelope and the aggregate envelope must reject it."""
    worker_case = (
        _phase_event(
            node_id="analyze", marker="start", phase_key="file:src/a.py#0", sequence_number=1
        ),
        _llm_call_event(sequence_number=2, phase_key=None),  # forgot to stamp
        _phase_event(
            node_id="analyze", marker="end", phase_key="file:src/a.py#0", sequence_number=3
        ),
    )
    with pytest.raises(ReplayEquivalenceError, match="stamp-omission"):
        _verify_phase_wellformed(worker_case)
    aggregate_case = (
        _phase_event(node_id="analyze", marker="start", phase_key="aggregate#0", sequence_number=1),
        _finding_event(sequence_number=2, phase_key=None),  # forgot to stamp
        _phase_event(node_id="analyze", marker="end", phase_key="aggregate#0", sequence_number=2),
    )
    with pytest.raises(ReplayEquivalenceError, match="stamp-omission"):
        _verify_phase_wellformed(aggregate_case)


def test_keyed_event_with_no_matching_open_phase_fails_loud() -> None:
    """Pin 4: a keyed event must sit inside an open phase matching BOTH its
    derived owner and its key — the wrong worker's envelope does not count
    (identity, never adjacency)."""
    events = (
        _phase_event(
            node_id="analyze", marker="start", phase_key="file:src/a.py#0", sequence_number=1
        ),
        _llm_call_event(sequence_number=2, phase_key="file:src/OTHER.py#0"),
        _phase_event(
            node_id="analyze", marker="end", phase_key="file:src/a.py#0", sequence_number=3
        ),
    )
    with pytest.raises(ReplayEquivalenceError, match=r"no open\s+phase matches"):
        _verify_phase_wellformed(events)


def test_sequential_era_stream_groups_and_verifies_unchanged() -> None:
    """Pin 5: legacy streams (every key None) keep their exact pre-fan-out
    semantics — adjacency grouping, two analyze passes never merged, and
    un-keyed phases still strictly non-nested."""
    legacy = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:p0", sequence_number=1),
        _llm_call_event(sequence_number=2),
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:p0", sequence_number=3),
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:p1", sequence_number=4),
        _llm_call_event(sequence_number=5),
        _phase_event(node_id="analyze", marker="end", phase_id="analyze:p1", sequence_number=6),
    )
    _verify_phase_wellformed(legacy)  # no raise: None keys outside keyed phases are legacy
    phases = _group_phases(legacy)
    assert [p.phase_id for p in phases] == ["analyze:p0", "analyze:p1"]  # never merged
    assert [len(p.events) for p in phases] == [1, 1]
    # Un-keyed overlap is still rejected (sequential-era rule intact).
    nested = (
        _phase_event(node_id="analyze", marker="start", phase_id="analyze:p0", sequence_number=1),
        _phase_event(node_id="trace", marker="start", phase_id="trace:0", sequence_number=2),
    )
    with pytest.raises(ReplayEquivalenceError, match="non-nested"):
        _verify_phase_wellformed(nested)
