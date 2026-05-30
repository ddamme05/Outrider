"""Unit tests for the pure (DB-free) surface of `audit/replay.py`.

Covers phase grouping, mode classification, and the verify-only checklist
helpers over in-memory reconstructed objects. The async DB-backed pieces
(`reconstruct`, `assert_replay_equivalent`, historical-policy severity) are
covered in `tests/integration/test_audit_replay.py`.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal
from uuid import UUID, uuid4

import pytest

from outrider.audit.events import (
    AgentTransitionEvent,
    FindingEvent,
    LLMCallEvent,
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
    _verify_cross_event_refs,
    _verify_full_finding,
    _verify_mode_consistency,
    _verify_phase_wellformed,
    _verify_proof_boundary,
    _verify_row_consistent,
    _verify_sequence_monotonic,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension

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
    )


def _phase_event(
    *,
    node_id: Literal["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"],
    marker: Literal["start", "end"],
    phase_id: str | None = None,
    sequence_number: int | None = None,
) -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=_REVIEW_ID,
        phase_id=phase_id or f"{node_id}:0",
        node_id=node_id,
        marker=marker,
        phase_key=None,
        sequence_number=sequence_number,
    )


def _llm_call_event(*, sequence_number: int | None = None) -> LLMCallEvent:
    return LLMCallEvent(
        review_id=_REVIEW_ID,
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
        repo_id=100,
        pr_number=1,
        head_sha="sha1",
        files_examined=1,
        files_traced_beyond_diff=0,
        llm_calls_made=1,
        total_input_tokens=100,
        total_output_tokens=50,
        total_cost_usd=Decimal("0.01"),
        wall_clock_seconds=Decimal("1.5"),
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
) -> ReconstructedReview:
    return ReconstructedReview(
        review_id=_REVIEW_ID,
        mode=mode,
        is_eval=False,
        review=review,
        events=(),
        phases=(),
        findings=findings,
        llm_exchanges=llm_exchanges,
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
    # The legitimate 90-180d window: findings (180d) present, LLM content (90d) gone.
    finding = ReconstructedFinding(event=_finding_event(), content=None)
    finding = finding.model_copy(update={"content": _content_for(finding.event)})
    exchange = ReconstructedLLMExchange(event=_llm_call_event(), prompt=None, completion=None)
    assert (
        _classify_mode(review_present=True, findings=(finding,), llm_exchanges=(exchange,))
        == ReplayMode.MIXED
    )


def test_classify_mode_review_absent_with_llm_content_raises() -> None:
    # Impossible under retention (LLM 90d ≤ review 180d): a purged review with a
    # surviving LLM content row is corruption, not a legitimate mixed window.
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
    # Sibling of the review-absent guard: LLM content (90d) purges no later than
    # finding content (180d), so surviving LLM content with a purged finding is
    # an out-of-order purge — corruption, not a legitimate MIXED window.
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
    _verify_full_finding(ReconstructedFinding(event=event, content=_content_for(event)))  # no raise


def test_verify_full_finding_stub_raises() -> None:
    with pytest.raises(ReplayEquivalenceError, match="expected full content but is a stub"):
        _verify_full_finding(ReconstructedFinding(event=_finding_event(), content=None))


def test_verify_full_finding_content_mismatch_raises() -> None:
    event = _finding_event()
    content = _content_for(event).model_copy(update={"severity": FindingSeverity.LOW})
    with pytest.raises(ReplayEquivalenceError, match="disagrees with audit event"):
        _verify_full_finding(ReconstructedFinding(event=event, content=content))


def test_verify_full_finding_proof_artifact_mismatch_raises() -> None:
    # The content row's proof artifact (query_match_id) disagrees with the
    # canonical FindingEvent — full mode must catch it.
    event = _finding_event(
        evidence_tier=EvidenceTier.OBSERVED, query_match_id="python.function_definition"
    )
    content = _content_for(event).model_copy(update={"query_match_id": "python.class_definition"})
    with pytest.raises(ReplayEquivalenceError, match="query_match_id"):
        _verify_full_finding(ReconstructedFinding(event=event, content=content))


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
