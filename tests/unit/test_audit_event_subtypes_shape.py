"""Every spec §8.2 audit subtype admits with valid fields + correct event_type.

Parametrized over the 12 V1 event types per spec §8.2 (`AgentTransitionEvent`,
`ReviewPhaseEvent`, `LLMCallEvent`, `FileExaminationEvent`, `FindingEvent`,
`TraceDecisionEvent`, `HITLRequestEvent`, `HITLDecisionEvent`,
`PublishEvent`, `PublishRoutingEvent`, `PublishEligibilityEvent`,
`PublishAttemptEvent`). Each tuple is `(event_class, expected_literal,
minimal_kwargs)`; construction must succeed and `event.event_type` must
equal the literal — confirms the discriminator value is wired correctly
on every subtype.

**Scope note (post-PR review fold):** the three analyze-foundation
event additions (`AnalyzeCompletedEvent`, `FindingProposalRejectedEvent`,
`AnalyzeResponseRejectedEvent` per `specs/2026-05-19-analyze-foundation.md`
§5) are NOT in this parametrized SUBTYPES tuple — they have their own
dedicated test file at `tests/unit/test_analyze_audit_events.py` that
exercises their validators end-to-end. This file's central parametrize
covers §8.2 only; "all subtypes" framings on the file's docstring used
to overclaim and were corrected here.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEventBase,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishAttemptEvent,
    PublishAttemptOutcome,
    PublishEligibility,
    PublishEligibilityEvent,
    PublishEvent,
    PublishRoutingEvent,
    PublishRoutingReason,
    ReviewPhaseEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
    compute_publish_attempt_content_hash,
    compute_publish_eligibility_decision_hash,
    compute_publish_routing_decision_hash,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import (
    PerFindingDecision,
    PerFindingOutcome,
    PublishDestination,
    ReviewDimension,
)


def _agent_transition_kwargs() -> dict[str, Any]:
    return {"from_node": "intake", "to_node": "triage", "latency_ms": 12}


def _review_phase_kwargs() -> dict[str, Any]:
    return {"phase_id": "p1", "node_id": "analyze", "marker": "start"}


def _llm_call_kwargs() -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-6",
        "node_id": "analyze",
        "input_tokens": 1000,
        "output_tokens": 200,
        "cached_tokens": 0,
        "cost_usd": 0.01,
        "latency_ms": 800,
        "prompt_hash": "a" * 64,
        "cache_hit": False,
        "context_summary": (
            ContextManifestEntry(
                file_path="src/foo.py",
                scope_unit_name="Foo.bar",
                line_start=1,
                line_end=10,
                inclusion_reason="changed_scope",
            ),
        ),
        "prompt_template_version": "analyze@1.0.0",
        "pricing_version": "v1",
        "system_prompt_hash": "b" * 64,
        "degraded_mode": False,
    }


def _file_examination_kwargs() -> dict[str, Any]:
    return {
        "file_path": "src/foo.py",
        "examination_type": "analyze",
        "node_id": "analyze",
        "parse_status": "clean",
    }


def _finding_kwargs() -> dict[str, Any]:
    return {
        "finding_id": uuid4(),
        "finding_type": FindingType.SQL_INJECTION,
        "severity": FindingSeverity.CRITICAL,
        "file_path": "src/foo.py",
        "line_start": 10,
        "line_end": 12,
        "dimension": ReviewDimension.SECURITY,
        "finding_content_hash": compute_finding_content_hash(
            file_path="src/foo.py",
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        "evidence_tier": EvidenceTier.JUDGED,
        "policy_version": "1.0.0",
        "proposal_hash": "a" * 64,  # Per DECISIONS.md#025; FindingEvent mirror.
    }


def _trace_decision_kwargs() -> dict[str, Any]:
    return {
        "source_finding_id": uuid4(),
        "target_file": "src/bar.py",
        "reason": "called from middleware/auth.py:42",
        "resolution_status": "resolved",
        "candidates_considered": ("src/bar.py", "src/baz.py"),
    }


def _hitl_request_kwargs() -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "findings_requiring_approval": (uuid4(),),
        "auto_post_findings": (uuid4(),),
        "expires_at": now + timedelta(minutes=30),
    }


def _hitl_decision_kwargs() -> dict[str, Any]:
    decision = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="",
    )
    return {
        "reviewer_id": "reviewer@example.com",
        "decisions": (decision,),
        "decision_latency_seconds": 42.5,
    }


def _publish_kwargs() -> dict[str, Any]:
    return {
        "github_review_id": 12345,
        "comments_posted": 3,
        "review_status": "COMMENT",
    }


def _publish_routing_kwargs() -> dict[str, Any]:
    file_path = "src/app.py"
    line_start = 10
    line_end = 12
    destination = PublishDestination.INLINE_COMMENT
    reason = PublishRoutingReason.REVIEWABLE_DIFF_LINE
    finding_type = FindingType.MISSING_INPUT_VALIDATION
    return {
        "finding_id": uuid4(),
        "destination": destination,
        "reason": reason,
        "coordinate_error_kind": None,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "finding_type": finding_type,
        "finding_content_hash": compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        "decision_content_hash": compute_publish_routing_decision_hash(
            destination=destination,
            reason=reason,
            coordinate_error_kind=None,
        ),
    }


def _publish_eligibility_kwargs() -> dict[str, Any]:
    file_path = "src/app.py"
    line_start = 10
    line_end = 12
    eligibility = PublishEligibility.ELIGIBLE
    reason = None
    return {
        "finding_id": uuid4(),
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "finding_type": FindingType.MISSING_INPUT_VALIDATION,
        "severity": FindingSeverity.MEDIUM,
        "original_severity": None,
        "finding_content_hash": compute_finding_content_hash(
            file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=FindingType.MISSING_INPUT_VALIDATION,
        ),
        "decision_content_hash": compute_publish_eligibility_decision_hash(
            eligibility=eligibility,
            reason=reason,
        ),
        "eligibility": eligibility,
        "reason": reason,
        "policy_version": "1.0.0",
    }


def _publish_attempt_kwargs() -> dict[str, Any]:
    review_id = uuid4()
    attempt_index = 1
    sorted_finding_ids: tuple[Any, ...] = ()
    outcome = PublishAttemptOutcome.NO_OP_EMPTY
    status_code: int | None = None
    failure_class: str | None = None
    comments_attempted = 0
    return {
        "review_id": review_id,
        "attempt_index": attempt_index,
        "outcome": outcome,
        "status_code": status_code,
        "failure_class": failure_class,
        "comments_attempted": comments_attempted,
        "sorted_finding_ids": sorted_finding_ids,
        "attempt_content_hash": compute_publish_attempt_content_hash(
            review_id=review_id,
            attempt_index=attempt_index,
            sorted_finding_ids=sorted_finding_ids,
            outcome=outcome,
            status_code=status_code,
            failure_class=failure_class,
            comments_attempted=comments_attempted,
        ),
    }


SUBTYPES: tuple[tuple[type[AuditEventBase], str, dict[str, Any]], ...] = (
    (AgentTransitionEvent, "agent_transition", _agent_transition_kwargs()),
    (ReviewPhaseEvent, "review_phase", _review_phase_kwargs()),
    (LLMCallEvent, "llm_call", _llm_call_kwargs()),
    (FileExaminationEvent, "file_examination", _file_examination_kwargs()),
    (FindingEvent, "finding", _finding_kwargs()),
    (TraceDecisionEvent, "trace_decision", _trace_decision_kwargs()),
    (HITLRequestEvent, "hitl_request", _hitl_request_kwargs()),
    (HITLDecisionEvent, "hitl_decision", _hitl_decision_kwargs()),
    (PublishEvent, "publish", _publish_kwargs()),
    (PublishRoutingEvent, "publish_routing", _publish_routing_kwargs()),
    (PublishEligibilityEvent, "publish_eligibility", _publish_eligibility_kwargs()),
    (PublishAttemptEvent, "publish_attempt", _publish_attempt_kwargs()),
)


@pytest.mark.parametrize(("event_class", "expected_event_type", "kwargs"), SUBTYPES)
def test_subtype_admits_with_valid_fields_and_event_type_literal_correct(
    event_class: type[AuditEventBase],
    expected_event_type: str,
    kwargs: dict[str, Any],
) -> None:
    """Every subtype constructs cleanly and reports the canonical event_type.

    Most fixtures don't include `review_id` in their kwargs and rely on the
    harness's `uuid4()`. `PublishAttemptEvent` is the exception: its
    `attempt_content_hash` recipe includes `review_id`, so the fixture
    pre-computes the hash against a known review_id and provides both in
    its kwargs. The fixture's review_id wins via the dict-merge order
    below — kwargs override the harness's default.
    """
    harness_kwargs: dict[str, Any] = {"review_id": uuid4()}
    harness_kwargs.update(kwargs)
    event = event_class(**harness_kwargs)
    assert event.event_type == expected_event_type
    assert event.review_id is not None


@pytest.mark.parametrize("field_name", ["prompt_hash", "system_prompt_hash"])
def test_llm_call_event_hash_fields_reject_non_hex(field_name: str) -> None:
    """`LLMCallEvent.prompt_hash` and `system_prompt_hash` must be SHA-256
    lowercase hex (matches sibling `FindingEvent.finding_content_hash`).
    Catches a producer-side bug at construction time rather than waiting
    for the persister's pre-tx recomputation guard to catch it at INSERT.
    """
    kwargs = _llm_call_kwargs()
    kwargs[field_name] = "sha256-abc"  # legacy literal; not lowercase hex
    with pytest.raises(ValidationError):
        LLMCallEvent(review_id=uuid4(), **kwargs)


# ---------------------------------------------------------------------------
# §0b crazy-audit fold: LLMCallEvent.degradation_reason provenance pairing.
# Mirrors the LLMRequest._enforce_degradation_provenance bidirectional rule
# at the event boundary. Without these tests the wrapper could silently drop
# the reason mid-pipeline (sharp-edges SE-1 + adversarial HIGH + data-int F1
# — three-agent convergent finding).
# ---------------------------------------------------------------------------


def test_llm_call_event_degradation_reason_defaults_none() -> None:
    """Backward-compat (F4): historical rows without `degradation_reason`
    still validate under the new schema. The field defaults to None and
    `degraded_mode=False` in the kwargs helper, so existing fixtures
    construct cleanly without explicit pass-through."""
    event = LLMCallEvent(review_id=uuid4(), **_llm_call_kwargs())
    assert event.degraded_mode is False
    assert event.degradation_reason is None


def test_llm_call_event_degraded_without_reason_raises() -> None:
    """`degraded_mode=True` + `degradation_reason=None` fails the mirror
    validator — same shape as the LLMRequest provenance rule. Prevents
    wrapper drift dropping the typed cause."""
    kwargs = _llm_call_kwargs()
    kwargs["degraded_mode"] = True
    kwargs["degradation_reason"] = None
    with pytest.raises(ValidationError, match="degraded_mode=True requires"):
        LLMCallEvent(review_id=uuid4(), **kwargs)


def test_llm_call_event_reason_without_degraded_raises() -> None:
    """`degraded_mode=False` + `degradation_reason='parse_failed'` fails —
    reason-without-mode is the inverse asymmetry. Either flag set without
    the other is a wrapper-drift signal."""
    kwargs = _llm_call_kwargs()
    kwargs["degraded_mode"] = False
    kwargs["degradation_reason"] = "parse_failed"
    with pytest.raises(ValidationError, match="degradation_reason requires"):
        LLMCallEvent(review_id=uuid4(), **kwargs)


@pytest.mark.parametrize("reason", ["parse_failed", "tree_has_error_in_changed_regions"])
def test_llm_call_event_degraded_with_typed_reason_admits(reason: str) -> None:
    """The happy path: both flags set consistently."""
    kwargs = _llm_call_kwargs()
    kwargs["degraded_mode"] = True
    kwargs["degradation_reason"] = reason
    event = LLMCallEvent(review_id=uuid4(), **kwargs)
    assert event.degraded_mode is True
    assert event.degradation_reason == reason


def test_llm_call_event_degradation_reason_rejects_arbitrary_string() -> None:
    """Same narrow-Literal contract as LLMRequest — extending the reason
    enumeration must happen in lockstep across LLMRequest AND LLMCallEvent."""
    kwargs = _llm_call_kwargs()
    kwargs["degraded_mode"] = True
    kwargs["degradation_reason"] = "some_new_reason"
    with pytest.raises(ValidationError):
        LLMCallEvent(review_id=uuid4(), **kwargs)


# ---------------------------------------------------------------------------
# Pre-emptive sweep #3: cross-file constraint coherence on node_id /
# review_status / reviewer_id. Each rejection test pairs with an admit test
# at the inclusive boundary so a future tightening shifts both sides loudly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_id", ["triage", "analyze", "synthesize", "trace"])
def test_llm_call_event_admits_canonical_node_ids(node_id: str) -> None:
    """The four LLM-calling nodes per spec §4.1 — matches the
    `LLMRequest.node_id` Literal in `llm/base.py`."""
    kwargs = _llm_call_kwargs()
    kwargs["node_id"] = node_id
    event = LLMCallEvent(review_id=uuid4(), **kwargs)
    assert event.node_id == node_id


@pytest.mark.parametrize("bad_node_id", ["intake", "hitl", "publish", "TRIAGE", "", "analyse"])
def test_llm_call_event_rejects_non_llm_calling_node_id(bad_node_id: str) -> None:
    """A non-LLM-calling node (intake/hitl/publish), wrong casing, empty
    string, or typo (analyse) — all rejected. Pre-sweep this field was
    `str` and admitted any value."""
    kwargs = _llm_call_kwargs()
    kwargs["node_id"] = bad_node_id
    with pytest.raises(ValidationError):
        LLMCallEvent(review_id=uuid4(), **kwargs)


@pytest.mark.parametrize("node_id", ["intake", "analyze"])
def test_file_examination_event_admits_canonical_node_ids(node_id: str) -> None:
    """FileExaminationEvent fires from intake (per-file fetch) and analyze
    (per-file examination). Other graph nodes do not emit it in V1."""
    kwargs = _file_examination_kwargs()
    kwargs["node_id"] = node_id
    event = FileExaminationEvent(review_id=uuid4(), **kwargs)
    assert event.node_id == node_id


@pytest.mark.parametrize("bad_node_id", ["triage", "trace", "hitl", "INTAKE", ""])
def test_file_examination_event_rejects_non_canonical_node_id(bad_node_id: str) -> None:
    kwargs = _file_examination_kwargs()
    kwargs["node_id"] = bad_node_id
    with pytest.raises(ValidationError):
        FileExaminationEvent(review_id=uuid4(), **kwargs)


@pytest.mark.parametrize("status", ["APPROVE", "REQUEST_CHANGES", "COMMENT"])
def test_publish_event_admits_canonical_review_status(status: str) -> None:
    """The three GitHub-side `event` parameter values for the create-review
    REST endpoint. V1 omits PENDING (draft state) deliberately."""
    kwargs = _publish_kwargs()
    kwargs["review_status"] = status
    event = PublishEvent(review_id=uuid4(), **kwargs)
    assert event.review_status == status


@pytest.mark.parametrize("bad_status", ["approve", "PENDING", "MERGED", "CLOSED", ""])
def test_publish_event_rejects_non_github_review_status(bad_status: str) -> None:
    """Wrong casing, GitHub draft state (PENDING), or non-review states
    (MERGED/CLOSED come from the PR-merge endpoint, not the review one) —
    all rejected at the schema layer."""
    kwargs = _publish_kwargs()
    kwargs["review_status"] = bad_status
    with pytest.raises(ValidationError):
        PublishEvent(review_id=uuid4(), **kwargs)


def test_hitl_decision_event_reviewer_id_max_length() -> None:
    """100-char cap on `reviewer_id`. Without the bound, a malformed or
    attacker-supplied reviewer id could fill the audit row arbitrarily."""
    kwargs = _hitl_decision_kwargs()
    kwargs["reviewer_id"] = "x" * 101
    with pytest.raises(ValidationError, match="reviewer_id"):
        HITLDecisionEvent(review_id=uuid4(), **kwargs)


def test_hitl_decision_event_reviewer_id_admits_at_max() -> None:
    """Inclusive boundary admits — a future tightening to 99 shifts BOTH
    the admit and reject sides loudly."""
    kwargs = _hitl_decision_kwargs()
    kwargs["reviewer_id"] = "x" * 100
    event = HITLDecisionEvent(review_id=uuid4(), **kwargs)
    assert event.reviewer_id == "x" * 100


# ---------------------------------------------------------------------------
# Pre-emptive sweep #4: ReviewPhaseEvent node_id + ContextManifestEntry caps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id",
    ["intake", "triage", "analyze", "trace", "synthesize", "hitl", "publish"],
)
def test_review_phase_event_admits_all_seven_graph_nodes(node_id: str) -> None:
    """All seven graph nodes per spec §4. V1 only intake + triage emit
    today; the other five land with their respective node specs."""
    kwargs = _review_phase_kwargs()
    kwargs["node_id"] = node_id
    event = ReviewPhaseEvent(review_id=uuid4(), **kwargs)
    assert event.node_id == node_id


@pytest.mark.parametrize("bad_node_id", ["webhook", "analyse", "INTAKE", "Hitl", "synthesise", ""])
def test_review_phase_event_rejects_non_canonical_node_id(bad_node_id: str) -> None:
    """Synthetic webhook source (used by AgentTransitionEvent, not Phase),
    typos, wrong-case all rejected."""
    kwargs = _review_phase_kwargs()
    kwargs["node_id"] = bad_node_id
    with pytest.raises(ValidationError):
        ReviewPhaseEvent(review_id=uuid4(), **kwargs)


def test_context_manifest_entry_file_path_max_length() -> None:
    """1024-char per-element cap matches AnalysisRound.files_examined."""
    with pytest.raises(ValidationError, match="file_path"):
        ContextManifestEntry(
            file_path="x" * 1025,
            scope_unit_name="foo",
            line_start=1,
            line_end=2,
            inclusion_reason="changed_scope",
        )


def test_context_manifest_entry_file_path_admits_at_max() -> None:
    entry = ContextManifestEntry(
        file_path="x" * 1024,
        scope_unit_name="foo",
        line_start=1,
        line_end=2,
        inclusion_reason="changed_scope",
    )
    assert len(entry.file_path) == 1024


def test_context_manifest_entry_scope_unit_name_max_length() -> None:
    with pytest.raises(ValidationError, match="scope_unit_name"):
        ContextManifestEntry(
            file_path="src/foo.py",
            scope_unit_name="x" * 1025,
            line_start=1,
            line_end=2,
            inclusion_reason="changed_scope",
        )


# ---------------------------------------------------------------------------
# Set-semantic tuple-field duplicates: rejected at the schema layer so
# logically identical rows can't hash to distinct shapes.
# ---------------------------------------------------------------------------


def test_hitl_request_event_rejects_duplicate_in_findings_requiring_approval() -> None:
    """Each finding appears once across the two tuples."""
    finding_id = uuid4()
    kwargs = _hitl_request_kwargs()
    kwargs["findings_requiring_approval"] = (finding_id, finding_id)
    kwargs["auto_post_findings"] = ()
    with pytest.raises(ValidationError, match="duplicate ids"):
        HITLRequestEvent(review_id=uuid4(), **kwargs)


def test_hitl_request_event_rejects_finding_in_both_tuples() -> None:
    finding_id = uuid4()
    kwargs = _hitl_request_kwargs()
    kwargs["findings_requiring_approval"] = (finding_id,)
    kwargs["auto_post_findings"] = (finding_id,)
    with pytest.raises(ValidationError, match="both"):
        HITLRequestEvent(review_id=uuid4(), **kwargs)


def test_hitl_decision_event_rejects_multiple_decisions_for_same_finding() -> None:
    """One decision per finding_id — duplicates are conflicting verdicts."""
    from outrider.policy import FindingSeverity
    from outrider.schemas import PerFindingDecision
    from outrider.schemas.hitl import PerFindingOutcome

    shared_finding_id = uuid4()
    decision_a = PerFindingDecision(
        finding_id=shared_finding_id,
        outcome=PerFindingOutcome.APPROVE,
        reason="lgtm",
    )
    decision_b = PerFindingDecision(
        finding_id=shared_finding_id,  # same finding — two verdicts
        outcome=PerFindingOutcome.SEVERITY_OVERRIDE,
        reason="changed mind",
        override_severity=FindingSeverity.LOW,
        original_severity=FindingSeverity.HIGH,
    )
    kwargs = _hitl_decision_kwargs()
    kwargs["decisions"] = (decision_a, decision_b)
    with pytest.raises(ValidationError, match="multiple decisions"):
        HITLDecisionEvent(review_id=uuid4(), **kwargs)


def test_trace_decision_event_rejects_duplicate_candidates() -> None:
    """`candidates_considered` is set-semantic — each candidate is one
    consideration, not many."""
    kwargs = _trace_decision_kwargs()
    kwargs["candidates_considered"] = ("src/a.py", "src/a.py")
    kwargs["resolution_status"] = "unresolved"
    kwargs["target_file"] = None
    with pytest.raises(ValidationError, match="candidates_considered contains duplicates"):
        TraceDecisionEvent(review_id=uuid4(), **kwargs)


def test_llm_call_event_rejects_duplicate_context_summary_entries() -> None:
    """Same (file_path, scope_unit_name) twice in one prompt's manifest
    is a producer bug — context attribution becomes ambiguous."""
    from datetime import UTC, datetime

    from outrider.audit.events import ContextManifestEntry, LLMCallEvent

    entry = ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )
    with pytest.raises(ValidationError, match="duplicate"):
        LLMCallEvent(
            review_id=uuid4(),
            timestamp=datetime.now(UTC),
            model="claude-sonnet-4-6",
            node_id="analyze",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=0.01,
            pricing_version="v2",
            latency_ms=1000,
            prompt_hash="a" * 64,
            cache_hit=False,
            context_summary=(entry, entry),  # same (file_path, scope_unit_name)
            prompt_template_version="v1",
            system_prompt_hash="b" * 64,
            degraded_mode=False,
        )
