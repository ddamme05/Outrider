"""Every audit subtype admits with valid fields + has correct event_type literal.

Parametrized over all 10 V1 event types per spec §8.2. Each tuple is
(event_class, expected_literal, minimal_kwargs); construction must succeed
and `event.event_type` must equal the literal — confirms the discriminator
value is wired correctly on every subtype.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from outrider.audit.events import (
    AgentTransitionEvent,
    AuditEventBase,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    LLMCallEvent,
    PublishEvent,
    PublishRoutingEvent,
    ReviewPhaseEvent,
    TraceDecisionEvent,
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
        "prompt_hash": "sha256-abc",
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
        "system_prompt_hash": "sha256-def",
        "degraded_mode": False,
    }


def _file_examination_kwargs() -> dict[str, Any]:
    return {
        "file_path": "src/foo.py",
        "examination_type": "deep",
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
        "finding_content_hash": "sha256-h",
        "evidence_tier": EvidenceTier.JUDGED,
        "policy_version": "1.0.0",
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
    return {
        "finding_id": uuid4(),
        "destination": PublishDestination.INLINE_COMMENT,
        "reason": "reviewable_diff_line",
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
)


@pytest.mark.parametrize(("event_class", "expected_event_type", "kwargs"), SUBTYPES)
def test_subtype_admits_with_valid_fields_and_event_type_literal_correct(
    event_class: type[AuditEventBase],
    expected_event_type: str,
    kwargs: dict[str, Any],
) -> None:
    """Every subtype constructs cleanly and reports the canonical event_type."""
    event = event_class(review_id=uuid4(), **kwargs)
    assert event.event_type == expected_event_type
    assert event.review_id is not None
