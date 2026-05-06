"""Numeric-field bounds across audit events.

Backs the cost-budget anomaly's input-sanitization claim: the V1
post-hoc cost cap sums `LLMCallEvent.cost_usd`; a malformed negative-cost
event would understate review cost. Token counts, latencies, and
GitHub-side counts all carry `ge=0` (or `ge=1` for entity IDs) at the
schema layer so the budget-summing code can trust its inputs.
"""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    AgentTransitionEvent,
    ContextManifestEntry,
    HITLDecisionEvent,
    LLMCallEvent,
    PublishEvent,
)
from outrider.schemas import PerFindingDecision, PerFindingOutcome


def _llm_call_kwargs(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
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
        "pricing_version": "v1",
        "system_prompt_hash": "sha256-def",
        "degraded_mode": False,
    }
    fields.update(overrides)
    return fields


def test_llm_call_event_negative_input_tokens_rejects() -> None:
    """Negative input_tokens raises (Field(ge=0))."""
    with pytest.raises(ValidationError):
        LLMCallEvent(**_llm_call_kwargs(input_tokens=-1))


def test_llm_call_event_negative_output_tokens_rejects() -> None:
    with pytest.raises(ValidationError):
        LLMCallEvent(**_llm_call_kwargs(output_tokens=-5))


def test_llm_call_event_negative_cached_tokens_rejects() -> None:
    with pytest.raises(ValidationError):
        LLMCallEvent(**_llm_call_kwargs(cached_tokens=-1))


def test_llm_call_event_negative_cost_rejects() -> None:
    """Cost-budget anomaly input sanitization: negative cost_usd raises."""
    with pytest.raises(ValidationError):
        LLMCallEvent(**_llm_call_kwargs(cost_usd=-0.01))


def test_llm_call_event_negative_latency_rejects() -> None:
    with pytest.raises(ValidationError):
        LLMCallEvent(**_llm_call_kwargs(latency_ms=-1))


def test_llm_call_event_zero_values_admit() -> None:
    """ge=0 means zero admits (cache_hit case can have 0 tokens, $0 cost)."""
    event = LLMCallEvent(
        **_llm_call_kwargs(
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0.0,
            latency_ms=0,
        )
    )
    assert event.cost_usd == 0.0


def test_agent_transition_event_negative_latency_rejects() -> None:
    with pytest.raises(ValidationError):
        AgentTransitionEvent(
            review_id=uuid4(),
            from_node="intake",
            to_node="triage",
            latency_ms=-1,
        )


def test_hitl_decision_event_negative_decision_latency_rejects() -> None:
    decision = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.APPROVE,
        reason="",
    )
    with pytest.raises(ValidationError):
        HITLDecisionEvent(
            review_id=uuid4(),
            reviewer_id="reviewer@example.com",
            decisions=(decision,),
            decision_latency_seconds=-0.01,
        )


def test_publish_event_github_review_id_must_be_positive() -> None:
    """GitHub review IDs are positive integers; 0 and negative raise."""
    with pytest.raises(ValidationError):
        PublishEvent(
            review_id=uuid4(),
            github_review_id=0,
            comments_posted=3,
            review_status="COMMENT",
        )
    with pytest.raises(ValidationError):
        PublishEvent(
            review_id=uuid4(),
            github_review_id=-1,
            comments_posted=3,
            review_status="COMMENT",
        )


def test_publish_event_negative_comments_posted_rejects() -> None:
    with pytest.raises(ValidationError):
        PublishEvent(
            review_id=uuid4(),
            github_review_id=12345,
            comments_posted=-1,
            review_status="COMMENT",
        )


def test_hitl_request_event_admits_future_expires_at() -> None:
    """HITLRequestEvent has no numeric bounds beyond the base; this test
    documents the negative-space — expires_at is just AwareDatetime.
    """
    # The actual constraint (expires_at > created_at) lives at the
    # construction-time concern, not the event schema. This test asserts
    # the schema admits a plausible future-dated expires_at to ensure
    # the absence of an unintended constraint.
    from outrider.audit.events import HITLRequestEvent

    event = HITLRequestEvent(
        review_id=uuid4(),
        findings_requiring_approval=(uuid4(),),
        auto_post_findings=(),
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    assert event.expires_at > event.timestamp
