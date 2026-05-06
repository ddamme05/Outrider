"""AuditEvent discriminated union: round-trip through TypeAdapter reconstructs concrete subtypes.

Backs the replay contract: `audit/replay.py` (separate spec) reads an
`audit_events` row, merges the row-level `sequence_number` into the
JSONB payload, then calls `AuditEventAdapter.validate_python(merged)` to
reconstruct the right concrete event. The discriminator (`event_type`)
is what makes this fan-out replayable.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    AuditEventAdapter,
    ContextManifestEntry,
    FindingEvent,
    LLMCallEvent,
    compute_finding_content_hash,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension


def _build_llm_call() -> LLMCallEvent:
    return LLMCallEvent(
        review_id=uuid4(),
        model="claude-sonnet-4-6",
        node_id="analyze",
        input_tokens=1000,
        output_tokens=200,
        cached_tokens=0,
        cost_usd=0.01,
        latency_ms=800,
        prompt_hash="sha256-abc",
        cache_hit=False,
        context_summary=(
            ContextManifestEntry(
                file_path="src/foo.py",
                scope_unit_name="Foo.bar",
                line_start=1,
                line_end=10,
                inclusion_reason="changed_scope",
            ),
        ),
        prompt_template_version="analyze@1.0.0",
        pricing_version="v1",
        system_prompt_hash="sha256-def",
        degraded_mode=False,
    )


def _build_finding(
    *,
    evidence_tier: EvidenceTier = EvidenceTier.OBSERVED,
    query_match_id: str | None = "py.security.placeholder",
    trace_path: tuple[str, ...] | None = None,
) -> FindingEvent:
    file_path = "src/foo.py"
    line_start = 10
    line_end = 12
    finding_type = FindingType.SQL_INJECTION
    return FindingEvent(
        review_id=uuid4(),
        finding_id=uuid4(),
        finding_type=finding_type,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        dimension=ReviewDimension.SECURITY,
        finding_content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            finding_type=finding_type,
        ),
        evidence_tier=evidence_tier,
        query_match_id=query_match_id,
        trace_path=trace_path,
        policy_version="1.0.0",
    )


def test_json_round_trip_reconstructs_concrete_subtype() -> None:
    """Dump LLMCallEvent with mode='json', exclude sequence_number,
    merge a row-level sequence_number into the payload, then
    AuditEventAdapter.validate_python(merged) reconstructs LLMCallEvent
    with the sequence number restored.
    """
    original = _build_llm_call()
    payload: dict[str, Any] = original.model_dump(mode="json", exclude={"sequence_number"})

    merged = {**payload, "sequence_number": 42}
    reconstructed = AuditEventAdapter.validate_python(merged)

    assert isinstance(reconstructed, LLMCallEvent)
    assert reconstructed.sequence_number == 42
    assert reconstructed.event_id == original.event_id
    assert reconstructed.review_id == original.review_id
    assert reconstructed.model == original.model
    assert reconstructed.context_summary == original.context_summary


def test_json_round_trip_for_finding_event() -> None:
    """FindingEvent round-trip preserves proof artifacts (evidence_tier,
    query_match_id, trace_path) — the proof-boundary preservation test.
    """
    original = _build_finding(
        evidence_tier=EvidenceTier.INFERRED,
        query_match_id=None,
        trace_path=("scope_a", "scope_b"),
    )
    payload = original.model_dump(mode="json", exclude={"sequence_number"})
    merged = {**payload, "sequence_number": 7}

    reconstructed = AuditEventAdapter.validate_python(merged)

    assert isinstance(reconstructed, FindingEvent)
    assert reconstructed.evidence_tier == EvidenceTier.INFERRED
    assert reconstructed.query_match_id is None
    assert reconstructed.trace_path == ("scope_a", "scope_b")
    assert isinstance(reconstructed.trace_path, tuple)


def test_finding_event_trace_path_round_trips_as_tuple() -> None:
    """trace_path JSON-decodes from array back to tuple, not list.

    Critical because replay reconstructs from JSONB; if the decoded type
    were list, in-place mutations would bypass the immutability claim.
    """
    original = _build_finding(
        evidence_tier=EvidenceTier.INFERRED,
        query_match_id=None,
        trace_path=("a", "b", "c"),
    )
    json_payload = original.model_dump_json(exclude={"sequence_number"})
    reconstructed = AuditEventAdapter.validate_json(json_payload)

    assert isinstance(reconstructed, FindingEvent)
    assert isinstance(reconstructed.trace_path, tuple)
    assert reconstructed.trace_path == ("a", "b", "c")


def test_unknown_event_type_rejects() -> None:
    """A discriminator value not in the union raises with a clear error."""
    payload = {
        "event_id": str(uuid4()),
        "review_id": str(uuid4()),
        "event_type": "not_a_real_event",
        "timestamp": datetime.now(UTC).isoformat(),
        "is_eval": False,
    }
    with pytest.raises(ValidationError):
        AuditEventAdapter.validate_python(payload)


def test_wrong_payload_for_valid_event_type_rejects() -> None:
    """event_type='finding' with LLMCallEvent-shaped payload raises.

    The discriminator selects FindingEvent's schema, which then fails
    on missing required fields (finding_id, finding_type, etc.).
    """
    llm_call = _build_llm_call()
    payload = llm_call.model_dump(mode="json", exclude={"sequence_number"})
    payload["event_type"] = "finding"

    with pytest.raises(ValidationError):
        AuditEventAdapter.validate_python(payload)
