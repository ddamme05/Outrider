"""Emitter serialization contract: JSON-mode dump excluding sequence_number, then replay merge.

The emitter writes payload-only JSONB rows; sequence_number lives on the
audit_events row as a DB-assigned BIGSERIAL column. Replay reconstructs
by merging the row metadata with the payload before validating through
the discriminated union.
"""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from outrider.audit.events import AuditEventAdapter, LLMCallEvent
from outrider.audit.events import ContextManifestEntry as Entry


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
        prompt_hash="a" * 64,
        cache_hit=False,
        context_summary=(
            Entry(
                file_path="src/foo.py",
                scope_unit_name="Foo.bar",
                line_start=1,
                line_end=10,
                inclusion_reason="changed_scope",
            ),
        ),
        prompt_template_version="analyze@1.0.0",
        pricing_version="v1",
        system_prompt_hash="b" * 64,
        degraded_mode=False,
    )


def test_event_dumps_to_json_mode_for_jsonb() -> None:
    """mode='json' emits UUID as str and datetime as ISO string (JSONB-compatible).

    Default Python mode would emit Python objects which JSONB cannot store.
    """
    event = _build_llm_call()
    payload = event.model_dump(mode="json")

    assert isinstance(payload["event_id"], str)
    UUID(payload["event_id"])

    assert isinstance(payload["review_id"], str)
    UUID(payload["review_id"])

    assert isinstance(payload["timestamp"], str)
    datetime.fromisoformat(payload["timestamp"])


def test_event_dump_excludes_sequence_number() -> None:
    """exclude={'sequence_number'} drops the field from the payload.

    Documents the row-vs-payload boundary: sequence_number is an
    audit_events row column, not part of the JSONB payload.
    """
    event = _build_llm_call()
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    assert "sequence_number" not in payload


def test_replay_merges_row_sequence_into_payload() -> None:
    """Given a payload (no sequence_number) and a row-level sequence_number,
    merge then validate through the discriminated union to get the right
    concrete subtype with the sequence number restored.
    """
    original = _build_llm_call()
    payload: dict[str, Any] = original.model_dump(mode="json", exclude={"sequence_number"})

    row_sequence = 12345
    merged = {**payload, "sequence_number": row_sequence}
    reconstructed = AuditEventAdapter.validate_python(merged)

    assert isinstance(reconstructed, LLMCallEvent)
    assert reconstructed.sequence_number == row_sequence
    assert reconstructed.event_id == original.event_id
