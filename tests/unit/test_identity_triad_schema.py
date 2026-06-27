"""Identity-triad schema slots (DECISIONS.md#056, arc 1a step 4).

`profile_id` / `reasoning_enabled` / `profile_contract_digest` are additive + nullable on
`LLMResponse` and the three completion events. The write-time fail-closed validator + the
provider/graph stamping land in later step-4 commits; this pins the schema slots (default
`None` = the UNQUALIFIED pre-#056 state, read-tolerant per DECISIONS#032) + the digest
sha256 pattern on the events.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AuditEventBase,
    LLMCallEvent,
    SynthesizeCompletedEvent,
)
from outrider.llm.base import LLMResponse

_TRIAD = ("profile_id", "reasoning_enabled", "profile_contract_digest")
_COMPLETION_EVENTS: tuple[type[AuditEventBase], ...] = (
    LLMCallEvent,
    AnalyzeCompletedEvent,
    SynthesizeCompletedEvent,
)


def _llm_response(**overrides: Any) -> LLMResponse:
    base: dict[str, Any] = {
        "text": "x",
        "model": "m",
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "finish_reason": "end_turn",
        "latency_ms": 1,
    }
    return LLMResponse(**(base | overrides))


def _llm_call_event(**overrides: Any) -> LLMCallEvent:
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "timestamp": datetime.now(UTC),
        "model": "m",
        "node_id": "analyze",
        "input_tokens": 1,
        "output_tokens": 1,
        "cached_tokens": 0,
        "cost_usd": 0.0,
        "pricing_version": "v3",
        "latency_ms": 1,
        "prompt_hash": "a" * 64,
        "cache_hit": False,
        "context_summary": (),
        "prompt_template_version": "analyze@1.0.0",
        "system_prompt_hash": "b" * 64,
        "degraded_mode": False,
    }
    return LLMCallEvent(**(base | overrides))


def test_llm_response_triad_defaults_none_and_accepts_values() -> None:
    r = _llm_response()
    assert (r.profile_id, r.reasoning_enabled, r.profile_contract_digest) == (None, None, None)
    r2 = _llm_response(
        profile_id="baseten", reasoning_enabled=False, profile_contract_digest="c" * 64
    )
    assert r2.profile_id == "baseten"
    assert r2.reasoning_enabled is False
    assert r2.profile_contract_digest == "c" * 64


def test_llm_call_event_triad_defaults_none_and_accepts_values() -> None:
    e = _llm_call_event()
    assert (e.profile_id, e.reasoning_enabled, e.profile_contract_digest) == (None, None, None)
    e2 = _llm_call_event(
        profile_id="anthropic", reasoning_enabled=True, profile_contract_digest="d" * 64
    )
    assert e2.profile_id == "anthropic"
    assert e2.reasoning_enabled is True


def test_llm_call_event_digest_rejects_non_sha256() -> None:
    with pytest.raises(ValidationError):
        _llm_call_event(profile_contract_digest="not-a-sha256")


@pytest.mark.parametrize("event_cls", _COMPLETION_EVENTS)
def test_completion_events_carry_nullable_triad_slots(event_cls: type[AuditEventBase]) -> None:
    """All three completion events declare the triad, each defaulting None — the
    UNQUALIFIED pre-#056 state (read-tolerant per DECISIONS#032)."""
    for field in _TRIAD:
        assert field in event_cls.model_fields, f"{event_cls.__name__} missing {field}"
        assert event_cls.model_fields[field].default is None
