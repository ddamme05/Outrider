"""Protocol shape + schema-config invariants.

Covers:
  - `LLMProvider` Protocol shape (no agent-state in signatures)
  - `LLMExchangePersister` Protocol shape (async)
  - Every Pydantic model in `llm/` has `extra="forbid"` AND `frozen=True`
  - `LLMResponse` has NO `severity`/`evidence_tier`/`confidence`/`cost_usd`
  - `LLMMessage.role` is `Literal["user", "assistant"]` (NOT `"system"`)
  - `_canonical_prompt_hash` AC#15 — pinned hex digest for known input
  - `_canonical_system_prompt_hash` separate cache lifecycle helper
"""

from __future__ import annotations

import inspect
from typing import Protocol, get_type_hints
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.llm.base import (
    LLMExchangePersister,
    LLMMessage,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.config import ModelConfig

# ---------------------------------------------------------------------------
# Protocol shape.
# ---------------------------------------------------------------------------


def test_llm_provider_is_protocol() -> None:
    assert issubclass(LLMProvider, Protocol)


def test_llm_provider_has_async_complete_method() -> None:
    assert hasattr(LLMProvider, "complete")
    sig = inspect.signature(LLMProvider.complete)
    # `(self, request: LLMRequest)` after dropping `self`.
    assert list(sig.parameters.keys()) == ["self", "request"]
    # Return annotation should be LLMResponse.
    hints = get_type_hints(LLMProvider.complete)
    assert hints.get("return") is LLMResponse


def test_llm_provider_complete_does_not_mention_review_state() -> None:
    """Per `state-is-pure-data` and the transport-only boundary, the
    Protocol must not mention agent state."""
    sig = inspect.signature(LLMProvider.complete)
    annotations = {p.annotation for p in sig.parameters.values()}
    annotations.add(sig.return_annotation)
    text = " ".join(repr(a) for a in annotations)
    assert "ReviewState" not in text
    assert "outrider.agent" not in text


def test_llm_exchange_persister_is_protocol() -> None:
    assert issubclass(LLMExchangePersister, Protocol)


def test_llm_exchange_persister_persist_is_async() -> None:
    """The persister's `persist` must be async — `complete()` is async and
    sync persist would block the event loop on DB I/O."""
    assert hasattr(LLMExchangePersister, "persist")
    method = LLMExchangePersister.persist
    assert inspect.iscoroutinefunction(method) or asyncio_protocol_check(method)


def asyncio_protocol_check(method: object) -> bool:
    """Best-effort: Protocol methods may register as functions; the
    annotation `async def` is what matters in practice."""
    src = inspect.getsource(method)  # type: ignore[arg-type]
    return src.lstrip().startswith("async def")


# ---------------------------------------------------------------------------
# Pydantic config — every model in llm/ has frozen=True + extra="forbid".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [LLMRequest, LLMResponse, LLMMessage])
def test_pydantic_models_are_frozen(cls: type) -> None:
    assert cls.model_config.get("frozen") is True


@pytest.mark.parametrize("cls", [LLMRequest, LLMResponse, LLMMessage])
def test_pydantic_models_forbid_extra(cls: type) -> None:
    assert cls.model_config.get("extra") == "forbid"


def test_model_config_is_frozen_and_forbids_extra() -> None:
    """`ModelConfig` is `BaseSettings`, not `BaseModel`, but the same
    discipline applies via `SettingsConfigDict`."""
    assert ModelConfig.model_config.get("frozen") is True
    assert ModelConfig.model_config.get("extra") == "forbid"


def test_pydantic_models_reject_unknown_fields() -> None:
    """`extra="forbid"` rejects unknown keys at construction."""
    with pytest.raises(ValidationError):
        LLMMessage(role="user", content="hello", unknown_field="x")  # type: ignore[call-arg]


def test_pydantic_models_are_immutable_post_construction() -> None:
    """`frozen=True` blocks attribute reassignment."""
    msg = LLMMessage(role="user", content="hello")
    with pytest.raises(ValidationError):
        msg.role = "assistant"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AC#7 — LLMResponse has no model-set proof-boundary fields.
# ---------------------------------------------------------------------------


def test_llm_response_has_no_severity_field() -> None:
    assert "severity" not in LLMResponse.model_fields


def test_llm_response_has_no_evidence_tier_field() -> None:
    assert "evidence_tier" not in LLMResponse.model_fields


def test_llm_response_has_no_confidence_field() -> None:
    assert "confidence" not in LLMResponse.model_fields


def test_llm_response_has_no_cost_usd_field() -> None:
    """Cost is computed by the provider at step 8 from token counts ×
    pricing table; lands on `LLMCallEvent`, NOT on the wrapper response."""
    assert "cost_usd" not in LLMResponse.model_fields


def test_llm_response_construction_with_severity_field_raises() -> None:
    """`extra="forbid"` rejects a model-set severity attempt at construction."""
    with pytest.raises(ValidationError):
        LLMResponse(  # type: ignore[call-arg]
            text="x",
            model="claude-sonnet-4-6",
            input_tokens=1,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=1,
            severity="HIGH",
        )


# ---------------------------------------------------------------------------
# LLMMessage.role narrowing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_llm_message_admits_canonical_roles(role: str) -> None:
    msg = LLMMessage(role=role, content="hi")  # type: ignore[arg-type]
    assert msg.role == role


def test_llm_message_rejects_system_role() -> None:
    """System content goes via `LLMRequest.system_prompt`, not as a message
    role — Anthropic's `MessageParam.role` doesn't accept it."""
    with pytest.raises(ValidationError):
        LLMMessage(role="system", content="hi")  # type: ignore[arg-type]


def test_llm_message_rejects_arbitrary_role() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role="developer", content="hi")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC#15 — `_canonical_prompt_hash` pinned digest.
# ---------------------------------------------------------------------------


def test_canonical_prompt_hash_pinned_digest() -> None:
    """Drift in the canonicalization function silently breaks replay
    equivalence. Pinning a known (system, user) pair to a known hex digest
    fails loud if any future change introduces normalization, trimming, or
    delimiter drift.
    """
    sys_prompt = "You are a code reviewer."
    user_prompt = "Review this PR."
    digest = _canonical_prompt_hash(sys_prompt, user_prompt)
    expected = "06a3f1c3ed69f279e780742b5762f8609818741d0c3408beb61c80111d2c7709"
    # If this digest changes, EITHER the canonicalization changed (which
    # breaks replay — bug), OR you intentionally re-pinned and need to
    # update the expected digest with a same-commit comment justifying
    # why replay-equivalence guarantees are intact.
    assert digest == expected, (
        f"Canonicalization drifted: got {digest}, expected {expected}. "
        f"This breaks replay equivalence. Investigate before re-pinning."
    )


def test_canonical_prompt_hash_is_deterministic() -> None:
    """Same inputs → same digest, every time."""
    digest_a = _canonical_prompt_hash("sys", "user")
    digest_b = _canonical_prompt_hash("sys", "user")
    assert digest_a == digest_b


def test_canonical_prompt_hash_distinguishes_system_vs_user() -> None:
    """Delimiter prevents trivial collision between
    ('a', 'bc') vs ('ab', 'c')."""
    digest_a = _canonical_prompt_hash("a", "bc")
    digest_b = _canonical_prompt_hash("ab", "c")
    assert digest_a != digest_b


def test_canonical_prompt_hash_no_unicode_normalization() -> None:
    """NFC and NFD forms of the same Unicode produce DIFFERENT hashes — the
    canonicalization explicitly does not normalize. Using explicit escape
    sequences so editor / file-system normalization can't silently make
    the two strings equivalent."""
    nfc = "café"  # composed: e-acute as single codepoint U+00E9
    nfd = "café"  # decomposed: 'e' + combining acute U+0301
    # Verify they really are different byte sequences:
    assert nfc.encode("utf-8") != nfd.encode("utf-8")
    digest_a = _canonical_prompt_hash(nfc, "user")
    digest_b = _canonical_prompt_hash(nfd, "user")
    assert digest_a != digest_b


def test_canonical_system_prompt_hash_is_separate() -> None:
    """`system_prompt` has its own cache lifecycle — separate hash."""
    sys_prompt = "You are a code reviewer."
    user_prompt = "Review this PR."
    full_hash = _canonical_prompt_hash(sys_prompt, user_prompt)
    sys_hash = _canonical_system_prompt_hash(sys_prompt)
    assert full_hash != sys_hash


def test_canonical_system_prompt_hash_pinned_digest() -> None:
    sys_prompt = "You are a code reviewer."
    digest = _canonical_system_prompt_hash(sys_prompt)
    expected = "e9ad20798df06f6c732e8f3643b9905197825dd57468d00197d78ded56c2bcca"
    assert digest == expected, (
        f"system-prompt canonicalization drifted: got {digest}, expected {expected}."
    )


# ---------------------------------------------------------------------------
# Sanity: a well-formed LLMRequest constructs cleanly.
# ---------------------------------------------------------------------------


def test_llm_request_minimal_well_formed_construction() -> None:
    """Smoke test: construct a triage-tier request with the minimum fields."""
    req = LLMRequest(
        system_prompt="You are a triage classifier.",
        user_prompt="Classify this PR.",
        model="claude-haiku-4-5",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",
        prompt_template_version="triage@1.0.0",
        degraded_mode=False,
    )
    assert req.node_id == "triage"
    assert req.is_eval is False
    assert req.context_summary == ()
    assert req.messages is None
    assert req.cache_control is True  # round-20 default per DECISIONS#013 point 4


def test_llm_response_minimal_well_formed_construction() -> None:
    resp = LLMResponse(
        text="ok",
        model="claude-haiku-4-5",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=200,
    )
    assert resp.text == "ok"
    assert resp.finish_reason == "end_turn"
