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


def test_llm_provider_declares_exact_method_set() -> None:
    """Protocol surface check — exact membership, not just presence.

    Class-10 (centrally-pinned-contract registration) doctrine: a new
    method on `LLMProvider` (e.g., a V1.5 `stream` for token-streaming
    or `embeddings` for retrieval) must surface here AND at every
    provider implementation + test fixture. Exact-membership check
    fails loudly on silent drift.
    """
    # `aclose` formalized on the Protocol per DECISIONS.md#035 (the lifespan
    # has always called provider.aclose(); the Protocol now declares it so the
    # TracingLLMProvider decorator is a clean drop-in).
    expected = {"complete", "aclose"}
    actual = {name for name in dir(LLMProvider) if not name.startswith("_")}
    assert actual == expected, (
        f"LLMProvider method set drift: missing={expected - actual}, "
        f"extra={actual - expected}. Update this pin AND every provider impl "
        f"(AnthropicProvider, V1.5 OpenAIProvider) if adding a method."
    )


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


def test_llm_exchange_persister_declares_exact_method_set() -> None:
    """Protocol surface check — exact membership, not just presence.

    Class-10 doctrine: a new method on `LLMExchangePersister` (e.g.,
    `persist_retry_attempt` if FUP-025's retry-policy work lands) must
    surface here AND at the durable `AuditPersister` + test doubles.
    """
    expected = {"persist"}
    actual = {name for name in dir(LLMExchangePersister) if not name.startswith("_")}
    assert actual == expected, (
        f"LLMExchangePersister method set drift: missing={expected - actual}, "
        f"extra={actual - expected}."
    )


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

    Re-pinned: the canonicalization switched from a fixed `\\x1e`
    delimiter to length-prefixed encoding to close a collision class
    where `\\x1e`-bearing payloads could move the prompt boundary
    across two distinct (system, user) pairs and share a digest. The
    new digest is the SHA-256 of
    `f"{len(sp)}:".encode() + sp + f"{len(up)}:".encode() + up` for the
    inputs below. Replay equivalence is preserved for events written
    after this re-pin; any historical events still reference the old
    digest under their original recipe and replay through the
    persister/replay layer (which is the canonical replay path, not
    this re-computation).
    """
    sys_prompt = "You are a code reviewer."
    user_prompt = "Review this PR."
    digest = _canonical_prompt_hash(system_prompt=sys_prompt, user_prompt=user_prompt)
    expected = "39bb3fcd4b56c0cf833ce282503c47603ece6f482694932e1d09cd5ef1665834"
    assert digest == expected, (
        f"Canonicalization drifted: got {digest}, expected {expected}. "
        f"If this drift is intentional, re-pin with a justification."
    )


def test_canonical_prompt_hash_rejects_delimiter_smuggle() -> None:
    """Length-prefix encoding resists the collision class where a fixed
    delimiter inside the prompt body could move the boundary across two
    distinct (system, user) pairs that share a digest.

    Under the OLD `\\x1e`-delimiter recipe, these two pairs collided.
    Under the length-prefix recipe, they produce distinct digests.
    """
    digest_a = _canonical_prompt_hash(system_prompt="A\x1eB", user_prompt="C")
    digest_b = _canonical_prompt_hash(system_prompt="A", user_prompt="B\x1eC")
    assert digest_a != digest_b, (
        "Length-prefix encoding must produce distinct digests for "
        "different (system, user) pairs even when their bytes contain "
        "the historical delimiter character."
    )


def test_canonical_prompt_hash_is_deterministic() -> None:
    """Same inputs → same digest, every time."""
    digest_a = _canonical_prompt_hash(system_prompt="sys", user_prompt="user")
    digest_b = _canonical_prompt_hash(system_prompt="sys", user_prompt="user")
    assert digest_a == digest_b


def test_canonical_prompt_hash_distinguishes_system_vs_user() -> None:
    """Delimiter prevents trivial collision between
    ('a', 'bc') vs ('ab', 'c')."""
    digest_a = _canonical_prompt_hash(system_prompt="a", user_prompt="bc")
    digest_b = _canonical_prompt_hash(system_prompt="ab", user_prompt="c")
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
    digest_a = _canonical_prompt_hash(system_prompt=nfc, user_prompt="user")
    digest_b = _canonical_prompt_hash(system_prompt=nfd, user_prompt="user")
    assert digest_a != digest_b


def test_canonical_system_prompt_hash_is_separate() -> None:
    """`system_prompt` has its own cache lifecycle — separate hash."""
    sys_prompt = "You are a code reviewer."
    user_prompt = "Review this PR."
    full_hash = _canonical_prompt_hash(system_prompt=sys_prompt, user_prompt=user_prompt)
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
