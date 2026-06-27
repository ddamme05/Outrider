"""Unit tests for `llm/host_profiles.py` (DECISIONS.md#056, arc 1a step 2).

Pure — no SDK, no network. Covers: BASETEN_PROFILE reproduces the spike constants; the
reasoning shapers; `read_usage` §8a fork (incl. the `unverified` fail-loud); the
`profile_contract_digest`; slug validation; `HOST_DEFAULT_MODELS`; resolution.
"""

from __future__ import annotations

import pytest

from outrider.llm.base import LLMInvalidResponseError
from outrider.llm.host_profiles import (
    BASETEN_PROFILE,
    HOST_DEFAULT_MODELS,
    HOST_PROFILES,
    HostPrivacy,
    JsonMode,
    ReasoningMechanism,
    TokenAccounting,
    read_usage,
    resolve_host_profile,
)


def test_baseten_profile_reproduces_the_spike_constants() -> None:
    p = BASETEN_PROFILE
    assert p.host_id == "baseten"
    assert p.base_url == "https://inference.baseten.co/v1"
    assert p.api_key_env == "BASETEN_API_KEY"
    assert p.model_slug_pattern == r"^zai-org/GLM-\d+(\.\d+)?$"
    assert p.json_mode is JsonMode.SOFT_FENCED  # Baseten shared API is soft (FUP-196).
    assert p.token_accounting is TokenAccounting.PROMPT_INCLUDES_CACHED  # §8a.
    assert p.reasoning_mechanism is ReasoningMechanism.CHAT_TEMPLATE_ARGS


def test_baseten_privacy_carries_retention_and_no_training() -> None:
    # #013/#015: the posture is retention + no-training, not just egress (audit-8 #3).
    priv = BASETEN_PROFILE.privacy
    assert priv.trains_on_inputs is False
    assert priv.retention  # non-empty retention statement
    assert priv.source_url.startswith("https://")
    assert priv.verified_date == "2026-06-27"
    assert priv.egress_host == "inference.baseten.co"
    assert priv.model_origin == "zhipu"


def test_slug_validation_accepts_glm_rejects_claude() -> None:
    BASETEN_PROFILE.validate_model_slug("zai-org/GLM-5.2")  # no raise
    with pytest.raises(ValueError, match="does not match host 'baseten'"):
        BASETEN_PROFILE.validate_model_slug("claude-sonnet-4-6")


def test_reasoning_off_chat_template_args_matches_the_spike_shape() -> None:
    kwargs: dict[str, object] = {}
    BASETEN_PROFILE.apply_reasoning_off(kwargs)
    assert kwargs == {"extra_body": {"chat_template_args": {"enable_thinking": False}}}


@pytest.mark.parametrize(
    ("mechanism", "expected"),
    [
        (ReasoningMechanism.REASONING_EFFORT_NONE, {"reasoning_effort": "none"}),
        (
            ReasoningMechanism.REASONING_ENABLED_FALSE,
            {"extra_body": {"reasoning": {"enabled": False}}},
        ),
        (ReasoningMechanism.THINKING_DISABLED, {"extra_body": {"thinking": {"type": "disabled"}}}),
        (ReasoningMechanism.NONE, {}),
    ],
)
def test_reasoning_shapers_each_produce_the_documented_wire_shape(
    mechanism: ReasoningMechanism, expected: dict[str, object]
) -> None:
    profile = BASETEN_PROFILE.model_copy(update={"reasoning_mechanism": mechanism})
    kwargs: dict[str, object] = {}
    profile.apply_reasoning_off(kwargs)
    assert kwargs == expected


def test_read_usage_prompt_includes_cached_subtracts() -> None:
    # §8a: prompt_tokens(100) includes cached(30) -> input=70, cache_read=30.
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=30,
        completion_tokens=50,
        accounting=TokenAccounting.PROMPT_INCLUDES_CACHED,
    ) == (70, 30, 50)


def test_read_usage_caps_cached_at_prompt_tokens() -> None:
    # A malformed cached > prompt can't drive input negative or break input+cache==prompt.
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=130,
        completion_tokens=50,
        accounting=TokenAccounting.PROMPT_INCLUDES_CACHED,
    ) == (0, 100, 50)


def test_read_usage_prompt_excludes_cached_passes_through() -> None:
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=30,
        completion_tokens=50,
        accounting=TokenAccounting.PROMPT_EXCLUDES_CACHED,
    ) == (100, 30, 50)


def test_read_usage_unverified_raises_on_cached_but_passes_on_zero() -> None:
    # unverified never guesses: a real cache hit is a loud failure, not a silent miscost.
    with pytest.raises(LLMInvalidResponseError):
        read_usage(
            prompt_tokens=100,
            raw_cached_tokens=1,
            completion_tokens=50,
            accounting=TokenAccounting.UNVERIFIED,
        )
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=0,
        completion_tokens=50,
        accounting=TokenAccounting.UNVERIFIED,
    ) == (100, 0, 50)


def test_profile_contract_digest_is_deterministic_and_rotates_on_wire_change() -> None:
    base = BASETEN_PROFILE.profile_contract_digest
    assert base == BASETEN_PROFILE.profile_contract_digest  # deterministic
    assert len(base) == 64  # sha256 hex
    # A wire-affecting change rotates it (so a warm cache invalidates).
    moved = BASETEN_PROFILE.model_copy(
        update={"reasoning_mechanism": ReasoningMechanism.THINKING_DISABLED}
    )
    assert moved.profile_contract_digest != base
    # A NON-wire field (privacy provenance) does NOT rotate it.
    reprivacy = BASETEN_PROFILE.model_copy(
        update={
            "privacy": HostPrivacy(
                egress_host="inference.baseten.co",
                model_origin="zhipu",
                direct_hosted=True,
                trains_on_inputs=False,
                retention="different wording",
                source_url="https://docs.baseten.co/observability/security",
                verified_date="2026-06-28",
            )
        }
    )
    assert reprivacy.profile_contract_digest == base


def test_host_default_models_anthropic_matches_canonical_tiers() -> None:
    assert HOST_DEFAULT_MODELS["anthropic"] == {
        "triage_model": "claude-haiku-4-5",
        "analyze_model": "claude-sonnet-4-6",
        "standard_analyze_model": "claude-haiku-4-5",
        "synthesize_model": "claude-haiku-4-5",
        "trace_model": "claude-haiku-4-5",
        "patch_model": "claude-haiku-4-5",
    }
    assert set(HOST_DEFAULT_MODELS["baseten"].values()) == {"zai-org/GLM-5.2"}
    assert HOST_DEFAULT_MODELS["anthropic"].keys() == HOST_DEFAULT_MODELS["baseten"].keys()


def test_resolve_host_profile() -> None:
    assert resolve_host_profile("baseten") is BASETEN_PROFILE
    assert set(HOST_PROFILES) == {"baseten"}  # arc 1a ships only Baseten
    with pytest.raises(ValueError, match="unknown OpenAI-compatible host 'deepinfra'"):
        resolve_host_profile("deepinfra")
