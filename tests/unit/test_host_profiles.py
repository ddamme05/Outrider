"""Unit tests for `llm/host_profiles.py` (DECISIONS.md#056, arc 1a step 2).

Pure — no SDK, no network. Covers: BASETEN_PROFILE reproduces the spike constants; the
reasoning shapers; `read_usage` §8a fork (incl. the `unverified` fail-loud); the
`profile_contract_digest`; slug validation; `HOST_DEFAULT_MODELS`; resolution.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.llm.base import LLMInvalidResponseError
from outrider.llm.host_profiles import (
    BASETEN_PROFILE,
    FIREWORKS_PROFILE,
    HOST_DEFAULT_MODELS,
    HOST_PROFILES,
    OPENAI_PROFILE,
    HostPrivacy,
    JsonMode,
    ReasoningMechanism,
    TokenAccounting,
    TokenLimitParam,
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
    # The GLM hosts' verified wire takes `max_tokens` — the SHAPER v3 default holds.
    assert p.token_limit_param is TokenLimitParam.MAX_TOKENS


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
    ) == (70, 30, 0, 50)


def test_read_usage_caps_cached_at_prompt_tokens() -> None:
    # A malformed cached > prompt can't drive input negative or break input+cache==prompt.
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=130,
        completion_tokens=50,
        accounting=TokenAccounting.PROMPT_INCLUDES_CACHED,
    ) == (0, 100, 0, 50)


def test_read_usage_prompt_excludes_cached_passes_through() -> None:
    assert read_usage(
        prompt_tokens=100,
        raw_cached_tokens=30,
        completion_tokens=50,
        accounting=TokenAccounting.PROMPT_EXCLUDES_CACHED,
    ) == (100, 30, 0, 50)


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
    ) == (100, 0, 0, 50)


def test_profile_contract_digest_is_deterministic_and_rotates_on_wire_change() -> None:
    base = BASETEN_PROFILE.profile_contract_digest
    assert base == BASETEN_PROFILE.profile_contract_digest  # deterministic
    assert len(base) == 64  # sha256 hex
    # A wire-affecting change rotates it (so a warm cache invalidates).
    moved = BASETEN_PROFILE.model_copy(
        update={"reasoning_mechanism": ReasoningMechanism.THINKING_DISABLED}
    )
    assert moved.profile_contract_digest != base
    # The token-ceiling kwarg name is wire-affecting (SHAPER v3): flipping it rotates.
    relimit = BASETEN_PROFILE.model_copy(
        update={"token_limit_param": TokenLimitParam.MAX_COMPLETION_TOKENS}
    )
    assert relimit.profile_contract_digest != base
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
        "analyze_model": "claude-sonnet-5",
        "standard_analyze_model": "claude-haiku-4-5",
        "synthesize_model": "claude-haiku-4-5",
        "trace_model": "claude-haiku-4-5",
        "patch_model": "claude-haiku-4-5",
    }
    assert set(HOST_DEFAULT_MODELS["baseten"].values()) == {"zai-org/GLM-5.2"}
    assert set(HOST_DEFAULT_MODELS["fireworks"].values()) == {"accounts/fireworks/models/glm-5p2"}
    assert HOST_DEFAULT_MODELS["anthropic"].keys() == HOST_DEFAULT_MODELS["baseten"].keys()
    assert HOST_DEFAULT_MODELS["fireworks"].keys() == HOST_DEFAULT_MODELS["baseten"].keys()


def test_resolve_host_profile() -> None:
    assert resolve_host_profile("baseten") is BASETEN_PROFILE
    assert resolve_host_profile("fireworks") is FIREWORKS_PROFILE
    assert resolve_host_profile("openai") is OPENAI_PROFILE
    assert set(HOST_PROFILES) == {"baseten", "fireworks", "openai"}
    with pytest.raises(ValueError, match="unknown OpenAI-compatible host 'deepinfra'"):
        resolve_host_profile("deepinfra")


def test_fireworks_profile_is_wire_verified() -> None:
    """The Fireworks profile encodes the 2026-07-06 captured wire contract
    (DECISIONS.md#056 amendment): strict schema (raw, no adapter), OpenAI cached-token
    accounting, reasoning_effort='none', and the no-training / open-model-ZDR privacy."""
    p = FIREWORKS_PROFILE
    assert p.host_id == "fireworks"
    assert p.base_url == "https://api.fireworks.ai/inference/v1"
    assert p.api_key_env == "FIREWORKS_API_KEY"
    assert p.json_mode is JsonMode.STRICT_JSON_SCHEMA  # raw schema accepted, no adapter
    assert p.token_accounting is TokenAccounting.PROMPT_INCLUDES_CACHED
    assert p.reasoning_mechanism is ReasoningMechanism.REASONING_EFFORT_NONE
    assert p.token_limit_param is TokenLimitParam.MAX_TOKENS  # verified GLM wire
    assert not p.reasoning_forced_on  # it HAS an off-switch
    assert p.privacy.trains_on_inputs is False  # hard-fail field: confirmed no-training
    assert p.privacy.egress_host == "api.fireworks.ai"
    assert p.privacy.verified_date == "2026-07-06"
    # The slug pattern admits the real Fireworks GLM path and rejects Baseten's shape.
    p.validate_model_slug("accounts/fireworks/models/glm-5p2")
    with pytest.raises(ValueError, match="does not match host 'fireworks'"):
        p.validate_model_slug("zai-org/GLM-5.2")
    # Its digest differs from Baseten's (cache/replay separation across the two GLM hosts).
    assert p.profile_contract_digest != BASETEN_PROFILE.profile_contract_digest


def test_fireworks_reasoning_off_shapes_reasoning_effort_none() -> None:
    """The wire-verified shaper: extra reasoning_effort='none' key, accepted by Fireworks."""
    kwargs: dict[str, object] = {}
    FIREWORKS_PROFILE.apply_reasoning_off(kwargs)
    assert kwargs == {"reasoning_effort": "none"}


def test_fireworks_usage_maps_cached_as_subset_of_prompt() -> None:
    """PROMPT_INCLUDES_CACHED, grounded in the captured wire (adapted.json:
    prompt_tokens=159, cached_tokens=158): cache_read is the cached subset, input is the
    remainder, and input + cache_read == prompt_tokens."""
    input_tokens, cache_read, _cache_write, output_tokens = read_usage(
        prompt_tokens=159,
        raw_cached_tokens=158,
        completion_tokens=410,
        accounting=FIREWORKS_PROFILE.token_accounting,
    )
    assert (input_tokens, cache_read, output_tokens) == (1, 158, 410)
    assert input_tokens + cache_read == 159


# The EXACT raw response body Fireworks GLM-5.2 returned for the analyze schema on the
# 2026-07-06 paid wire (spikes/fireworks/fixtures/raw.json — gitignored, so this inline
# copy is the SOLE in-repo record; kept byte-verbatim so a future re-capture can diff
# against it). Direct, un-fenced JSON; nullable proof fields
# (query_match_id/trace_path/trace_candidates) OMITTED, not fabricated — the reason the
# profile ships the RAW schema with no adapter.
_FIREWORKS_RAW_WIRE_BODY = (
    '{\n"findings": [\n{\n"finding_type": "sql_injection",\n'
    '"evidence_tier": "judged",\n"title": "SQL Injection via string concatenation",\n'
    '"description": "The code constructs a SQL query by directly concatenating the '
    "`user_id` parameter into the query string. This allows an attacker to inject "
    "arbitrary SQL commands. To fix this, use parameterized queries instead of string "
    'concatenation, e.g., `db.execute(\\"SELECT * FROM users WHERE id = ?\\", '
    '(user_id,))`.",\n'
    '"evidence": "    query = \\"SELECT * FROM users WHERE id = \'\\" + user_id + '
    '\\"\'\\"",\n"line_start": 4,\n"line_end": 4\n}\n]\n}'
)


def test_fireworks_raw_wire_body_parses_and_omits_proof_fields() -> None:
    """Regression on the captured wire (DECISIONS.md#056 amendment): the RAW-schema
    response parses through `AnalyzeResponseRaw` unfenced, and the optional proof fields
    are ABSENT (→ None), not fabricated — the harm the rejected required-completion adapter
    would have introduced (a `query_match_id`/`trace_path` invented on a JUDGED finding)."""
    from outrider.schemas.llm.analyze import AnalyzeResponseRaw

    parsed = AnalyzeResponseRaw.model_validate_json(_FIREWORKS_RAW_WIRE_BODY)
    (finding,) = parsed.findings
    assert finding.evidence_tier == "judged"
    assert finding.query_match_id is None  # omitted by the raw schema, not fabricated
    assert finding.trace_path is None
    assert finding.trace_candidates == ()  # default; the model left it out


def test_reasoning_forced_on_is_true_only_for_none_mechanism() -> None:
    # forced_on is the PROFILE's half of reasoning_enabled (provider OR-combines it with the
    # requested flag): an off-switch host is forced-on False; NONE is forced-on True. Stamping
    # forced_on alone would mis-audit an operator-enabled off-switch run as reasoning-off.
    assert BASETEN_PROFILE.reasoning_forced_on is False
    none_host = BASETEN_PROFILE.model_copy(update={"reasoning_mechanism": ReasoningMechanism.NONE})
    assert none_host.reasoning_forced_on is True


def test_profile_contract_digest_folds_in_shaper_contract_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A shaper-FUNCTION-body change (no DATA field change) must rotate the digest; the
    # version is the only lever for that, so bumping it has to move the digest.
    base = BASETEN_PROFILE.profile_contract_digest
    monkeypatch.setattr("outrider.llm.host_profiles.SHAPER_CONTRACT_VERSION", "v999")
    assert BASETEN_PROFILE.profile_contract_digest != base


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1, "raw_cached_tokens": 0, "completion_tokens": 5},
        {"prompt_tokens": 10, "raw_cached_tokens": -1, "completion_tokens": 5},
        {"prompt_tokens": 10, "raw_cached_tokens": 0, "completion_tokens": -1},
    ],
)
def test_read_usage_rejects_negative_components(usage: dict[str, int]) -> None:
    # A negative usage component would drive a negative token/cost — reject at the boundary.
    with pytest.raises(LLMInvalidResponseError):
        read_usage(accounting=TokenAccounting.PROMPT_INCLUDES_CACHED, **usage)


def _valid_privacy_kwargs() -> dict[str, object]:
    return {
        "egress_host": "inference.baseten.co",
        "model_origin": "zhipu",
        "direct_hosted": True,
        "trains_on_inputs": False,
        "retention": "no storage",
        "source_url": "https://docs.baseten.co/observability/security",
        "verified_date": "2026-06-27",
    }


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("retention", ""),
        ("egress_host", "   "),
        ("source_url", "http://insecure.example"),
        ("source_url", ""),
        ("verified_date", "2026/06/27"),
        ("verified_date", "June 27"),
    ],
)
def test_host_privacy_rejects_malformed_provenance(field: str, bad: str) -> None:
    # #015 publish-or-refuse needs an auditable claim — empty/malformed provenance fails loud.
    kwargs = _valid_privacy_kwargs()
    kwargs[field] = bad
    with pytest.raises(ValidationError):
        HostPrivacy(**kwargs)


# ---------------------------------------------------------------------------
# OpenAI native host (specs/2026-07-18-openai-native-host.md).
# ---------------------------------------------------------------------------


def test_openai_profile_wire_contract() -> None:
    """The profile encodes the mirror-verified 5.6 contract: JSON_OBJECT (strict
    would 400 the partial-required analyze schema), writes-reported accounting,
    top-level reasoning_effort=none, and the four digest-folded behaviors —
    including the wire-captured `max_completion_tokens` requirement (the 5.6
    family 400s on `max_tokens`; paid probe, 13/13 rows)."""
    p = OPENAI_PROFILE
    assert p.host_id == "openai"
    assert p.base_url == "https://api.openai.com/v1"
    assert p.api_key_env == "OPENAI_API_KEY"
    assert p.json_mode is JsonMode.JSON_OBJECT
    assert p.token_accounting is TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED
    assert p.reasoning_mechanism is ReasoningMechanism.REASONING_EFFORT_NONE
    assert p.flat_rate_input_ceiling_tokens == 272_000
    assert p.sends_prompt_cache_key is True
    assert p.requested_service_tier == "default"
    assert p.token_limit_param is TokenLimitParam.MAX_COMPLETION_TOKENS
    assert p.privacy.trains_on_inputs is False
    assert p.reasoning_forced_on is False


def test_openai_slug_pattern_rejects_alias_and_foreign_slugs() -> None:
    """Explicit slugs only — the bare `gpt-5.6` alias routes to Sol server-side
    and would desync the request-side pricing key; each admitted slug is pinned
    individually (per-variant, not a union assertion)."""
    for ok in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
        OPENAI_PROFILE.validate_model_slug(ok)
    for bad in ("gpt-5.6", "gpt-5.5", "gpt-5.6-sol\n", "claude-sonnet-5", "gpt-5.6-mini"):
        with pytest.raises(ValueError):
            OPENAI_PROFILE.validate_model_slug(bad)


def test_openai_default_models_tiered_sol_deep_luna_cheap() -> None:
    assert HOST_DEFAULT_MODELS["openai"] == {
        "triage_model": "gpt-5.6-luna",
        "analyze_model": "gpt-5.6-sol",
        "standard_analyze_model": "gpt-5.6-luna",
        "synthesize_model": "gpt-5.6-luna",
        "trace_model": "gpt-5.6-luna",
        "patch_model": "gpt-5.6-luna",
    }
    assert HOST_DEFAULT_MODELS["openai"].keys() == HOST_DEFAULT_MODELS["anthropic"].keys()


def test_new_profile_fields_are_digest_folded() -> None:
    """Each of the three new fields rotates the contract digest independently —
    they are wire/billing-affecting host DATA, not decoration."""
    base = OPENAI_PROFILE.profile_contract_digest
    for update in (
        {"flat_rate_input_ceiling_tokens": 300_000},
        {"sends_prompt_cache_key": False},
        {"requested_service_tier": None},
    ):
        moved = OPENAI_PROFILE.model_copy(update=update)
        assert moved.profile_contract_digest != base, update


def test_read_usage_writes_reported_arm() -> None:
    """5.6 accounting: reads subtracted (subset, documented); writes carried as
    their own class, NOT subtracted (conservation equation is probe-pinned);
    writes exceeding prompt_tokens is malformed wire."""
    assert read_usage(
        prompt_tokens=2000,
        raw_cached_tokens=1500,
        completion_tokens=300,
        accounting=TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED,
        raw_cache_write_tokens=400,
    ) == (500, 1500, 400, 300)
    with pytest.raises(LLMInvalidResponseError):
        read_usage(
            prompt_tokens=100,
            raw_cached_tokens=0,
            completion_tokens=10,
            accounting=TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED,
            raw_cache_write_tokens=101,
        )
    with pytest.raises(LLMInvalidResponseError):
        read_usage(
            prompt_tokens=100,
            raw_cached_tokens=0,
            completion_tokens=10,
            accounting=TokenAccounting.PROMPT_INCLUDES_CACHED_WRITES_REPORTED,
            raw_cache_write_tokens=-1,
        )


def test_read_usage_unverified_raises_on_write_tokens() -> None:
    """UNVERIFIED never guesses about EITHER cache class."""
    with pytest.raises(LLMInvalidResponseError):
        read_usage(
            prompt_tokens=100,
            raw_cached_tokens=0,
            completion_tokens=10,
            accounting=TokenAccounting.UNVERIFIED,
            raw_cache_write_tokens=5,
        )
