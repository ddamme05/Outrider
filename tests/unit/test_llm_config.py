"""ModelConfig env reading + validators.

Covers AC#2: env-prefix reading, regex validation, deprecated-model
rejection. Per spec §4.2.
"""

from __future__ import annotations

import os
from collections.abc import Iterator  # noqa: TC003 — used at runtime by `@contextmanager`
from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from outrider.llm.config import (
    ModelConfig,
    is_anthropic_family_model,
    model_uses_adaptive_thinking,
)


@contextmanager
def _env(**kvs: str) -> Iterator[None]:
    """Set env vars for the duration of the block; restore on exit.

    Important: we set in os.environ AND in the process env so
    BaseSettings (which reads at construction) sees them.
    """
    sentinel = object()
    saved: dict[str, str | object] = {}
    try:
        for k, v in kvs.items():
            saved[k] = os.environ.get(k, sentinel)
            os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is sentinel:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Defaults match canonical spec §4.2.
# ---------------------------------------------------------------------------


def test_defaults_match_canonical_spec() -> None:
    cfg = ModelConfig()
    assert cfg.triage_model == "claude-haiku-4-5"
    assert cfg.analyze_model == "claude-sonnet-5"
    # STANDARD-tier analyze defaults to Haiku (the eval-gated cost flip, DECISIONS.md#041);
    # DEEP-tier files stay on analyze_model (Sonnet).
    assert cfg.standard_analyze_model == "claude-haiku-4-5"
    # Synthesize defaults to Haiku (DECISIONS.md#043 — one bounded summary call
    # per review; findings/severity/dedup are not model-dependent).
    assert cfg.synthesize_model == "claude-haiku-4-5"
    assert cfg.trace_model == "claude-haiku-4-5"


def test_standard_analyze_model_env_override_and_validated() -> None:
    """`standard_analyze_model` reads its own env var and runs the shared
    `_validate_model_string` (the eval-gated flip is an env/default change, not a code
    edit — `model-strings-from-config-not-hardcoded`)."""
    # Override to Sonnet (NOT the Haiku default per #041) so this proves the env actually
    # changes the value — i.e. the rollback path `=claude-sonnet-4-6` works, not just that
    # the value happens to equal the new default.
    with _env(OUTRIDER_MODEL_STANDARD_ANALYZE_MODEL="claude-sonnet-4-6"):
        assert ModelConfig().standard_analyze_model == "claude-sonnet-4-6"
    with pytest.raises(ValidationError):
        ModelConfig(standard_analyze_model="not-a-real-model")


# ---------------------------------------------------------------------------
# Env-prefix reading.
# ---------------------------------------------------------------------------


def test_reads_env_prefix() -> None:
    with _env(
        OUTRIDER_MODEL_TRIAGE_MODEL="claude-haiku-4-5",
        OUTRIDER_MODEL_ANALYZE_MODEL="claude-opus-4-1",
    ):
        cfg = ModelConfig()
        assert cfg.triage_model == "claude-haiku-4-5"
        assert cfg.analyze_model == "claude-opus-4-1"


def test_each_field_independently_overridable() -> None:
    with _env(OUTRIDER_MODEL_TRACE_MODEL="claude-haiku-4-5"):
        cfg = ModelConfig()
        # Other fields keep defaults
        assert cfg.triage_model == "claude-haiku-4-5"
        assert cfg.analyze_model == "claude-sonnet-5"
        assert cfg.trace_model == "claude-haiku-4-5"


# ---------------------------------------------------------------------------
# Validators — regex pattern.
# ---------------------------------------------------------------------------


def test_rejects_non_anthropic_family_string() -> None:
    """V1 wrapper is Anthropic-only; an OpenAI model name configured by
    env var is a real-world bug worth catching at startup."""
    with pytest.raises(ValidationError, match="V1 Anthropic family"):
        ModelConfig(triage_model="gpt-4")


def test_rejects_arbitrary_string() -> None:
    with pytest.raises(ValidationError, match="V1 Anthropic family"):
        ModelConfig(triage_model="hunter2")


def test_rejects_empty_string() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(triage_model="")


def test_rejects_almost_valid_string() -> None:
    """Common typo: missing the major-version suffix."""
    with pytest.raises(ValidationError, match="V1 Anthropic family"):
        ModelConfig(triage_model="claude-haiku")


@pytest.mark.parametrize(
    "model",
    [
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-5",  # without minor
        # Round-21 fold per Codex finding: SDK 0.100 catalog publishes
        # dated "exact pin" forms (e.g., claude-haiku-4-5-20251001)
        # alongside the undated alias. Regex must accept both shapes
        # so operators pinning to a specific build via OUTRIDER_MODEL_*
        # env vars don't get rejected at construction.
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6-20251015",
        "claude-opus-4-7-20251020",
    ],
)
def test_admits_anthropic_family_strings(model: str) -> None:
    cfg = ModelConfig(triage_model=model)
    assert cfg.triage_model == model


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        ("claude-haiku-4-5", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-7-20251001", True),
        ("zai-org/GLM-5.2", False),
        ("accounts/fireworks/models/glm-5p2", False),
        ("stub-model", False),
    ],
)
def test_is_anthropic_family_model(slug: str, expected: bool) -> None:
    """The public predicate build_graph uses to flag a non-anthropic model_config that
    must carry the host-identity triad (FUP-194). Mirrors `_VALID_MODEL_PATTERN`."""
    assert is_anthropic_family_model(slug) is expected


@pytest.mark.parametrize(
    ("slug", "expected"),
    [
        # Adaptive-thinking generation: Sonnet 5+, Opus 4.7+.
        ("claude-sonnet-5", True),
        ("claude-sonnet-5-20260615", True),  # dated pin, no minor
        ("claude-opus-4-7", True),
        ("claude-opus-4-8", True),
        ("claude-opus-4-7-20251020", True),
        # Current generation: accepts temperature, thinking off by default.
        ("claude-sonnet-4-6", False),
        ("claude-haiku-4-5", False),
        ("claude-opus-4-6", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-haiku-5", False),  # no next-gen Haiku yet
        # Non-anthropic / malformed → legacy shape (False).
        ("zai-org/GLM-5.2", False),
        ("stub-model", False),
    ],
)
def test_model_uses_adaptive_thinking(slug: str, expected: bool) -> None:
    """The generation predicate `AnthropicProvider._build_sdk_kwargs` reads to
    shape the request: omit sampling params + disable adaptive thinking for
    Sonnet 5+ / Opus 4.7+; keep the legacy shape (temperature, no `thinking`
    kwarg) for current-gen and non-anthropic models."""
    assert model_uses_adaptive_thinking(slug) is expected


def test_rejects_dated_form_with_wrong_date_length() -> None:
    """Dated form requires exactly 8 digits (YYYYMMDD); other digit
    lengths are rejected so a typo like `-2025` (4 digits) is caught."""
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="V1 Anthropic family"):
        ModelConfig(triage_model="claude-haiku-4-5-2025")
    with pytest.raises(pydantic.ValidationError, match="V1 Anthropic family"):
        ModelConfig(triage_model="claude-haiku-4-5-202510")


# ---------------------------------------------------------------------------
# Validators — DEPRECATED_MODELS rejection.
# ---------------------------------------------------------------------------


def test_rejects_deprecated_model() -> None:
    """SDK's `DEPRECATED_MODELS` constant lists models Anthropic has
    deprecated. Construction should fail at startup, not at first call."""
    from anthropic.resources.messages import DEPRECATED_MODELS

    # Reuse `ModelConfig`'s own canonical regex for the filter so the
    # test never drifts from what the validator actually accepts (Codex
    # follow-on per Copilot review). The earlier inline regex omitted
    # the optional dated `-YYYYMMDD` suffix that `_VALID_MODEL_PATTERN`
    # accepts (round-21 widening), which would have caused this test to
    # skip in SDK versions where Anthropic ships deprecated models only
    # in dated form. Importing the underscore-prefixed module-level
    # constant is a deliberate test-side breach of convention to lock
    # the test to the validator's source of truth.
    from outrider.llm.config import _VALID_MODEL_PATTERN

    if not DEPRECATED_MODELS:
        pytest.skip("DEPRECATED_MODELS is empty in this SDK version")

    # Filter using the canonical regex itself. Older deprecated models
    # like `claude-2.0` predate the family-suffix shape and won't match;
    # this test fires only when at least one deprecated entry passes
    # both gates (regex + DEPRECATED_MODELS membership).
    pattern_compatible = [m for m in DEPRECATED_MODELS if _VALID_MODEL_PATTERN.match(m)]
    if not pattern_compatible:
        pytest.skip(
            "no DEPRECATED_MODELS entries match _VALID_MODEL_PATTERN; "
            "the deprecation-rejection path can't fire on this SDK"
        )

    target = pattern_compatible[0]
    with pytest.raises(ValidationError, match="deprecated by Anthropic"):
        ModelConfig(triage_model=target)


# ---------------------------------------------------------------------------
# frozen=True discipline.
# ---------------------------------------------------------------------------


def test_config_is_frozen() -> None:
    cfg = ModelConfig()
    with pytest.raises(ValidationError):
        cfg.triage_model = "claude-sonnet-4-6"  # type: ignore[misc]


def test_config_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        ModelConfig(unknown_field="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Host-aware selection — ModelConfig.for_host (DECISIONS.md#056).
# ---------------------------------------------------------------------------


def test_for_host_anthropic_matches_default_construction() -> None:
    """`for_host("anthropic")` is byte-identical to `ModelConfig()`: same per-node
    defaults, same claude-family validation path."""
    assert ModelConfig.for_host("anthropic").model_dump() == ModelConfig().model_dump()


def test_for_host_baseten_uses_glm_slugs() -> None:
    cfg = ModelConfig.for_host("baseten")
    assert cfg.analyze_model == "zai-org/GLM-5.2"
    assert {
        cfg.triage_model,
        cfg.analyze_model,
        cfg.standard_analyze_model,
        cfg.synthesize_model,
        cfg.trace_model,
        cfg.patch_model,
    } == {"zai-org/GLM-5.2"}


def test_for_host_baseten_skips_claude_validation() -> None:
    """The GLM slug would fail ModelConfig's claude-family field validator; `for_host`
    must NOT run it for a non-anthropic host (the provider validates the slug)."""
    with pytest.raises(ValidationError):
        ModelConfig(analyze_model="zai-org/GLM-5.2")  # claude validator rejects it
    # for_host("baseten") holds the same slug without raising.
    assert ModelConfig.for_host("baseten").analyze_model == "zai-org/GLM-5.2"


def test_for_host_env_override_wins_over_host_default() -> None:
    """`OUTRIDER_MODEL_*` set by the operator beats the host default, both branches."""
    with _env(OUTRIDER_MODEL_TRIAGE_MODEL="claude-sonnet-4-6"):
        assert ModelConfig.for_host("anthropic").triage_model == "claude-sonnet-4-6"
    # On a non-anthropic host the override is taken verbatim (no claude validation),
    # so a GLM-shaped override survives.
    with _env(OUTRIDER_MODEL_TRIAGE_MODEL="zai-org/GLM-4.6"):
        assert ModelConfig.for_host("baseten").triage_model == "zai-org/GLM-4.6"


def test_for_host_unknown_host_raises() -> None:
    with pytest.raises(ValueError, match="unknown OUTRIDER_LLM_HOST 'deepinfra'"):
        ModelConfig.for_host("deepinfra")


def test_for_host_field_lists_stay_in_lockstep() -> None:
    """`for_host` indexes `_EnvModelOverrides` and `HOST_DEFAULT_MODELS[host]` by
    `ModelConfig.model_fields`; a field added to one but not the others would crash at
    `for_host()` (AttributeError / KeyError). Pin set-equality so the gap fails here instead."""
    from outrider.llm.config import _EnvModelOverrides
    from outrider.llm.host_profiles import HOST_DEFAULT_MODELS

    model_fields = set(ModelConfig.model_fields)
    assert set(_EnvModelOverrides.model_fields) == model_fields
    for host, defaults in HOST_DEFAULT_MODELS.items():
        assert set(defaults) == model_fields, f"{host} default-model keys drifted from ModelConfig"
