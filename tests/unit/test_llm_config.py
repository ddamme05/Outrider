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

from outrider.llm.config import ModelConfig


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
    assert cfg.analyze_model == "claude-sonnet-4-6"
    # STANDARD-tier analyze defaults to Haiku (the eval-gated cost flip, DECISIONS.md#041);
    # DEEP-tier files stay on analyze_model (Sonnet).
    assert cfg.standard_analyze_model == "claude-haiku-4-5"
    assert cfg.synthesize_model == "claude-sonnet-4-6"
    assert cfg.trace_model == "claude-haiku-4-5"


def test_standard_analyze_model_env_override_and_validated() -> None:
    """`standard_analyze_model` reads its own env var and runs the shared
    `_validate_model_string` (the eval-gated flip is an env/default change, not a code
    edit — `model-strings-from-config-not-hardcoded`)."""
    with _env(OUTRIDER_MODEL_STANDARD_ANALYZE_MODEL="claude-haiku-4-5"):
        assert ModelConfig().standard_analyze_model == "claude-haiku-4-5"
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
        assert cfg.analyze_model == "claude-sonnet-4-6"
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
