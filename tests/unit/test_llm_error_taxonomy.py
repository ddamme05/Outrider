"""Exception hierarchy tests — see AC for `LLMProviderError` shape.

Covers:
  - Direct instantiation of `LLMProviderError` raises (abstract-by-construction)
  - `__init_subclass__` enforces `retry_at_layer` presence at class-def
  - `__init_subclass__` enforces `retry_at_layer` value membership
  - All 12 concrete subclasses have correct `retry_at_layer`
  - `LLMUnknownError` exists for the Anthropic-fall-through path
  - Anthropic exception → Outrider mapping table is complete and well-typed
"""

from __future__ import annotations

import pytest

from outrider.llm.base import (
    LLMAuthError,
    LLMConflictError,
    LLMInvalidRequestError,
    LLMInvalidResponseError,
    LLMMissingAPIKeyError,
    LLMPersisterError,
    LLMPersisterNotWiredError,
    LLMPricingMissingError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMUnexpectedContentBlocksError,
    LLMUnknownError,
    LLMUpstreamError,
)

# ---------------------------------------------------------------------------
# Abstract-by-construction enforcement.
# ---------------------------------------------------------------------------


def test_llm_provider_error_is_not_directly_instantiable() -> None:
    """The base class must reject direct instantiation per spec."""
    with pytest.raises(TypeError, match="abstract"):
        LLMProviderError("boom")


def test_concrete_subclasses_are_instantiable() -> None:
    """Every concrete subclass admits direct instantiation."""
    for cls in (
        LLMUnknownError,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMConflictError,
        LLMUpstreamError,
        LLMAuthError,
        LLMInvalidRequestError,
        LLMInvalidResponseError,
        LLMUnexpectedContentBlocksError,
        LLMMissingAPIKeyError,
        LLMPersisterNotWiredError,
        LLMPersisterError,
        LLMPricingMissingError,
    ):
        # Should not raise.
        instance = cls("test message")
        assert isinstance(instance, LLMProviderError)
        assert isinstance(instance, Exception)


# ---------------------------------------------------------------------------
# `__init_subclass__` enforcement (presence + value).
# ---------------------------------------------------------------------------


def test_subclass_missing_retry_at_layer_raises_at_class_definition() -> None:
    """Defining a subclass without `retry_at_layer` fails at class-def time
    (not at first runtime use)."""
    with pytest.raises(TypeError, match="must set retry_at_layer"):
        # Define inside the assertion so the failure fires at class-creation.
        class _ForgotRetryLayer(LLMProviderError):  # noqa: N818
            pass


def test_subclass_with_invalid_retry_at_layer_value_raises() -> None:
    """A subclass setting `retry_at_layer = "invalid"` fails at class-def
    time with a message naming the offending value."""
    with pytest.raises(TypeError, match="not in"):

        class _BadValue(LLMProviderError):  # noqa: N818
            retry_at_layer = "invalid"  # type: ignore[assignment]


def test_subclass_with_typo_retry_at_layer_value_raises() -> None:
    """Common typo: trailing whitespace `"node "` does NOT match `"node"`."""
    with pytest.raises(TypeError, match="not in"):

        class _Typo(LLMProviderError):  # noqa: N818
            retry_at_layer = "node "  # type: ignore[assignment]


def test_subclass_with_wrong_case_retry_at_layer_value_raises() -> None:
    """Common typo: wrong case `"NODE"` does NOT match `"node"`."""
    with pytest.raises(TypeError, match="not in"):

        class _WrongCase(LLMProviderError):  # noqa: N818
            retry_at_layer = "NODE"  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# `retry_at_layer` per-subclass values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,expected_layer",
    [
        (LLMUnknownError, "none"),
        (LLMTimeoutError, "node"),
        (LLMRateLimitError, "node"),
        (LLMConflictError, "node"),  # round-21: 409 in SDK retry set
        (LLMUpstreamError, "node"),
        (LLMAuthError, "none"),
        (LLMInvalidRequestError, "none"),
        (LLMInvalidResponseError, "none"),
        (LLMUnexpectedContentBlocksError, "none"),
        (LLMMissingAPIKeyError, "none"),
        (LLMPersisterNotWiredError, "none"),
        (LLMPersisterError, "none"),
        (LLMPricingMissingError, "none"),
    ],
)
def test_retry_at_layer_per_subclass(cls: type[LLMProviderError], expected_layer: str) -> None:
    """Each concrete subclass sets the documented `retry_at_layer`."""
    assert cls.retry_at_layer == expected_layer


def test_recoverable_subclasses_are_node_layer() -> None:
    """Timeout/RateLimit/Conflict/Upstream are the recoverable cases per
    spec (round-21 fold for 409 ConflictError per SDK retry set);
    all surface as `retry_at_layer="node"` (the calling node retries)."""
    recoverable = {LLMTimeoutError, LLMRateLimitError, LLMConflictError, LLMUpstreamError}
    for cls in recoverable:
        assert cls.retry_at_layer == "node"


def test_terminal_subclasses_are_none_layer() -> None:
    """Auth/InvalidRequest/InvalidResponse/Unexpected/MissingAPIKey/
    PersisterNotWired/Persister/Unknown/PricingMissing are terminal."""
    terminal = {
        LLMAuthError,
        LLMInvalidRequestError,
        LLMInvalidResponseError,
        LLMUnexpectedContentBlocksError,
        LLMMissingAPIKeyError,
        LLMPersisterNotWiredError,
        LLMPersisterError,
        LLMPricingMissingError,
        LLMUnknownError,
    }
    for cls in terminal:
        assert cls.retry_at_layer == "none"


# ---------------------------------------------------------------------------
# Hierarchy + LLMUnknownError.
# ---------------------------------------------------------------------------


def test_all_subclasses_inherit_from_provider_error() -> None:
    for cls in (
        LLMUnknownError,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMUpstreamError,
        LLMAuthError,
        LLMInvalidRequestError,
        LLMInvalidResponseError,
        LLMUnexpectedContentBlocksError,
        LLMMissingAPIKeyError,
        LLMPersisterNotWiredError,
        LLMPersisterError,
        LLMPricingMissingError,
    ):
        assert issubclass(cls, LLMProviderError)


def test_llm_unknown_error_is_concrete() -> None:
    """Fall-through case: unmapped Anthropic APIError → LLMUnknownError.
    The class must be instantiable so the wrapper can raise it."""
    instance = LLMUnknownError("unmapped APIError subclass")
    assert isinstance(instance, LLMProviderError)
    assert instance.retry_at_layer == "none"


# ---------------------------------------------------------------------------
# isinstance discrimination (the whole point of the typed taxonomy).
# ---------------------------------------------------------------------------


def test_caller_can_isinstance_check_for_recoverability() -> None:
    """Downstream node specs read `retry_at_layer` directly; this asserts
    the access pattern works on raised instances (not just classes)."""
    err = LLMTimeoutError("timeout after 30s")
    assert err.retry_at_layer == "node"
    err_terminal = LLMAuthError("401 unauthorized")
    assert err_terminal.retry_at_layer == "none"


# ---------------------------------------------------------------------------
# Structured attrs — round-19 fold per Codex finding (low-confidence,
# fixed for spec-vs-code parity). Spec promises kwargs that the previous
# implementation didn't honor.
# ---------------------------------------------------------------------------


def test_pricing_missing_error_carries_structured_missing_models() -> None:
    err = LLMPricingMissingError(
        "two models missing",
        missing_models=["claude-fake-1", "claude-fake-2"],
    )
    assert err.missing_models == ("claude-fake-1", "claude-fake-2")
    assert isinstance(err.missing_models, tuple)


def test_pricing_missing_error_default_missing_models_empty_tuple() -> None:
    """Backward-compat: positional message still works; `missing_models`
    defaults to an empty tuple."""
    err = LLMPricingMissingError("simple message")
    assert err.missing_models == ()


def test_unexpected_content_blocks_error_carries_structured_block_types() -> None:
    err = LLMUnexpectedContentBlocksError(
        "got two blocks",
        actual_block_types=["ThinkingBlock", "TextBlock"],
    )
    assert err.actual_block_types == ("ThinkingBlock", "TextBlock")
    assert isinstance(err.actual_block_types, tuple)


def test_unexpected_content_blocks_error_default_block_types_empty_tuple() -> None:
    err = LLMUnexpectedContentBlocksError("simple message")
    assert err.actual_block_types == ()
