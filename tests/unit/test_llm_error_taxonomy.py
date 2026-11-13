"""Check LLM error classes, retry declarations, and typed Anthropic error mappings.

HTTP 408/409/429/5xx errors may retry; structured refusal is terminal."""

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
    LLMPricingContractError,
    LLMPricingMissingError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
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
        LLMRefusalError,
        LLMMissingAPIKeyError,
        LLMPersisterNotWiredError,
        LLMPersisterError,
        LLMPricingMissingError,
        LLMPricingContractError,
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
        (LLMConflictError, "node"),  # HTTP 409 is retry-eligible.
        (LLMUpstreamError, "node"),
        (LLMAuthError, "none"),
        (LLMInvalidRequestError, "none"),
        (LLMInvalidResponseError, "none"),
        (LLMUnexpectedContentBlocksError, "none"),
        (LLMRefusalError, "none"),
        (LLMMissingAPIKeyError, "none"),
        (LLMPersisterNotWiredError, "none"),
        (LLMPersisterError, "none"),
        (LLMPricingMissingError, "none"),
        # openai-native-host spec: post-completion pricing-contract deviation —
        # persisted first, then raised; terminal so retry can't bill a second call.
        (LLMPricingContractError, "none"),
    ],
)
def test_retry_at_layer_per_subclass(cls: type[LLMProviderError], expected_layer: str) -> None:
    """Each concrete subclass sets the documented `retry_at_layer`."""
    assert cls.retry_at_layer == expected_layer


def test_provider_error_docstring_names_every_node_layer_class() -> None:
    """Keep documented retry semantics complete as the taxonomy grows (FUP-025).

    LLMProviderError.__doc__ must name every concrete subclass whose
    retry_at_layer is "node".
    """
    from outrider.llm.base import LLMProviderError

    # Compute the actual set of "node"-layer classes by walking the
    # subclass tree (covers both direct subclasses and any future
    # sub-subclasses that explicitly set `retry_at_layer="node"`).
    def _all_subclasses(cls: type) -> set[type]:
        result: set[type] = set()
        for sub in cls.__subclasses__():
            result.add(sub)
            result.update(_all_subclasses(sub))
        return result

    # Rejected class definitions may remain in __subclasses__ until garbage
    # collection and lack retry_at_layer. Read defensively so test order and
    # collection timing cannot break this walk.
    node_layer_classes = {
        cls
        for cls in _all_subclasses(LLMProviderError)
        if getattr(cls, "retry_at_layer", None) == "node"
    }
    assert node_layer_classes, "fixture sanity: at least one node-layer class must exist"

    doc = LLMProviderError.__doc__
    assert doc is not None, "LLMProviderError must have a docstring"
    for cls in node_layer_classes:
        assert cls.__name__ in doc, (
            f"LLMProviderError.__doc__ does not name {cls.__name__!r} (a "
            f"`retry_at_layer=node` class). The docstring's retry-semantics "
            f"section must enumerate every node-layer class. Add the name "
            f"to the bullet under `retry_at_layer semantics: - 'node': ...`."
        )


# ---------------------------------------------------------------------------
# Hierarchy + LLMUnknownError.
# ---------------------------------------------------------------------------


def test_all_subclasses_inherit_from_provider_error() -> None:
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
        LLMRefusalError,
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


def test_refusal_error_carries_structured_category() -> None:
    """A refusal (stop_reason="refusal") carries the `stop_details` category
    (e.g. "cyber") for structured caller inspection; terminal (`retry_at_layer`
    "none") — retrying the same prompt won't help."""
    err = LLMRefusalError("model declined", category="cyber")
    assert err.category == "cyber"
    assert err.retry_at_layer == "none"


def test_refusal_error_default_category_is_none() -> None:
    """Backward-compat: positional message works; `category` defaults to None
    (a pre-output refusal may carry no `stop_details`)."""
    err = LLMRefusalError("model declined")
    assert err.category is None
