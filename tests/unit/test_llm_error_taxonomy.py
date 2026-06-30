"""Exception hierarchy tests — see AC for `LLMProviderError` shape.

Covers:
  - Direct instantiation of `LLMProviderError` raises (abstract-by-construction)
  - `__init_subclass__` enforces `retry_at_layer` presence at class-def
  - `__init_subclass__` enforces `retry_at_layer` value membership
  - All 14 concrete subclasses have correct `retry_at_layer`
    (round-21 added `LLMConflictError` for 409, in the SDK's
    default-retry set alongside 408/429/5xx; the Sonnet 5 migration
    added `LLMRefusalError` for stop_reason="refusal" — terminal)
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
        (LLMRefusalError, "none"),
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


def test_provider_error_docstring_names_every_node_layer_class() -> None:
    """Pins the round-30 codex audit fold: `LLMProviderError.__doc__`'s
    `retry_at_layer` semantics section MUST name every concrete subclass
    where `retry_at_layer == "node"`. Otherwise the doc drifts behind
    the actual taxonomy as new retry-eligible classes are added.

    This is the same class-omission bug pattern FUP-025 has been
    defending against — round-14 caught `LLMConflictError` missing from
    the retry-eligible list when 409 was promoted to `"node"`; round-30
    caught the same omission in this docstring. The test fails-loud the
    moment a new `"node"`-layer class is added without updating the
    docstring.
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

    # Use `getattr(..., None)` not `cls.retry_at_layer` directly: the
    # failing-subclass tests above (`test_subclass_missing_retry_at_layer_*`)
    # define orphan classes inside `pytest.raises` whose definitions raise
    # in `__init_subclass__`. Python registers them in
    # `LLMProviderError.__subclasses__()` via weakref BEFORE
    # `__init_subclass__` runs (registration happens in `type.__new__`;
    # `__init_subclass__` runs in `type.__init__`). The orphans live in
    # the subclass registry until GC, which is non-deterministic across
    # pytest test orderings — locally they may GC before this test runs,
    # in CI they may not. Filter defensively so missing-attribute orphans
    # don't crash the walk; `None != "node"` excludes them naturally.
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


def test_terminal_subclasses_are_none_layer() -> None:
    """Auth/InvalidRequest/InvalidResponse/Unexpected/MissingAPIKey/
    PersisterNotWired/Persister/Unknown/PricingMissing are terminal."""
    terminal = {
        LLMAuthError,
        LLMInvalidRequestError,
        LLMInvalidResponseError,
        LLMUnexpectedContentBlocksError,
        LLMRefusalError,
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
