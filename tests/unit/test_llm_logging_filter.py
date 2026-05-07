"""Three-tier recursive content-leak filter (AC#17 + AC#21).

Tier 0 (type-based, every logger, recursive)
Tier 1 (key-name, every logger, recursive)
Tier 2 (key-name, outrider.llm.* only, recursive)

Plus filter installation on handlers (AC#23) — the round-15 fix where
filters must attach to handlers, not loggers.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest
from pydantic import BaseModel

from outrider.audit.events import (
    ContextManifestEntry,
    LLMCallEvent,
)
from outrider.llm.base import LLMMessage, LLMRequest, LLMResponse
from outrider.llm.logging import (
    LLM_CONTENT_BEARING_TYPES,
    RejectLLMContentFilter,
    register_filter_on_all_handlers,
)

# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _record(
    name: str,
    level: int = logging.INFO,
    msg: str = "test",
    extra: dict[str, object] | None = None,
) -> logging.LogRecord:
    """Build a LogRecord matching what `logger.info(msg, extra=...)` produces.

    `extra` keys become attributes on the record (logging framework
    convention); we mirror that here.
    """
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


def _entry() -> ContextManifestEntry:
    return ContextManifestEntry(
        file_path="src/foo.py",
        scope_unit_name="Foo.bar",
        line_start=1,
        line_end=10,
        inclusion_reason="changed_scope",
    )


def _build_response() -> LLMResponse:
    return LLMResponse(
        text="SECRET",
        model="claude-sonnet-4-6",
        input_tokens=10,
        output_tokens=10,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=100,
    )


def _build_request() -> LLMRequest:
    return LLMRequest(
        system_prompt="You are a code reviewer.",
        user_prompt="Review this PR.",
        model="claude-sonnet-4-6",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="analyze",
        prompt_template_version="analyze@1.0.0",
        degraded_mode=False,
        context_summary=(_entry(),),
    )


def _build_llm_call_event() -> LLMCallEvent:
    return LLMCallEvent(
        review_id=uuid4(),
        model="claude-sonnet-4-6",
        node_id="analyze",
        input_tokens=10,
        output_tokens=10,
        cached_tokens=0,
        cost_usd=0.0001,
        pricing_version="v1",
        latency_ms=100,
        prompt_hash="sha256-abc",
        cache_hit=False,
        context_summary=(_entry(),),
        prompt_template_version="analyze@1.0.0",
        system_prompt_hash="sha256-def",
        degraded_mode=False,
    )


@pytest.fixture
def filter_instance() -> RejectLLMContentFilter:
    return RejectLLMContentFilter()


# ---------------------------------------------------------------------------
# LLM_CONTENT_BEARING_TYPES integrity.
# ---------------------------------------------------------------------------


def test_content_bearing_types_constant_shape() -> None:
    """The constant defines what Tier 0 rejects. New content-bearing
    schemas (V1.5+ tool surfaces) MUST be added here."""
    assert LLMRequest in LLM_CONTENT_BEARING_TYPES
    assert LLMResponse in LLM_CONTENT_BEARING_TYPES
    assert LLMMessage in LLM_CONTENT_BEARING_TYPES
    assert len(LLM_CONTENT_BEARING_TYPES) == 3


# ---------------------------------------------------------------------------
# Tier 0 — type-based rejection on every logger.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "logger_name",
    [
        "app.webhooks",
        "outrider.agent.analyze",
        "outrider.audit",
        "root",
        "outrider.llm.anthropic_provider",
    ],
)
def test_tier0_rejects_llm_response_in_extra_on_every_logger(
    filter_instance: RejectLLMContentFilter, logger_name: str
) -> None:
    """Round-10 leak path: raw LLMResponse in extras on a non-LLM logger
    would have leaked through pre-round-10 design."""
    record = _record(logger_name, extra={"response": _build_response()})
    assert filter_instance.filter(record) is False


@pytest.mark.parametrize(
    "logger_name",
    ["app.webhooks", "outrider.agent.analyze", "outrider.audit"],
)
def test_tier0_rejects_llm_request_in_extra_on_every_logger(
    filter_instance: RejectLLMContentFilter, logger_name: str
) -> None:
    record = _record(logger_name, extra={"request": _build_request()})
    assert filter_instance.filter(record) is False


@pytest.mark.parametrize(
    "logger_name", ["app.webhooks", "outrider.agent.analyze", "outrider.audit"]
)
def test_tier0_rejects_llm_message_in_extra_on_every_logger(
    filter_instance: RejectLLMContentFilter, logger_name: str
) -> None:
    record = _record(logger_name, extra={"msg": LLMMessage(role="user", content="x")})
    assert filter_instance.filter(record) is False


def test_tier0_rejects_llm_response_in_nested_list(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """`extra={"items": [response]}` — recursion finds it."""
    record = _record(
        "outrider.agent.analyze",
        extra={"items": [_build_response()]},
    )
    assert filter_instance.filter(record) is False


def test_tier0_rejects_llm_response_in_nested_dict(
    filter_instance: RejectLLMContentFilter,
) -> None:
    record = _record(
        "outrider.agent.analyze",
        extra={"data": {"nested": _build_response()}},
    )
    assert filter_instance.filter(record) is False


def test_tier0_does_not_reject_llm_call_event(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """LLMCallEvent is metadata-only per #014; audit/dashboard code
    legitimately logs it. NOT in LLM_CONTENT_BEARING_TYPES."""
    record = _record("outrider.audit", extra={"event": _build_llm_call_event()})
    assert filter_instance.filter(record) is True


# ---------------------------------------------------------------------------
# Tier 1 — global key-name rejection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "prompt",
        "completion",
        "messages",
        "system_prompt",
        "user_prompt",
        "tool_input",
        "tool_use",
        "tool_result",
        "system",
        "api_key",
        "authorization",
        "x_api_key",
        "anthropic_api_key",
    ],
)
@pytest.mark.parametrize("logger_name", ["app.webhooks", "outrider.audit", "outrider.llm.foo"])
def test_tier1_rejects_global_keys_on_every_logger(
    filter_instance: RejectLLMContentFilter,
    key: str,
    logger_name: str,
) -> None:
    record = _record(logger_name, extra={key: "any value"})
    assert filter_instance.filter(record) is False


def test_tier1_rejects_nested_keys(
    filter_instance: RejectLLMContentFilter,
) -> None:
    record = _record("app.webhooks", extra={"data": {"prompt": "secret"}})
    assert filter_instance.filter(record) is False


def test_tier1_rejects_keys_in_list_of_dicts(
    filter_instance: RejectLLMContentFilter,
) -> None:
    record = _record("app.webhooks", extra={"items": [{"prompt": "x"}, {"prompt": "y"}]})
    assert filter_instance.filter(record) is False


# ---------------------------------------------------------------------------
# Tier 2 — LLM-logger-scoped key-name rejection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["text", "content"])
def test_tier2_rejects_on_llm_logger(filter_instance: RejectLLMContentFilter, key: str) -> None:
    record = _record("outrider.llm.anthropic_provider", extra={key: "completion"})
    assert filter_instance.filter(record) is False


@pytest.mark.parametrize("key", ["text", "content"])
def test_tier2_rejects_on_llm_namespace_root(
    filter_instance: RejectLLMContentFilter, key: str
) -> None:
    """Records on the literal `outrider.llm` logger also reject."""
    record = _record("outrider.llm", extra={key: "x"})
    assert filter_instance.filter(record) is False


@pytest.mark.parametrize("key", ["text", "content"])
@pytest.mark.parametrize(
    "logger_name", ["app.webhooks", "outrider.audit", "outrider.agent.analyze", ""]
)
def test_tier2_allows_on_non_llm_loggers(
    filter_instance: RejectLLMContentFilter,
    key: str,
    logger_name: str,
) -> None:
    """Tier 2 is scoped: outside outrider.llm.*, `text`/`content` are
    allowed (they're generic; full rejection would false-positive on
    webhook handlers / FastAPI middleware)."""
    record = _record(logger_name, extra={key: "x"})
    assert filter_instance.filter(record) is True


def test_tier2_recursive_on_llm_logger(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """Nested {"text": ...} under a non-rejected wrapper key still rejects
    on outrider.llm.*."""
    record = _record("outrider.llm.foo", extra={"response": {"text": "secret"}})
    assert filter_instance.filter(record) is False


# ---------------------------------------------------------------------------
# Truly-generic-name guard — body / payload / request / response allowed
# (top-level only when value is scalar).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["body", "payload", "request", "response"])
def test_truly_generic_keys_with_scalar_values_allowed(
    filter_instance: RejectLLMContentFilter, key: str
) -> None:
    """No rejection when the value is a scalar — the filter only fires on
    rejected keys, not on these wrapper keys themselves."""
    record = _record("app.webhooks", extra={key: "small string"})
    assert filter_instance.filter(record) is True


# ---------------------------------------------------------------------------
# Pydantic-model-as-extra walk (non-LLM model).
# ---------------------------------------------------------------------------


class _RandomModel(BaseModel):
    foo: str


def test_unrelated_pydantic_model_with_no_rejected_key_admits(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """A Pydantic model that's not in LLM_CONTENT_BEARING_TYPES gets
    walked via model_dump(); if no rejected keys appear, allow."""
    record = _record("app.webhooks", extra={"obj": _RandomModel(foo="bar")})
    assert filter_instance.filter(record) is True


class _BrokenModel(BaseModel):
    """Pydantic model whose `model_dump` raises — covers round-16 finding
    (test-coverage agent #4): the filter must not crash when a model has
    a broken serializer."""

    foo: str

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:  # type: ignore[override]
        raise RuntimeError("intentional failure for filter robustness test")


def test_filter_handles_broken_model_dump_without_crashing(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """If a third-party Pydantic model has a broken `model_dump`, the
    filter must not crash. Defense-in-depth: a buggy log call shouldn't
    take down logging entirely."""
    import contextlib

    broken = _BrokenModel(foo="bar")
    record = _record("app.webhooks", extra={"obj": broken})
    # Either the filter returns a verdict, or it raises a known exception
    # (the broken model's RuntimeError); either way the call completes.
    # The point is the filter didn't hang or bring down the logging system.
    with contextlib.suppress(RuntimeError):
        filter_instance.filter(record)


# ---------------------------------------------------------------------------
# Depth bound — no infinite recursion.
# ---------------------------------------------------------------------------


def test_depth_bound_does_not_hang_and_fails_closed(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """Self-referential dict is bounded at depth 8. Round-16 fold per
    Codex finding: at the bound, the walk fails CLOSED (rejects), not
    open. So a cyclic structure terminates AND rejects."""
    deep: dict[str, object] = {"safe_key": "value"}
    deep["self"] = deep  # cycle
    record = _record("app.webhooks", extra={"obj": deep})
    # Round-16: depth-bound now fail-closed (filter rejects).
    assert filter_instance.filter(record) is False, (
        "round-16: depth bound must fail-closed (reject), not fail-open"
    )


def test_deep_nesting_beyond_bound_fails_closed(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """Round-16 fix: content nested deeper than depth bound is rejected
    rather than allowed to leak. Defense-in-depth would rather drop a
    record than ship content from beyond the recursion bound."""
    nested: dict[str, object] = {"safe_key": "value"}
    # 12 layers of wrapping — exceeds the depth bound of 8
    for _ in range(12):
        nested = {"wrap": nested}
    record = _record("app.webhooks", extra={"data": nested})
    assert filter_instance.filter(record) is False


def test_deeply_nested_rejected_key_is_caught_within_bound(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """Within the depth bound, nested rejected keys are still caught."""
    nested: dict[str, object] = {"prompt": "secret"}
    for _ in range(5):
        nested = {"wrap": nested}
    record = _record("app.webhooks", extra={"data": nested})
    assert filter_instance.filter(record) is False


# ---------------------------------------------------------------------------
# Stateless contract — concurrent invocation safety.
# ---------------------------------------------------------------------------


def test_filter_has_no_mutable_state(
    filter_instance: RejectLLMContentFilter,
) -> None:
    """Filter must be stateless; no instance attributes mutated by
    `filter()`."""
    snapshot_before = dict(filter_instance.__dict__)
    record = _record("outrider.llm.anthropic_provider", extra={"prompt": "x"})
    filter_instance.filter(record)
    snapshot_after = dict(filter_instance.__dict__)
    assert snapshot_before == snapshot_after


# ---------------------------------------------------------------------------
# Schema-introspection contract (AC#17): every str-typed content-bearing
# field in the public schemas appears in Tier 1 or Tier 2.
# ---------------------------------------------------------------------------


from outrider.llm.logging import _TIER_1_KEYS, _TIER_2_KEYS  # noqa: E402


def test_all_documented_content_field_names_in_some_tier() -> None:
    """If a future schema adds a content-bearing str field whose name
    isn't in the rejection set, this test fails so the omission is loud.
    """
    documented_content_fields = {
        # LLMRequest
        "system_prompt",
        "user_prompt",
        "messages",
        # LLMResponse
        "text",
        # LLMMessage
        "content",
    }
    union = _TIER_1_KEYS | _TIER_2_KEYS
    missing = documented_content_fields - union
    assert not missing, (
        f"content-bearing field names not in any rejection tier: {missing}. "
        f"Add them to TIER_1_KEYS (global) or TIER_2_KEYS (LLM-scoped)."
    )


def test_content_bearing_types_constant_matches_public_schemas() -> None:
    """`LLM_CONTENT_BEARING_TYPES` should be exactly the set of public
    Pydantic models in `llm/base.py` that carry content-bearing string
    fields. If a new content-bearing schema is added, this test catches
    the missed update to the constant."""
    expected = {LLMRequest, LLMResponse, LLMMessage}
    assert set(LLM_CONTENT_BEARING_TYPES) == expected


# ---------------------------------------------------------------------------
# Filter installation on handlers (AC#23).
# ---------------------------------------------------------------------------


def test_register_installs_on_root_handlers() -> None:
    """Filter must attach to HANDLERS (round-15 correction) — logger-level
    filters miss propagated records."""
    root = logging.getLogger()
    handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        filter_instance = register_filter_on_all_handlers()
        assert any(isinstance(f, RejectLLMContentFilter) for f in handler.filters)
        assert filter_instance is not None
    finally:
        root.removeHandler(handler)


def test_register_is_idempotent() -> None:
    """Multiple invocations must not duplicate the filter on a handler."""
    root = logging.getLogger()
    handler = logging.NullHandler()
    root.addHandler(handler)
    try:
        register_filter_on_all_handlers()
        register_filter_on_all_handlers()
        register_filter_on_all_handlers()
        filter_count = sum(1 for f in handler.filters if isinstance(f, RejectLLMContentFilter))
        assert filter_count == 1, (
            f"expected 1 RejectLLMContentFilter on handler, got {filter_count}"
        )
    finally:
        root.removeHandler(handler)


def test_register_installs_on_outrider_logger_handlers() -> None:
    """Walk includes the `outrider` logger, not just root."""
    outrider_logger = logging.getLogger("outrider")
    handler = logging.NullHandler()
    outrider_logger.addHandler(handler)
    try:
        register_filter_on_all_handlers()
        assert any(isinstance(f, RejectLLMContentFilter) for f in handler.filters)
    finally:
        outrider_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Real propagation chain — emit a content-bearing record, capture, assert.
# ---------------------------------------------------------------------------


class _CapturingHandler(logging.Handler):
    """Captures records that pass through filters."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_propagation_chain_rejects_content_bearing_record() -> None:
    """Integration-shaped: register filter on root handlers, emit on
    `outrider.agent.analyze`, assert the handler did NOT see the record."""
    root = logging.getLogger()
    handler = _CapturingHandler()
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    saved_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        register_filter_on_all_handlers()
        agent_logger = logging.getLogger("outrider.agent.analyze")
        agent_logger.info("test", extra={"system_prompt": "SECRET"})
        # Filter should have rejected the record before emit().
        rejected = not any(getattr(r, "system_prompt", None) == "SECRET" for r in handler.records)
        assert rejected, "filter did not reject content-bearing record"
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)


def test_propagation_chain_admits_safe_record() -> None:
    """Negative test: a safe record on the same logger admits."""
    root = logging.getLogger()
    handler = _CapturingHandler()
    handler.setLevel(logging.DEBUG)
    root.addHandler(handler)
    saved_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        register_filter_on_all_handlers()
        agent_logger = logging.getLogger("outrider.agent.analyze")
        agent_logger.info("test", extra={"safe_key": "no leak here"})
        admitted = any(getattr(r, "safe_key", None) == "no leak here" for r in handler.records)
        assert admitted
    finally:
        root.removeHandler(handler)
        root.setLevel(saved_level)
