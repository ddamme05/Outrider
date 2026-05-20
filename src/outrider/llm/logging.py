# Logging filter that rejects records carrying LLM content.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#016.
"""Three-tier recursive content-leak filter.

Defense-in-depth backup to the schema-level default-redaction on
`LLMRequest`/`LLMResponse`/`LLMMessage`. The schema-level redaction is
the primary defense for the `model_dump()` path — `model_dump()` elides
content unless a caller explicitly opts in by passing the
`INCLUDE_TEXT_OPT_IN` sentinel as the serialization context (no
production caller does so today; the persister persists raw content
via direct attribute access into the `llm_call_content` side-table,
bypassing `model_dump()` entirely). This filter catches leak paths
that don't go through `model_dump()`: ad-hoc dicts, third-party SDK
debug logging, raw Pydantic instances dropped into `extra={...}`.

Three rejection tiers (see AC#21):

  - **Tier 0 (type-based, every logger, recursive):** any value in record
    attrs/extras matching `LLM_CONTENT_BEARING_TYPES` rejects the record.
    Catches `extra={"response": llm_response}` from `outrider.agent.*`
    where `text` is generic-named but the OBJECT is content-bearing.
  - **Tier 1 (key-name, every logger, recursive):** unambiguous content/
    secret keys at any depth.
  - **Tier 2 (key-name, `outrider.llm.*` only, recursive):** `text` /
    `content` keys scoped to LLM-namespace loggers (broader scoping
    would produce false positives from FastAPI middleware / webhook
    handlers using `text` for legitimate non-content payloads).

Walks nested `dict`/`list`/`tuple`/Pydantic-model values to depth 8.
Stateless; sync; safe for concurrent invocation. Install via
`register_filter_on_all_handlers()` (handler-level, NOT logger-level —
logger-level filters miss propagated records).
"""

import logging
from typing import Any, Final

from pydantic import BaseModel

from outrider.llm.base import LLMMessage, LLMRequest, LLMResponse

__all__ = [
    "LLM_CONTENT_BEARING_TYPES",
    "RejectLLMContentFilter",
    "register_filter_on_all_handlers",
]


# Tier 0: any value matching one of these types in attrs/extras rejects
# the record on every logger. Constant rather than runtime-computed so
# imports + tests can grep for it; future content-bearing schemas
# (V1.5 LLMToolCall, etc.) MUST be added here.
LLM_CONTENT_BEARING_TYPES: Final[tuple[type[BaseModel], ...]] = (
    LLMRequest,
    LLMResponse,
    LLMMessage,
)

# Tier 1 — global key-name rejection (every logger, recursive).
# Unambiguous content/secret indicators.
_TIER_1_KEYS: Final[frozenset[str]] = frozenset(
    {
        # Prompt / completion content
        "prompt",
        "completion",
        "messages",
        "system_prompt",
        "user_prompt",
        "tool_input",
        "tool_use",
        "tool_result",
        # SDK-side request shape (Anthropic's `messages.create(system=...)`)
        "system",
        # Auth / credential
        "api_key",
        "authorization",
        "x_api_key",
        "anthropic_api_key",
    }
)

# Tier 2 — LLM-logger-scoped (records on `outrider.llm.*`, recursive).
# Generic-named so cannot be Tier-1 globally without false positives.
_TIER_2_KEYS: Final[frozenset[str]] = frozenset({"text", "content"})

# Bound to prevent pathological self-referential structures from hanging
# the filter. records nesting deeper than this
# are REJECTED (fail-closed) — defense-in-depth means we'd rather drop a
# record than ship content from beyond the recursion bound. The earlier
# fail-open behavior was the real bug.
_RECURSION_DEPTH_LIMIT: Final[int] = 8

_LLM_LOGGER_PREFIX: Final[str] = "outrider.llm"


class RejectLLMContentFilter(logging.Filter):
    """Three-tier recursive content-leak filter.

    Stateless: no instance attributes mutated during `filter()`. Concurrent
    invocation from multiple loggers/tasks/threads is exercised in
    production; future contributors MUST NOT add memoization-via-instance
    state.

    Sync by design: the logging framework is sync; an async filter would
    deadlock or drop records.

    Performance budget: <100µs per typical log record on Python 3.13.
    Depth bound + breadth-bounded ad-hoc dicts make pathological cost
    impossible.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True to allow the record, False to reject.

        Walks `record.__dict__` recursively; if any rejected element is
        found at any depth, the record is rejected.
        """
        is_llm_scoped = record.name == _LLM_LOGGER_PREFIX or record.name.startswith(
            _LLM_LOGGER_PREFIX + "."
        )
        active_keys = _TIER_1_KEYS | _TIER_2_KEYS if is_llm_scoped else _TIER_1_KEYS

        # `record.__dict__` includes standard log attributes (msg, args,
        # levelname, etc.) plus any extras attached via `extra={...}`.
        # Standard attributes are scalars / strings — the walk treats them
        # as leaves. Custom extras (top-level keys from `extra=...`) are
        # the substantive payload.
        return not _walk(record.__dict__, active_keys, depth=0)


def _walk(obj: Any, active_keys: frozenset[str], depth: int) -> bool:
    """Return True if `obj` contains a rejected element at any depth.

    Recursion bound: `_RECURSION_DEPTH_LIMIT`. **At the bound, return True
    (REJECT).** returning False would
    fail-open — content nested at exactly depth 8 would pass the filter
    silently. The defense-in-depth role demands fail-closed at the bound,
    so a record with deeply-nested rejected content (or a pathological
    self-reference) is dropped, not leaked. The spec's test description
    says deep nesting is "handled gracefully (rejected without hanging)";
    rejection is the closed state.
    """
    if depth >= _RECURSION_DEPTH_LIMIT:
        return True

    # Tier 0 — type-based rejection.
    if isinstance(obj, LLM_CONTENT_BEARING_TYPES):
        return True

    # Other Pydantic models — walk their default-redacted dump.
    # Spec note: NOT passing `INCLUDE_TEXT_OPT_IN` here; the filter mirrors
    # what would actually appear in a serialized record, which is the
    # default-redacted form.
    #
    # : a third-party model with a broken
    # serializer (custom `__get_pydantic_core_schema__`, unsupported
    # types in nested fields, etc.) could raise from `model_dump()`. A
    # logging filter that raises during emission disrupts logging and
    # can spam stderr depending on `logging.raiseExceptions`. Fail
    # closed — reject the record on serialization failure rather than
    # letting the exception propagate up through the logging stack.
    if isinstance(obj, BaseModel):
        try:
            dumped = obj.model_dump()
        except Exception:
            return True
        return _walk(dumped, active_keys, depth + 1)

    # Dict — check keys + recurse into values.
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in active_keys:
                return True
            if _walk(value, active_keys, depth + 1):
                return True
        return False

    # List / tuple — recurse into elements.
    if isinstance(obj, (list, tuple)):
        return any(_walk(item, active_keys, depth + 1) for item in obj)

    # Scalar — done.
    return False


def register_filter_on_all_handlers(
    filter_instance: RejectLLMContentFilter | None = None,
) -> None:
    """Walk the logger chain and install the filter on every reachable handler.

    Filters MUST be attached to handlers, NOT to loggers — Python's
    logger-level filters only see records emitted directly on that logger;
    propagated records bypass them. This helper walks the root logger plus
    `outrider` AND every `outrider.*` descendant logger, finds every
    handler reachable via propagation OR directly attached, and calls
    `handler.addFilter(...)` on each.

    walking only root + `outrider` missed
    handlers attached directly to `outrider.agent.*`, `outrider.llm.*`,
    `outrider.audit.*` etc. when those loggers had `propagate=False` set
    (test fixtures using `caplog`, third-party adapters, structlog).
    Walking `manager.loggerDict` covers every logger that has been touched
    by a `getLogger(...)` call — which in practice is every logger Outrider
    code emits on, since they're all created at import time.

    Idempotent: handlers already carrying a `RejectLLMContentFilter`
    instance are left alone (no duplicate adds). Re-invocable after later
    handler registration (e.g., FastAPI/uvicorn registering theirs at
    startup).

    Copilot follow-on fix: the previous signature returned
    `RejectLLMContentFilter` so callers could "verify which instance is
    active," but the conditional skip path (handler already had a
    different `RejectLLMContentFilter`) made the returned object
    potentially uninstalled — the misleading return-value pattern.
    Returns `None` now; callers that want the active instance should
    introspect a handler's `.filters` directly. Test surface confirmed
    no caller needs the instance for substantive work.
    """
    if filter_instance is None:
        filter_instance = RejectLLMContentFilter()

    seen_handlers: set[int] = set()
    loggers_to_walk: list[logging.Logger] = [logging.getLogger()]  # root
    loggers_to_walk.append(logging.getLogger("outrider"))
    # Every logger registered under the `outrider.*` namespace. Python's
    # logging Manager keeps a flat dict of all loggers ever created via
    # `getLogger(name)`; we pick out our project's namespace only so we
    # don't accidentally addFilter to third-party handlers we don't own.
    # **Snapshot via `list(...)` first** (per audit-agent
    # finding M3): walking the live dict races concurrent
    # `getLogger(...)` calls (RuntimeError: dict changed size during
    # iteration). V1.5's parallel-analyze workers will create child
    # loggers on first use; the snapshot eliminates the race entirely.
    for name, logger_obj in list(logging.getLogger().manager.loggerDict.items()):
        if name.startswith("outrider.") and isinstance(logger_obj, logging.Logger):
            loggers_to_walk.append(logger_obj)

    for logger in loggers_to_walk:
        for handler in logger.handlers:
            handler_id = id(handler)
            if handler_id in seen_handlers:
                continue
            seen_handlers.add(handler_id)
            already_present = any(isinstance(f, RejectLLMContentFilter) for f in handler.filters)
            if not already_present:
                handler.addFilter(filter_instance)
