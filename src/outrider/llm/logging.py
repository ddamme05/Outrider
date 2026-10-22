# Logging filter that rejects records carrying LLM content.
# See specs/2026-05-05-llm-provider-wrapper.md and DECISIONS.md #013/#016.
"""Reject log records that carry model-call content or credentials.

Schema serializers redact by default; this filter covers raw objects and
ad-hoc containers. It rejects content-bearing model types on any logger,
explicit content/secret keys globally, and generic `text`/`content` keys
under `outrider.llm`. Nested values are checked; the depth limit fails closed.

Install on handlers with `register_filter_on_all_handlers`, because
logger filters do not see propagated records (`DECISIONS.md#013/#016`).
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
# the filter. Records nesting deeper than this are REJECTED
# (fail-closed) — defense-in-depth means we'd rather drop a record than
# ship content from beyond the recursion bound. The earlier fail-open
# behavior was the real bug.
_RECURSION_DEPTH_LIMIT: Final[int] = 8

_LLM_LOGGER_PREFIX: Final[str] = "outrider.llm"


class RejectLLMContentFilter(logging.Filter):
    """Synchronous content filter with no mutable per-record instance state.

    The depth limit stops cycles and excessive nesting. Keep record-specific
    state local so concurrent logging does not share it.
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
    """Return True for rejected content or nesting at the depth limit.

    The limit fails closed, including self-referential containers. Other
    Pydantic models are traversed through their default-redacted dump;
    serialization failure also rejects the record.
    """
    if depth >= _RECURSION_DEPTH_LIMIT:
        return True

    # Tier 0 — type-based rejection.
    if isinstance(obj, LLM_CONTENT_BEARING_TYPES):
        return True

    # Use the default-redacted dump, without INCLUDE_TEXT_OPT_IN. A broken
    # serializer rejects the record rather than disrupting logging or leaking text.
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
    """Install the filter on root, `outrider`, and registered descendant handlers.

    Handler filters cover propagated records and loggers with `propagate=False`.
    Snapshot the logger registry before walking it, since concurrent logger
    creation can resize it. Existing content filters are left in place; call
    again after new handlers are registered. Inspect `handler.filters` to see
    the installed instance.
    """
    if filter_instance is None:
        filter_instance = RejectLLMContentFilter()

    seen_handlers: set[int] = set()
    loggers_to_walk: list[logging.Logger] = [logging.getLogger()]  # root
    loggers_to_walk.append(logging.getLogger("outrider"))
    # Snapshot our logger registry before iteration; concurrent getLogger calls
    # can resize the live dict. Include descendants with propagate=False.
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
