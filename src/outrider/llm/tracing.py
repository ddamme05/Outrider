# LLM tracing — composition-root decorator. See DECISIONS.md#035.
"""Provider-agnostic LangSmith tracing for any `LLMProvider`.

Tracing is observability *policy*, not transport: it does not belong inside a
concrete provider's constructor (where it made provider behavior depend on
ambient env and would be duplicated per provider as V1.5 adds `OpenAIProvider`).
Per DECISIONS.md#035, concrete providers are tracing-agnostic; the composition
root applies `wrap_provider_if_tracing()` once, which wraps the provider in a
`TracingLLMProvider` decorator when `LANGSMITH_TRACING=true` (+ a key) is set.

`import langsmith` lives ONLY in this file (lazily, inside the decorator's
`__init__`, so the no-trace path never imports it) — honoring
`vendor-sdks-only-in-wrappers`. The env var NAMES are langsmith config, kept
here with the decorator for cohesion; reading `os.environ` is not an SDK import.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from outrider.llm.base import LLMProvider, LLMRequest, LLMResponse

__all__ = [
    "TracingLLMProvider",
    "langsmith_tracing_enabled",
    "wrap_provider_if_tracing",
]

_LOGGER = logging.getLogger(__name__)


class TracingLLMProvider:
    """`LLMProvider` decorator that adds LangSmith tracing at the `.complete()`
    Protocol boundary.

    Traces the domain `LLMRequest`/`LLMResponse` at the Protocol seam rather
    than `wrap_anthropic`'s native Anthropic SDK spans (DECISIONS.md#035
    trade-off): one tracing seam for every current and future provider, at the
    cost of native LLM-run fields (token usage, model params), which are
    re-addable as span metadata if LangSmith UI fidelity later requires it.

    Constructed ONLY when the composition root has decided tracing is on
    (`wrap_provider_if_tracing`), so `langsmith` is imported here — lazily, to
    keep the no-trace path import-free (a cold-start optimization, NOT optionality:
    langsmith is a hard dependency). `aclose()` forwards to the wrapped provider
    (the decorator owns no transport resources of its own).
    """

    def __init__(self, inner: LLMProvider) -> None:
        self._inner = inner
        # Lazy import: only the trace-on path (this constructor) pulls langsmith,
        # keeping cold-start + the no-trace path import-free (langsmith is a hard
        # dep; this is a latency optimization, not graceful-degradation).
        # `import langsmith` confined to llm/ per vendor-sdks-only-in-wrappers.
        from langsmith import traceable

        # Wrap once at construction (not per call). `traceable` detects the
        # async callable and returns an async wrapper that records the bound
        # `complete`'s input (the LLMRequest) and output (the LLMResponse) as a
        # LangSmith run. Name the run by the concrete provider class so mixed
        # Anthropic/OpenAI (V1.5) traffic is distinguishable in the UI.
        self._traced_complete = traceable(run_type="llm", name=f"{type(inner).__name__}.complete")(
            inner.complete
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        result: LLMResponse = await self._traced_complete(request)
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()


def langsmith_tracing_enabled() -> bool:
    """Whether LangSmith LLM-call tracing should be applied.

    Mirrors the env contract LangGraph itself reads, so node-level traces
    (LangGraph) and LLM-call traces (this decorator) activate together:
    `LANGSMITH_TRACING=true` (literal "true", case-insensitive +
    whitespace-tolerant — NOT a general truthy-string parser, matching
    LangSmith's own SDK convention) AND a non-empty `LANGSMITH_API_KEY`.

    Both are required: with tracing on but the key unset, the LangSmith client
    accepts traces and silently drops them in a background thread, wasting
    per-call CPU on tracing that never surfaces in the UI — so a `true`-but-no-key
    config is surfaced as a WARN and treated as OFF, rather than activating a
    silent no-op trace.
    """
    tracing_on = os.environ.get("LANGSMITH_TRACING", "").strip().lower() == "true"
    if not tracing_on:
        return False
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        _LOGGER.warning(
            "llm.tracing: LANGSMITH_TRACING=true but LANGSMITH_API_KEY is "
            "unset/empty; tracing NOT enabled and traces would silently drop in "
            "the LangSmith client's background thread. Set LANGSMITH_API_KEY to "
            "enable tracing, or unset LANGSMITH_TRACING to silence this warning."
        )
        return False
    return True


def wrap_provider_if_tracing(provider: LLMProvider) -> LLMProvider:
    """Apply the tracing decorator to `provider` iff tracing is enabled.

    The single call the composition root makes (DECISIONS.md#035): the enable
    decision lives here, once, not in N concrete providers. Returns the provider
    unchanged when tracing is off.
    """
    if langsmith_tracing_enabled():
        _LOGGER.info("llm.tracing: LangSmith tracing enabled via TracingLLMProvider")
        return TracingLLMProvider(provider)
    return provider
