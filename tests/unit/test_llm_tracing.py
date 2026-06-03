"""Unit tests for `llm/tracing.py` — the composition-root LangSmith tracing
decorator (DECISIONS.md#035).

Two surfaces: the enable-decision (`langsmith_tracing_enabled` /
`wrap_provider_if_tracing`, the env contract that used to live in
`AnthropicProvider.__init__`) and the `TracingLLMProvider` decorator
(delegation + that it applies `langsmith.traceable`). LangSmith is mocked
throughout — no real tracing I/O. Each test controls its own env explicitly, so
no autouse env-scrub fixture is needed (the old `_hermetic_langsmith_env`
band-aid is gone: the provider no longer reads tracing env).
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from outrider.llm.tracing import (
    TracingLLMProvider,
    langsmith_tracing_enabled,
    wrap_provider_if_tracing,
)

_SENTINEL_RESPONSE = object()
# `traceable(...)` returns a decorator; this passthrough stand-in returns the
# wrapped function unchanged, so the decorator delegates without real langsmith.
_PASSTHROUGH_TRACEABLE = lambda **_kwargs: lambda fn: fn  # noqa: E731


class _StubProvider:
    """Minimal `LLMProvider` stub: records complete/aclose calls."""

    def __init__(self) -> None:
        self.complete_calls: list[object] = []
        self.aclose_calls = 0

    async def complete(self, request: object) -> object:
        self.complete_calls.append(request)
        return _SENTINEL_RESPONSE

    async def aclose(self) -> None:
        self.aclose_calls += 1


# ---------------------------------------------------------------------------
# The enable decision — langsmith_tracing_enabled / wrap_provider_if_tracing.
# ---------------------------------------------------------------------------


def test_tracing_disabled_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-off: no `LANGSMITH_TRACING` → disabled, provider unwrapped."""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert langsmith_tracing_enabled() is False
    stub = _StubProvider()
    assert wrap_provider_if_tracing(stub) is stub  # returned unchanged


@pytest.mark.parametrize("raw", ["true", "TRUE", "True", " true ", "tRuE"])
def test_tracing_enabled_when_env_truthy(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    """`LANGSMITH_TRACING=true` (case-insensitive, whitespace-tolerant) + a key
    enables tracing; `wrap_provider_if_tracing` returns a `TracingLLMProvider`
    and emits the activation INFO."""
    monkeypatch.setenv("LANGSMITH_TRACING", raw)
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test_key")
    assert langsmith_tracing_enabled() is True
    caplog.set_level(logging.INFO, logger="outrider.llm.tracing")
    with patch("langsmith.traceable", side_effect=_PASSTHROUGH_TRACEABLE):
        wrapped = wrap_provider_if_tracing(_StubProvider())
    assert isinstance(wrapped, TracingLLMProvider)
    assert any("LangSmith tracing enabled" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("raw", ["false", "False", "0", "", "yes", "1"])
def test_tracing_disabled_when_env_non_true(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Only literal "true" activates — `yes`/`1`/`0`/empty do NOT, matching
    LangSmith's own env convention. Pins the negative set against a future
    widening to a general truthy parser."""
    monkeypatch.setenv("LANGSMITH_TRACING", raw)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert langsmith_tracing_enabled() is False
    stub = _StubProvider()
    assert wrap_provider_if_tracing(stub) is stub


@pytest.mark.parametrize("missing_key", ["", "   ", None])
def test_tracing_disabled_and_warns_when_api_key_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, missing_key: str | None
) -> None:
    """`LANGSMITH_TRACING=true` with an empty/whitespace/unset key MUST NOT
    enable tracing (the LangSmith client would accept traces and silently drop
    them in its background thread). Surfaced as a WARN naming the key, treated
    as OFF."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    if missing_key is None:
        monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    else:
        monkeypatch.setenv("LANGSMITH_API_KEY", missing_key)
    caplog.set_level(logging.WARNING, logger="outrider.llm.tracing")
    assert langsmith_tracing_enabled() is False
    assert any("LANGSMITH_API_KEY" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# The decorator — delegation + traceable application.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decorator_delegates_complete_and_aclose() -> None:
    """`complete()` forwards to the wrapped provider and returns its response;
    `aclose()` forwards too (the decorator owns no transport resources)."""
    stub = _StubProvider()
    with patch("langsmith.traceable", side_effect=_PASSTHROUGH_TRACEABLE):
        provider = TracingLLMProvider(stub)
    request = object()
    result = await provider.complete(request)
    assert result is _SENTINEL_RESPONSE
    assert stub.complete_calls == [request]
    await provider.aclose()
    assert stub.aclose_calls == 1


def test_decorator_applies_traceable_with_llm_run_type() -> None:
    """The decorator wraps the inner `complete` with `langsmith.traceable`
    once, as an `llm` run."""
    stub = _StubProvider()
    with patch("langsmith.traceable", return_value=(lambda fn: fn)) as mock_traceable:
        TracingLLMProvider(stub)
    mock_traceable.assert_called_once()
    assert mock_traceable.call_args.kwargs.get("run_type") == "llm"
