"""Lifespan FUP-011 closure — AnthropicProvider.aclose() called on shutdown.

Pins the FUP-011 exit rule: AnthropicProvider.aclose() exists and is
invoked from the FastAPI lifespan teardown. Verifies via a spy on the
provider's aclose method.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan

# PEM + env-var injection + LLM provider stub are centralized in
# `tests/conftest.py` per round-31 fold (DevEx audit, HIGH). Tests grab
# fresh stubs via the `make_stub_llm_provider` factory fixture.


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001 — fixture activates env
    """Lifespan hard-requires `GitHubAppSettings()` at startup; the shared
    `github_app_env` fixture from `tests/conftest.py` provides the three
    env vars. Module-local autouse wrapper saves per-test argument plumbing.
    """


async def test_lifespan_calls_provider_aclose_on_shutdown(
    make_stub_llm_provider: type,
) -> None:
    """Lifespan teardown awaits `provider.aclose()`."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # Round-39 strict `is True` gate requires the exact bool, not
    # MagicMock's truthy default.
    mock_engine.sync_engine.hide_parameters = True

    stub_provider = make_stub_llm_provider()

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: stub_provider,
    )

    app = FastAPI()
    async with lifespan(app):
        # Provider is constructed at startup; aclose not called yet.
        stub_provider.aclose.assert_not_called()

    # After exiting the context manager, aclose was called exactly once.
    stub_provider.aclose.assert_awaited_once()


async def test_lifespan_calls_engine_dispose_on_shutdown(
    make_stub_llm_provider: type,
) -> None:
    """Lifespan teardown awaits `engine.dispose()`."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    mock_engine.sync_engine.hide_parameters = True

    stub_provider = make_stub_llm_provider()

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: stub_provider,
    )

    app = FastAPI()
    async with lifespan(app):
        mock_engine.dispose.assert_not_called()

    mock_engine.dispose.assert_awaited_once()


async def test_anthropic_provider_aclose_method_exists() -> None:
    """Sanity test: the production AnthropicProvider exposes aclose().

    Without this, FUP-011 isn't actually closed even if the lifespan
    test (which uses a MagicMock provider) passes.
    """
    import inspect

    from outrider.llm.anthropic_provider import AnthropicProvider

    assert hasattr(AnthropicProvider, "aclose")
    assert inspect.iscoroutinefunction(AnthropicProvider.aclose)


async def test_anthropic_provider_aclose_is_idempotent() -> None:
    """M3 regression: aclose() called twice calls the underlying SDK once.

    Without the `_closed` guard, a future code path calling
    `provider.aclose()` outside the lifespan teardown would stack a
    second close on top of the lifespan callback; httpx's behavior on
    repeated `aclose()` is version-dependent.
    """
    import inspect
    from unittest.mock import AsyncMock, patch

    from pydantic import SecretStr

    from outrider.llm.anthropic_provider import AnthropicProvider
    from outrider.llm.config import ModelConfig

    # Construct a real provider; mock only the underlying SDK client close.
    class _StubPersister:
        async def persist(self, *args: object, **kwargs: object) -> None: ...

    provider = AnthropicProvider(
        api_key=SecretStr("sk-test"),
        model_config=ModelConfig(),
        persister=_StubPersister(),  # type: ignore[arg-type]
    )
    assert inspect.iscoroutinefunction(provider._client.close)

    with patch.object(provider._client, "close", new=AsyncMock(return_value=None)) as mock_close:
        await provider.aclose()
        mock_close.assert_awaited_once()

        # Second call is a no-op via the _closed guard; underlying SDK
        # close should NOT be called again.
        await provider.aclose()
        mock_close.assert_awaited_once()  # still 1


async def test_aclose_invokes_underlying_close_once_under_concurrent_calls() -> None:
    """Concurrent aclose() calls invoke the underlying SDK close at most once.

    NOTE on test honesty: under Python's cooperative async with a mock that
    returns instantly, this test passes EVEN WITHOUT the `_close_lock`
    (the check-then-set runs synchronously; the second coroutine sees
    `_closed=True` before it can race). The behavior-level test can't
    distinguish "with lock" from "without lock" for the current impl —
    that's the structural contract; see
    `test_aclose_uses_close_lock_for_atomicity` below for the structural
    pin. This test exercises the user-visible behavior (concurrent
    callers don't stack underlying closes), not the impl mechanism.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    from pydantic import SecretStr

    from outrider.llm.anthropic_provider import AnthropicProvider
    from outrider.llm.config import ModelConfig

    class _StubPersister:
        async def persist(self, *args: object, **kwargs: object) -> None: ...

    provider = AnthropicProvider(
        api_key=SecretStr("sk-test"),
        model_config=ModelConfig(),
        persister=_StubPersister(),  # type: ignore[arg-type]
    )

    with patch.object(provider._client, "close", new=AsyncMock(return_value=None)) as mock_close:
        await asyncio.gather(*(provider.aclose() for _ in range(16)))
        mock_close.assert_awaited_once()


def test_aclose_uses_close_lock_for_atomicity() -> None:
    """Structural pin: `aclose()` MUST acquire `self._close_lock` to atomically
    bracket the check-then-set. The behavior test above cannot distinguish
    "with lock" from "without lock" for the current impl (synchronous
    check + synchronous set, no yield point under cooperative async); this
    test asserts the lock IS present in the source so a future refactor
    that removes it fails this gate.

    Empirically verified: removing the `async with self._close_lock:` line
    and rerunning the behavior test above produces an identical pass — the
    structural assertion is the only test that fires on lock removal.

    Round-4 audit-the-audit caught the behavior test as vacuous for proving
    the lock's contribution; this structural test is the fix.
    """
    import inspect

    from outrider.llm.anthropic_provider import AnthropicProvider

    source = inspect.getsource(AnthropicProvider.aclose)
    assert "async with self._close_lock:" in source, (
        "AnthropicProvider.aclose() must acquire self._close_lock to atomically "
        "bracket the check-then-set on self._closed. Removing the lock allows "
        "a future refactor that adds an await between the check and the set "
        "to silently introduce a TOCTOU race."
    )


async def test_complete_after_aclose_raises_loud() -> None:
    """Round-4 regression: provider.complete() called after aclose() raises
    a typed LLM error rather than surfacing an obscure httpx error from
    the closed client.

    Scenario: uvicorn graceful-shutdown finishes in-flight requests AFTER
    the lifespan yields back. A queued request handler holds a reference
    to `app.state.provider` and tries to call `complete()` on it. Without
    this guard the call fails deep inside the SDK with
    `RuntimeError("Cannot send a request, as the client has been closed.")`
    — opaque to log readers. The guard surfaces the misuse at the wrapper
    boundary instead.
    """
    from pydantic import SecretStr

    from outrider.llm.anthropic_provider import AnthropicProvider
    from outrider.llm.base import LLMRequest, LLMUnknownError
    from outrider.llm.config import ModelConfig

    class _StubPersister:
        async def persist(self, *args: object, **kwargs: object) -> None: ...

    provider = AnthropicProvider(
        api_key=SecretStr("sk-test"),
        model_config=ModelConfig(),
        persister=_StubPersister(),  # type: ignore[arg-type]
    )
    await provider.aclose()  # mark closed

    from uuid import uuid4

    request = LLMRequest(
        system_prompt="x",
        user_prompt="y",
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",
        prompt_template_version="triage:1",
        degraded_mode=False,
    )

    with pytest.raises(LLMUnknownError, match="provider is closed"):
        await provider.complete(request)


async def test_anthropic_provider_aclose_timeout_does_not_block_lifespan() -> None:
    """L3 regression: aclose() with a hung SDK close releases via timeout
    rather than blocking lifespan teardown indefinitely.

    Simulates a slow close by patching `self._client.close` to await an
    asyncio.Event that never fires within the timeout window. Asserts
    `aclose()` returns successfully (via the wait_for TimeoutError path)
    AND marks the provider as closed (no retry on next call).
    """
    import asyncio
    from unittest.mock import patch

    from pydantic import SecretStr

    from outrider.llm.anthropic_provider import _ACLOSE_TIMEOUT_SECONDS, AnthropicProvider
    from outrider.llm.config import ModelConfig

    class _StubPersister:
        async def persist(self, *args: object, **kwargs: object) -> None: ...

    provider = AnthropicProvider(
        api_key=SecretStr("sk-test"),
        model_config=ModelConfig(),
        persister=_StubPersister(),  # type: ignore[arg-type]
    )

    # Hung close: awaits an event that never fires.
    never_fires = asyncio.Event()

    async def _hung_close() -> None:
        await never_fires.wait()

    # Override the timeout to a tiny value so the test runs fast — the
    # default 10s would slow the suite. Patch the module-level constant.
    with (
        patch.object(provider._client, "close", new=_hung_close),
        patch("outrider.llm.anthropic_provider._ACLOSE_TIMEOUT_SECONDS", 0.05),
    ):
        # Without the timeout guard, this would hang forever.
        await asyncio.wait_for(provider.aclose(), timeout=2.0)

    # Provider marked closed; next call is a no-op (no retry of the hung close).
    await provider.aclose()
    # Sanity: 10s default is the production value (regression in case a
    # future refactor lowers it to something risky).
    assert _ACLOSE_TIMEOUT_SECONDS >= 5.0
