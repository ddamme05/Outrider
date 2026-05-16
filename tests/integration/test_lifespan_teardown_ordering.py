"""Lifespan H3 closure — engine.dispose() runs even if provider.aclose() raises.

Pins the spec's AsyncExitStack guarantee: every push_async_callback runs
on teardown, LIFO order, even if a prior callback raises. Without this,
a transient SDK error during rolling deploy could leak the engine's
connection pool.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan


class _SyntheticACloseError(RuntimeError):
    """Synthetic exception raised by the test's mock aclose."""


async def test_engine_dispose_runs_when_provider_aclose_raises() -> None:
    """provider.aclose() raises during teardown; engine.dispose() still runs.

    AsyncExitStack pushes callbacks in order: engine.dispose first, then
    provider.aclose. Teardown is LIFO: provider.aclose runs first, raises;
    engine.dispose runs SECOND, even though aclose raised. The exception
    propagates AFTER both callbacks have been attempted.
    """
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"

    mock_provider = MagicMock()
    mock_provider.aclose = AsyncMock(side_effect=_SyntheticACloseError("injected"))

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: mock_provider,
    )

    app = FastAPI()
    with pytest.raises(_SyntheticACloseError):
        async with lifespan(app):
            pass

    # Both teardown callbacks ran, despite the raise.
    mock_provider.aclose.assert_awaited_once()
    mock_engine.dispose.assert_awaited_once()


async def test_lifespan_rejects_real_engine_without_hide_parameters(
    migrated_db: str,
) -> None:
    """Round-4 regression: defeat MagicMock-masking of the M2 assertion.

    The existing `test_lifespan_rejects_engine_without_hide_parameters`
    uses MagicMock; MagicMock returns truthy MagicMock by default for any
    attribute access. That test verifies the assertion fires when an
    explicit override sets `hide_parameters=False`, but doesn't verify
    the assertion exercises the SAME attribute path a real SQLAlchemy
    engine exposes.

    This test constructs a REAL `AsyncEngine` via `create_async_engine`
    (no `hide_parameters=True` — production defaults). The lifespan
    assertion must fire. Together with the MagicMock test, both the
    contract AND the SQLAlchemy attribute path are pinned.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    real_engine_no_hide_params = create_async_engine(migrated_db)
    # Sanity: SQLAlchemy's default is hide_parameters=False on sync_engine.
    # If a future SQLAlchemy version flipped this default to True, this
    # assertion would PASS but the test's intended exercise (verifying the
    # lifespan rejects a misconfigured engine) would silently no-op — the
    # lifespan's `assert ... hide_parameters` check would never fire
    # because the engine no longer matches the "misconfigured" condition.
    # If that happens, replace this test with one that explicitly disables
    # hide_parameters via `execution_options` or similar. The MagicMock-
    # variant test above (which sets `sync_engine.hide_parameters = False`
    # explicitly) remains the primary regression gate regardless of
    # SQLAlchemy default.
    assert real_engine_no_hide_params.sync_engine.hide_parameters is False

    mock_provider = MagicMock()
    mock_provider.aclose = AsyncMock(return_value=None)

    lifespan = build_lifespan(
        engine_factory=lambda: real_engine_no_hide_params,
        provider_factory=lambda _persister: mock_provider,
    )

    app = FastAPI()
    try:
        with pytest.raises(RuntimeError, match="hide_parameters"):
            async with lifespan(app):
                pass
    finally:
        await real_engine_no_hide_params.dispose()


async def test_lifespan_rejects_engine_without_hide_parameters() -> None:
    """M2 regression: lifespan body asserts `engine.hide_parameters is True`
    on the constructed engine. A test seam (or future extraction) that
    returns a real engine without the setting must fail loud at lifespan
    entry rather than silently regressing DECISIONS#016 logs-stay-
    metadata-only at runtime.

    Constructs a real-shaped engine mock whose `hide_parameters` attribute
    is False; asserts lifespan startup raises AssertionError before yielding.
    """
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # The lifespan asserts `engine.sync_engine.hide_parameters` — set the
    # misconfiguration on the nested sync_engine, not the AsyncEngine.
    mock_engine.sync_engine.hide_parameters = False

    mock_provider = MagicMock()
    mock_provider.aclose = AsyncMock(return_value=None)

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: mock_provider,
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="hide_parameters"):
        async with lifespan(app):
            pass


async def test_engine_dispose_runs_when_provider_constructor_fails() -> None:
    """If `provider_factory` raises DURING lifespan startup (after engine
    construction), the AsyncExitStack still runs the engine.dispose
    callback that was already pushed before the failure.
    """
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"

    def _failing_provider_factory(_persister: object) -> object:
        raise RuntimeError("provider construction failed")

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=_failing_provider_factory,
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="provider construction failed"):
        async with lifespan(app):
            pass

    # engine.dispose ran on teardown despite the startup failure — the
    # callback was already pushed onto the stack before the raise.
    mock_engine.dispose.assert_awaited_once()
