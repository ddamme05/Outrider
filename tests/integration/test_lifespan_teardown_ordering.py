"""Lifespan cleanup disposes the engine even when provider shutdown fails.

AsyncExitStack callbacks run in LIFO order despite an earlier callback error,
so provider-close failures cannot skip connection-pool cleanup."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan

# §0c fingerprint-bypass helper (`noop_severity_policy_fingerprint_check`)
# is in `tests/integration/conftest.py` per §0c-devex-H2.


class _SyntheticACloseError(RuntimeError):
    """Synthetic exception raised by the test's mock aclose."""


@pytest.mark.usefixtures("github_app_env")
async def test_engine_dispose_runs_when_provider_aclose_raises(
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
    make_stub_llm_provider: type,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engine disposal follows a provider-close failure after successful startup."""
    monkeypatch.setenv("OUTRIDER_GITHUB_CREDENTIAL_SOURCE", "env")
    monkeypatch.setenv("OUTRIDER_SWEEP_DISABLED", "1")

    events: list[str] = []
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(side_effect=lambda: events.append("engine dispose"))
    mock_engine.url.drivername = "postgresql+psycopg"
    # Satisfy the strict boolean gate so this case reaches teardown ordering.
    mock_engine.sync_engine.hide_parameters = True

    provider = make_stub_llm_provider()

    def close_provider() -> None:
        events.append("provider close")
        raise _SyntheticACloseError("injected")

    provider.aclose.side_effect = close_provider

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister, _model_config, _host, _reasoning: provider,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(_SyntheticACloseError):
        async with lifespan(app):
            events.append("lifespan entered")

    provider.aclose.assert_awaited_once()
    mock_engine.dispose.assert_awaited_once()
    assert events == ["lifespan entered", "provider close", "engine dispose"]


async def test_lifespan_rejects_real_engine_without_hide_parameters(
    migrated_db: str,
) -> None:
    """A real AsyncEngine without hide_parameters=True fails startup.

    This checks SQLAlchemy's actual attribute path independently of MagicMock defaults.
    The constructed engine is rejected before opening a connection."""
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
        provider_factory=lambda _persister, _model_config, _host, _reasoning: mock_provider,
    )

    app = FastAPI()
    try:
        with pytest.raises(RuntimeError, match="hide_parameters"):
            async with lifespan(app):
                pass
    finally:
        await real_engine_no_hide_params.dispose()


async def test_lifespan_rejects_engine_without_hide_parameters() -> None:
    """Reject an engine whose sync_engine.hide_parameters is False.

    The runtime startup gate must enforce DECISIONS#016 before yielding, including
    under python -O where an assert would be stripped."""
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
        provider_factory=lambda _persister, _model_config, _host, _reasoning: mock_provider,
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="hide_parameters"):
        async with lifespan(app):
            pass


async def test_lifespan_rejects_engine_with_truthy_non_bool_hide_parameters() -> None:
    """Reject truthy non-booleans at the engine redaction gate.

    The string "true" is not the bool True. Requiring the exact boolean keeps an
    injected factory from bypassing the hide_parameters contract."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # A truthy string must not satisfy the boolean redaction gate.
    mock_engine.sync_engine.hide_parameters = "true"

    mock_provider = MagicMock()
    mock_provider.aclose = AsyncMock(return_value=None)

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister, _model_config, _host, _reasoning: mock_provider,
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="hide_parameters"):
        async with lifespan(app):
            pass


async def test_engine_dispose_runs_when_provider_constructor_fails(
    noop_severity_policy_fingerprint_check: object,
) -> None:
    """If `provider_factory` raises DURING lifespan startup (after engine
    construction), the AsyncExitStack still runs the engine.dispose
    callback that was already pushed before the failure.
    """
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # The redaction gate requires the exact bool.
    mock_engine.sync_engine.hide_parameters = True

    def _failing_provider_factory(
        _persister: object, _model_config: object, _host: object, _reasoning: object
    ) -> object:
        raise RuntimeError("provider construction failed")

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=_failing_provider_factory,
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
    )

    app = FastAPI()
    with pytest.raises(RuntimeError, match="provider construction failed"):
        async with lifespan(app):
            pass

    # engine.dispose ran on teardown despite the startup failure — the
    # callback was already pushed onto the stack before the raise.
    mock_engine.dispose.assert_awaited_once()
