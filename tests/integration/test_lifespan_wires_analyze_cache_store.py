"""Production wiring guard — lifespan injects the analyze-cache store.

Pins the lever-8 Stage-B production contract
(specs/2026-06-11-file-hash-analyze-cache.md): store-or-None is the
enable switch, so a `build_graph(...)` call that omits
`analyze_cache_store` silently ships an inert shadow stage — no
`CacheLookupEvent` telemetry and no `analyze_file_cache` writes ever
accrue in the one environment the flip evidence must come from. The
spy wraps the REAL `build_graph` (delegates, then captures kwargs) so
the guard also proves the wired store is accepted by the actual graph
builder, not just named in the call.
"""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan
from outrider.cache import AnalyzeCacheStore

# `outrider.api`'s package namespace re-exports a `lifespan` FUNCTION that
# shadows the submodule attribute, so `import outrider.api.lifespan as m`
# binds the function. Resolve the actual module for monkeypatching.
lifespan_module = importlib.import_module("outrider.api.lifespan")


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001 — fixture activates env
    """Lifespan hard-requires `GitHubAppSettings()` at startup; the shared
    `github_app_env` fixture from `tests/conftest.py` provides the three
    env vars. Module-local autouse wrapper saves per-test argument plumbing.
    """


async def test_lifespan_wires_analyze_cache_store_into_build_graph(
    monkeypatch: pytest.MonkeyPatch,
    make_stub_llm_provider: type,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
) -> None:
    """The production `build_graph(...)` call passes a live
    `AnalyzeCacheStore`, not the inert `None` default."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    mock_engine.sync_engine.hide_parameters = True

    real_build_graph = lifespan_module.build_graph
    captured: dict[str, Any] = {}

    def spying_build_graph(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_build_graph(**kwargs)

    monkeypatch.setattr(lifespan_module, "build_graph", spying_build_graph)

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister, _model_config, _host, _reasoning: (
            make_stub_llm_provider()
        ),  # noqa: E501
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,
    )

    app = FastAPI()
    async with lifespan(app):
        pass

    assert "analyze_cache_store" in captured, (
        "lifespan's build_graph call omits analyze_cache_store — Stage B "
        "shadow is inert in production (store-or-None is the enable switch)"
    )
    store = captured["analyze_cache_store"]
    assert isinstance(store, AnalyzeCacheStore), (
        f"expected a wired AnalyzeCacheStore, got {type(store).__name__} — "
        "a None here means no CacheLookupEvent telemetry and no cache "
        "writes ever accrue in production"
    )
