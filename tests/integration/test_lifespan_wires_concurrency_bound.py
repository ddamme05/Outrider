# See DECISIONS.md#045 — V1 per-process concurrent-review ceiling (FUP-164).
"""Production wiring guard — lifespan installs the concurrency bound on run_graph.

Pins the FUP-164 / DECISIONS.md#045 production contract: `app.state.run_graph`
must be the semaphore-BOUNDED wrapper, not the bare closure. The unit tests for
`concurrency_limited` prove the wrapper works; they do NOT prove the lifespan
installs it at the load-bearing seam. A future edit could build the wrapper but
stash bare `run_graph` (or drop the wrap entirely) and every helper test would
still pass — the same silent-wiring class already guarded for `AnalyzeCacheStore`.

The guard is behavioral, not structural: it sets the ceiling to 1, injects a fake
compiled graph whose `ainvoke` blocks, and drives two concurrent invocations
through `app.state.run_graph`. A correctly-bounded wrapper serializes them (only
one enters `ainvoke` at a time); a bare closure would let both run concurrently
and the assertion fires.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan

# `outrider.api`'s package namespace re-exports a `lifespan` FUNCTION that
# shadows the submodule attribute, so resolve the actual module for
# monkeypatching `build_graph` (mirrors test_lifespan_wires_analyze_cache_store).
lifespan_module = importlib.import_module("outrider.api.lifespan")


@pytest.fixture(autouse=True)
def _activate_github_app_env(github_app_env: None) -> None:  # noqa: ARG001 — fixture activates env
    """Lifespan hard-requires `GitHubAppSettings()` + `DashboardSettings()` at
    startup; the shared `github_app_env` fixture provides those env vars."""


async def test_lifespan_installs_concurrency_bound_on_run_graph(
    monkeypatch: pytest.MonkeyPatch,
    make_stub_llm_provider: type,
    noop_severity_policy_fingerprint_check: object,
    in_memory_checkpointer_factory: object,
) -> None:
    """With the ceiling at 1, two concurrent `app.state.run_graph` calls
    serialize — proving the bounded wrapper, not the bare closure, is wired."""
    monkeypatch.setenv("OUTRIDER_MAX_CONCURRENT_REVIEWS", "1")
    # Disable the periodic sweep so its background task never touches the fake
    # graph during the test window.
    monkeypatch.setenv("OUTRIDER_SWEEP_DISABLED", "1")

    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    mock_engine.sync_engine.hide_parameters = True

    running = 0
    peak = 0
    release = asyncio.Event()  # held; ainvoke bodies block until set

    async def blocking_ainvoke(_state: Any, *, config: Any = None) -> dict[str, bool]:  # noqa: ARG001
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1
        return {"ok": True}

    def fake_build_graph(**_kwargs: Any) -> Any:
        fake = MagicMock()
        fake.ainvoke = blocking_ainvoke
        return fake

    monkeypatch.setattr(lifespan_module, "build_graph", fake_build_graph)

    lifespan = build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister, _model_config: make_stub_llm_provider(),
        severity_policy_fingerprint_check=noop_severity_policy_fingerprint_check,  # type: ignore[arg-type]
        checkpointer_factory=in_memory_checkpointer_factory,
    )

    app = FastAPI()
    async with lifespan(app):
        run_graph = app.state.run_graph
        # `run_graph` builds its RunnableConfig from `state.review_id`; a bare
        # namespace with that attribute is enough to reach `ainvoke`.
        t1 = asyncio.create_task(run_graph(SimpleNamespace(review_id=UUID(int=1))))
        t2 = asyncio.create_task(run_graph(SimpleNamespace(review_id=UUID(int=2))))

        # Spin the loop: with the ceiling at 1, exactly one call may enter
        # ainvoke; the other must park on the semaphore.
        for _ in range(10):
            await asyncio.sleep(0)
        assert running == 1, (
            "both run_graph calls entered ainvoke concurrently — app.state.run_graph "
            "is the BARE closure, not the semaphore-bounded wrapper (FUP-164 regression)"
        )
        assert peak == 1
        assert not (t1.done() or t2.done())

        # Release: the first exits, freeing the slot; the second enters and runs.
        release.set()
        await asyncio.wait_for(asyncio.gather(t1, t2), timeout=1.0)
        assert peak == 1  # the second waited; the ceiling was never crossed
