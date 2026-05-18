"""Lifespan FUP-006 closure — RejectLLMContentFilter re-registered post-handler-add.

`outrider/__init__.py` calls `register_filter_on_all_handlers()` at import
time, which covers handlers present at that moment. uvicorn / FastAPI
register their own handlers later, during app startup; without a
re-invocation hook, those handlers are unfiltered.

The lifespan's `register_filter_on_all_handlers()` call closes the gap.
This test boots a minimal app whose lifespan runs the call, simulates a
late handler addition, asserts the filter rejects content records on
the actual handler chain.
"""

from __future__ import annotations

import logging
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


def _make_test_lifespan(stub_provider_cls: type) -> object:
    """Build a lifespan with mock engine + stub provider so the real DB/SDK
    aren't required for filter-re-registration testing.

    `stub_provider_cls` is the StubLLMProvider CLASS from the
    `make_stub_llm_provider` factory fixture — the helper instantiates
    a fresh stub per call.
    """
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # Round-39 strict `is True` gate requires the exact bool, not
    # MagicMock's truthy default.
    mock_engine.sync_engine.hide_parameters = True

    stub_provider = stub_provider_cls()

    return build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: stub_provider,
    )


async def test_lifespan_installs_filter_on_late_registered_handler(
    make_stub_llm_provider: type,
) -> None:
    """Simulate uvicorn registering its handler AFTER `import outrider` (which
    already called `register_filter_on_all_handlers()` once). The lifespan
    body's call re-runs the install and the late-registered handler now
    carries the filter.

    Without this, the late handler is unfiltered and content-bearing
    records emitted on outrider.* loggers leak through.
    """
    # Capture a fresh handler on the `outrider.llm` logger AFTER import —
    # this simulates the uvicorn-late-registration scenario.
    late_handler = logging.StreamHandler()
    target_logger = logging.getLogger("outrider.llm.test_filter_re_registration")

    # Loggers are process-global; save state and restore in `finally` so
    # this test's handler-add / propagate mutation does not leak into
    # any other test that touches the same logger tree.
    saved_handlers = list(target_logger.handlers)
    saved_propagate = target_logger.propagate
    saved_level = target_logger.level
    try:
        target_logger.addHandler(late_handler)
        target_logger.propagate = False  # isolate from root for the test

        # Before lifespan runs, the late handler has NO filter installed.
        from outrider.llm.logging import RejectLLMContentFilter

        pre_filter_count = sum(
            1 for f in late_handler.filters if isinstance(f, RejectLLMContentFilter)
        )
        assert pre_filter_count == 0

        # Enter the lifespan body.
        app = FastAPI()
        lifespan_cm = _make_test_lifespan(make_stub_llm_provider)(app)
        async with lifespan_cm:
            # Inside lifespan body, the filter IS installed on the late handler.
            post_filter_count = sum(
                1 for f in late_handler.filters if isinstance(f, RejectLLMContentFilter)
            )
            assert post_filter_count == 1
    finally:
        target_logger.removeHandler(late_handler)
        target_logger.handlers = saved_handlers
        target_logger.propagate = saved_propagate
        target_logger.level = saved_level


async def test_lifespan_filter_actually_rejects_content_records(
    make_stub_llm_provider: type,
) -> None:
    """End-to-end behavior: after lifespan setup, content-bearing records
    emitted on outrider.* loggers are rejected by the actual handler
    chain. Pins the FUP-006 exit rule: integration test boots a minimal
    app, emits a record on outrider.llm.foo, asserts rejection."""
    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    target_logger = logging.getLogger("outrider.llm.test_capture")
    capture_handler = _CaptureHandler()

    # Save process-global logger state so this test cannot contaminate
    # other tests that touch the outrider.llm.* tree.
    saved_handlers = list(target_logger.handlers)
    saved_propagate = target_logger.propagate
    saved_level = target_logger.level
    try:
        target_logger.addHandler(capture_handler)
        target_logger.setLevel(logging.DEBUG)
        target_logger.propagate = False

        app = FastAPI()
        lifespan_cm = _make_test_lifespan(make_stub_llm_provider)(app)
        async with lifespan_cm:
            # Emit a record carrying content fields the filter rejects.
            target_logger.info(
                "test message",
                extra={"prompt": "secret prompt content"},
            )

        # Filter rejected → emit() was never called.
        assert len(captured) == 0, f"Filter failed to reject content record; captured: {captured}"
    finally:
        target_logger.removeHandler(capture_handler)
        target_logger.handlers = saved_handlers
        target_logger.propagate = saved_propagate
        target_logger.level = saved_level
