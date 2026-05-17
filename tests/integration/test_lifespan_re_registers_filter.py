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

from fastapi import FastAPI

from outrider.api.lifespan import build_lifespan


def _make_test_lifespan() -> object:
    """Build a lifespan with mock engine + provider so the real DB/SDK
    aren't required for filter-re-registration testing."""
    mock_engine = MagicMock()
    mock_engine.dispose = AsyncMock(return_value=None)
    mock_engine.url.drivername = "postgresql+psycopg"
    # Round-39 strict `is True` gate requires the exact bool, not
    # MagicMock's truthy default.
    mock_engine.sync_engine.hide_parameters = True

    mock_provider = MagicMock()
    mock_provider.aclose = AsyncMock(return_value=None)

    return build_lifespan(
        engine_factory=lambda: mock_engine,
        provider_factory=lambda _persister: mock_provider,
    )


async def test_lifespan_installs_filter_on_late_registered_handler() -> None:
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
    target_logger.addHandler(late_handler)
    target_logger.propagate = False  # isolate from root for the test

    # Before lifespan runs, the late handler has NO filter installed.
    from outrider.llm.logging import RejectLLMContentFilter

    pre_filter_count = sum(1 for f in late_handler.filters if isinstance(f, RejectLLMContentFilter))
    assert pre_filter_count == 0

    # Enter the lifespan body.
    app = FastAPI()
    lifespan_cm = _make_test_lifespan()(app)
    async with lifespan_cm:
        # Inside lifespan body, the filter IS installed on the late handler.
        post_filter_count = sum(
            1 for f in late_handler.filters if isinstance(f, RejectLLMContentFilter)
        )
        assert post_filter_count == 1


async def test_lifespan_filter_actually_rejects_content_records() -> None:
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
    target_logger.addHandler(capture_handler)
    target_logger.setLevel(logging.DEBUG)
    target_logger.propagate = False

    app = FastAPI()
    lifespan_cm = _make_test_lifespan()(app)
    async with lifespan_cm:
        # Emit a record carrying content fields the filter rejects.
        target_logger.info(
            "test message",
            extra={"prompt": "secret prompt content"},
        )

    # Filter rejected → emit() was never called.
    assert len(captured) == 0, f"Filter failed to reject content record; captured: {captured}"
