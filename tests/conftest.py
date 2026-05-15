"""Top-level pytest fixtures shared across unit/integration/eval tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from outrider.audit.events import LLMCallEvent, ReviewPhaseEvent
    from outrider.llm.base import LLMRequest, LLMResponse


@pytest.fixture(scope="session")
def canonical_python_source() -> bytes:
    """Bytes of the canonical Python fixture per the V1 ast_facts/ spec.

    Read via `Path.read_bytes()` rather than `import` because the file
    contains source code we parse, not a Python module to load
    (and `tests/` is not on `pythonpath` per `docs/conventions.md`).
    """
    return (Path(__file__).parent / "fixtures" / "python_canonical.py").read_bytes()


@pytest.fixture(scope="session")
def canonical_python_path() -> str:
    """Repo-relative path to the canonical fixture, for ScopeUnit.file_path."""
    return "tests/fixtures/python_canonical.py"


# ---------------------------------------------------------------------------
# Triage-node spec fixtures: NoOpPersister + RecordingPhaseEventSink
# ---------------------------------------------------------------------------
#
# Per the triage-node spec, both are root-conftest sibling fixtures rather
# than `tests/_helpers/` modules because `pyproject.toml:64-68` forbids
# cross-tier test imports. Pytest's parent-conftest discovery auto-
# inherits these into unit/, integration/, and eval/ test bodies.


class NoOpPersister:
    """No-op LLMExchangePersister for tests; matches the Protocol exactly.

    Used by integration tests that construct a real `AnthropicProvider`
    instance (which requires a persister at __init__ time per
    LLMPersisterNotWiredError fail-closed design) but don't care about
    durable persistence. Tests that need to assert persistence
    (LLMCallEvent shape, content-row content) use a recording variant
    landing with FUP-007.
    """

    async def persist(
        self,
        event: LLMCallEvent,
        request: LLMRequest,
        response: LLMResponse,
    ) -> None:
        return None


class RecordingPhaseEventSink:
    """PhaseEventSink that captures emissions in a list for assertion.

    Function-scoped via the `recording_phase_event_sink` fixture so each
    test gets a fresh `.events` list. Direct construction in tests works
    too — useful when a test needs more than one sink instance.
    """

    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)


@pytest.fixture(scope="session")
def no_op_persister() -> NoOpPersister:
    """Session-scoped: the persister is stateless, safe to share."""
    return NoOpPersister()


@pytest.fixture
def recording_phase_event_sink() -> RecordingPhaseEventSink:
    """Function-scoped: each test asserts against a fresh .events list."""
    return RecordingPhaseEventSink()
