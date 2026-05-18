"""Top-level pytest fixtures shared across unit/integration/eval tests."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

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


# ---------------------------------------------------------------------------
# GitHub-App test fixtures — shared across auth/lifespan/filter tests.
# ---------------------------------------------------------------------------
#
# Centralized so a PEM rotation or env-var rename touches one place.


@functools.cache
def _generate_test_rsa_pem() -> str:
    """Generate a one-shot RSA-2048 PEM at first call; cache thereafter.

    Tests need a structurally-valid PEM so githubkit's `AppInstallationAuthStrategy`
    accepts it at construction (no JWT mint happens until the first API
    call, but the constructor parses the PEM). Generating at import-time
    instead of committing the PEM avoids tripping secret scanners on a
    repo-committed `BEGIN RSA PRIVATE KEY` block. The key is process-local
    and never leaves the test interpreter.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem_bytes.decode("ascii")


TEST_GITHUB_APP_PRIVATE_KEY_PEM = _generate_test_rsa_pem()


@pytest.fixture
def github_app_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the three OUTRIDER_GITHUB_APP_* env vars to test values.

    Opt-in (not autouse): tests that need GitHubAppSettings() to load
    cleanly request this fixture explicitly. Lifespan integration tests
    that hit GitHubAppSettings() during startup must call this; pure
    unit tests that mock the settings object do not need it.
    """
    monkeypatch.setenv("OUTRIDER_GITHUB_APP_ID", "12345")
    monkeypatch.setenv("OUTRIDER_GITHUB_APP_PRIVATE_KEY", TEST_GITHUB_APP_PRIVATE_KEY_PEM)
    monkeypatch.setenv("OUTRIDER_GITHUB_WEBHOOK_SECRET", "test-secret")


class StubLLMProvider:
    """Satisfies the LLMProvider Protocol (has `complete` + `aclose`) so
    integration tests of lifespan can pass the runtime-checkable
    `isinstance(provider, LLMProvider)` gate without instantiating a
    real provider client.

    A plain MagicMock would fail the runtime-checkable Protocol check
    because MagicMock's auto-generated attributes don't carry method
    signatures Pydantic/Protocol introspection recognizes.

    Consumed via the `stub_llm_provider` fixture or the
    `make_stub_llm_provider` factory fixture below — never imported
    directly (tests/ is not on `pythonpath`, only `src/`).
    """

    def __init__(self) -> None:
        self.aclose = AsyncMock(return_value=None)

    async def complete(self, request: object) -> object:  # noqa: ARG002
        msg = "StubLLMProvider does not implement complete; tests should not call it"
        raise NotImplementedError(msg)


@pytest.fixture
def stub_llm_provider() -> StubLLMProvider:
    """Function-scoped: fresh StubLLMProvider per test.

    Use when a test needs ONE stub instance to inject as the provider.
    For tests that build multiple lifespans (each needing its own stub),
    use `make_stub_llm_provider` instead.
    """
    return StubLLMProvider()


@pytest.fixture
def make_stub_llm_provider() -> type[StubLLMProvider]:
    """Function-scoped factory: returns the StubLLMProvider class so the
    test can instantiate as many fresh stubs as it needs.

    The fixture returns the CLASS, not an instance — tests call
    `make_stub_llm_provider()` like a constructor. Pytest discovers this
    fixture by name even though it's not annotated as `Callable[..., T]`.
    """
    return StubLLMProvider
