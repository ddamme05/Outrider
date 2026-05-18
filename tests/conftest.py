"""Top-level pytest fixtures shared across unit/integration/eval tests."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

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


# ---------------------------------------------------------------------------
# GitHub-App test fixtures — shared across auth/lifespan/filter tests.
# ---------------------------------------------------------------------------
#
# Per round-31 multi-lens audit (DevEx HIGH): the same PEM block + env
# fixture + LLM provider stub previously appeared inline in three test
# files (test_github_auth_wrapper.py, test_lifespan_re_registers_filter.py,
# test_lifespan_calls_anthropic_provider_aclose.py). Centralized here so
# a PEM rotation or env-var rename touches one place.


TEST_GITHUB_APP_PRIVATE_KEY_PEM = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAyV3jByXmtRDdMVQuQzZBzZ4WK/wXf6OhV79IfYxOpaA/D87T
+9yzhRgI3OqDt6w8GdW8b62Bnlcj+JpUlDeJWj99H6OYDcOQXTjp2qsdoUFXrSqi
ZpL9JSf25LxoY/AyJ7+yLLEgEgYzgvKM/CdAh1FUDH4xKK8WTpQRYjzn9zywV3qa
RUFOVMyW/9MGlxoGgF+JU/Q4S7P5tBNgrAUbzpsfX23pPKpsWPYbT2qIMOgN/Cu5
qPp/v34UM6IIWQYDejaeapwUjvFvXNvy/aLk78qsiLcQ1OZALwBTwIptCG6mlFiM
TwoFlSbCV+sQ4OFB44d5tHkYrkrPgKAOhrZ4VQIDAQABAoIBAEDVwSVTGCC2BPlS
xJq2KQUbCjnL1Wq6gAOJZuh84xVR/zKR4UvRrSDRxe6P9DqEv8RvfXm6rl/63oCp
e0d6Sb1G2lU+IUcIRTpJg/9XYL5KqkZQjlGfnTpoOTumOgX9NeAaoeRSwLYz3GW0
nIvr3DBftxq2KIsB8nbQy+i07ngzZRRBb9wRcDPRGsR45fl/HOUjEXLcYUR+QSrT
2DUjDmCYr/ohG7VtuVRrM7tWSEjqYpZi8oxbODHWyMOEf3GtSF4o8DfImfQDLY6h
ie+0Ndnu1FlRxQ7QkrjqcjeQ1ATBYzdpvPMpovDxnyDg7Z0+W8VCkH2bbtSE6kfQ
fmF2u4ECgYEA8Xg26WeOipvLBzfdMz/Ckdq3K9Zh2vJ+rGCMUw5+VgB/2HrjP3RR
ev2y2WtwO+i1ZbN8b5MlDoZKKvKpw/cWZHbdL7BNAzMz0Bq+UoeZdRZAjqBYjGEG
xx+1cKzc7CTxoBQQKMlbS5GqlswtPK5xLF7uG0POoLsxr0BkkAdwUgUCgYEA1ZGT
WrLPgrPMVlmuypIVYj04vCY7VLpRRGBI7/UfqJzVKdrJDfx7nLgcwy+QQTrxJgxJ
B8N5GU1HvUFGD2pHpA9MMakgX79+8s12CRyJBwxbpO4lkkjqLrkb5SO2OHRsklrP
yE6XzAZ/x4UmTuvKTAQfMTSC0bQVRFRwymvA7tECgYBy3wEqBjKtFmZcdwIVlblG
KHODYAVUuvf+Egn1IFRDfsLDtgQK/2QFV3lt+KqlspsOiTbZ9MTbB9NdMpiBcrA+
F5fyXOAQ1qLrnHsklUVdcGjf0EwTzZ8ufWFFJVo+9PVTPDywbVUNs+UVjY6/JFGV
TaUuPF+sGOaDiojuGBKhrQKBgF13eUzy9KIBxKHK1lZSRiOmYthBpJF0vJX+R6KO
e0pj5/yPg7M9NCnXdfUjLfGm3WBQPnsl/sgL8MqwbgxbeYfYNiOZGNgcENrIlcJU
2GjlKgrSDgyKqNcF74OkH7SgC8oM4/wHocSpqgD8sH6XlBE7yJUcUf3DXR2C2x9w
UO6BAoGBAJa66xH4Yi54JeZdT3i9BiPL6QPLOG2g8O09JL2gZsHnVjyEopgsHnsZ
KwzcQXSEMXyAdvAGTQVf1qiQTuI8KO0iijUOnLgwJ97kAVtw01Z3SqimbBSpEK4n
e9Wj1IM7r5h2YqnYbVL4S26vNQyfA0lKZ5T9q8X/eYIWGowm1zUI
-----END RSA PRIVATE KEY-----
"""


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
