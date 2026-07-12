"""Top-level pytest fixtures shared across unit/integration/eval tests."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from langgraph.checkpoint.base import BaseCheckpointSaver

import outrider.policy.dimensions as _policy_dimensions
import outrider.policy.severity as _policy_severity
from outrider.agent.checkpoint_serde import is_outrider_checkpoint_serde

if TYPE_CHECKING:
    from collections.abc import Iterator

    from outrider.audit.events import LLMCallEvent, ReviewPhaseEvent
    from outrider.llm.base import LLMRequest, LLMResponse


# Captured at import time. Any test that mutates the module attribute
# (`unittest.mock.patch`, `monkeypatch.setattr`, direct assignment) breaks
# the startup-fingerprint-check invariant — the live mapping would no
# longer match the DB row at ACTIVE_POLICY_VERSION. The autouse fixture
# below fails-loud at teardown so the offending test surfaces immediately.
# Per §0c of specs/2026-05-19-analyze-foundation.md (also DI-M3 from the
# round-2 crazy audit). To exercise alternate policies:
#   - for replay-tier tests: seed an additional row in severity_policies
#     with a distinct version and pass that version through the
#     `policy_version` injection path used by `load_policy_for_version`.
#   - for unit-tier tests of `lookup_severity`: construct a separate dict
#     and test against it directly without mutating the module attribute.
#
# Per §0c data-integrity audit M-4: also check that nothing rebound
# `outrider.api.lifespan.SEVERITY_POLICY` (the lifespan module's own
# `from outrider.policy.severity import SEVERITY_POLICY` creates an
# independent binding which `monkeypatch.setattr("outrider.api.lifespan.SEVERITY_POLICY", ...)`
# can swap without tripping the severity-module check).
_ORIGINAL_SEVERITY_POLICY = _policy_severity.SEVERITY_POLICY
_ORIGINAL_FINDING_TYPE_TO_DIMENSION = _policy_dimensions.FINDING_TYPE_TO_DIMENSION


@pytest.fixture(autouse=True)
def _checkpointer_serde_guard() -> Iterator[None]:
    """Fail any test that constructs a checkpointer without the Outrider serde.

    FUP-220: every LangGraph checkpointer must carry the Outrider serde, or HITL
    resume + replay-equivalence break under LANGGRAPH_STRICT_MSGPACK=true.

    Patches `BaseCheckpointSaver.__init__` — which every `InMemorySaver` /
    `AsyncPostgresSaver` (and any subclass that calls `super().__init__`) routes
    through — so ANY checkpointer built during a test is checked regardless of
    class name, import alias, module-qualification, or factory. Fires on every
    run (not only under strict), so a saver constructed with the default
    (permissive) serde surfaces immediately rather than at a future
    langgraph-checkpoint bump. Complements the static whole-repo AST scan in
    `tests/unit/test_checkpoint_serde.py`, which covers code paths NOT exercised
    by tests (e.g. `scripts/`). `MagicMock()` checkpointers don't subclass
    `BaseCheckpointSaver`, so they don't trip this guard.

    The identity check goes through `is_outrider_checkpoint_serde` (a marker
    subclass), NOT a langgraph-private serde attribute — a private-attr read
    would make this guard raise for every wired checkpointer if langgraph renamed
    the attribute in a bump, the exact fragility FUP-220 exists to remove.

    Deliberately patches/restores `__init__` directly rather than via the
    `monkeypatch` fixture: requesting `monkeypatch` here would pull its setup
    forward and reorder its teardown after `_no_severity_policy_patching`'s
    check, spuriously tripping that guard.
    """
    original_init = BaseCheckpointSaver.__init__

    @functools.wraps(original_init)
    def _guarded_init(self: BaseCheckpointSaver, *args: object, **kwargs: object) -> None:
        original_init(self, *args, **kwargs)  # type: ignore[arg-type]
        if not is_outrider_checkpoint_serde(self.serde):
            raise RuntimeError(
                f"{type(self).__name__} was constructed without "
                "serde=build_checkpoint_serde(). Under LANGGRAPH_STRICT_MSGPACK=true this "
                "checkpointer refuses Outrider state types on deserialize, breaking HITL "
                "resume + replay. Pass serde=build_checkpoint_serde() from "
                "outrider.agent.checkpoint_serde."
            )

    BaseCheckpointSaver.__init__ = _guarded_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        BaseCheckpointSaver.__init__ = original_init  # type: ignore[method-assign]


@pytest.fixture(autouse=True)
def _no_severity_policy_patching() -> Iterator[None]:
    """Assert SEVERITY_POLICY identity at teardown (production + lifespan re-export).

    Per §8 of `specs/2026-05-19-analyze-foundation.md`: also guards
    `outrider.policy.dimensions.FINDING_TYPE_TO_DIMENSION` identity.
    Patching the dimension mapping breaks the module-load lockstep
    guard in `outrider.policy.dimensions._verify_lockstep` and would
    silently misroute findings to the wrong dimension at classification
    time. Same rule applies: never patch the module attribute. Construct
    a separate dict locally if you need to test alternate mappings.
    """
    yield
    # `if/raise` (not `assert`) so the guard survives `python -O` which
    # strips `assert` statements; same rationale as the lifespan
    # `hide_parameters` gate (§0c sharp-edges audit #3).
    if _policy_severity.SEVERITY_POLICY is not _ORIGINAL_SEVERITY_POLICY:
        raise RuntimeError(
            "outrider.policy.severity.SEVERITY_POLICY was rebound during this test. "
            "Patching/monkeypatching SEVERITY_POLICY breaks the lifespan startup "
            "fingerprint check (live mapping no longer matches the DB row at "
            "ACTIVE_POLICY_VERSION). For replay tests, seed an additional "
            "severity_policies row and use load_policy_for_version; for unit tests "
            "of lookup_severity, construct a separate dict locally and test against "
            "it directly. See specs/2026-05-19-analyze-foundation.md §0c."
        )
    # Lifespan's own `from outrider.policy.severity import SEVERITY_POLICY`
    # creates a second binding; check it too if the module was imported.
    lifespan_mod = sys.modules.get("outrider.api.lifespan")
    if lifespan_mod is not None:
        lifespan_bound = getattr(lifespan_mod, "SEVERITY_POLICY", _ORIGINAL_SEVERITY_POLICY)
        if lifespan_bound is not _ORIGINAL_SEVERITY_POLICY:
            raise RuntimeError(
                "outrider.api.lifespan.SEVERITY_POLICY was rebound during this test "
                "(the lifespan module's own re-export, distinct from the "
                "outrider.policy.severity binding). Same rule applies: never "
                "patch this binding either. See §0c data-integrity audit M-4."
            )
    # Per §8: dimension mapping identity must hold too.
    if _policy_dimensions.FINDING_TYPE_TO_DIMENSION is not _ORIGINAL_FINDING_TYPE_TO_DIMENSION:
        raise RuntimeError(
            "outrider.policy.dimensions.FINDING_TYPE_TO_DIMENSION was rebound "
            "during this test. Patching the dimension mapping breaks the "
            "module-load lockstep guard and would silently misroute findings "
            "to wrong dimensions at classification time. Construct a separate "
            "dict locally if you need alternate mappings — "
            "`{FindingType.X: ReviewDimension.Y, ...}` "
            "(import ReviewDimension from `outrider.schemas.review_finding` "
            "or via `outrider.schemas`; FindingType from `outrider.policy`). "
            "See specs/2026-05-19-analyze-foundation.md §8 + foundation-wide "
            "DevEx audit F8."
        )


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

    Also sets `OUTRIDER_ADMIN_API_KEY` since the lifespan's
    `DashboardSettings()` construction is in the same startup chain —
    a missing key surfaces as a Pydantic ValidationError at lifespan
    Step 7c (see `api/lifespan.py` admin_api_key wiring).
    """
    monkeypatch.setenv("OUTRIDER_GITHUB_APP_ID", "12345")
    monkeypatch.setenv("OUTRIDER_GITHUB_APP_PRIVATE_KEY", TEST_GITHUB_APP_PRIVATE_KEY_PEM)
    monkeypatch.setenv("OUTRIDER_GITHUB_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("OUTRIDER_ADMIN_API_KEY", "test-admin-key")


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
        # Instance-level AsyncMock so tests can assert `.assert_awaited_once()`
        # AND it satisfies the runtime-checkable `LLMProvider.aclose` member
        # (DECISIONS.md#035 formalized `aclose` on the Protocol; retained #058).
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
