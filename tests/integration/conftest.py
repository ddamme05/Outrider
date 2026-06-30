"""Integration-test fixtures: fresh-DB-per-test isolation.

Each integration test runs against a brand-new Postgres database, created
at fixture setup and dropped at teardown. The pattern is committed to per
the schema-layer spec — sharing a DB across tests would re-introduce
ordering coupling, and tests like test_severity_policies_seeded depend on
a fresh post-migration state.

**Test isolation is enforced at two layers** (see docs/testing.md
"Two-container model" for the full rationale):

  1. *Process-level.* Tests connect to ``TEST_DATABASE_URL`` exclusively
     — a separate ``postgres-test`` container on port 5433 with
     ephemeral tmpfs data. The dev ``outrider`` DB on port 5432 is
     never touched by the automated test suite.
  2. *Fail-loud URL guard.* The ``fresh_db`` fixture asserts the URL
     matches the expected pattern (port 5433, "test" in the database
     name) before any DDL. A misconfigured ``.env`` that points the
     fixture at the dev DB by mistake is caught at fixture setup with
     a clear error, not silently swallowed.

Two fixtures cover the integration-test surface:

  - ``fresh_db``     — yields a URL to an empty Postgres database. The
                      caller chooses what to do with it (run alembic,
                      create raw SQL, etc.). Used by tests that exercise
                      the migration itself.
  - ``migrated_db``  — depends on ``fresh_db``; runs ``alembic upgrade
                      head`` against it. Yields the URL of a fully-
                      migrated DB. Used by tests that insert/query
                      against the schema.

Alembic is invoked programmatically (``alembic.command.upgrade``) rather
than via subprocess because env.py reads ``DATABASE_URL`` from
``os.environ`` on each invocation (runpy re-executes env.py from
scratch), so the fixture can switch the URL by setting the env var
before calling. ``asyncio.to_thread`` wraps the sync alembic call so the
internal ``asyncio.run`` in env.py doesn't collide with the pytest-asyncio
event loop.

Future maintainers: do NOT "optimize" this by sharing a DB across tests.
Test ordering becomes load-bearing the moment any test mutates state, and
diagnosing ordering failures is much harder than the per-test DB
creation cost (a few hundred milliseconds; CREATE/DROP DATABASE on a
local Postgres is fast).
"""

import asyncio
import os
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister
from outrider.eval_support import (
    assert_test_url_is_isolated,
    ephemeral_database,
    run_alembic_upgrade_head,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"


async def _noop_severity_policy_fingerprint_check(_engine: object) -> None:
    """Async no-op for MagicMock-engine lifespan tests.

    Lifespan tests that inject a MagicMock engine cannot satisfy the
    default `_verify_severity_policy_fingerprint` (which opens a real DB
    connection at lifespan Step 1b). Pass this helper via
    `build_lifespan(severity_policy_fingerprint_check=...)` to bypass.

    The §0c fingerprint behavior itself is exercised against a real
    `migrated_db` engine in `test_lifespan_startup_fingerprint.py` —
    this no-op exists so OTHER lifespan tests (teardown ordering, filter
    re-registration, provider aclose) aren't blocked on having a real DB.

    Discoverability per §0c-devex-H2: lifted from three duplicated
    in-file definitions into this conftest so a future fourth lifespan
    test finds the canonical helper.
    """


@pytest.fixture(autouse=True)
def _ensure_truncation_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide `OUTRIDER_TRUNCATION_HMAC_SECRET` when the ambient env lacks it.

    Integration tests that enter the app lifespan hit
    `require_truncation_secret()` (landed 2026-06-19, `policy/output_sanitizer.py`),
    which raises if the secret is unset. These tests exercise OTHER lifespan
    behavior (teardown ordering, startup fingerprint, build_graph wiring), not the
    secret requirement itself — that is covered at the unit tier by
    `test_startup_secret_validation.py`. Without this, the whole lifespan-test class
    fails in any shell that didn't source a `.env` carrying the secret (a fragility
    introduced when the startup check landed but the lifespan tests were not made
    self-contained). The `if not set` guard respects a real ambient secret (CI or a
    sourced `.env`); it only fills the gap.
    """
    if not os.environ.get("OUTRIDER_TRUNCATION_HMAC_SECRET"):
        monkeypatch.setenv("OUTRIDER_TRUNCATION_HMAC_SECRET", "integration-test-truncation-secret")


@pytest.fixture(scope="session")
def noop_severity_policy_fingerprint_check() -> Callable[[object], Awaitable[None]]:
    """Session-scoped: the no-op is stateless, safe to share."""
    return _noop_severity_policy_fingerprint_check


def _in_memory_checkpointer_factory() -> Any:
    """Async-context-manager-yielding checkpointer for lifespan tests.

    The production `_default_checkpointer_factory` builds an
    `AsyncPostgresSaver` against `DATABASE_URL`; tests that inject a
    MagicMock engine cannot supply a real DB connection. Returning an
    `InMemorySaver` wrapped in a no-op async context manager satisfies
    the lifespan body's `stack.enter_async_context(...)` call without
    a real psycopg parse.
    """
    from contextlib import asynccontextmanager  # noqa: PLC0415

    from langgraph.checkpoint.memory import InMemorySaver  # noqa: PLC0415

    @asynccontextmanager
    async def _cm() -> AsyncIterator[InMemorySaver]:
        yield InMemorySaver()

    return _cm()


@pytest.fixture
def in_memory_checkpointer_factory() -> Callable[[], Any]:
    """Returns a factory the lifespan body can call to enter the saver
    via AsyncExitStack."""
    return _in_memory_checkpointer_factory


# The pyproject_async template puts source-code config (script_location etc.)
# in pyproject.toml under [tool.alembic]; alembic.ini holds DB connection +
# logging only. The CLI auto-discovers both, but a path-constructed Config
# needs the toml_file passed explicitly for ScriptDirectory.from_config to
# find script_location.
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"


async def _run_alembic_action(action: str, target: str, db_url: str) -> None:
    """Run an alembic command (upgrade/downgrade) with DATABASE_URL overridden.

    env.py reads os.environ["DATABASE_URL"] each time it's exec'd via
    runpy, so setting it here before calling alembic is the URL-injection
    seam. ``asyncio.to_thread`` runs the sync call in a fresh thread so
    env.py's internal ``asyncio.run(run_async_migrations())`` doesn't try
    to nest event loops.

    The upgrade-to-head case delegates to the shared
    `run_alembic_upgrade_head`; the DOWNGRADE branch (and upgrade to a
    non-head target) stays local because the shared helper is upgrade-only
    by design — `alembic_runner` / the genesis-migration round-trip drive
    downgrade and must keep that capability here.
    """
    if action == "upgrade" and target == "head":
        await run_alembic_upgrade_head(db_url)
        return

    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    try:
        config = Config(str(ALEMBIC_INI), toml_file=str(PYPROJECT_TOML))
        if action == "upgrade":
            await asyncio.to_thread(command.upgrade, config, target)
        elif action == "downgrade":
            await asyncio.to_thread(command.downgrade, config, target)
        else:
            raise ValueError(f"unsupported alembic action: {action!r}")
    finally:
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url


@pytest_asyncio.fixture
async def fresh_db() -> AsyncGenerator[str]:
    """Yield a URL to a brand-new empty Postgres database.

    The DB is created at fixture setup against the postgres-test container
    (port 5433, ephemeral tmpfs data) and force-dropped at teardown. The
    URL pattern guard runs first; a misconfigured TEST_DATABASE_URL fails
    loud before any DDL.
    """
    try:
        main_url = os.environ["TEST_DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError(
            "TEST_DATABASE_URL is not set. Run `set -a && source .env && set +a` "
            "before pytest, and confirm .env has the TEST_ block (see "
            ".env.example). Integration tests require the postgres-test "
            "container, not the dev postgres container."
        ) from exc

    # The isolation guard, CREATE/DROP DATABASE (AUTOCOMMIT admin engine), and
    # the pg_terminate_backend teardown sweep all live in the shared
    # `ephemeral_database` CM (see outrider.eval_support). The explicit guard
    # here is belt-and-suspenders: `ephemeral_database` runs it too, but
    # calling it first keeps the fail-loud message at the fixture entrypoint.
    assert_test_url_is_isolated(main_url)
    async with ephemeral_database(base_url=main_url, name_prefix="outrider_test_") as test_url:
        yield test_url


AlembicRunner = Callable[[str, str, str], Awaitable[None]]


@pytest_asyncio.fixture
async def alembic_runner() -> AlembicRunner:
    """Return an awaitable callable: ``runner(action, target, db_url)``.

    Tests that drive alembic themselves (e.g., test_genesis_migration's
    upgrade + downgrade round-trip) use this fixture to invoke alembic
    against their fresh DB. ``migrated_db`` uses it internally too.
    """
    return _run_alembic_action


@pytest_asyncio.fixture
async def migrated_db(fresh_db: str) -> str:
    """Yield a URL to a DB upgraded to the latest migration head.

    Suitable for tests that insert/query against the schema and don't
    care how the schema got there. Schema-shape tests (genesis migration
    round-trip, append-only trigger introspection) should use
    ``fresh_db`` directly so they own the alembic invocation.
    """
    await _run_alembic_action("upgrade", "head", fresh_db)
    return fresh_db


# ---------------------------------------------------------------------------
# AuditPersister fixtures — shared across persister integration tests.
# ---------------------------------------------------------------------------
#
# Each persister test needs the same setup: a seeded `installations` row,
# a seeded `reviews` row (so `persister.persist()`'s SELECT-installation_id
# lookup resolves), a live `AsyncEngine`, an `AuditPersister` instance.
# These fixtures consolidate that boilerplate; per-test customization
# (e.g., overriding the retention TTL) goes via direct construction.

# Canonical installation_id for seeded test data. A test that needs a
# distinct installation_id constructs its own seed inline.
PERSISTER_TEST_INSTALLATION_ID = 12345


@dataclass(frozen=True, slots=True)
class PersisterTestSetup:
    """Ready-to-go fixture bundle for AuditPersister integration tests.

    Carries the engine (caller is responsible for nothing — disposal
    handled by the fixture), the persister, the seeded review id, and
    the installation_id. Tests acquire this fixture and proceed straight
    to exercising `persist()` / `emit_phase()`.
    """

    engine: AsyncEngine
    persister: AuditPersister
    review_id: UUID
    installation_id: int


async def _seed_install_and_review(
    engine: AsyncEngine,
    installation_id: int = PERSISTER_TEST_INSTALLATION_ID,
) -> UUID:
    """Insert installations + reviews rows; return the reviews.id UUID."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"id": installation_id},
        )
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'running', "
                "  NOW() + INTERVAL '90 days'"
                ") RETURNING id"
            ),
            {"id": installation_id},
        )
        return UUID(str(result.scalar_one()))


if TYPE_CHECKING:
    from outrider.audit.events import FileExaminationEvent, LLMCallEvent, ReviewPhaseEvent
    from outrider.llm.base import LLMRequest, LLMResponse


# Type aliases for the factory fixtures.
LLMCallEventFactory = Callable[..., "LLMCallEvent"]
LLMRequestFactory = Callable[..., "LLMRequest"]
LLMResponseFactory = Callable[..., "LLMResponse"]
ReviewPhaseEventFactory = Callable[..., "ReviewPhaseEvent"]
FileExaminationEventFactory = Callable[..., "FileExaminationEvent"]


# Canonical prompts used by the request + event factories so their
# prompt_hash / system_prompt_hash values agree by construction. The
# persister's pre-tx guard checks `event.prompt_hash ==
# _canonical_prompt_hash(system_prompt=request.system_prompt,
# user_prompt=request.user_prompt)` — the factories share these prompts
# so hashes line up unless a test deliberately diverges them.
_FACTORY_SYSTEM_PROMPT = "the system prompt"
_FACTORY_USER_PROMPT = "the user prompt"


@pytest.fixture
def llm_call_event_factory() -> LLMCallEventFactory:
    """Factory: `factory(review_id, **kwargs) -> LLMCallEvent` with canonical
    defaults. Tests that need divergence on a cross-checked field use
    `event.model_copy(update={...})` and expect the corresponding
    pre-tx or in-tx guard to raise (e.g., diverging `cost_usd` trips
    `AuditPersisterEventResponseFieldMismatchError`; diverging
    `timestamp` trips `AuditPersisterIdempotencyConflict` on re-emit).
    """
    from datetime import UTC, datetime

    from outrider.audit.events import LLMCallEvent
    from outrider.llm.anthropic_provider import (
        _ANTHROPIC_CONTRACT_DIGEST,
        _ANTHROPIC_PROFILE_ID,
    )
    from outrider.llm.base import _canonical_prompt_hash, _canonical_system_prompt_hash
    from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd

    # Default factory matches the response factory's tokens/model/latency/
    # cache state so the persister's pre-tx STABLE-field response check
    # passes by construction. cost_usd recomputed canonically + pricing_version
    # pinned to the module constant so the in-tx fresh-write pricing check
    # also passes. Tests that need divergence on any of these use
    # `model_copy(update=...)` and expect the corresponding guard to fire.
    default_input_tokens = 100
    default_output_tokens = 50
    default_model = "claude-haiku-4-5"
    canonical_cost_usd = float(
        compute_cost_usd(
            "anthropic",
            default_model,
            input_tokens=default_input_tokens,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=default_output_tokens,
        )
    )

    def _build(
        review_id: UUID,
        *,
        latency_ms: int = 250,
        is_eval: bool = False,
        user_prompt: str = _FACTORY_USER_PROMPT,
        system_prompt: str = _FACTORY_SYSTEM_PROMPT,
    ) -> LLMCallEvent:
        return LLMCallEvent(
            review_id=review_id,
            model=default_model,
            # Matches llm_response_factory's finish_reason so the persister's
            # response-vs-event cross-check passes by construction
            # (DECISIONS.md#016 Amended 2026-06-30).
            finish_reason="end_turn",
            node_id="triage",
            input_tokens=default_input_tokens,
            output_tokens=default_output_tokens,
            cached_tokens=0,
            cost_usd=canonical_cost_usd,
            pricing_version=PRICING_VERSION,
            latency_ms=latency_ms,
            prompt_hash=_canonical_prompt_hash(
                system_prompt=system_prompt, user_prompt=user_prompt
            ),
            cache_hit=False,
            context_summary=(),
            prompt_template_version="triage:1",
            system_prompt_hash=_canonical_system_prompt_hash(system_prompt),
            degraded_mode=False,
            is_eval=is_eval,
            timestamp=datetime.now(UTC),
            # Host-qualified per #056 — matches llm_response_factory's triad so the
            # persister's event-vs-response cross-check passes by construction.
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )

    return _build


@pytest.fixture
def llm_request_factory() -> LLMRequestFactory:
    """Factory: `factory(review_id, **kwargs) -> LLMRequest`."""
    from outrider.llm.base import LLMRequest

    def _build(
        review_id: UUID,
        *,
        user_prompt: str = _FACTORY_USER_PROMPT,
        is_eval: bool = False,
    ) -> LLMRequest:
        return LLMRequest(
            system_prompt=_FACTORY_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model="claude-haiku-4-5",
            max_tokens=1024,
            temperature=0.0,
            review_id=review_id,
            node_id="triage",
            is_eval=is_eval,
            prompt_template_version="triage:1",
            degraded_mode=False,
        )

    return _build


@pytest.fixture
def llm_response_factory() -> LLMResponseFactory:
    """Factory: `factory(**kwargs) -> LLMResponse`."""
    from outrider.llm.anthropic_provider import (
        _ANTHROPIC_CONTRACT_DIGEST,
        _ANTHROPIC_PROFILE_ID,
    )
    from outrider.llm.base import LLMResponse

    def _build(*, text_value: str = "the completion text") -> LLMResponse:
        # Host-qualified per #056 (claude → anthropic) so the response can be
        # priced + persister-cross-checked like a real AnthropicProvider response.
        return LLMResponse(
            text=text_value,
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=250,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )

    return _build


@pytest.fixture
def review_phase_event_factory() -> ReviewPhaseEventFactory:
    """Factory: `factory(review_id, **kwargs) -> ReviewPhaseEvent`."""
    from uuid import uuid4

    from outrider.audit.events import ReviewPhaseEvent

    def _build(
        review_id: UUID,
        *,
        marker: str = "start",
        phase_id: str | None = None,
        phase_key: str | None = None,
        is_eval: bool = False,
    ) -> ReviewPhaseEvent:
        return ReviewPhaseEvent(
            review_id=review_id,
            phase_id=phase_id or str(uuid4()),
            node_id="triage",
            marker=marker,  # type: ignore[arg-type]
            is_eval=is_eval,
            phase_key=phase_key,
        )

    return _build


@pytest.fixture
def file_examination_event_factory() -> FileExaminationEventFactory:
    """Factory: `factory(review_id, **kwargs) -> FileExaminationEvent`.

    Defaults to a clean parse — overrides let tests construct skipped
    rows with a `skip_reason`. Used by FUP-029 integration tests
    (round-31 fold) to exercise `AuditPersister.emit_file_examination`.
    """
    from outrider.audit.events import FileExaminationEvent

    def _build(
        review_id: UUID,
        *,
        file_path: str = "src/example.py",
        examination_type: str = "intake_fetch",
        node_id: str = "intake",
        parse_status: str = "clean",
        skip_reason: object | None = None,
        is_eval: bool = False,
    ) -> FileExaminationEvent:
        return FileExaminationEvent(
            review_id=review_id,
            file_path=file_path,
            examination_type=examination_type,  # type: ignore[arg-type]
            node_id=node_id,  # type: ignore[arg-type]
            parse_status=parse_status,  # type: ignore[arg-type]
            skip_reason=skip_reason,  # type: ignore[arg-type]
            is_eval=is_eval,
        )

    return _build


@pytest_asyncio.fixture
async def persister_setup(migrated_db: str) -> AsyncGenerator[PersisterTestSetup]:
    """Seeded DB + engine + persister + review_id, scoped to one test.

    Disposes the engine on teardown. Tests that need custom retention
    TTLs or sessionmaker settings construct their own setup inline; this
    fixture covers the default case.

    `hide_parameters=True` mirrors the production engine factory at
    `src/outrider/api/lifespan.py::_default_engine_factory`. Without it,
    SQLAlchemy exception strings would include bound `prompt`/`completion`
    values from failing content INSERTs. The real residual leak vector
    this defends against is any log handler that renders `str(exc)` on
    the raw SQLAlchemy exception BEFORE the wrapper at
    `AnthropicProvider.complete()` Step 9 sees it (engine-level logging,
    third-party pool handlers, ad-hoc `logger.exception(...)` calls in
    test code). The wrapper-chain vector is separately closed via the
    round-9 + round-26 hardening: unknown exception types render only
    as `<TypeName>` and raise with `from None`, suppressing the cause
    chain via `__suppress_context__`. Tests that exercise failure paths
    rely on this engine setting being live so the assertions match
    production behavior.
    """
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_install_and_review(engine)
        persister = AuditPersister(
            session_factory=async_sessionmaker(engine, expire_on_commit=False),
            retention_settings=RetentionSettings(),
        )
        yield PersisterTestSetup(
            engine=engine,
            persister=persister,
            review_id=review_id,
            installation_id=PERSISTER_TEST_INSTALLATION_ID,
        )
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def eval_review_id(persister_setup: PersisterTestSetup) -> UUID:
    """An `is_eval=True` reviews row under `persister_setup`'s installation.

    FUP-130: `persist()` / `emit_finding()` now require `event.is_eval` to match
    the reviews row's `is_eval`. Tests exercising the eval-tagged happy path emit
    against THIS review (is_eval=True) instead of the production-default
    `persister_setup.review_id` (is_eval=False), which would now be a mismatch.
    Distinct pr_number/head_sha so it never collides with the default review.
    """
    async with persister_setup.engine.begin() as conn:
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  is_eval, retention_expires_at"
                ") VALUES ("
                "  :iid, 100, 2, 'sha-eval', 'running', "
                "  true, NOW() + INTERVAL '90 days'"
                ") RETURNING id"
            ),
            {"iid": persister_setup.installation_id},
        )
        return UUID(str(result.scalar_one()))
