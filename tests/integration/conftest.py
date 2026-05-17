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
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
# The pyproject_async template puts source-code config (script_location etc.)
# in pyproject.toml under [tool.alembic]; alembic.ini holds DB connection +
# logging only. The CLI auto-discovers both, but a path-constructed Config
# needs the toml_file passed explicitly for ScriptDirectory.from_config to
# find script_location.
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"

# Defense-in-depth URL pattern guard. The integration suite must connect
# to the dedicated postgres-test container (port 5433, db name containing
# "test"), never to the dev DB. If TEST_DATABASE_URL is missing or
# misconfigured (e.g., copy-pasted from DATABASE_URL), the fixture refuses
# to run rather than silently issuing CREATE/DROP DATABASE against the
# dev container. Same posture as docker-compose.yml's `:?` requirement on
# TEST_POSTGRES_*: fail loud at the entrypoint, not somewhere downstream.
_EXPECTED_TEST_PORT = "5433"
_EXPECTED_TEST_DB_NAME_FRAGMENT = "test"

# Mask the password segment of a postgres URL before surfacing it in error
# messages. A misconfigured TEST_DATABASE_URL pointing at the dev or prod
# DB would otherwise spill its password into CI logs and exception traces.
_PASSWORD_REDACTION = re.compile(r"(://[^:/@\s]+:)([^@]+)(@)")


def _redact_url_password(url: str) -> str:
    return _PASSWORD_REDACTION.sub(r"\1***\3", url)


def _assert_test_url_is_isolated(url: str) -> None:
    """Refuse to run if TEST_DATABASE_URL doesn't point at the test container.

    Two checks: the host:port segment must end in :5433, and the database
    name must contain the literal "test". Both are properties of the
    postgres-test container's intended configuration. A URL that fails
    either check is almost certainly a misconfigured .env that points
    the test fixture at the dev DB.

    The error message redacts any password component so a copy-pasted
    dev/prod URL doesn't leak credentials into CI logs.
    """
    safe_url = _redact_url_password(url)
    if f":{_EXPECTED_TEST_PORT}" not in url:
        raise RuntimeError(
            f"TEST_DATABASE_URL must target port {_EXPECTED_TEST_PORT} "
            f"(the postgres-test container); got: {safe_url!r}. "
            "Refusing to run integration tests against an unexpected URL — "
            "see docs/testing.md 'Two-container model' for the rationale."
        )
    db_segment = url.rsplit("/", 1)[-1]
    if _EXPECTED_TEST_DB_NAME_FRAGMENT not in db_segment.lower():
        raise RuntimeError(
            f"TEST_DATABASE_URL database name must contain '"
            f"{_EXPECTED_TEST_DB_NAME_FRAGMENT}' (canonical: outrider_test); "
            f"got database segment: {db_segment!r}. "
            "Refusing to run integration tests against an unexpected DB."
        )


def _replace_db(url: str, new_db: str) -> str:
    """Swap the database name in a postgresql+psycopg:// URL.

    Naive but adequate for the dev URL shape we control. Would need
    revisiting if the URL ever included query parameters after the db
    segment.
    """
    base, _ = url.rsplit("/", 1)
    return f"{base}/{new_db}"


async def _run_alembic_action(action: str, target: str, db_url: str) -> None:
    """Run an alembic command (upgrade/downgrade) with DATABASE_URL overridden.

    env.py reads os.environ["DATABASE_URL"] each time it's exec'd via
    runpy, so setting it here before calling alembic is the URL-injection
    seam. ``asyncio.to_thread`` runs the sync call in a fresh thread so
    env.py's internal ``asyncio.run(run_async_migrations())`` doesn't try
    to nest event loops.
    """
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

    _assert_test_url_is_isolated(main_url)

    test_db_name = f"outrider_test_{uuid4().hex[:8]}"
    test_url = _replace_db(main_url, test_db_name)

    # CREATE DATABASE / DROP DATABASE cannot run inside a transaction
    # block, so the admin engine uses AUTOCOMMIT isolation.
    admin_engine = create_async_engine(main_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    finally:
        await admin_engine.dispose()

    try:
        yield test_url
    finally:
        # Force-disconnect any leftover backends, then drop the DB. This
        # makes teardown robust against tests that leave engines open;
        # without the pg_terminate_backend call, DROP DATABASE fails with
        # "database is being accessed by other users."
        admin_engine = create_async_engine(main_url, isolation_level="AUTOCOMMIT")
        try:
            async with admin_engine.connect() as conn:
                await conn.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :name AND pid <> pg_backend_pid()"
                    ),
                    {"name": test_db_name},
                )
                await conn.execute(text(f'DROP DATABASE IF EXISTS "{test_db_name}"'))
        finally:
            await admin_engine.dispose()


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
                "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                "  total_input_tokens, total_output_tokens, total_cost_usd, "
                "  wall_clock_seconds, retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'running', 0, 0, 0, 0, 0, 0, 0, "
                "  NOW() + INTERVAL '90 days'"
                ") RETURNING id"
            ),
            {"id": installation_id},
        )
        return UUID(str(result.scalar_one()))


if TYPE_CHECKING:
    from outrider.audit.events import LLMCallEvent, ReviewPhaseEvent
    from outrider.llm.base import LLMRequest, LLMResponse


# Type aliases for the factory fixtures.
LLMCallEventFactory = Callable[..., "LLMCallEvent"]
LLMRequestFactory = Callable[..., "LLMRequest"]
LLMResponseFactory = Callable[..., "LLMResponse"]
ReviewPhaseEventFactory = Callable[..., "ReviewPhaseEvent"]


# Canonical prompts used by the request + event factories so their
# prompt_hash / system_prompt_hash values agree by construction. The
# persister's pre-tx guard checks `event.prompt_hash ==
# _canonical_prompt_hash(request.system_prompt, request.user_prompt)`
# — the factories share these prompts so hashes line up unless a test
# deliberately diverges them.
_FACTORY_SYSTEM_PROMPT = "the system prompt"
_FACTORY_USER_PROMPT = "the user prompt"


@pytest.fixture
def llm_call_event_factory() -> LLMCallEventFactory:
    """Factory: `factory(review_id, **kwargs) -> LLMCallEvent` with canonical
    defaults. Tests pass kwargs to override per-case (e.g., distinct
    `cost_usd` to trigger the idempotency-conflict path).
    """
    from datetime import UTC, datetime

    from outrider.audit.events import LLMCallEvent
    from outrider.llm.base import _canonical_prompt_hash, _canonical_system_prompt_hash

    def _build(
        review_id: UUID,
        *,
        cost_usd: float = 0.001,
        latency_ms: int = 250,
        is_eval: bool = False,
        user_prompt: str = _FACTORY_USER_PROMPT,
        system_prompt: str = _FACTORY_SYSTEM_PROMPT,
    ) -> LLMCallEvent:
        return LLMCallEvent(
            review_id=review_id,
            model="claude-haiku-4-5",
            node_id="triage",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=cost_usd,
            pricing_version="1.0.0",
            latency_ms=latency_ms,
            prompt_hash=_canonical_prompt_hash(system_prompt, user_prompt),
            cache_hit=False,
            context_summary=(),
            prompt_template_version="triage:1",
            system_prompt_hash=_canonical_system_prompt_hash(system_prompt),
            degraded_mode=False,
            is_eval=is_eval,
            timestamp=datetime.now(UTC),
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
    from outrider.llm.base import LLMResponse

    def _build(*, text_value: str = "the completion text") -> LLMResponse:
        return LLMResponse(
            text=text_value,
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=250,
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
