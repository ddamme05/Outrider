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
from collections.abc import AsyncGenerator, Awaitable, Callable
from pathlib import Path
from uuid import uuid4

import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

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


def _assert_test_url_is_isolated(url: str) -> None:
    """Refuse to run if TEST_DATABASE_URL doesn't point at the test container.

    Two checks: the host:port segment must end in :5433, and the database
    name must contain the literal "test". Both are properties of the
    postgres-test container's intended configuration. A URL that fails
    either check is almost certainly a misconfigured .env that points
    the test fixture at the dev DB.
    """
    if f":{_EXPECTED_TEST_PORT}" not in url:
        raise RuntimeError(
            f"TEST_DATABASE_URL must target port {_EXPECTED_TEST_PORT} "
            f"(the postgres-test container); got: {url!r}. "
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
