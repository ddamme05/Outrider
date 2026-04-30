"""Eval-harness conftest per spec §11.2 + DECISIONS.md#008 Amended 2026-04-29.

Three responsibilities:

1. **`--is-eval` CLI option.** Sets `OUTRIDER_IS_EVAL=1` so the rest of the
   pipeline can detect eval mode if it cares to (see `docs/testing.md`).
2. **`eval_db` fixture.** Fresh-DB-per-test pattern with the URL-pattern
   guard from `_assert_test_url_is_isolated`, mirroring the integration-tier
   `migrated_db` fixture. Created at fixture setup, dropped at teardown.
3. **`is_eval_injection` autouse fixture.** Loud-failure integrity gate per
   the eval-harness spec: at test teardown, if the test used `eval_db`,
   queries `reviews` + `audit_events` for rows where `is_eval = FALSE` and
   raises if any are found. Does NOT auto-coerce — that would mask the
   exact bug class the gate exists to catch (factories that forget the
   flag). Setting `is_eval=True` is the factory's responsibility (see
   `tests/eval/fixtures/factories.py`); this fixture is the after-the-fact
   check. Pattern matches the project's loud-failure discipline
   (`PerFindingDecision.reason` required-no-default,
   `candidates_considered` required-no-default,
   `FindingEvent.finding_content_hash` equality verifier).

Note on conftest scope: this file ONLY applies to tests under `tests/eval/`.
The harness-internal integration test at
`tests/integration/test_eval_harness_is_eval_flag.py` does NOT consume the
autouse fixture (different conftest tree); it tests the propagation
contract directly via explicit assertions.

Helper functions duplicated from `tests/integration/conftest.py` rather
than imported (tests/ is not a package and conftest.py is pytest-special;
cross-conftest imports add fragility for marginal DRY benefit). If both
conftests need to evolve in lockstep, refactor to a shared
`tests/_db_helpers.py` module then.
"""

import asyncio
import os
import re
from collections.abc import AsyncGenerator
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

REPO_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_INI = REPO_ROOT / "alembic.ini"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"

_EXPECTED_TEST_PORT = "5433"
_EXPECTED_TEST_DB_NAME_FRAGMENT = "test"
_PASSWORD_REDACTION = re.compile(r"(://[^:/@\s]+:)([^@]+)(@)")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --is-eval CLI option per `docs/testing.md`.

    Sets `OUTRIDER_IS_EVAL=1` in the environment when present. Eval-harness
    fixture FACTORIES are responsible for setting `is_eval=True` on every
    review and audit row they construct (loud-failure pattern); this flag
    is the explicit signal that we're running in eval mode.
    """
    parser.addoption(
        "--is-eval",
        action="store_true",
        default=False,
        help="Run in eval mode (sets OUTRIDER_IS_EVAL=1).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Honor --is-eval by setting the env var early."""
    if config.getoption("--is-eval"):
        os.environ["OUTRIDER_IS_EVAL"] = "1"


def _redact_url_password(url: str) -> str:
    return _PASSWORD_REDACTION.sub(r"\1***\3", url)


def _assert_test_url_is_isolated(url: str) -> None:
    """Refuse to run if TEST_DATABASE_URL doesn't target the postgres-test container."""
    safe_url = _redact_url_password(url)
    if f":{_EXPECTED_TEST_PORT}" not in url:
        raise RuntimeError(
            f"TEST_DATABASE_URL must target port {_EXPECTED_TEST_PORT} "
            f"(the postgres-test container); got: {safe_url!r}. "
            "Refusing to run eval tests against an unexpected URL — "
            "see docs/testing.md 'Two-container model' for the rationale."
        )
    db_segment = url.rsplit("/", 1)[-1]
    if _EXPECTED_TEST_DB_NAME_FRAGMENT not in db_segment.lower():
        raise RuntimeError(
            f"TEST_DATABASE_URL database name must contain '"
            f"{_EXPECTED_TEST_DB_NAME_FRAGMENT}'; got database segment: "
            f"{db_segment!r}. Refusing to run eval tests against an "
            "unexpected DB."
        )


def _replace_db(url: str, new_db: str) -> str:
    base, _ = url.rsplit("/", 1)
    return f"{base}/{new_db}"


async def _run_alembic_upgrade_head(db_url: str) -> None:
    """Run `alembic upgrade head` against db_url with DATABASE_URL overridden."""
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    try:
        config = Config(str(ALEMBIC_INI), toml_file=str(PYPROJECT_TOML))
        await asyncio.to_thread(command.upgrade, config, "head")
    finally:
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url


@pytest_asyncio.fixture
async def eval_db() -> AsyncGenerator[str]:
    """Fresh migrated Postgres DB per test, dropped at teardown.

    Yields a URL to a brand-new postgres-test database, alembic-upgraded to
    head. The URL pattern guard runs first — a misconfigured
    TEST_DATABASE_URL fails loud before any DDL.
    """
    try:
        main_url = os.environ["TEST_DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError(
            "TEST_DATABASE_URL is not set. Run `set -a && source .env && "
            "set +a` before pytest, and confirm .env has the TEST_ block."
        ) from exc

    _assert_test_url_is_isolated(main_url)

    test_db_name = f"outrider_eval_{uuid4().hex[:8]}"
    test_url = _replace_db(main_url, test_db_name)

    admin_engine = create_async_engine(main_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{test_db_name}"'))
    finally:
        await admin_engine.dispose()

    await _run_alembic_upgrade_head(test_url)

    try:
        yield test_url
    finally:
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


@pytest_asyncio.fixture(autouse=True)
async def is_eval_injection(request: pytest.FixtureRequest) -> AsyncGenerator[None]:
    """Loud-failure integrity gate for the is_eval discipline.

    Yields (test runs); at teardown, if the test used `eval_db`, queries
    `reviews` + `audit_events` for any row with `is_eval = FALSE` and
    raises. Tests that don't use `eval_db` (pure-Pydantic factory tests,
    metrics-shape tests) are no-ops — there's nothing to query.

    Runs BEFORE eval_db's teardown drops the database (autouse fixtures
    teardown in reverse-setup order; if the test requested eval_db, this
    fixture's post-yield code runs first while the DB is still alive).
    """
    yield

    if "eval_db" not in request.fixturenames:
        return

    db_url: str = request.getfixturevalue("eval_db")
    engine = create_async_engine(db_url)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT 'reviews' AS table_name, id::text AS row_id "
                    "FROM reviews WHERE is_eval = FALSE "
                    "UNION ALL "
                    "SELECT 'audit_events' AS table_name, "
                    "event_id::text AS row_id "
                    "FROM audit_events WHERE is_eval = FALSE"
                )
            )
            violations = result.all()
            if violations:
                raise AssertionError(
                    f"is_eval discipline violation: "
                    f"{len(violations)} row(s) with is_eval=False in eval-test DB. "
                    "Factories MUST set is_eval=True; the autouse integrity "
                    "gate is the loud-failure check. Violations: "
                    f"{[(v.table_name, v.row_id) for v in violations]}"
                )
    finally:
        await engine.dispose()
