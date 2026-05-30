"""Eval-harness conftest per spec §11.2 + DECISIONS.md#008 Amended 2026-04-30.

Two responsibilities:

1. **`--is-eval` CLI option.** Sets `OUTRIDER_IS_EVAL=1` so the rest of the
   pipeline can detect eval mode if it cares to (see `docs/testing.md`).
2. **`eval_db` fixture with bundled integrity gate.** Fresh-DB-per-test
   pattern with the URL-pattern guard from `_assert_test_url_is_isolated`,
   mirroring the integration-tier `migrated_db` fixture. Created at fixture
   setup, integrity-gate query at the start of teardown, dropped at the
   end of teardown. The integrity gate is loud-failure: it queries every
   `is_eval`-bearing table (`reviews`, `audit_events`, `findings`,
   `llm_call_content`, `anomalies` per `docs/schema.md` "Eval isolation")
   for rows where `is_eval` is not TRUE (FALSE or NULL, via `IS DISTINCT
   FROM TRUE` — robust against a future nullable `is_eval` column) and
   raises `AssertionError` if any are found. Does NOT auto-coerce — that
   would mask the exact bug class the
   gate exists to catch (factories that forget the flag). Setting
   `is_eval=True` is the factory's responsibility (see
   `tests/eval/fixtures/factories.py`); this gate is the after-the-fact
   check. Pattern matches the project's loud-failure discipline
   (`PerFindingDecision.reason` required-no-default,
   `proposed_import_strings` required-no-default,
   `FindingEvent.finding_content_hash` equality verifier).

   Earlier drafts split this into a separate `is_eval_injection` autouse
   fixture; folded into `eval_db`'s teardown to make the order
   deterministic (autouse fixtures of the same scope can set up before
   explicit fixtures, putting the autouse's post-yield query AFTER
   `eval_db`'s drop — querying a dropped DB).

Conftest scope: this file applies to tests under `tests/eval/`. The
harness-internal tests at `tests/eval/test_*.py` (not under
`tests/eval/scenarios/`) consume `eval_db` directly when they need a DB.

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
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

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
    """Honor --is-eval by setting the env var; refuse to run without it.

    Fail-loud rationale: this conftest only loads when pytest collects tests
    under `tests/eval/`, so reaching this hook means eval tests are in scope.
    Today's row-level isolation comes from factories setting `is_eval=True`
    plus the `eval_db` teardown integrity gate; the `--is-eval` flag is the
    entry-point declaration that future graph code (LLM provider wrappers,
    dispatchers, sweep jobs) reads via `OUTRIDER_IS_EVAL=1` to alter behavior.
    Failing loud here keeps `docs/testing.md`'s "do not run eval without
    --is-eval" claim true today AND when the future readers land — soft
    convention rots the moment a graph node starts branching on the env var.
    """
    if not config.getoption("--is-eval"):
        raise pytest.UsageError(
            "tests/eval/ requires the --is-eval flag. Run: pytest tests/eval --is-eval"
        )
    os.environ["OUTRIDER_IS_EVAL"] = "1"


def _redact_url_password(url: str) -> str:
    return _PASSWORD_REDACTION.sub(r"\1***\3", url)


def _assert_test_url_is_isolated(url: str) -> None:
    """Refuse to run if TEST_DATABASE_URL doesn't target the postgres-test container.

    Parses the URL with SQLAlchemy `make_url` and checks the port and database
    as structured components. Substring matching could false-match a port-like
    or "test"-like sequence embedded in a password or host.
    """
    safe_url = _redact_url_password(url)
    try:
        parsed = make_url(url)
    except (ArgumentError, ValueError) as exc:
        # make_url raises ArgumentError for an unparseable URL, but a bare
        # builtins.ValueError (NOT an ArgumentError subclass) when the port
        # component is non-numeric (int() on the port) — catch both so a
        # typo'd/non-numeric port surfaces this RuntimeError, not a raw ValueError.
        raise RuntimeError(
            f"TEST_DATABASE_URL is not a parseable database URL; got: "
            f"{safe_url!r}. See docs/testing.md 'Two-container model'."
        ) from exc
    if parsed.port != int(_EXPECTED_TEST_PORT):
        raise RuntimeError(
            f"TEST_DATABASE_URL must target port {_EXPECTED_TEST_PORT} "
            f"(the postgres-test container); got port {parsed.port!r} in "
            f"{safe_url!r}. Refusing to run eval tests against an unexpected "
            "URL — see docs/testing.md 'Two-container model' for the rationale."
        )
    db_name = parsed.database or ""
    if _EXPECTED_TEST_DB_NAME_FRAGMENT not in db_name.lower():
        raise RuntimeError(
            f"TEST_DATABASE_URL database name must contain "
            f"'{_EXPECTED_TEST_DB_NAME_FRAGMENT}'; got database {db_name!r}. "
            "Refusing to run eval tests against an unexpected DB."
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


async def _assert_no_is_eval_violations(conn: AsyncConnection) -> None:
    """Raise AssertionError if any is_eval-bearing table holds a non-eval row.

    Loud-failure integrity gate: every row in a table that carries `is_eval`
    (per `docs/schema.md` "Eval isolation": reviews, audit_events, findings,
    llm_call_content, anomalies) must have is_eval=True. Factories own setting
    the flag; this gate catches a factory or direct insertion that forgot it.
    The predicate is `IS DISTINCT FROM TRUE` so it flags FALSE and NULL alike
    (robust against a future nullable column). Adding a new is_eval-bearing
    table requires extending this UNION. Extracted from `eval_db`'s teardown so
    the gate is directly testable (see tests/eval/test_is_eval_flag.py).
    """
    result = await conn.execute(
        text(
            "SELECT 'reviews' AS table_name, id::text AS row_id "
            "FROM reviews WHERE is_eval IS DISTINCT FROM TRUE "
            "UNION ALL "
            "SELECT 'audit_events' AS table_name, "
            "event_id::text AS row_id "
            "FROM audit_events WHERE is_eval IS DISTINCT FROM TRUE "
            "UNION ALL "
            "SELECT 'findings' AS table_name, "
            "finding_id::text AS row_id "
            "FROM findings WHERE is_eval IS DISTINCT FROM TRUE "
            "UNION ALL "
            "SELECT 'llm_call_content' AS table_name, "
            "event_id::text AS row_id "
            "FROM llm_call_content WHERE is_eval IS DISTINCT FROM TRUE "
            "UNION ALL "
            "SELECT 'anomalies' AS table_name, id::text AS row_id "
            "FROM anomalies WHERE is_eval IS DISTINCT FROM TRUE"
        )
    )
    violations = result.all()
    if violations:
        raise AssertionError(
            f"is_eval discipline violation: {len(violations)} "
            "row(s) where is_eval is not TRUE (FALSE or NULL) in eval-test DB. "
            "Factories MUST set is_eval=True; this gate is the "
            "loud-failure check (per `PerFindingDecision.reason` "
            "no-default + `proposed_import_strings` no-default + "
            "`FindingEvent.finding_content_hash` equality "
            "verifier discipline). Violations: "
            f"{[(v.table_name, v.row_id) for v in violations]}"
        )


@pytest_asyncio.fixture
async def eval_db() -> AsyncGenerator[str]:
    """Fresh migrated Postgres DB per test, dropped at teardown.

    Yields a URL to a brand-new postgres-test database, alembic-upgraded to
    head. The URL pattern guard runs first — a misconfigured
    TEST_DATABASE_URL fails loud before any DDL.

    **Integrity gate runs in this fixture's own teardown** (not via a
    separate autouse fixture). Pytest fixture teardown is reverse-setup
    order, so a separate autouse fixture's post-yield query could race
    against the eval_db post-yield drop (autouse fixtures of the same
    scope can set up before explicit fixtures, putting eval_db drop
    BEFORE the autouse query). Baking the check into eval_db's own
    finalizer makes the order deterministic: query, THEN drop.
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

    # DB exists at this point. Wrap alembic + yield + integrity gate in
    # a single try/finally so the DROP cleanup runs on ANY failure path
    # after CREATE — including a migration error. Earlier draft put
    # `await _run_alembic_upgrade_head(test_url)` outside the try block,
    # which leaked outrider_eval_* DBs on postgres-test whenever migrations
    # failed (the function exited before reaching the yield's try/finally).
    # The "fresh-DB-per-test" pattern the spec invokes only holds if the
    # cleanup is unconditional — fix surfaces the implicit cleanup contract
    # the spec assumed but didn't enumerate.
    try:
        await _run_alembic_upgrade_head(test_url)
        yield test_url

        # Integrity gate (extracted to `_assert_no_is_eval_violations` so the
        # gate is directly testable — see tests/eval/test_is_eval_flag.py):
        # query the live DB BEFORE the drop. Pure-Pydantic tests that don't use
        # eval_db never reach this code.
        check_engine = create_async_engine(test_url)
        try:
            async with check_engine.connect() as conn:
                await _assert_no_is_eval_violations(conn)
        finally:
            await check_engine.dispose()
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


@pytest_asyncio.fixture
async def eval_db_session_factory(
    eval_db: str,
) -> AsyncGenerator[async_sessionmaker[AsyncSession]]:
    """An ``async_sessionmaker`` bound to the per-test eval database.

    The HITL-resume scenario (FUP-105) injects this into ``AuditReplayer``
    so replay reconstructs from the same eval DB the run was recorded in.
    Built from ``eval_db``'s URL with ``expire_on_commit=False`` (mirrors the
    ``AuditPersister`` construction discipline) and ``NullPool`` so no idle
    connection lingers to race ``eval_db``'s teardown DROP — the engine is
    disposed here (inner fixture) before ``eval_db`` drops the database.
    """
    engine = create_async_engine(eval_db, poolclass=NullPool)
    try:
        yield async_sessionmaker(engine, expire_on_commit=False)
    finally:
        await engine.dispose()
