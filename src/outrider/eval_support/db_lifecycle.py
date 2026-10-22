# Ephemeral postgres-test lifecycle shared by run_review + the test conftests + smoke runner.
# See specs/2026-06-01-eval-graph-driver.md (resolution A) for why this is `src`-resident.
"""Create, migrate and drop ephemeral test databases.

Drivers, test fixtures and the smoke runner share this lifecycle. Each exported
create/drop primitive checks test URL markers (port 5433 and "test" in the
database name) and validates the SQL identifier. Eval-only entry points call
require_eval_mode separately; integration fixtures also use these primitives.
URL errors redact passwords.

CREATE/DROP use AUTOCOMMIT, and drop terminates target connections first.
Alembic runs in a worker thread with DATABASE_URL temporarily set, because its
env.py starts its own event loop. See docs/testing.md "Two-container model"."""

from __future__ import annotations

import asyncio
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Repo-root-relative alembic config. This module is eval/test infrastructure
# and assumes a repo checkout (alembic.ini lives at the repo root, not beside
# an installed wheel) — the same assumption the conftests make.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_INI = _REPO_ROOT / "alembic.ini"
_PYPROJECT_TOML = _REPO_ROOT / "pyproject.toml"

EXPECTED_TEST_PORT = 5433
EXPECTED_TEST_DB_NAME_FRAGMENT = "test"
# Name prefix for per-test ephemeral eval DBs (minted by `ephemeral_database` and
# the eval_db fixture). The single source of truth so a guard that refuses
# anything-but-a-per-test-DB can't drift from what actually creates them.
EVAL_DB_NAME_PREFIX = "outrider_eval_"

_EVAL_MODE_ENV_VAR = "OUTRIDER_IS_EVAL"
_PASSWORD_REDACTION = re.compile(r"(://[^:/@\s]+:)([^@]+)(@)")
# Ephemeral DB names are generated as `<prefix>_<uuid4-hex>` (lowercase letters,
# digits, underscore). create/drop interpolate the name into a quoted SQL
# identifier; this pattern guards the exported primitives against a
# quote-breaking name from a future caller.
_VALID_DB_NAME = re.compile(r"[a-z0-9_]+")


class EvalDBIsolationError(RuntimeError):
    """A URL failed the test port/name checks before database DDL."""


class EvalModeNotEnabledError(RuntimeError):
    """`OUTRIDER_IS_EVAL=1` was not set when an eval-only path was entered.

    Guards `run_review`, which is reachable from production `src/` and must
    refuse to create/drop databases outside a declared eval session.
    """


def redact_url_password(url: str) -> str:
    """Render a database URL with the password masked, for safe logging."""
    return _PASSWORD_REDACTION.sub(r"\1***\3", url)


def require_eval_mode() -> None:
    """Raise `EvalModeNotEnabledError` unless `OUTRIDER_IS_EVAL=1` is set."""
    if os.environ.get(_EVAL_MODE_ENV_VAR) != "1":
        raise EvalModeNotEnabledError(
            f"{_EVAL_MODE_ENV_VAR}=1 is required before any eval database "
            "lifecycle operation (run_review / ephemeral_database). This "
            "guard prevents a stray call from issuing CREATE/DROP DATABASE "
            "outside a declared eval session. Run under pytest with "
            "--is-eval, or export the variable explicitly."
        )


def assert_test_url_is_isolated(url: str) -> None:
    """Require port 5433 and "test" in the database name.

    Parse with make_url so password/host substrings cannot satisfy those checks.
    Malformed URLs raise EvalDBIsolationError with password-redacted detail."""
    safe_url = redact_url_password(url)
    try:
        parsed = make_url(url)
    except (ArgumentError, ValueError) as exc:
        raise EvalDBIsolationError(
            f"the provided value is not a parseable database URL; got {safe_url!r}. "
            "See docs/testing.md 'Two-container model'."
        ) from exc
    if parsed.port != EXPECTED_TEST_PORT:
        raise EvalDBIsolationError(
            f"database URL must target port {EXPECTED_TEST_PORT} (the "
            f"postgres-test container); got port {parsed.port!r} in {safe_url!r}. "
            "Refusing to run against an unexpected URL — see docs/testing.md "
            "'Two-container model'."
        )
    db_name = parsed.database or ""
    if EXPECTED_TEST_DB_NAME_FRAGMENT not in db_name.lower():
        raise EvalDBIsolationError(
            f"database name must contain {EXPECTED_TEST_DB_NAME_FRAGMENT!r}; "
            f"got database {db_name!r}. Refusing to run against an unexpected DB."
        )


def replace_db_name(url: str, new_db: str) -> str:
    """Return `url` with its database component swapped for `new_db`.

    Uses `make_url(...).set(database=...)` so query parameters and the rest
    of the URL are preserved; the password is NOT hidden in the result (the
    returned URL is used to connect, not to log).
    """
    return make_url(url).set(database=new_db).render_as_string(hide_password=False)


def _validate_db_name(db_name: str) -> None:
    """Reject a db_name that isn't a generated `[a-z0-9_]+` identifier.

    `create_database` / `drop_database` interpolate `db_name` into a quoted SQL
    identifier (`"<name>"`). These are exported primitives, so guard against a
    future caller passing a quote-breaking name. All current callers generate
    `<prefix>_<uuid4-hex>`, which matches.
    """
    if not _VALID_DB_NAME.fullmatch(db_name):
        raise ValueError(
            f"db_name must match [a-z0-9_]+ (ephemeral test DB names are "
            f"generated as <prefix>_<uuid4-hex>); got {db_name!r}"
        )


async def create_database(admin_url: str, db_name: str) -> None:
    """`CREATE DATABASE "<db_name>"` via an AUTOCOMMIT admin connection.

    Runs `assert_test_url_is_isolated` first. These DDL primitives are exported,
    so the guard lives INSIDE them (not only in `ephemeral_database`) — an
    arbitrary-URL caller must not be able to CREATE against a non-test DB.
    """
    assert_test_url_is_isolated(admin_url)
    _validate_db_name(db_name)
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    finally:
        await engine.dispose()


async def drop_database(admin_url: str, db_name: str) -> None:
    """Terminate other backends, then `DROP DATABASE IF EXISTS "<db_name>"`.

    The `pg_terminate_backend` sweep is required because an idle pooled
    connection to the target DB makes `DROP DATABASE` fail; it excludes the
    admin connection itself (`pid <> pg_backend_pid()`).

    Runs `assert_test_url_is_isolated` first — same exported-primitive
    fail-closed reasoning as `create_database`: no DROP against a non-test DB.
    """
    assert_test_url_is_isolated(admin_url)
    _validate_db_name(db_name)
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as conn:
            await conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": db_name},
            )
            await conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
    finally:
        await engine.dispose()


async def run_alembic_upgrade_head(db_url: str) -> None:
    """Run `alembic upgrade head` against `db_url`.

    `DATABASE_URL` is overridden for the duration of the call (env.py reads
    it on each invocation) and restored afterward; the sync alembic command
    runs in a worker thread so it doesn't collide with the event loop.
    """
    original_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    try:
        config = Config(str(_ALEMBIC_INI), toml_file=str(_PYPROJECT_TOML))
        await asyncio.to_thread(command.upgrade, config, "head")
    finally:
        if original_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original_url


@asynccontextmanager
async def ephemeral_database(
    *, base_url: str, name_prefix: str = EVAL_DB_NAME_PREFIX
) -> AsyncIterator[str]:
    """Create a uniquely-named ephemeral DB on the test container; drop on exit.

    Yields the URL of the freshly-created (NOT yet migrated) database — the
    caller decides whether to run `run_alembic_upgrade_head` against it. The
    isolation guard runs first; the `DROP` is unconditional (runs on any exit
    path after the `CREATE`, including a migration or run-time failure), so
    the create→drop pairing holds even when the body raises.

    Does NOT call `require_eval_mode()` — the integration conftest's non-eval
    `fresh_db` path uses this CM too. Eval-only entry points (`run_review`)
    call `require_eval_mode()` themselves before entering.
    """
    assert_test_url_is_isolated(base_url)
    db_name = f"{name_prefix}{uuid4().hex[:8]}"
    db_url = replace_db_name(base_url, db_name)
    await create_database(base_url, db_name)
    try:
        yield db_url
    finally:
        await drop_database(base_url, db_name)
