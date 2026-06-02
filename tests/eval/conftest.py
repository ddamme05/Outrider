"""Eval-harness conftest per spec §11.2 + DECISIONS.md#008 Amended 2026-04-30.

Two responsibilities:

1. **`--is-eval` CLI option.** Sets `OUTRIDER_IS_EVAL=1` so the rest of the
   pipeline can detect eval mode if it cares to (see `docs/testing.md`).
2. **`eval_db` fixture with bundled integrity gate.** Fresh-DB-per-test
   pattern with the URL-pattern guard from `assert_test_url_is_isolated`,
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

The ephemeral-DB lifecycle primitives (URL-isolation guard, create/drop,
alembic upgrade, integrity gate) live in `outrider.eval_support` — the
shared module `run_review` and the integration conftest also consume.
The two test-private aliases below (`_assert_test_url_is_isolated`,
`_assert_no_is_eval_violations`) re-export the shared functions under the
names the harness-internal tests import (`tests/eval/test_eval_db_isolation_guard.py`,
`tests/eval/test_is_eval_flag.py`); the shared `EvalDBIsolationError` /
`EvalIsolationViolationError` subclass `RuntimeError` / `AssertionError`
respectively, so those tests' `pytest.raises(...)` contracts hold unchanged.
"""

import os
from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from outrider.eval_support import (
    assert_no_is_eval_violations,
    assert_test_url_is_isolated,
    create_database,
    drop_database,
    replace_db_name,
    run_alembic_upgrade_head,
)

# Re-export under the private names the harness-internal tests import. The
# shared `EvalDBIsolationError`/`EvalIsolationViolationError` subclass
# `RuntimeError`/`AssertionError` and preserve the pinned message substrings,
# so the `pytest.raises(...)`/`match=...` contracts in
# `test_eval_db_isolation_guard.py` and `test_is_eval_flag.py` hold unchanged.
_assert_test_url_is_isolated = assert_test_url_is_isolated
_assert_no_is_eval_violations = assert_no_is_eval_violations


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

    Lifecycle primitives come from `outrider.eval_support`; the guard +
    create + migrate + integrity-gate + drop sequence is preserved exactly,
    including the unconditional DROP on any failure path after CREATE.
    """
    try:
        main_url = os.environ["TEST_DATABASE_URL"]
    except KeyError as exc:
        raise RuntimeError(
            "TEST_DATABASE_URL is not set. Run `set -a && source .env && "
            "set +a` before pytest, and confirm .env has the TEST_ block."
        ) from exc

    assert_test_url_is_isolated(main_url)

    test_db_name = f"outrider_eval_{uuid4().hex[:8]}"
    test_url = replace_db_name(main_url, test_db_name)

    await create_database(main_url, test_db_name)

    # DB exists at this point. Wrap alembic + yield + integrity gate in
    # a single try/finally so the DROP cleanup runs on ANY failure path
    # after CREATE — including a migration error. The "fresh-DB-per-test"
    # pattern the spec invokes only holds if the cleanup is unconditional.
    try:
        await run_alembic_upgrade_head(test_url)
        yield test_url

        # Integrity gate (`assert_no_is_eval_violations`, re-exported as the
        # test-private `_assert_no_is_eval_violations` so the gate is directly
        # testable — see tests/eval/test_is_eval_flag.py): query the live DB
        # BEFORE the drop. Pure-Pydantic tests that don't use eval_db never
        # reach this code.
        check_engine = create_async_engine(test_url)
        try:
            async with check_engine.connect() as conn:
                await assert_no_is_eval_violations(conn)
        finally:
            await check_engine.dispose()
    finally:
        await drop_database(main_url, test_db_name)


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
