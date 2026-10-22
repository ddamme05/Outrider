"""Shared eval/test database lifecycle and isolation checks.

Eval drivers live in src and cannot import tests (docs/conventions.md). Drivers,
test fixtures and the smoke runner therefore share these create/migrate/drop
helpers; see `specs/2026-06-01-eval-graph-driver.md` (resolution A).

Database lifecycle helpers require test URL markers: port 5433 and "test" in the
database name. Eval-only entry points additionally require OUTRIDER_IS_EVAL=1.
The post-run integrity check rejects rows whose is_eval flag is not True."""

from outrider.eval_support.db_lifecycle import (
    EVAL_DB_NAME_PREFIX,
    EXPECTED_TEST_DB_NAME_FRAGMENT,
    EXPECTED_TEST_PORT,
    EvalDBIsolationError,
    EvalModeNotEnabledError,
    assert_test_url_is_isolated,
    create_database,
    drop_database,
    ephemeral_database,
    redact_url_password,
    replace_db_name,
    require_eval_mode,
    run_alembic_upgrade_head,
)
from outrider.eval_support.isolation import (
    EvalIsolationViolationError,
    assert_no_is_eval_violations,
)

__all__ = [
    "EVAL_DB_NAME_PREFIX",
    "EXPECTED_TEST_DB_NAME_FRAGMENT",
    "EXPECTED_TEST_PORT",
    "EvalDBIsolationError",
    "EvalIsolationViolationError",
    "EvalModeNotEnabledError",
    "assert_no_is_eval_violations",
    "assert_test_url_is_isolated",
    "create_database",
    "drop_database",
    "ephemeral_database",
    "redact_url_password",
    "replace_db_name",
    "require_eval_mode",
    "run_alembic_upgrade_head",
]
