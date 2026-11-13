"""Database setup and isolation checks shared by evaluation runners.

Helpers accept only isolated test database URLs. Evaluation entry points must
also set OUTRIDER_IS_EVAL=1 and mark their records as evaluation data.
"""

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
