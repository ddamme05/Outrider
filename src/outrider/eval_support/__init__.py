"""Eval-harness infrastructure that must be importable from `src`.

This subpackage exists for one structural reason: the eval graph driver
(`run_review`, exported from `outrider.agent`) is reachable from production
`src/outrider/`, the eval scenarios call it with only a fixture path (no
injection seam), so it must self-construct its dependencies and run its own
ephemeral-database lifecycle. Because `pythonpath = ["src"]` only — `tests/`
is NOT importable from `src/` (see `docs/conventions.md`) — that lifecycle
code cannot live under `tests/`. It lives here instead so that `run_review`
can call it, and it is the single shared implementation of the
create/migrate/drop/guard lifecycle: `run_review`, `tests/integration/conftest.py`,
`tests/eval/conftest.py`, and `scripts/smoke_e2e.py` all import it *from* `src`
(tests may import `src`, not vice-versa) rather than each keeping a copy.

**This is eval/test infrastructure, walled off from the real subsystems.**
It is not an `agent/` concern and not normal production database behavior —
it issues `CREATE DATABASE` / `DROP DATABASE` against the ephemeral
`postgres-test` container only. The fail-closed guards make that explicit:

- `require_eval_mode()` refuses to proceed unless `OUTRIDER_IS_EVAL=1`.
- `assert_test_url_is_isolated()` refuses any URL that is not the test
  container (port 5433, "test" in the database name).

See `specs/2026-06-01-eval-graph-driver.md` (resolution A) for why this
package exists.
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
