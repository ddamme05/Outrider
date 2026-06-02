# is_eval integrity gate for run_review; consolidation target for the eval_db conftest gate.
# See specs/2026-06-01-eval-graph-driver.md (resolution A) + docs/testing.md "Eval isolation".
"""Loud-failure integrity gate for eval-database isolation.

Every row in an `is_eval`-bearing table (`reviews`, `audit_events`,
`findings`, `llm_call_content`, `anomalies` per `docs/schema.md` "Eval
isolation") produced during an eval run MUST carry `is_eval=True`. Factories
own setting the flag; this gate is the after-the-fact check that catches a
factory — or a direct insert — that forgot it.

The predicate is `IS DISTINCT FROM TRUE` so it flags `FALSE` and `NULL`
alike (robust against a future nullable `is_eval` column). It does NOT
auto-coerce — auto-coercion would mask exactly the bug class the gate exists
to catch.

`run_review` (the eval graph driver) uses it today; `tests/eval/conftest.py`'s
`eval_db` fixture migrates onto it next, so both run the identical query.
Adding a new `is_eval`-bearing table requires extending the UNION here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


class EvalIsolationViolationError(AssertionError):
    """An `is_eval`-bearing table held a row with `is_eval` not TRUE.

    Subclass of `AssertionError` so existing `pytest.raises(AssertionError)`
    contracts in the eval-harness tests continue to hold after migration.
    """


_IS_EVAL_VIOLATIONS_QUERY = text(
    "SELECT 'reviews' AS table_name, id::text AS row_id "
    "FROM reviews WHERE is_eval IS DISTINCT FROM TRUE "
    "UNION ALL "
    "SELECT 'audit_events' AS table_name, event_id::text AS row_id "
    "FROM audit_events WHERE is_eval IS DISTINCT FROM TRUE "
    "UNION ALL "
    "SELECT 'findings' AS table_name, finding_id::text AS row_id "
    "FROM findings WHERE is_eval IS DISTINCT FROM TRUE "
    "UNION ALL "
    "SELECT 'llm_call_content' AS table_name, event_id::text AS row_id "
    "FROM llm_call_content WHERE is_eval IS DISTINCT FROM TRUE "
    "UNION ALL "
    "SELECT 'anomalies' AS table_name, id::text AS row_id "
    "FROM anomalies WHERE is_eval IS DISTINCT FROM TRUE"
)


async def assert_no_is_eval_violations(conn: AsyncConnection) -> None:
    """Raise `EvalIsolationViolationError` if any row has `is_eval` not TRUE.

    Queries the live database; the caller must run it BEFORE dropping the DB.
    """
    result = await conn.execute(_IS_EVAL_VIOLATIONS_QUERY)
    violations = result.all()
    if violations:
        raise EvalIsolationViolationError(
            f"is_eval discipline violation: {len(violations)} row(s) where "
            "is_eval is not TRUE (FALSE or NULL) in the eval database. Eval "
            "writers MUST set is_eval=True; this gate is the loud-failure "
            "check. Violations: "
            f"{[(v.table_name, v.row_id) for v in violations]}"
        )
