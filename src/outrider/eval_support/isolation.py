"""Detect evaluation records that lack is_eval=True.

The checks include NULL flags and report contaminated tables rather than
silently changing data. Runners use them to verify database isolation before
teardown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection


class EvalIsolationViolationError(AssertionError):
    """An eval table contained an is_eval flag other than True.

    AssertionError preserves the eval harness's failure contract."""


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
    "FROM anomalies WHERE is_eval IS DISTINCT FROM TRUE "
    "UNION ALL "
    "SELECT 'analyze_file_cache' AS table_name, cache_key AS row_id "
    "FROM analyze_file_cache WHERE is_eval IS DISTINCT FROM TRUE"
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
