"""Persist anomalies once per `(review_id, rule_name)`.

Each emit opens a fresh session and inserts with `on_conflict_do_nothing`
against the rule's partial unique index. `_RULE_NAME_INDEX_WHERE` supplies
literal SQL predicates: a bound predicate can fail PostgreSQL index inference
under prepared generic plans. Every rule needs a matching index migration.

Sweep callers hold `SWEEP_LOCK_ID` around their status changes and emit the
anomaly first, so a failed emit leaves expiry retryable. Graph callers rely
on insert idempotency without a sweep lock. Both pass `severity` and `is_eval`
explicitly; the database default is only a backstop.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final
from uuid import UUID

from sqlalchemy import TextClause
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.db.models.anomalies import Anomaly

# Per-rule partial-index predicates as LITERAL SQL, never ORM expressions.
# The ON CONFLICT arbiter only matches a rule's partial unique index when
# `index_where` is literal text identical to the migration's `CREATE UNIQUE
# INDEX ... WHERE rule_name = '<value>'`. An ORM expression
# (`Anomaly.rule_name == rule_name.value`) renders a bind parameter, which
# psycopg3 generic plans can't prove implies the index's constant predicate,
# so arbiter inference fails (42P10) once the statement is server-prepared —
# silently defeating the idempotent no-op the HITL sweep depends on. Mirrors
# the literal `sa_text(...)` form the audit natural-key path uses. Every
# AnomalyRuleName MUST have an entry (enforced at import below); a new rule
# without its literal predicate fails loud rather than shipping the bug.
_RULE_NAME_INDEX_WHERE: Final[Mapping[AnomalyRuleName, TextClause]] = MappingProxyType(
    {
        AnomalyRuleName.HITL_TIMEOUT: sa_text("rule_name = 'hitl_timeout'"),
        AnomalyRuleName.CROSS_ROUND_SEVERITY_DIVERGENCE: sa_text(
            "rule_name = 'cross_round_severity_divergence'"
        ),
        AnomalyRuleName.COST_BUDGET_STARVATION: sa_text("rule_name = 'cost_budget_starvation'"),
        AnomalyRuleName.GATED_FINDINGS_OVER_CAP: sa_text("rule_name = 'gated_findings_over_cap'"),
    }
)

_UNMAPPED_RULES = set(AnomalyRuleName) - set(_RULE_NAME_INDEX_WHERE)
if _UNMAPPED_RULES:  # pragma: no cover - import-time guard against a new rule
    raise RuntimeError(
        "AnomalyRuleName members missing a literal partial-index predicate in "
        f"_RULE_NAME_INDEX_WHERE: {sorted(r.value for r in _UNMAPPED_RULES)}. "
        "Add one matching the rule's `CREATE UNIQUE INDEX ... WHERE rule_name "
        "= '<value>'` migration."
    )


class AnomalyPersisterConfigError(ValueError):
    """Raised when `AnomalyPersister` is constructed without
    `session_factory`. Fail-loud at construction time per the
    sibling persister precedents.
    """

    def __init__(self) -> None:
        super().__init__(
            "AnomalyPersister requires session_factory: "
            "pass an async_sessionmaker[AsyncSession] at "
            "startup time (sweep-runner OR build_graph)."
        )


class AnomalyPersister:
    """Durable `AnomalySink` with one fresh transaction per emit.

    The matching partial index deduplicates retries. Callers supply severity,
    details, and eval scope; new rows start with `status='open'`.
    """

    def __init__(self, *, session_factory: async_sessionmaker[AsyncSession]) -> None:
        if session_factory is None:
            raise AnomalyPersisterConfigError()
        self._session_factory = session_factory

    async def emit_anomaly(
        self,
        *,
        review_id: UUID,
        rule_name: AnomalyRuleName,
        severity: AnomalySeverity,
        details: dict[str, Any],
        is_eval: bool,
    ) -> None:
        """Insert an anomaly or accept an existing `(review_id, rule_name)`.

        The literal `index_where` must match the rule's partial unique index.
        A new rule must ship its predicate and index migration together; a missing
        index makes PostgreSQL reject the conflict target rather than deduplicate.
        """
        async with self._session_factory() as session, session.begin():
            stmt = (
                postgresql_insert(Anomaly)
                .values(
                    review_id=review_id,
                    rule_name=rule_name.value,
                    severity=severity.value,
                    details=details,
                    status="open",
                    is_eval=is_eval,
                )
                .on_conflict_do_nothing(
                    index_elements=["review_id"],
                    # LITERAL partial-index predicate keyed by the runtime
                    # rule_name (each rule_name has its own partial unique
                    # index). MUST be literal SQL, not an ORM expression —
                    # see `_RULE_NAME_INDEX_WHERE` above for why a bind
                    # parameter breaks arbiter inference under generic plans.
                    index_where=_RULE_NAME_INDEX_WHERE[rule_name],
                )
            )
            await session.execute(stmt)


__all__ = ["AnomalyPersister", "AnomalyPersisterConfigError"]
