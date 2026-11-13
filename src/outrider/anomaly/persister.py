"""Save each operator alert once per review and rule.

AnomalyPersister opens a session for each insert and uses the rule's database
index to ignore retries. New rules need matching indexes. Sweep callers record
the alert before changing status so a failed write can be retried safely.
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

# Match each migration's partial-index predicate with literal SQL. A bound
# parameter cannot establish that predicate under PostgreSQL generic plans,
# causing ON CONFLICT inference to fail (42P10). The import guard requires a
# predicate for every AnomalyRuleName; each new rule also needs its migration.
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
                    # Literal predicate for this rule's partial unique index.
                    index_where=_RULE_NAME_INDEX_WHERE[rule_name],
                )
            )
            await session.execute(stmt)


__all__ = ["AnomalyPersister", "AnomalyPersisterConfigError"]
