"""Durable AnomalySink implementation.

Per-emit fresh `AsyncSession`; uses `postgresql_insert(...).
on_conflict_do_nothing(...)` against the partial unique index
`uq_anomalies_hitl_timeout_natural_key` (from Group 3's HITL
migration: `WHERE rule_name='hitl_timeout'`).

Idempotency contract: same `(review_id, rule_name='hitl_timeout')`
pair is admitted by the partial unique index AT MOST ONCE; retries
collapse to a no-op. The sweep job's anomaly-first ordering depends
on this — if the anomaly emit raises spuriously, the sweep
short-circuits and the row stays in `awaiting_approval` for the next
sweep tick. With `on_conflict_do_nothing`, a retry of an already-
recorded anomaly returns cleanly and the sweep proceeds to the
status flip.
"""

from typing import Any
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.anomaly.rule_names import AnomalyRuleName
from outrider.db.models.anomalies import Anomaly


class AnomalyPersisterConfigError(ValueError):
    """Raised when `AnomalyPersister` is constructed without
    `session_factory`. Fail-loud at construction time per the
    sibling persister precedents.
    """

    def __init__(self) -> None:
        super().__init__(
            "AnomalyPersister requires session_factory: "
            "pass an async_sessionmaker[AsyncSession] at "
            "sweep-runner-startup time."
        )


class AnomalyPersister:
    """Durable implementation of `AnomalySink`.

    Per-emit fresh `AsyncSession`. `on_conflict_do_nothing` against
    the partial unique index makes the emit idempotent on
    `(review_id, rule_name='hitl_timeout')`. `severity` and
    `status='open'` are V1-hardcoded for the `hitl_timeout` rule;
    future rules extending `AnomalyRuleName` can carry per-rule
    defaults inline at the emit call-site.
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
        severity: str,
        details: dict[str, Any],
    ) -> None:
        """Insert one anomaly row with on-conflict-do-nothing on the
        partial unique index `(review_id, rule_name)` (rule-name-
        partitioned).

        For `rule_name=hitl_timeout`, the partial unique index is
        `uq_anomalies_hitl_timeout_natural_key` from Group 3's
        migration. Other rules added later need matching partial
        unique indexes; this persister's conflict-handling assumes
        such an index exists for every `AnomalyRuleName` value.
        """
        async with self._session_factory() as session, session.begin():
            stmt = (
                postgresql_insert(Anomaly)
                .values(
                    review_id=review_id,
                    rule_name=rule_name.value,
                    severity=severity,
                    details=details,
                    status="open",
                )
                .on_conflict_do_nothing()
            )
            await session.execute(stmt)


__all__ = ["AnomalyPersister", "AnomalyPersisterConfigError"]
