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

        V1 ONLY supports `rule_name=AnomalyRuleName.HITL_TIMEOUT`.
        The on-conflict target — `index_elements=["review_id"]` +
        `index_where=(Anomaly.rule_name == "hitl_timeout")` —
        mirrors the partial unique index
        `uq_anomalies_hitl_timeout_natural_key` from Group 3's
        migration (`ON anomalies (review_id) WHERE rule_name =
        'hitl_timeout'`). Without explicit index targeting,
        PostgreSQL's conflict-arbiter inference may fail to match
        the partial index and incorrectly treat a same-review_id
        retry as a new insert — defeating the idempotency contract
        the sweep's anomaly-first ordering depends on.

        Future rule_names ship with matching partial unique indexes
        AND require this method to dispatch on rule_name (V1.5
        refactor). Fail-loud here when a non-HITL_TIMEOUT rule is
        emitted so the gap is caught at runtime rather than
        silently producing a non-idempotent INSERT.
        """
        if rule_name is not AnomalyRuleName.HITL_TIMEOUT:
            msg = (
                f"AnomalyPersister.emit_anomaly only supports "
                f"AnomalyRuleName.HITL_TIMEOUT in V1; got {rule_name!r}. "
                f"New rule_names need their own partial unique index + "
                f"a dispatch update to the on_conflict target."
            )
            raise NotImplementedError(msg)
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
                .on_conflict_do_nothing(
                    index_elements=["review_id"],
                    index_where=(Anomaly.rule_name == "hitl_timeout"),
                )
            )
            await session.execute(stmt)


__all__ = ["AnomalyPersister", "AnomalyPersisterConfigError"]
