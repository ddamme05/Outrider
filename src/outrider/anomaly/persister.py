"""Durable AnomalySink implementation.

Per-emit fresh `AsyncSession`; uses `postgresql_insert(...).
on_conflict_do_nothing(...)` against per-rule partial unique indexes
(e.g. `uq_anomalies_hitl_timeout_natural_key`,
`uq_anomalies_cross_round_severity_divergence_natural_key`). The
`index_where` predicate is computed from the runtime `rule_name`
value — each rule's partial index has the form
`WHERE rule_name='<rule_value>'`, and PostgreSQL's conflict-arbiter
needs the exact `index_where` form to match a partial index.

Idempotency contract: same `(review_id, rule_name)` pair is admitted
by the partial unique index AT MOST ONCE; retries collapse to a
no-op. Per `AnomalySink` Protocol docstring, the persister serves
two caller classes:

- **Sweep callers** (`sweep/hitl_expiry.py`): rely on `SWEEP_LOCK_ID`
  for surrounding-operation serialization; the partial unique index
  here provides per-row idempotency. Anomaly-first ordering depends
  on this — if the anomaly emit raises spuriously, the sweep
  short-circuits and the row stays in `awaiting_approval` for the
  next sweep tick. With `on_conflict_do_nothing`, a retry of an
  already-recorded anomaly returns cleanly and the sweep proceeds
  to the status flip.

- **Graph callers** (`agent/nodes/synthesize.py`): no surrounding
  advisory lock — anomaly emission has no non-idempotent external
  side effect; concurrent or replayed inserts collapse via the
  partial unique index. Same DB-layer idempotency mechanism; the
  caller-class distinction lives entirely in the surrounding
  concurrency story (sweep needs the advisory lock for the
  surrounding status flip; graph does not).

The `is_eval` column is written explicitly per `docs/testing.md`'s
loud-failure convention (every is_eval-bearing row's flag is set
by the producer; the column's `server_default=text("false")` is
defense in depth, NOT the primary contract). Eval-scenario emissions
land with `is_eval=True` and are filtered out of the production
anomaly queue + pass the eval-DB teardown integrity gate.
"""

from typing import Any
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
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
            "startup time (sweep-runner OR build_graph)."
        )


class AnomalyPersister:
    """Durable implementation of `AnomalySink`.

    Per-emit fresh `AsyncSession`. `on_conflict_do_nothing` against
    the runtime-`rule_name`'s partial unique index makes the emit
    idempotent on `(review_id, rule_name)`. `severity` and
    `is_eval` are caller-controlled (no V1 defaults — both are
    loud-failure surfaces); `status='open'` is V1-hardcoded because
    V1 has no other terminal state at emit-time.

    Serves both sweep callers (HITL_TIMEOUT) and graph callers
    (CROSS_ROUND_SEVERITY_DIVERGENCE). The dispatch is implicit in
    the `index_where=(Anomaly.rule_name == rule_name.value)` clause
    — each rule's partial unique index has a matching predicate.
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
        """Insert one anomaly row with on-conflict-do-nothing on the
        partial unique index `(review_id, rule_name)` matching the
        runtime `rule_name`.

        The on-conflict target — `index_elements=["review_id"]` +
        `index_where=(Anomaly.rule_name == rule_name.value)` —
        dispatches on the runtime rule_name. PostgreSQL's
        conflict-arbiter requires the `index_where` predicate to
        match the partial unique index exactly; without it the
        inference may fall through to a non-partial unique index or
        treat a same-review_id retry as a new insert — defeating the
        idempotency contract.

        Every AnomalyRuleName value must have a matching partial
        unique index in the DB (created by a Group 3-or-later
        migration with `WHERE rule_name = '<value>'`). If the
        migration is missing for a new rule_name, the
        on_conflict_do_nothing falls through silently — the insert
        succeeds AS A NEW ROW each time, breaking idempotency.
        Producer-side discipline: every new AnomalyRuleName ships
        with a matching migration in the same PR.
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
                    # `.value` extracts the `Text` form the DB column
                    # stores; the partial-index predicate is matched
                    # by the runtime rule_name (each rule_name has its
                    # own partial unique index).
                    index_where=(Anomaly.rule_name == rule_name.value),
                )
            )
            await session.execute(stmt)


__all__ = ["AnomalyPersister", "AnomalyPersisterConfigError"]
