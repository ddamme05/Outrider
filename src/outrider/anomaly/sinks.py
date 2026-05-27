"""AnomalySink Protocol — typed seam for anomaly emission.

Same shape as `audit/sinks.py` Protocols + the `db/sinks.py`
ReviewStatusSink: `@runtime_checkable`, kwargs-only method,
closure-injected at sweep-runner-startup time.

V1 has one emit method (`emit_anomaly`). Idempotency is owned by the
durable `AnomalyPersister` via `postgresql_insert(...).
on_conflict_do_nothing(...)` against the partial unique index
`uq_anomalies_hitl_timeout_natural_key` (from Group 3's migration).
A retry of the same `(review_id, rule_name)` is a clean no-op.

Both `rule_name` and `severity` are typed StrEnums (not bare str),
so a typo at call-site fails mypy before reaching the DB. The DB
columns remain `Text` so future rule/severity additions don't force
a migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity


@runtime_checkable
class AnomalySink(Protocol):
    """Emit one anomaly row. Idempotent on `(review_id, rule_name)`
    via the persister-side partial unique index.

    Per `sweep-jobs-use-advisory-locks`: callers (the sweep job)
    acquire `SWEEP_LOCK_ID` BEFORE invoking `emit_anomaly`, so
    cross-process serial-correctness is enforced at the lock layer
    AND the DB-level partial unique index. Within a single sweep
    process, the loop body is sequential.

    Contract:
      - Returns `None` on success (no payload echo; the audit shadow
        is the `anomalies` row).
      - `severity` (`AnomalySeverity` StrEnum) and `details` are
        caller-controlled — the StrEnum prevents typos at the
        producer boundary (e.g. "Medium" with uppercase or "med")
        even though the DB column is Text. `details` is a JSON-
        native dict.
      - `status` defaults to "open" inside the persister; not
        exposed on the Protocol because V1 has no other terminal
        state at emit-time.
    """

    async def emit_anomaly(
        self,
        *,
        review_id: UUID,
        rule_name: AnomalyRuleName,
        severity: AnomalySeverity,
        details: dict[str, Any],
    ) -> None:
        """Persist one anomaly row.

        Idempotent on `(review_id, rule_name)` per the partial unique
        index — a retry of the same `(review_id, rule_name)` pair is
        a no-op (returns successfully without raising).
        """
        ...


__all__ = ["AnomalySink"]
