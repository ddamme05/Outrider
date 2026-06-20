"""AnomalySink Protocol — typed seam for anomaly emission.

Same shape as `audit/sinks.py` Protocols + the `db/sinks.py`
ReviewStatusSink: `@runtime_checkable`, kwargs-only method,
closure-injected at startup time.

V1 has one emit method (`emit_anomaly`). Idempotency is owned by the
durable `AnomalyPersister` via `postgresql_insert(...).
on_conflict_do_nothing(...)` against per-rule partial unique indexes
(e.g. `uq_anomalies_hitl_timeout_natural_key`,
`uq_anomalies_cross_round_severity_divergence_natural_key`). A retry
of the same `(review_id, rule_name)` is a clean no-op regardless of
caller class.

Both `rule_name` and `severity` are typed StrEnums (not bare str),
so a typo at call-site fails mypy before reaching the DB. `is_eval`
is mandatory (no default) per the loud-failure convention in
`docs/testing.md` "Eval isolation end-to-end" — every is_eval-bearing
row's flag is set explicitly by the producer; a call site that omits
the flag is a bug. The DB columns remain `Text` (rule_name/severity)
so future rule/severity additions don't force a migration.
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

    **Two caller classes** with different concurrency contracts. The
    DB-side idempotency mechanism is the same for both; the
    surrounding lock semantics differ:

    - **Sweep callers** (`sweep/hitl_expiry.py` and future sweep
      jobs): acquire `SWEEP_LOCK_ID` BEFORE invoking `emit_anomaly`
      per the `sweep-jobs-use-advisory-locks` invariant
      (`docs/invariants.md:257`). The advisory lock enforces
      cross-process serial-correctness on the SURROUNDING work
      (typically a non-idempotent status flip like `awaiting_approval`
      → `expired`); the DB-level partial unique index enforces
      per-row idempotency on emission. Both layers are load-bearing.

    - **Graph callers** (`agent/nodes/synthesize.py` for
      CROSS_ROUND_SEVERITY_DIVERGENCE; `agent/nodes/analyze.py` for
      COST_BUDGET_STARVATION, Stage 2): do NOT acquire any advisory
      lock. The rationale is DB-layer idempotency, NOT a serialization
      premise — same-thread concurrent `ainvoke` serialization is
      explicitly NOT guaranteed per `DECISIONS.md#027` line 946 (the
      race that motivated the publish-side advisory lock).
      Anomaly emission is fully idempotent at the database layer
      regardless of ordering — `postgresql_insert(...)
      .on_conflict_do_nothing(...)` makes a re-emission a clean
      no-op. The `sweep-jobs-use-advisory-locks` invariant applies
      where the protected operation IS non-idempotent (state flips
      that interleave badly under concurrent workers); anomaly
      emission alone does not qualify.

    Contract:
      - Returns `None` on success (no payload echo; the audit shadow
        is the `anomalies` row).
      - `severity` (`AnomalySeverity` StrEnum) and `details` are
        caller-controlled — the StrEnum prevents typos at the
        producer boundary (e.g. "Medium" with uppercase or "med")
        even though the DB column is Text. `details` is a JSON-
        native dict.
      - `is_eval: bool` is MANDATORY (no default). Loud-failure
        convention per `docs/testing.md` — eval-scenario emissions
        must land with `is_eval=True` to be filtered out of the
        production anomaly queue and to pass the eval-DB teardown
        integrity gate. Sweep callers that pre-filter eval reviews
        via `WHERE Review.is_eval.is_(False)` pass `is_eval=False`
        explicitly (the filter justifies the value; the value is
        still explicit at the call site).
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
        is_eval: bool,
    ) -> None:
        """Persist one anomaly row.

        Idempotent on `(review_id, rule_name)` per the per-rule
        partial unique index — a retry of the same `(review_id,
        rule_name)` pair is a no-op (returns successfully without
        raising). `is_eval` is mandatory; see class-level Contract
        section for the loud-failure rationale.
        """
        ...


__all__ = ["AnomalySink"]
