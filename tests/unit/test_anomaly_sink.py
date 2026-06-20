"""Unit tests for the AnomalySink Protocol + the AnomalyPersister
config-error path.

Durable persister behavior (on_conflict_do_nothing semantics, partial-
index match) is covered in integration tests against postgres-test;
this file pins:
  - Protocol surface (runtime_checkable, single emit method)
  - AnomalyPersister config-error path (None session_factory raises)
  - AnomalyRuleName enum value pin
"""

from __future__ import annotations

from typing import Any
from uuid import UUID  # noqa: TC003  (runtime: Protocol method signature)

import pytest

from outrider.anomaly import (
    AnomalyPersister,
    AnomalyPersisterConfigError,
    AnomalyRuleName,
    AnomalySeverity,
    AnomalySink,
)


def test_anomaly_rule_name_hitl_timeout_value_pinned() -> None:
    """Canonical anomaly rule name string per docs/spec.md §16."""
    assert AnomalyRuleName.HITL_TIMEOUT.value == "hitl_timeout"


def test_anomaly_rule_name_v1_value_set() -> None:
    """V1 ships three rules. New rules extend the enum + need matching
    partial unique indexes on the anomalies table.

    HITL_TIMEOUT — sweep-emitted from `sweep/hitl_expiry.py` per
    `docs/spec.md` §16.
    CROSS_ROUND_SEVERITY_DIVERGENCE — graph-emitted from
    `agent/nodes/synthesize.py::_detect_and_report_divergence` per
    the synthesize-node spec; surfaces cross-round divergence on
    EITHER `severity` OR `policy_version` for the same content_hash
    as corruption (severity-set-by-policy +
    severity-policy-versioned-for-replay invariants + per-element
    validator chain guarantee same content_hash + same finding_type
    + same policy_version => same severity). Either axis triggers
    the same rule; recovery action is identical.
    COST_BUDGET_STARVATION — graph-emitted from `agent/nodes/analyze.py`
    when a pass skips >= COST_BUDGET_STARVATION_THRESHOLD files with
    COST_BUDGET_EXHAUSTED (FUP-044 ext 3 / analyze-cost-fairness Stage 2).
    """
    assert {m.value for m in AnomalyRuleName} == {
        "hitl_timeout",
        "cross_round_severity_divergence",
        "cost_budget_starvation",
    }


def test_persister_rejects_none_session_factory() -> None:
    """Fail-loud at construction time per the sibling persister
    precedents (ReviewStatusPersister, AuditPersister)."""
    with pytest.raises(AnomalyPersisterConfigError):
        AnomalyPersister(session_factory=None)  # type: ignore[arg-type]


def test_recording_sink_satisfies_protocol_structurally() -> None:
    """A duck-typed recording double satisfies AnomalySink."""

    class _Recording:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def emit_anomaly(
            self,
            *,
            review_id: UUID,
            rule_name: AnomalyRuleName,
            severity: AnomalySeverity,
            details: dict[str, Any],
            is_eval: bool,
        ) -> None:
            self.calls.append(
                {
                    "review_id": review_id,
                    "rule_name": rule_name,
                    "severity": severity,
                    "details": details,
                    "is_eval": is_eval,
                }
            )

    sink = _Recording()
    assert isinstance(sink, AnomalySink)


def test_partial_sink_rejected_by_runtime_check() -> None:
    """A class missing emit_anomaly fails isinstance(AnomalySink)."""

    class _Partial:
        pass

    assert not isinstance(_Partial(), AnomalySink)


def test_anomaly_sink_declares_one_method() -> None:
    """Protocol surface check — exact membership, not just presence.

    Pins the public API to `{"emit_anomaly"}`. A new public method
    added to `AnomalySink` (e.g., a V1.5 `emit_recovery` for the
    `hitl_recovery` anomaly the dashboard might consume) fails this
    test loudly — same shape as the
    `tests/unit/test_github_publisher.py::test_publish_event_sink_method_set`
    pin per the `Class-10` doctrine (centrally-pinned contracts
    require call-side registration).
    """
    actual_public_methods = {name for name in AnomalySink.__dict__ if not name.startswith("_")}
    assert actual_public_methods == {"emit_anomaly"}, (
        f"AnomalySink public surface drift: missing={ {'emit_anomaly'} - actual_public_methods }, "
        f"extra={actual_public_methods - {'emit_anomaly'}}. Update this test AND verify "
        f"every AnomalySink consumer + test fixture carries the new method."
    )


def test_rule_name_index_where_covers_every_rule_with_literal_sql() -> None:
    """Completeness + literalness of the per-rule predicate map: every
    AnomalyRuleName has an entry, and each compiles to literal SQL
    (`rule_name = '<value>'`) with no bind params. The completeness arm
    guards against a new rule shipping without its literal predicate."""
    from sqlalchemy.dialects import postgresql

    from outrider.anomaly.persister import _RULE_NAME_INDEX_WHERE

    assert set(_RULE_NAME_INDEX_WHERE) == set(AnomalyRuleName), (
        "every AnomalyRuleName must have a literal partial-index predicate; "
        f"missing={set(AnomalyRuleName) - set(_RULE_NAME_INDEX_WHERE)}"
    )
    for rule, clause in _RULE_NAME_INDEX_WHERE.items():
        compiled = clause.compile(dialect=postgresql.dialect())
        assert not compiled.params, (
            f"{rule.value}: index_where must be literal SQL; got bind params "
            f"{dict(compiled.params)} — arbiter inference will fail under generic plans"
        )
        assert f"rule_name = '{rule.value}'" in str(compiled)


def test_rule_name_index_where_matches_orm_model_partial_index_predicate() -> None:
    """Drift guard: each `_RULE_NAME_INDEX_WHERE[rule]` predicate must byte-match
    the `postgresql_where` of that rule's partial unique index on the ORM model
    (`Anomaly.__table__`), which in turn mirrors the migration's `CREATE UNIQUE
    INDEX ... WHERE`. The predicate literal lives in THREE places that must stay
    identical (migration, ORM model, this map); map-vs-model is the only pair no
    other test compares. Without this, tightening an index predicate (e.g. adding
    `AND status = 'open'`) on the model + migration but missing the map silently
    desyncs `index_where` from the partial index — arbiter inference falls
    through and duplicate rows return, the exact idempotency break this map
    exists to prevent. (The completeness test only checks the map contains
    `rule_name = '<enum.value>'`; the integration test only checks the live DB
    index — neither catches predicate drift between the map and the model.)
    """
    from sqlalchemy.dialects import postgresql

    from outrider.anomaly.persister import _RULE_NAME_INDEX_WHERE
    from outrider.db.models.anomalies import Anomaly

    pg = postgresql.dialect()
    model_predicate_by_index = {
        ix.name: ix.dialect_options["postgresql"].get("where") for ix in Anomaly.__table__.indexes
    }
    for rule in AnomalyRuleName:
        index_name = f"uq_anomalies_{rule.value}_natural_key"
        model_where = model_predicate_by_index.get(index_name)
        assert model_where is not None, (
            f"{rule.value}: no partial index {index_name!r} with a postgresql_where "
            f"predicate on Anomaly.__table__ — the map predicate has nothing to mirror"
        )
        model_sql = str(model_where.compile(dialect=pg))
        map_sql = str(_RULE_NAME_INDEX_WHERE[rule].compile(dialect=pg))
        assert map_sql == model_sql, (
            f"{rule.value}: _RULE_NAME_INDEX_WHERE predicate {map_sql!r} drifted from "
            f"the ORM model's partial-index predicate {model_sql!r} ({index_name}). "
            f"The ON CONFLICT index_where must byte-match the partial index, or "
            f"on_conflict_do_nothing falls through and idempotency breaks."
        )


class _NullAsyncCtx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _CapturingSession:
    """Fake AsyncSession recording statements passed to `execute`, so the
    real `emit_anomaly`'s rendered SQL is inspectable without a DB."""

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def __aenter__(self) -> _CapturingSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def begin(self) -> _NullAsyncCtx:
        return _NullAsyncCtx()

    async def execute(self, stmt: Any) -> None:
        self.statements.append(stmt)


@pytest.mark.parametrize("rule_name", list(AnomalyRuleName))
async def test_emit_anomaly_renders_literal_on_conflict_where(rule_name: AnomalyRuleName) -> None:
    """The REAL `emit_anomaly` must render its ON CONFLICT partial-index
    predicate as literal SQL (`WHERE rule_name = '<value>'`), never a bind
    parameter, for every rule. An ORM expression
    (`Anomaly.rule_name == rule_name.value`) renders `WHERE rule_name = $1`,
    which psycopg3 generic plans can't prove implies a rule's partial unique
    index predicate — arbiter inference then fails (42P10) once the statement
    is server-prepared, silently defeating the idempotent no-op the HITL sweep
    depends on. Integration tests use NullPool (no prepared statements), so
    this is only catchable here. Invokes the method (not just the map) so a
    revert of the method body to an ORM expression fails this test. Twin of
    the audit replay-verdict check.
    """
    from uuid import uuid4

    from sqlalchemy.dialects import postgresql

    session = _CapturingSession()
    persister = AnomalyPersister(session_factory=lambda: session)  # type: ignore[arg-type, return-value]
    await persister.emit_anomaly(
        review_id=uuid4(),
        rule_name=rule_name,
        severity=AnomalySeverity.MEDIUM,
        details={},
        is_eval=False,
    )
    assert len(session.statements) == 1
    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert f"WHERE rule_name = '{rule_name.value}'" in sql, sql
    assert "%(rule_name_1)s" not in sql, f"predicate rendered as bind param: {sql}"
