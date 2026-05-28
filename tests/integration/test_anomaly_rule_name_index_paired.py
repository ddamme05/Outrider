"""Enum-vs-index pairing test for AnomalyRuleName.

The `AnomalyPersister.emit_anomaly` dispatch refactor at
`src/outrider/anomaly/persister.py` removed the HITL_TIMEOUT-only
fail-loud check and replaced the hardcoded `index_where=(Anomaly.rule_name
== AnomalyRuleName.HITL_TIMEOUT.value)` with the dynamic
`index_where=(Anomaly.rule_name == rule_name.value)`. The conflict-arbiter
now resolves against the partial unique index matching the RUNTIME
rule_name — every `AnomalyRuleName` value MUST have a matching partial
unique index in the live schema, or `on_conflict_do_nothing` silently
falls through and idempotency breaks.

This test pre-empts the Class-10 (centrally-pinned contract requires
call-side registration) failure mode catalogued in
`docs/CODE_REVIEW_STYLES.md`: a contributor adding a new
`AnomalyRuleName` value WITHOUT the paired Alembic migration creates a
silent INSERT-duplicate window for the new rule. The producer-side
discipline ("ship the migration in the same PR") is documented at
`anomaly/persister.py` and in the migration files, but documentation
alone is not the gate — this test is the gate.

Convergent finding from the Phase 2 multi-lens audit
(adversarial + sharp-edges + CR-styles all flagged it independently);
adding it here closes that gap before the synthesize node body lands.

Tier: integration (queries `pg_index` against a real migrated DB).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from outrider.anomaly.rule_names import AnomalyRuleName


@pytest.mark.parametrize("rule_name", list(AnomalyRuleName))
@pytest.mark.asyncio
async def test_every_anomaly_rule_has_paired_partial_unique_index(
    rule_name: AnomalyRuleName,
    migrated_db: str,
) -> None:
    """Every AnomalyRuleName value has a matching partial unique index.

    Index naming convention: `uq_anomalies_<rule_value>_natural_key`
    with predicate `WHERE rule_name = '<rule_value>'`. The persister's
    `on_conflict_do_nothing(index_elements=["review_id"],
    index_where=(Anomaly.rule_name == rule_name.value))` requires this
    exact shape for the conflict-arbiter to match.

    Failure mode this test catches: developer adds a new
    `AnomalyRuleName` value but forgets the paired Alembic migration.
    The persister would dispatch on the new value, find no matching
    partial index, and `on_conflict_do_nothing` would fall through —
    every retry would land a new row, breaking the idempotency contract
    the sweep + graph callers BOTH depend on (sweep relies on the
    advisory lock + the index; graph callers rely on the index alone
    because they don't acquire any lock).
    """
    expected_index_name = f"uq_anomalies_{rule_name.value}_natural_key"
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        async with engine.begin() as conn:
            result = await conn.execute(
                text(
                    """
                    SELECT i.indisvalid, i.indisready
                    FROM pg_index i
                    JOIN pg_class c ON c.oid = i.indexrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = :index_name
                      AND n.nspname = current_schema()
                    """
                ),
                {"index_name": expected_index_name},
            )
            row = result.first()
        if row is None:
            pytest.fail(
                f"AnomalyRuleName.{rule_name.name} (value={rule_name.value!r}) "
                f"has no matching partial unique index in the live schema. "
                f"Expected index name: {expected_index_name!r}. "
                f"Add an Alembic migration creating "
                f"`CREATE UNIQUE INDEX CONCURRENTLY {expected_index_name} "
                f"ON anomalies (review_id) WHERE rule_name = '{rule_name.value}';` "
                f"in the same PR that introduced the enum value. Without this "
                f"index, `AnomalyPersister.emit_anomaly` silently loses "
                f"idempotency for this rule_name (on_conflict_do_nothing "
                f"falls through and every retry lands a new row)."
            )
        indisvalid, indisready = row
        assert indisvalid, (
            f"Partial unique index {expected_index_name!r} exists but "
            f"`indisvalid=false`. A `CREATE INDEX CONCURRENTLY` build failed "
            f"and left an INVALID index. Run "
            f"`DROP INDEX CONCURRENTLY IF EXISTS {expected_index_name};` "
            f"then re-run `alembic upgrade head`."
        )
        assert indisready, (
            f"Partial unique index {expected_index_name!r} exists but "
            f"`indisready=false`. Same recovery as INVALID — drop and "
            f"recreate."
        )
    finally:
        await engine.dispose()
