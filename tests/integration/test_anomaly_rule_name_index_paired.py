"""Enum-vs-index pairing test for AnomalyRuleName.

The `AnomalyPersister.emit_anomaly` dispatch looks up a LITERAL-SQL
partial-index predicate from `_RULE_NAME_INDEX_WHERE` by the RUNTIME
rule_name (`index_where=_RULE_NAME_INDEX_WHERE[rule_name]`). The
conflict-arbiter resolves against the partial unique index matching that
rule_name — every `AnomalyRuleName` value MUST have a matching partial
unique index in the live schema, or `on_conflict_do_nothing` silently
falls through and idempotency breaks. (The predicate is literal `sa_text`,
not an ORM expression: an ORM-expression `index_where` renders a bind
parameter that fails arbiter inference under psycopg3 generic plans — the
defect this paired index + the persister's literal-SQL form guard against.)

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
    index_where=_RULE_NAME_INDEX_WHERE[rule_name])` (literal SQL) requires
    this exact shape for the conflict-arbiter to match.

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
            # Full shape check: index name + indisvalid + indisready +
            # indisunique + indexed columns + predicate text. Audit
            # finding: name-only check passes a wrongly-shaped index
            # (right name, wrong uniqueness, wrong columns, wrong
            # predicate) — that would silently break idempotency.
            result = await conn.execute(
                text(
                    """
                    SELECT
                        i.indisvalid,
                        i.indisready,
                        i.indisunique,
                        pg_get_indexdef(i.indexrelid) AS index_def
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
        indisvalid, indisready, indisunique, index_def = row
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
        assert indisunique, (
            f"Index {expected_index_name!r} exists but is NOT a UNIQUE "
            f"index (`indisunique=false`). The `on_conflict_do_nothing` "
            f"idempotency contract requires uniqueness on (review_id) "
            f"under the rule_name predicate; a non-unique index of the "
            f"same name silently lets duplicate INSERTs land."
        )
        # `pg_get_indexdef` returns the CREATE INDEX SQL. Verify it
        # targets `anomalies(review_id)` and has the correct
        # `WHERE rule_name = '<value>'` predicate. The exact normalized
        # form Postgres emits looks like:
        #   CREATE UNIQUE INDEX uq_... ON public.anomalies USING btree
        #   (review_id) WHERE (rule_name = '<value>'::text)
        index_def_str = str(index_def)
        assert "anomalies" in index_def_str.lower(), (
            f"Index {expected_index_name!r} is not on the `anomalies` table: "
            f"index_def={index_def_str!r}"
        )
        assert "(review_id)" in index_def_str, (
            f"Index {expected_index_name!r} does not target the "
            f"(review_id) column: index_def={index_def_str!r}. "
            f"The on-conflict arbiter requires `index_elements=['review_id']` "
            f"to match this index's column list."
        )
        expected_predicate_fragment = f"rule_name = '{rule_name.value}'"
        assert expected_predicate_fragment in index_def_str, (
            f"Index {expected_index_name!r} predicate does not match the "
            f"expected `WHERE {expected_predicate_fragment}` clause: "
            f"index_def={index_def_str!r}. A wrong predicate makes the "
            f"on-conflict arbiter fail to match this index — silent "
            f"idempotency break for this rule_name."
        )
    finally:
        await engine.dispose()
