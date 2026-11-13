"""Prevent repeated budget-shortage warnings for the same review.

Revision ID: 5786686ef937
Revises: 3bb6c2c3c0b1
Create Date: 2026-06-20

A unique index permits one cost_budget_starvation warning per non-null review
ID. The migration verifies the index definition and validity before finishing.
Warnings whose review was deleted may keep a null review ID.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "5786686ef937"
down_revision: str | Sequence[str] | None = "3bb6c2c3c0b1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the partial unique index for `cost_budget_starvation`."""
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_anomalies_cost_budget_starvation_natural_key
                ON anomalies (review_id)
                WHERE rule_name = 'cost_budget_starvation';
            """
        )

    # Fail-loud verification: index landed VALID + UNIQUE + on the right
    # table/column + with the expected partial predicate. Mirrors the
    # synthesize migration's targeted catalog checks (NOT a brittle
    # pg_get_indexdef string compare): an index with the same NAME but the
    # wrong SHAPE (manual creation, partial restore, drift) would make
    # `CREATE INDEX CONCURRENTLY IF NOT EXISTS` silently no-op while the
    # persister's `on_conflict_do_nothing` arbitrates against the wrong
    # index and loses idempotency.
    op.execute(
        """
        DO $$
        DECLARE
            expected_index_name text := 'uq_anomalies_cost_budget_starvation_natural_key';
            expected_table text := 'anomalies';
            expected_column text := 'review_id';
            expected_rule_value text := 'cost_budget_starvation';
            actual_table text;
            actual_unique boolean;
            actual_valid boolean;
            actual_ready boolean;
            actual_column text;
            actual_n_columns int;
            actual_predicate text;
        BEGIN
            SELECT
                rel.relname,
                i.indisunique,
                i.indisvalid,
                i.indisready,
                att.attname,
                array_length(i.indkey::int[], 1),
                pg_get_expr(i.indpred, i.indrelid)
            INTO
                actual_table,
                actual_unique,
                actual_valid,
                actual_ready,
                actual_column,
                actual_n_columns,
                actual_predicate
            FROM pg_index i
            JOIN pg_class idx ON idx.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = idx.relnamespace
            JOIN pg_class rel ON rel.oid = i.indrelid
            JOIN pg_attribute att
                ON att.attrelid = i.indrelid
                AND att.attnum = i.indkey[0]
            WHERE idx.relname = expected_index_name
              AND n.nspname = current_schema();

            IF NOT FOUND THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % is missing in schema %.',
                    expected_index_name, current_schema();
            END IF;

            IF NOT (actual_valid AND actual_ready) THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % is not valid+ready '
                    '(indisvalid=%, indisready=%). Recovery: '
                    'DROP INDEX CONCURRENTLY IF EXISTS %, then re-run alembic upgrade head.',
                    expected_index_name, actual_valid, actual_ready, expected_index_name;
            END IF;

            IF actual_table != expected_table THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % targets table % (expected %).',
                    expected_index_name, actual_table, expected_table;
            END IF;

            IF NOT actual_unique THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % is not UNIQUE '
                    '(indisunique=false). Partial-unique idempotency is broken; '
                    'on_conflict_do_nothing would arbitrate against the wrong index.',
                    expected_index_name;
            END IF;

            IF actual_n_columns != 1 OR actual_column != expected_column THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % indexes (cols=%, first=%) — '
                    'expected single column ''%''.',
                    expected_index_name, actual_n_columns, actual_column, expected_column;
            END IF;

            IF actual_predicate IS NULL THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % is not a partial index '
                    '(indpred is NULL). Expected predicate referencing rule_name = ''%''.',
                    expected_index_name, expected_rule_value;
            END IF;

            -- Quote-anchor the rule-value substring so collisions like
            -- `rule_name = 'cost_budget_starvation_extra'` fail the check.
            IF position('rule_name' in actual_predicate) = 0
               OR position('''' || expected_rule_value || '''' in actual_predicate) = 0 THEN
                RAISE EXCEPTION
                    'cost-fairness migration failed: index % predicate ''%'' does not '
                    'reference both ''rule_name'' AND the exactly-quoted literal ''%''.',
                    expected_index_name, actual_predicate, expected_rule_value;
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    """Drop the partial unique index. The index is a derived structure, not
    data — dropping leaves `anomalies` rows intact. Idempotency for any
    subsequent COST_BUDGET_STARVATION emission would silently break until
    the index is recreated (or the persister rolled back to not mapping the
    rule), but existing rows are preserved."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS uq_anomalies_cost_budget_starvation_natural_key;"
        )
