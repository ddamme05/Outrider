"""Prevent repeated warnings when approval-required findings exceed the limit.

Revision ID: e1f2a3b4c5d6
Revises: c7a2f5e1b9d4
Create Date: 2026-06-24

The gated_findings_over_cap index permits one warning per non-null review ID.
The migration verifies the full index definition and validity. Warnings whose
review was deleted may keep a null review ID.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e1f2a3b4c5d6"
down_revision: str | Sequence[str] | None = "c7a2f5e1b9d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the partial unique index for `gated_findings_over_cap`."""
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_anomalies_gated_findings_over_cap_natural_key
                ON anomalies (review_id)
                WHERE rule_name = 'gated_findings_over_cap';
            """
        )

    # Fail-loud verification: index landed VALID + UNIQUE + on the right
    # table/column + with the expected partial predicate (targeted catalog
    # checks, not a brittle pg_get_indexdef string compare).
    op.execute(
        """
        DO $$
        DECLARE
            expected_index_name text := 'uq_anomalies_gated_findings_over_cap_natural_key';
            expected_table text := 'anomalies';
            expected_column text := 'review_id';
            expected_rule_value text := 'gated_findings_over_cap';
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
                    'finding-cap migration failed: index % is missing in schema %.',
                    expected_index_name, current_schema();
            END IF;

            IF NOT (actual_valid AND actual_ready) THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % is not valid+ready '
                    '(indisvalid=%, indisready=%). Recovery: '
                    'DROP INDEX CONCURRENTLY IF EXISTS %, then re-run alembic upgrade head.',
                    expected_index_name, actual_valid, actual_ready, expected_index_name;
            END IF;

            IF actual_table != expected_table THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % targets table % (expected %).',
                    expected_index_name, actual_table, expected_table;
            END IF;

            IF NOT actual_unique THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % is not UNIQUE '
                    '(indisunique=false). Partial-unique idempotency is broken; '
                    'on_conflict_do_nothing would arbitrate against the wrong index.',
                    expected_index_name;
            END IF;

            IF actual_n_columns != 1 OR actual_column != expected_column THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % indexes (cols=%, first=%) — '
                    'expected single column ''%''.',
                    expected_index_name, actual_n_columns, actual_column, expected_column;
            END IF;

            IF actual_predicate IS NULL THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % is not a partial index '
                    '(indpred is NULL). Expected predicate referencing rule_name = ''%''.',
                    expected_index_name, expected_rule_value;
            END IF;

            -- Quote-anchor the rule-value substring so collisions like
            -- `rule_name = 'gated_findings_over_cap_extra'` fail the check.
            IF position('rule_name' in actual_predicate) = 0
               OR position('''' || expected_rule_value || '''' in actual_predicate) = 0 THEN
                RAISE EXCEPTION
                    'finding-cap migration failed: index % predicate ''%'' does not '
                    'reference both ''rule_name'' AND the exactly-quoted literal ''%''.',
                    expected_index_name, actual_predicate, expected_rule_value;
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    """Drop the partial unique index. The index is a derived structure, not
    data — dropping leaves `anomalies` rows intact. Idempotency for any
    subsequent GATED_FINDINGS_OVER_CAP emission would silently break until the
    index is recreated, but existing rows are preserved."""
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS uq_anomalies_gated_findings_over_cap_natural_key;"
        )
