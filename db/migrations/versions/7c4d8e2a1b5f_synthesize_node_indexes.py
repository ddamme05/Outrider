"""Synthesize node schema additions: partial unique index for the
`cross_round_severity_divergence` anomaly rule.

Revision ID: 7c4d8e2a1b5f
Revises: 4b9f1c5a7e21
Create Date: 2026-05-28

Schema addition for the synthesize node per `specs/2026-05-28-synthesize-
node.md` Phase 4:

  1. `uq_anomalies_cross_round_severity_divergence_natural_key` — partial
     unique index on `anomalies(review_id) WHERE rule_name =
     'cross_round_severity_divergence'`. Mirrors the HITL_TIMEOUT pattern
     from Group 3's migration; idempotency contract for the
     graph-emitted CROSS_ROUND_SEVERITY_DIVERGENCE anomaly that
     `agent/nodes/synthesize.py` raises when same-`content_hash` findings
     across analysis rounds carry divergent severity (corruption per
     `severity-set-by-policy` invariant + `compute_finding_content_hash`
     recipe + `ReviewFinding._verify_baseline_severity`).

The `AnomalyPersister.emit_anomaly` dispatch looks up a LITERAL-SQL
predicate `index_where=_RULE_NAME_INDEX_WHERE[rule_name]`
(→ `sa_text("rule_name = 'cross_round_severity_divergence'")`) keyed by
the runtime rule_name — without this paired partial unique index,
Postgres' conflict-arbiter would fail to match a partial index for the
new rule_name and the `on_conflict_do_nothing` falls through SILENTLY:
every retry would land a NEW row. (The predicate must be literal SQL, not
an ORM expression — a bind parameter fails arbiter inference under
psycopg3 generic plans; see `_RULE_NAME_INDEX_WHERE` in the persister.)
The integration test
`tests/integration/test_anomaly_rule_name_index_paired.py` enumerates
every `AnomalyRuleName` value and asserts a matching partial unique
index exists in `pg_index`, catching the enum-vs-migration drift class
at the integration tier per the Class-10 (centrally-pinned contract
requires call-side registration) pattern in `docs/CODE_REVIEW_STYLES.md`.

Sibling-pattern caveats from the HITL precedent
(`33f8fe051bec_hitl_node_indexes.py`) apply:

  - `CREATE UNIQUE INDEX CONCURRENTLY` keeps the build non-blocking on
    production `anomalies` scans. Recovery from a failed concurrent
    build: `DROP INDEX CONCURRENTLY IF EXISTS <name>` then re-run.

  - No backfill needed: CROSS_ROUND_SEVERITY_DIVERGENCE has never been
    emitted (synthesize node lands in Phase 5; this Phase 4 migration is
    a prerequisite). Zero existing rows for the predicate.

  - `anomalies.review_id` is nullable with `ondelete='SET NULL'` (per
    `db/models/anomalies.py`); the partial unique index admits at most
    one non-null `(review_id, rule_name='cross_round_severity_divergence')`
    per review. Orphans (post-purge NULL `review_id`) are not enforced
    — same trade-off as the HITL_TIMEOUT partial unique index.

See:
  - specs/2026-05-28-synthesize-node.md §Severity policy + §Audit append-only
  - specs/_2026-05-27-synthesize-pre-spec-gates.md gate #7
  - docs/invariants.md `idempotency-via-db-unique-constraint`
  - docs/CODE_REVIEW_STYLES.md Class 10 (centrally-pinned contract)
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7c4d8e2a1b5f"
down_revision: str | Sequence[str] | None = "4b9f1c5a7e21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the partial unique index for `cross_round_severity_divergence`."""
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_anomalies_cross_round_severity_divergence_natural_key
                ON anomalies (review_id)
                WHERE rule_name = 'cross_round_severity_divergence';
            """
        )

    # Fail-loud verification: index landed VALID + UNIQUE + on the right
    # table/column + with the expected partial predicate.
    #
    # The prior version of this block checked only existence + indisvalid
    # + indisready. Per CodeRabbit catch (narrowed by Codex), the broader
    # failure mode is: an index with the same NAME but the wrong SHAPE
    # could exist (manual creation, partial restore, drift from a prior
    # migration), and `CREATE INDEX CONCURRENTLY IF NOT EXISTS` would
    # silently no-op — the verification would pass, but the persister's
    # `on_conflict_do_nothing` would arbitrate against the wrong index
    # and lose idempotency.
    #
    # Targeted catalog checks (not a full `pg_get_indexdef` string compare,
    # which is brittle to whitespace / quoting drift):
    #   1. `indisvalid` + `indisready` — concurrent build completed.
    #   2. `indisunique` — partial-unique idempotency depends on it.
    #   3. Target table = `anomalies` (via pg_class on indrelid).
    #   4. Indexed column = `review_id` (via pg_attribute on indkey[0]).
    #   5. Predicate references both `rule_name` and the rule value
    #      `cross_round_severity_divergence` (via pg_get_expr on indpred —
    #      substring match, not exact string compare).
    op.execute(
        """
        DO $$
        DECLARE
            expected_index_name text := 'uq_anomalies_cross_round_severity_divergence_natural_key';
            expected_table text := 'anomalies';
            expected_column text := 'review_id';
            expected_rule_value text := 'cross_round_severity_divergence';
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
                    'Synthesize migration failed: index % is missing in schema %.',
                    expected_index_name, current_schema();
            END IF;

            IF NOT (actual_valid AND actual_ready) THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % is not valid+ready '
                    '(indisvalid=%, indisready=%). Recovery: '
                    'DROP INDEX CONCURRENTLY IF EXISTS %, then re-run alembic upgrade head.',
                    expected_index_name, actual_valid, actual_ready, expected_index_name;
            END IF;

            IF actual_table != expected_table THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % targets table % (expected %).',
                    expected_index_name, actual_table, expected_table;
            END IF;

            IF NOT actual_unique THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % is not UNIQUE '
                    '(indisunique=false). Partial-unique idempotency is broken; '
                    'on_conflict_do_nothing would arbitrate against the wrong index.',
                    expected_index_name;
            END IF;

            IF actual_n_columns != 1 OR actual_column != expected_column THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % indexes (cols=%, first=%) — '
                    'expected single column ''%''.',
                    expected_index_name, actual_n_columns, actual_column, expected_column;
            END IF;

            IF actual_predicate IS NULL THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % is not a partial index '
                    '(indpred is NULL). Expected predicate referencing rule_name = ''%''.',
                    expected_index_name, expected_rule_value;
            END IF;

            -- Quote-anchor the rule-value substring so collisions like
            -- `rule_name = 'cross_round_severity_divergence_extra'` or
            -- `rule_name LIKE 'cross_round_severity_divergence%'` fail
            -- the check. `pg_get_expr` renders the predicate with the
            -- literal wrapped in single quotes (PG-escaped as ''); the
            -- substring search now requires the CLOSING quote to be
            -- present immediately after the rule value, which rejects
            -- the collision class. Per Pass-1 multi-lens audit DB lens
            -- (substring-match collision LOW).
            IF position('rule_name' in actual_predicate) = 0
               OR position('''' || expected_rule_value || '''' in actual_predicate) = 0 THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: index % predicate ''%'' does not '
                    'reference both ''rule_name'' AND the exactly-quoted literal ''%''.',
                    expected_index_name, actual_predicate, expected_rule_value;
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    """Drop the partial unique index.

    No data guard needed (unlike the HITL `reviews.expires_at` column):
    the index is a derived structure, not data. Dropping leaves the
    `anomalies` rows intact. Idempotency for any subsequent
    CROSS_ROUND_SEVERITY_DIVERGENCE emission would silently break until
    the index is recreated (or the persister is rolled back to the
    HITL_TIMEOUT-only fail-loud check), but the existing rows are
    preserved.
    """
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS "
            "uq_anomalies_cross_round_severity_divergence_natural_key;"
        )
