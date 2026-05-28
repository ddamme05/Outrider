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

The `AnomalyPersister.emit_anomaly` dispatch refactor at
`src/outrider/anomaly/persister.py` reads `index_where=(Anomaly.rule_name
== rule_name.value)` dynamically — without this paired partial unique
index, Postgres' conflict-arbiter would fail to match a partial index for
the new rule_name and the `on_conflict_do_nothing` falls through SILENTLY:
every retry would land a NEW row. The integration test
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

    # Fail-loud verification: index landed VALID. `CREATE INDEX
    # CONCURRENTLY` failures leave an INVALID index in `pg_index`; a
    # subsequent `IF NOT EXISTS` would silently skip it and the persister
    # would silently lose idempotency. Verify the index is present AND
    # `indisvalid` before declaring the migration done. Recovery is
    # `DROP INDEX CONCURRENTLY IF EXISTS <name>` then re-run.
    op.execute(
        """
        DO $$
        DECLARE
            expected_indexes text[] := ARRAY[
                'uq_anomalies_cross_round_severity_divergence_natural_key'
            ];
            invalid_or_missing text[];
        BEGIN
            SELECT array_agg(name)
            INTO invalid_or_missing
            FROM unnest(expected_indexes) AS name
            WHERE NOT EXISTS (
                SELECT 1
                FROM pg_index i
                JOIN pg_class c ON c.oid = i.indexrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = name
                  AND n.nspname = current_schema()
                  AND i.indisvalid = true
                  AND i.indisready = true
            );
            IF invalid_or_missing IS NOT NULL THEN
                RAISE EXCEPTION
                    'Synthesize migration failed: indexes missing or INVALID: %. '
                    'Run DROP INDEX CONCURRENTLY IF EXISTS <name>, '
                    'then re-run `alembic upgrade head`.',
                    invalid_or_missing;
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
