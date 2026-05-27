"""HITL node schema additions: reviews.expires_at + four partial unique indexes.

Revision ID: 33f8fe051bec
Revises: 8f2a4c1e7b3d
Create Date: 2026-05-26

Schema additions for the HITL node per `specs/2026-05-26-hitl-node.md`
Implementation Group 3:

  1. `reviews.expires_at TIMESTAMPTZ NULL` — the HITL approval window
     timestamp, mirrored from `HITLRequest.expires_at` for sweep-query
     efficiency (`WHERE status='awaiting_approval' AND expires_at < NOW()`).
     The corresponding `reviews.hitl_request` JSONB carries the full
     HITLRequest snapshot; the typed column is a convenience cache for
     the sweep's HITL-expiry sub-job (see `sweep/hitl_expiry.py` per Group 8).

  2. `ix_reviews_awaiting_approval_expires_at` — partial index on
     `(expires_at)` filtered by `status='awaiting_approval'`. Powers the
     sweep query directly; sibling pattern to the existing
     `ix_reviews_active_status` partial index.

  3. `uq_audit_events_hitl_request_natural_key` — partial unique index
     on `(review_id)` filtered by `event_type='hitl_request'`. V1 HITL
     is single-shot per review (per Non-goals "No multi-gate semantics");
     the index lets the persister-side `_persist_keyed_by_natural_key`
     helper fire `IntegrityError(UniqueViolation)` (caught by
     `postgresql_insert(...).on_conflict_do_nothing(...)`) when a body
     re-run on resume re-emits the request event.

  4. `uq_audit_events_hitl_decision_natural_key` — same shape for
     `event_type='hitl_decision'`. Natural-key identity-subset includes
     `decisions_content_hash`; divergent re-submissions raise
     `AuditPersisterHITLDecisionNaturalKeyConflict`.

  5. `uq_anomalies_hitl_timeout_natural_key` — partial unique index on
     `anomalies(review_id) WHERE rule_name='hitl_timeout'`. Schema-side
     defense for the anomaly-first ordering's idempotency claim (see
     `sweep/hitl_expiry.py::transition_expired_hitl_reviews` per Group 8):
     the sweep emits the canonical `hitl_timeout` anomaly BEFORE flipping
     the review's status to `awaiting_approval_expired`, and the index
     makes `postgresql_insert(...).on_conflict_do_nothing(...)` a clean
     no-op on retry.

All four indexes are partial (event_type / rule_name predicate in WHERE)
so they restrict the constraint to the relevant event-type / rule-name
partition only; other rows stay under whatever idempotency mode their
producer chooses.

Sibling-pattern caveats from the trace-decision precedent
(`8f2a4c1e7b3d_trace_decision_natural_key_index.py`) apply:

  - `payload ? '...'` predicate is NOT needed on the HITL audit indexes
    because the natural-key uses `review_id` only (a top-level column,
    never NULL). For the trace index the natural-key crosses into JSONB
    via `payload->>'source_finding_id'`, requiring the existence check.

  - `CREATE UNIQUE INDEX CONCURRENTLY` keeps the build non-blocking on
    production audit-events scans. Recovery from a failed concurrent
    build: `DROP INDEX CONCURRENTLY IF EXISTS <name>` then re-run.

  - No backfill: HITL has not shipped, so no `hitl_request` /
    `hitl_decision` / `hitl_timeout` rows exist yet.

See:
  - specs/2026-05-26-hitl-node.md Implementation Group 3
  - DECISIONS.md#026 (natural-key idempotency mode)
  - docs/spec.md §4.1.6 line 1421 (canonical `hitl_timeout` anomaly)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "33f8fe051bec"
down_revision: str | Sequence[str] | None = "8f2a4c1e7b3d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add `reviews.expires_at` + four partial unique indexes."""
    # --- 1. Add reviews.expires_at column (nullable; only HITL rows set it) ---
    # Guard against retry-after-partial-failure: the `CREATE INDEX
    # CONCURRENTLY` blocks below can fail in production and leave an
    # INVALID index that the operator must DROP CONCURRENTLY before
    # re-running. Re-running with an unconditional `op.add_column`
    # would raise `DuplicateColumn` because the column already
    # landed in the first attempt. Inspect first; add only when absent.
    bind = op.get_bind()
    expires_at_exists = bind.execute(
        sa.text(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_name = 'reviews'
              AND column_name = 'expires_at'
            """
        )
    ).first()
    if not expires_at_exists:
        op.add_column(
            "reviews",
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        )

    # --- 2. Partial index on reviews.expires_at for the HITL-expiry sweep ---
    # `awaiting_approval` is the only status that should ever have
    # expires_at < NOW() AND still be sweepable; the partial index keeps
    # the index small and lets the sweep's predicate match index entries
    # directly.
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
                ix_reviews_awaiting_approval_expires_at
                ON reviews (expires_at)
                WHERE status = 'awaiting_approval';
            """
        )

    # --- 3 + 4. Partial unique indexes on audit_events for HITL natural keys ---
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_audit_events_hitl_request_natural_key
                ON audit_events (review_id)
                WHERE event_type = 'hitl_request';
            """
        )
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_audit_events_hitl_decision_natural_key
                ON audit_events (review_id)
                WHERE event_type = 'hitl_decision';
            """
        )

    # --- 5. Partial unique index on anomalies for the hitl_timeout rule ---
    # `anomalies.review_id` is nullable with `ondelete='SET NULL'` per
    # `db/models/anomalies.py`; the partial unique index admits at most
    # one non-null `(review_id, rule_name='hitl_timeout')` per review.
    # If `review_id` becomes NULL after a review purge, the index does
    # NOT enforce uniqueness on the orphan row — acceptable because
    # orphans are no longer review-actionable.
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_anomalies_hitl_timeout_natural_key
                ON anomalies (review_id)
                WHERE rule_name = 'hitl_timeout';
            """
        )

    # --- 6. Fail-loud verification: every index landed VALID ---
    # `CREATE INDEX CONCURRENTLY` failures leave an INVALID index in
    # `pg_index`; subsequent `IF NOT EXISTS` runs would silently skip it
    # and production code would sequential-scan. Verify all four indexes
    # are present AND `indisvalid` before declaring the migration done.
    # Recovery from an INVALID index is `DROP INDEX CONCURRENTLY IF EXISTS
    # <name>` then re-run; this check makes the failure visible at
    # `alembic upgrade head` time rather than at first query.
    op.execute(
        """
        DO $$
        DECLARE
            expected_indexes text[] := ARRAY[
                'ix_reviews_awaiting_approval_expires_at',
                'uq_audit_events_hitl_request_natural_key',
                'uq_audit_events_hitl_decision_natural_key',
                'uq_anomalies_hitl_timeout_natural_key'
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
                WHERE c.relname = name
                  AND i.indisvalid = true
                  AND i.indisready = true
            );
            IF invalid_or_missing IS NOT NULL THEN
                RAISE EXCEPTION
                    'HITL migration failed: indexes missing or INVALID: %. '
                    'Run DROP INDEX CONCURRENTLY IF EXISTS <name> per index, '
                    'then re-run `alembic upgrade head`.',
                    invalid_or_missing;
            END IF;
        END$$;
        """
    )


def downgrade() -> None:
    """Drop the four partial indexes + the expires_at column.

    Fail-loud guard on `reviews.expires_at`: if any row has a non-NULL
    `expires_at` value, the downgrade refuses — dropping the column
    would silently lose HITL approval-window data. Operator's recovery
    is to either (a) clear the values via
    `UPDATE reviews SET expires_at = NULL WHERE expires_at IS NOT NULL`
    after exporting whatever forensic data is wanted, OR (b) accept the
    loss and run a separate `ALTER TABLE reviews DROP COLUMN expires_at`
    manually. Per `audit-events-append-only` the audit-row record of HITL
    activity stays intact regardless — only the convenience cache is lost.
    """
    op.execute(
        """
        DO $$
        DECLARE
            populated_count bigint;
        BEGIN
            SELECT count(*) INTO populated_count
            FROM reviews WHERE expires_at IS NOT NULL;
            IF populated_count > 0 THEN
                RAISE EXCEPTION
                    'HITL downgrade refused: % review row(s) have non-NULL '
                    'expires_at. Dropping reviews.expires_at would lose this '
                    'data. Clear the values explicitly (UPDATE reviews SET '
                    'expires_at = NULL WHERE expires_at IS NOT NULL) before '
                    're-running downgrade, or drop the column manually if the '
                    'loss is accepted.', populated_count;
            END IF;
        END$$;
        """
    )

    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_anomalies_hitl_timeout_natural_key;")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_audit_events_hitl_decision_natural_key;")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_audit_events_hitl_request_natural_key;")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_reviews_awaiting_approval_expires_at;")

    op.drop_column("reviews", "expires_at")
