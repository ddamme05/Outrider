"""Prevent duplicate phase records when a review resumes.

Revision ID: 4b9f1c5a7e21
Revises: 33f8fe051bec
Create Date: 2026-05-27

The index identifies each review phase and its start or end marker, including
the per-file phase key. Existing duplicates or an invalid index stop the
migration with recovery instructions. Audit rows remain append-only; this
migration never rewrites them (DECISIONS.md#014, DECISIONS.md#026).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b9f1c5a7e21"
down_revision: str | Sequence[str] | None = "33f8fe051bec"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the review_phase natural-key partial unique index.

    Pre-flight duplicate detection FIRST. `CREATE UNIQUE INDEX
    CONCURRENTLY` scans existing rows during the build; if any
    historical duplicate phase rows exist on the natural key (the
    pre-fix HITL-resume duplicate-emit case), the build raises
    `UniqueViolation` and the migration fails with an opaque DB-level
    error. Audit rows are append-only per `DECISIONS.md#014` (the
    trigger from genesis migration forbids UPDATE / DELETE), so silent
    cleanup is impossible — explicit operator remediation is required.
    The detection query runs in its own statement (NOT in the
    autocommit block below) so the result is available before the
    concurrent index build starts.

    Uses `CREATE UNIQUE INDEX CONCURRENTLY` inside an Alembic autocommit
    block so the build does not block writes to `audit_events`. Postgres
    still scans the full table to identify rows matching the WHERE
    clause; the non-concurrent SHARE lock would block other event-type
    INSERTs for that duration. CONCURRENTLY trades that for a slower
    build that never blocks writers.

    IF NOT EXISTS-guarded — a partial-failure rerun lands cleanly.
    A failed concurrent build can leave an INVALID index in the
    catalog (a row in `pg_index` with `indisvalid=false`); `CREATE
    UNIQUE INDEX CONCURRENTLY IF NOT EXISTS` matches by NAME only and
    would skip the rebuild on retry, marking the migration complete
    without ever enforcing the natural-key uniqueness contract.
    Pre-flight check #2 below guards against this silent-success
    hole: query `pg_index.indisvalid` for the named index and raise
    with explicit operator guidance if it exists but is invalid (the
    standard recovery is `DROP INDEX CONCURRENTLY uq_audit_events_
    review_phase_natural_key` then re-run this migration).
    """
    # Pre-flight: count natural-key duplicate groups. A "group" is a
    # `(review_id, phase_id, COALESCE(phase_key, ''), marker)` tuple
    # that appears in more than one audit_events row.
    bind = op.get_bind()
    duplicate_groups = bind.execute(
        sa.text(
            """
            SELECT COUNT(*) FROM (
                SELECT
                    review_id,
                    (payload->>'phase_id') AS phase_id,
                    COALESCE(phase_key, '') AS phase_key_key,
                    (payload->>'marker') AS marker
                FROM audit_events
                WHERE event_type = 'review_phase'
                      AND payload ? 'phase_id'
                      AND payload ? 'marker'
                GROUP BY review_id, phase_id, phase_key_key, marker
                HAVING COUNT(*) > 1
            ) AS dup_groups
            """
        )
    ).scalar_one()
    if duplicate_groups > 0:
        raise RuntimeError(
            f"4b9f1c5a7e21_review_phase_natural_key_index: cannot proceed. "
            f"Found {duplicate_groups} natural-key duplicate group(s) in "
            f"audit_events for event_type='review_phase' (key="
            f"(review_id, phase_id, COALESCE(phase_key, ''), marker)). "
            f"`CREATE UNIQUE INDEX CONCURRENTLY` would raise UniqueViolation "
            f"during its existing-rows scan. Audit rows are append-only per "
            f"DECISIONS.md#014 (the trigger from genesis migration forbids "
            f"DELETE / UPDATE), and that invariant is load-bearing for replay "
            f"equivalence — it is NOT a candidate for `temporarily disable the "
            f"trigger and dedupe`. Approved remediation paths: (1) for "
            f"non-production environments (dev / staging / test DBs), drop and "
            f"recreate the database, replay any seed data from fresh, then "
            f"re-run `alembic upgrade head`; pre-HITL-ship environments should "
            f"not have phase duplicates anyway (HITL is the only path that "
            f"produced them, and HITL ships in the same PR as this migration). "
            f"(2) for production-shaped recovery (a DB that already had HITL "
            f"exercised pre-fix), a new DECISIONS entry must route an append-"
            f"only-compatible historical-duplicate strategy (candidates: a "
            f"superseding `event_type='review_phase_v2'` value emitted by "
            f"post-migration code with the partial unique index re-scoped, OR "
            f"a tombstone-event pattern carried in a sibling table) before any "
            f"data-touching cleanup runs. Do NOT detach the append-only "
            f"trigger — that defeats audit-events-append-only and "
            f"`replay-equivalence-window` simultaneously."
        )

    # Pre-flight #2: detect a stale INVALID index from a prior
    # failed `CREATE INDEX CONCURRENTLY` build. PostgreSQL leaves
    # the index row in `pg_index` with `indisvalid=false` after a
    # failed concurrent build; `CREATE UNIQUE INDEX CONCURRENTLY IF
    # NOT EXISTS` matches by NAME only and skips on retry, so the
    # migration would mark itself complete without ever rebuilding
    # the index — the natural-key uniqueness contract would be
    # silently absent. The PG documentation calls this out as the
    # standard recovery-procedure for the failed-CONCURRENT-build
    # case. Detect explicitly and direct the operator to drop the
    # stale index before retrying.
    invalid_index_exists = bind.execute(
        sa.text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_class c
                JOIN pg_index i ON i.indexrelid = c.oid
                WHERE c.relname = 'uq_audit_events_review_phase_natural_key'
                      AND i.indisvalid = false
            )
            """
        )
    ).scalar_one()
    if invalid_index_exists:
        raise RuntimeError(
            "4b9f1c5a7e21_review_phase_natural_key_index: cannot proceed. "
            "A stale INVALID index named "
            "`uq_audit_events_review_phase_natural_key` exists in pg_index "
            "(indisvalid=false) — left behind by a prior failed `CREATE "
            "UNIQUE INDEX CONCURRENTLY` build. `IF NOT EXISTS` matches by "
            "NAME only, so a naive retry would silently skip the rebuild "
            "and mark this migration complete without ever enforcing the "
            "natural-key uniqueness contract. Standard recovery: run "
            "`DROP INDEX CONCURRENTLY uq_audit_events_review_phase_natural_key;` "
            "as a manual DBA step (this is the PostgreSQL-documented "
            "remediation for failed-CONCURRENT-build state and does NOT "
            "touch any audit_events row — it only drops the index entry), "
            "then re-run `alembic upgrade head`."
        )

    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_audit_events_review_phase_natural_key
                ON audit_events (
                    review_id,
                    (payload->>'phase_id'),
                    COALESCE(phase_key, ''),
                    (payload->>'marker')
                )
                WHERE event_type = 'review_phase'
                      AND payload ? 'phase_id'
                      AND payload ? 'marker';
            """
        )

    # Post-build verification: confirm the index landed VALID. Even
    # with the pre-flight check above, a fresh `CREATE INDEX
    # CONCURRENTLY` can fail mid-build for transient reasons
    # (deadlock with concurrent INSERT, OOM kill, transaction
    # rollback), leaving an INVALID `pg_index` row behind. A
    # subsequent `alembic upgrade head` that thinks the migration
    # already ran would never catch this — production code would
    # then sequential-scan instead of using the unique index.
    # Mirror of the HITL migration's `--- 6. Fail-loud verification`
    # block at 33f8fe051bec_hitl_node_indexes.py:141.
    op.execute(
        """
        DO $$
        DECLARE
            expected_index text := 'uq_audit_events_review_phase_natural_key';
            is_valid_ready boolean;
        BEGIN
            SELECT i.indisvalid AND i.indisready
            INTO is_valid_ready
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = expected_index;
            IF is_valid_ready IS DISTINCT FROM true THEN
                RAISE EXCEPTION
                    'review_phase natural-key index migration failed: '
                    '% is missing or INVALID after CREATE INDEX CONCURRENTLY. '
                    'Recovery: DROP INDEX CONCURRENTLY IF EXISTS %, '
                    'then re-run `alembic upgrade head`.',
                    expected_index, expected_index;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    """Drop the partial unique index concurrently (mirror of upgrade)."""
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_audit_events_review_phase_natural_key;")
