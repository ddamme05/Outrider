"""Prevent duplicate records of the same trace decision.

Revision ID: 8f2a4c1e7b3d
Revises: 3d03bca7f2be
Create Date: 2026-05-24

A unique index identifies trace decisions by review and source finding. The
migration builds the index concurrently and checks its validity. An invalid
index requires the reported recovery step before retrying (DECISIONS.md#026).
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8f2a4c1e7b3d"
down_revision: str | Sequence[str] | None = "3d03bca7f2be"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the trace_decision natural-key partial unique index.

    Uses `CREATE UNIQUE INDEX CONCURRENTLY` inside an Alembic autocommit
    block so the build does not block writes to `audit_events`. While
    no `trace_decision` rows exist yet (trace hasn't shipped), Postgres
    still scans the full table to identify rows matching the WHERE
    clause; on a production audit-events table that scan can be long
    and the non-concurrent SHARE lock would block other event-type
    INSERTs for its duration. CONCURRENTLY trades that for a slower
    build that never blocks writers.

    IF NOT EXISTS-guarded — a partial-failure rerun lands cleanly.
    Note: a failed concurrent build can leave an INVALID index that
    IF NOT EXISTS would silently skip on retry; the standard recovery
    is `DROP INDEX CONCURRENTLY uq_audit_events_trace_decision_natural_key`
    then re-run this migration.

    The index applies to the trace-decision event-type partition only;
    other event types stay under event_id-PK idempotency per #026.
    """
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_audit_events_trace_decision_natural_key
                ON audit_events (review_id, (payload->>'source_finding_id'))
                WHERE event_type = 'trace_decision'
                      AND payload ? 'source_finding_id';
            """
        )

    # Post-build verification: confirm the index landed VALID. A
    # failed `CREATE INDEX CONCURRENTLY` leaves an INVALID `pg_index`
    # row that `IF NOT EXISTS` would silently skip on retry, marking
    # the migration complete without enforcing the uniqueness
    # contract — production trace-decision emits would then never
    # hit the natural-key conflict path. Mirror of the HITL
    # migration's `--- 6. Fail-loud verification` block at
    # 33f8fe051bec_hitl_node_indexes.py:141.
    op.execute(
        """
        DO $$
        DECLARE
            expected_index text := 'uq_audit_events_trace_decision_natural_key';
            is_valid_ready boolean;
        BEGIN
            SELECT i.indisvalid AND i.indisready
            INTO is_valid_ready
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = expected_index;
            IF is_valid_ready IS DISTINCT FROM true THEN
                RAISE EXCEPTION
                    'trace_decision natural-key index migration failed: '
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
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_audit_events_trace_decision_natural_key;")
