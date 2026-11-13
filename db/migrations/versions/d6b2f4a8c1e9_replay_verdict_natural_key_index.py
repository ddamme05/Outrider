"""Keep one saved replay verdict per review.

A unique index prevents duplicate replay_verdict events for the same review.
The migration builds it concurrently and checks validity before finishing.
Follow its recovery instructions if an earlier build left an invalid index.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d6b2f4a8c1e9"
down_revision: str | Sequence[str] | None = "b7e3f1a92c4d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the partial unique index concurrently, then verify it landed VALID."""
    with op.get_context().autocommit_block():
        op.execute(
            """
            CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS
                uq_audit_events_replay_verdict_natural_key
                ON audit_events (review_id)
                WHERE event_type = 'replay_verdict';
            """
        )

    # Post-build verification: a `CREATE INDEX CONCURRENTLY` can fail mid-build for
    # transient reasons (deadlock with a concurrent INSERT, OOM, rollback), leaving
    # an INVALID `pg_index` row a later `alembic upgrade head` would never re-attempt
    # — production would then seq-scan instead of using the unique index. Fail loud.
    op.execute(
        """
        DO $$
        DECLARE
            expected_index text := 'uq_audit_events_replay_verdict_natural_key';
            is_valid_ready boolean;
        BEGIN
            SELECT i.indisvalid AND i.indisready
            INTO is_valid_ready
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            WHERE c.relname = expected_index;
            IF is_valid_ready IS DISTINCT FROM true THEN
                RAISE EXCEPTION
                    'replay_verdict natural-key index migration failed: '
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
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS uq_audit_events_replay_verdict_natural_key;")
