"""Prevent truncating audit and purge history.

Revision ID: c22e2864d3d8
Revises: d6b2f4a8c1e9
Create Date: 2026-06-05

Statement-level triggers reject TRUNCATE on audit_events and purge_audit.
They use the same database guard as the row-level update/delete protections
(DECISIONS.md#014).
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c22e2864d3d8"
down_revision: str | Sequence[str] | None = "d6b2f4a8c1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add statement-level BEFORE TRUNCATE guards to audit_events + purge_audit.

    Both reuse the table-agnostic genesis guard function (keyed on TG_TABLE_NAME /
    TG_OP), so the message shape stays canonical with the row-level triggers.
    """
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_audit_events_no_truncate ON audit_events;
        CREATE TRIGGER trg_audit_events_no_truncate
            BEFORE TRUNCATE ON audit_events
            FOR EACH STATEMENT
            EXECUTE FUNCTION outrider_audit_append_only_guard();
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_purge_audit_no_truncate ON purge_audit;
        CREATE TRIGGER trg_purge_audit_no_truncate
            BEFORE TRUNCATE ON purge_audit
            FOR EACH STATEMENT
            EXECUTE FUNCTION outrider_audit_append_only_guard();
        """
    )


def downgrade() -> None:
    """Drop the statement-level TRUNCATE guards (mirror of upgrade)."""
    op.execute("DROP TRIGGER IF EXISTS trg_purge_audit_no_truncate ON purge_audit;")
    op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_truncate ON audit_events;")
