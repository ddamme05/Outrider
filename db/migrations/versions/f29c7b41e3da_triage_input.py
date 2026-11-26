"""Retain bounded triage input without changing the versioned event payload."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f29c7b41e3da"
down_revision = "c5a71b9de406"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "audit_events",
        sa.Column("triage_input", postgresql.JSONB(none_as_null=True), nullable=True),
    )
    # Every existing row has SQL NULL. Enforce the constraint on new writes
    # without scanning permanent audit history during the upgrade.
    op.create_check_constraint(
        "ck_audit_events_triage_input",
        "audit_events",
        "triage_input IS NULL OR (event_type = 'llm_call' "
        "AND (payload ->> 'node_id') IS NOT DISTINCT FROM 'triage' "
        "AND jsonb_typeof(triage_input) = 'object')",
        postgresql_not_valid=True,
    )


def downgrade() -> None:
    # Hold off concurrent writers until the column is dropped or the
    # transaction rolls back. A recorded audit receipt survives app rollbacks.
    op.execute("LOCK TABLE audit_events IN ACCESS EXCLUSIVE MODE")
    if op.get_bind().scalar(
        sa.text("SELECT EXISTS (SELECT 1 FROM audit_events WHERE triage_input IS NOT NULL)")
    ):
        raise RuntimeError(
            "Stored triage input prevents this downgrade. Keep the current database revision "
            "when rolling back the application."
        )
    op.drop_constraint("ck_audit_events_triage_input", "audit_events", type_="check")
    op.drop_column("audit_events", "triage_input")
