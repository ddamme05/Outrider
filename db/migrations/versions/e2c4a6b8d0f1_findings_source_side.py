"""Record whether a finding refers to current or removed source.

Revision ID: e2c4a6b8d0f1
Revises: 9d2e4f6a8c10
Create Date: 2026-11-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2c4a6b8d0f1"
down_revision: str | Sequence[str] | None = "9d2e4f6a8c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "findings",
        sa.Column("source_side", sa.Text(), server_default=sa.text("'head'"), nullable=False),
    )
    op.create_check_constraint(
        "ck_findings_source_side", "findings", "source_side IN ('head', 'base')"
    )


def downgrade() -> None:
    op.drop_constraint("ck_findings_source_side", "findings", type_="check")
    op.drop_column("findings", "source_side")
