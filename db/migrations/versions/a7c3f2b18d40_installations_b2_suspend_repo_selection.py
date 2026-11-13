"""Record installation suspension and repository selection.

Revision ID: a7c3f2b18d40
Revises: ff57dcf6fbd1
Create Date: 2026-07-06

suspended_at records a reversible pause. repository_selection distinguishes
all-repository access from individually selected repositories. Existing rows
default to selected, so missing repository membership does not grant access
(DECISIONS.md#065).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7c3f2b18d40"
down_revision: str | Sequence[str] | None = "ff57dcf6fbd1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add `suspended_at` (nullable) + `repository_selection` (NOT NULL, default 'selected')."""
    op.add_column(
        "installations",
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "installations",
        sa.Column(
            "repository_selection",
            sa.Text(),
            server_default=sa.text("'selected'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("installations", "repository_selection")
    op.drop_column("installations", "suspended_at")
