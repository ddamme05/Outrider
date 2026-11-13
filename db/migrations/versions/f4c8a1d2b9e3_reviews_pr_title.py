"""Save the pull-request title with each new review.

Revision ID: f4c8a1d2b9e3
Revises: e9a1c3d5b7f2
Create Date: 2026-06-13

The nullable pr_title column has no backfill, so older reviews keep null.
A later commit creates a new review; redelivering the same commit does not
rewrite the saved title.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f4c8a1d2b9e3"
down_revision: str | Sequence[str] | None = "e9a1c3d5b7f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable `pr_title` column to `reviews`."""
    op.add_column("reviews", sa.Column("pr_title", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the `pr_title` column (mirror of upgrade)."""
    op.drop_column("reviews", "pr_title")
