"""Keep each review's updated_at timestamp current.

Revision ID: 54bb7ed5f51a
Revises: c22e2864d3d8
Create Date: 2026-06-06

A database trigger sets updated_at on every row update, regardless of which
application path writes it. Reads and the creation-time default are unchanged.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "54bb7ed5f51a"
down_revision: str | Sequence[str] | None = "c22e2864d3d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the BEFORE UPDATE trigger that stamps reviews.updated_at = NOW()."""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION outrider_reviews_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at := NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_reviews_set_updated_at ON reviews;
        CREATE TRIGGER trg_reviews_set_updated_at
            BEFORE UPDATE ON reviews
            FOR EACH ROW
            EXECUTE FUNCTION outrider_reviews_set_updated_at();
        """
    )


def downgrade() -> None:
    """Drop the trigger and its function (mirror of upgrade)."""
    op.execute("DROP TRIGGER IF EXISTS trg_reviews_set_updated_at ON reviews;")
    op.execute("DROP FUNCTION IF EXISTS outrider_reviews_set_updated_at();")
