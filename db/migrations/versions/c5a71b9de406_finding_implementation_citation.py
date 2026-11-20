"""Keep a checked implementation citation with the finding's retained content."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c5a71b9de406"
down_revision = "e2c4a6b8d0f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "findings", sa.Column("implementation_citation", postgresql.JSONB(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("findings", "implementation_citation")
