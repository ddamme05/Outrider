"""Store optional Slack connection details for each installation.

Revision ID: 3bb6c2c3c0b1
Revises: b1d7e4a92c63
Create Date: 2026-06-15

The fields hold the workspace, encrypted bot token, destination channel and
setup details. They are nullable because Slack is optional; existing rows need
no backfill. Deleting an installation also removes its Slack settings
(DECISIONS.md#051).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3bb6c2c3c0b1"
down_revision: str | Sequence[str] | None = "b1d7e4a92c63"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable per-install Slack config columns to `installations`."""
    op.add_column("installations", sa.Column("slack_team_id", sa.Text(), nullable=True))
    op.add_column(
        "installations",
        sa.Column("slack_bot_token_ciphertext", sa.LargeBinary(), nullable=True),
    )
    op.add_column("installations", sa.Column("slack_channel_id", sa.Text(), nullable=True))
    op.add_column(
        "installations",
        sa.Column("slack_configured_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("installations", sa.Column("slack_configured_by", sa.Text(), nullable=True))


def downgrade() -> None:
    """Drop the per-install Slack config columns (mirror of upgrade)."""
    op.drop_column("installations", "slack_configured_by")
    op.drop_column("installations", "slack_configured_at")
    op.drop_column("installations", "slack_channel_id")
    op.drop_column("installations", "slack_bot_token_ciphertext")
    op.drop_column("installations", "slack_team_id")
