"""installations Slack config columns — per-install workspace / bot-token / channel.

Revision ID: 3bb6c2c3c0b1
Revises: b1d7e4a92c63
Create Date: 2026-06-15

Adds five nullable columns to `installations` for per-installation Slack
notifications (dashboard-in-Slack, commit 6):

  - slack_team_id              — Slack workspace/team id (from oauth.v2.access)
  - slack_bot_token_ciphertext — bot token ENCRYPTED at rest (DECISIONS.md#051;
                                 Fernet ciphertext via notify/token_crypto.py),
                                 never plaintext
  - slack_channel_id           — the channel the bot posts to
  - slack_configured_at        — when the install completed the Slack OAuth flow
  - slack_configured_by        — the dashboard admin who bound the install (V1: "admin")

All nullable: Slack is opt-in per install, and an install that never connects
Slack leaves them NULL — no backfill. The ciphertext column is bytea
(`LargeBinary`); decryption is confined to the Slack notifier boundary. The
columns live on `installations` (per the dashboard-in-Slack spec) so the #012
tombstone/purge lifecycle carries the encrypted token with the install on
hard-delete. Touches only `installations`; no append-only table is involved.
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
