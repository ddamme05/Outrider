"""Record the model host and reasoning setting on cached results.

Revision ID: ff57dcf6fbd1
Revises: e1f2a3b4c5d6
Create Date: 2026-06-29

profile_id and reasoning_enabled support cache reporting by host. Both are
nullable, with no backfill: null means no host information was supplied, not
that the result came from Anthropic. Host identity remains part of the cache
key (DECISIONS.md#056, FUP-194).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "ff57dcf6fbd1"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable host-triad telemetry columns to `analyze_file_cache`."""
    op.add_column("analyze_file_cache", sa.Column("profile_id", sa.Text(), nullable=True))
    op.add_column("analyze_file_cache", sa.Column("reasoning_enabled", sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Drop the host-triad telemetry columns (mirror of upgrade)."""
    op.drop_column("analyze_file_cache", "reasoning_enabled")
    op.drop_column("analyze_file_cache", "profile_id")
