"""reviews.pr_title — persist the PR title captured at review creation.

Revision ID: f4c8a1d2b9e3
Revises: e9a1c3d5b7f2
Create Date: 2026-06-13

Adds a nullable `pr_title` column to `reviews`. The PR title
(`pull_request.title` from the webhook payload) was previously only present
on the transient `PRContext` / `ReviewState`, never persisted — so the
dashboard reviews list could not show it (it rendered raw `repo_id` + PR
number). Captured at review creation (`api/webhooks/router.py`), immutable
thereafter: review creation is idempotent on `(repo_id, pr_number,
head_sha)`, so a new head SHA is a new row (new title) and the same triple
short-circuits without mutating the title. Nullable with NO backfill —
pre-existing rows stay NULL and the dashboard renders a fallback. Touches
only `reviews`; no append-only table is involved.
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
