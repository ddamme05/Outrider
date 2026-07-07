"""installations B2 columns — suspension state + repository_selection.

Revision ID: a7c3f2b18d40
Revises: ff57dcf6fbd1
Create Date: 2026-07-06

Arc B2 (DECISIONS.md#065) adds two columns to `installations`:

  - suspended_at        — nullable timestamptz; set on `installation.suspend`,
                          cleared on `unsuspend`. Suspension is a reversible
                          pause (#012 — never a purge); the active-membership
                          gate excludes suspended installs (`suspended_at IS NULL`).
  - repository_selection — NOT NULL text, `'all'` | `'selected'` (GitHub's
                          install `repository_selection`). `selected` requires a
                          per-repo `installation_repositories` row (fail-closed);
                          `all` authorizes at the install level. Server-default
                          `'selected'` so any pre-B2 row is fail-closed by default.

Touches only `installations`; no append-only table is involved.
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
