"""analyze_file_cache host-triad telemetry columns — profile_id + reasoning_enabled.

Revision ID: ff57dcf6fbd1
Revises: e1f2a3b4c5d6
Create Date: 2026-06-29

Adds two nullable columns to `analyze_file_cache` for group-by-host warm-cache
telemetry (DECISIONS.md#056 / FUP-194). The host-identity triad already folds
into the cache KEY (correctness/isolation); these denormalize the two
human-meaningful triad components for observability, mirroring `model` and the
other denormalized key-component columns:

  - profile_id        — the resolved host ("anthropic" for the native host,
                        "baseten" etc. for a registry host)
  - reasoning_enabled — whether the cached analysis ran with reasoning on

Both nullable: NULL = UNQUALIFIED (no host-identity triad supplied — a pre-#056
row or test wiring), NOT the anthropic host — a real anthropic run stamps
"anthropic". No backfill: pre-#056 rows predate host-qualification, so there is
no host to assign — NULL is genuinely correct for them. The opaque
`profile_contract_digest` is NOT denormalized (no human-meaningful group-by; it
stays in the cache_key for correctness only).
Plain ADD COLUMN ... NULL — metadata-only, no table rewrite. Touches only
`analyze_file_cache`; no append-only table involved.
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
