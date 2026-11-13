"""Store reusable per-file analysis results.

Cache rows hold a result, its model and policy versions, and an expiry time.
They can be replaced after expiry or deleted during retention cleanup; they
are not the append-only audit record. Deleting a source review removes its
cache rows (DECISIONS.md#014).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "e9a1c3d5b7f2"
down_revision: str | Sequence[str] | None = "54bb7ed5f51a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "analyze_file_cache",
        sa.Column("cache_key", sa.Text(), primary_key=True),
        sa.Column(
            "installation_id",
            sa.BigInteger(),
            sa.ForeignKey("installations.installation_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("repo_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "source_review_id",
            sa.Uuid(),
            sa.ForeignKey("reviews.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("payload", JSONB(), nullable=False),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("prompt_template_version", sa.Text(), nullable=False),
        sa.Column("trivial_filter_version", sa.Text(), nullable=False),
        sa.Column("query_registry_digest", sa.Text(), nullable=False),
        sa.Column("active_policy_version", sa.Text(), nullable=False),
        sa.Column("analyze_parser_version", sa.Text(), nullable=False),
        sa.Column("prompt_hash", sa.Text(), nullable=False),
        sa.Column("is_eval", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_analyze_file_cache_retention_expires_at",
        "analyze_file_cache",
        ["retention_expires_at"],
    )
    op.create_index(
        "ix_analyze_file_cache_installation_id",
        "analyze_file_cache",
        ["installation_id"],
    )
    op.create_index(
        "ix_analyze_file_cache_source_review_id",
        "analyze_file_cache",
        ["source_review_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_analyze_file_cache_source_review_id", table_name="analyze_file_cache")
    op.drop_index("ix_analyze_file_cache_installation_id", table_name="analyze_file_cache")
    op.drop_index("ix_analyze_file_cache_retention_expires_at", table_name="analyze_file_cache")
    op.drop_table("analyze_file_cache")
