"""analyze_file_cache: the lever-8 per-file analyze outcome cache.

A CACHE, not audit (specs/2026-06-11-file-hash-analyze-cache.md): rows
delete legally via TTL, retention sweep, and installation purge — the
append-only discipline lives on the companion audit events. The table
joins the DECISIONS.md#014 content-retention regime: `source_review_id`
FK CASCADE couples cache rows to their source review's purge,
`installation_id` FK RESTRICT forces purge-before-delete (with
purge_audit), and `retention_expires_at` carries the
min(TTL, source retention) write-time bound. `cache_key` (the
eight-component digest) is the PK and the `ON CONFLICT DO NOTHING`
arbiter for concurrent same-key writes.

Plain (non-concurrent) DDL: brand-new table, nothing to lock.
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
