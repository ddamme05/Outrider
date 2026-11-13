"""Remove stored review totals that are calculated from audit events.

Revision ID: b7e3f1a92c4d
Revises: 7c4d8e2a1b5f
Create Date: 2026-06-03

Drop the seven review metric columns and their checks. Dashboard totals come
from recorded events rather than these columns (DECISIONS.md#037). Downgrade
recreates the columns with their original types and defaults.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "b7e3f1a92c4d"
down_revision: str | Sequence[str] | None = "7c4d8e2a1b5f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (column, sa type, CHECK-constraint name) — the seven dead metric columns.
_METRIC_COLUMNS: tuple[tuple[str, sa.types.TypeEngine, str], ...] = (
    ("files_examined", sa.Integer(), "ck_reviews_files_examined_nonneg"),
    ("files_traced_beyond_diff", sa.Integer(), "ck_reviews_files_traced_beyond_diff_nonneg"),
    ("llm_calls_made", sa.Integer(), "ck_reviews_llm_calls_made_nonneg"),
    ("total_input_tokens", sa.Integer(), "ck_reviews_total_input_tokens_nonneg"),
    ("total_output_tokens", sa.Integer(), "ck_reviews_total_output_tokens_nonneg"),
    ("total_cost_usd", sa.Numeric(), "ck_reviews_total_cost_usd_nonneg"),
    ("wall_clock_seconds", sa.Numeric(), "ck_reviews_wall_clock_seconds_nonneg"),
)


def upgrade() -> None:
    for column, _sa_type, check_name in _METRIC_COLUMNS:
        op.drop_constraint(check_name, "reviews", type_="check")
        op.drop_column("reviews", column)


def downgrade() -> None:
    # Re-attach the (dead) columns. The server_default 0 is a transient backfill
    # so the NOT NULL add is valid on existing rows; drop it right after so the
    # re-created columns match genesis exactly (NOT NULL, no server default).
    for column, sa_type, check_name in _METRIC_COLUMNS:
        op.add_column(
            "reviews",
            sa.Column(column, sa_type, nullable=False, server_default=sa.text("0")),
        )
        op.alter_column("reviews", column, server_default=None)
        op.create_check_constraint(check_name, "reviews", f"{column} >= 0")
