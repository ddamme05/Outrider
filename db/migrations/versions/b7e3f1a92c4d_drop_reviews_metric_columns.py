"""Drop the seven seeded-zero `reviews` aggregate-metric columns (DECISIONS.md#037).

Revision ID: b7e3f1a92c4d
Revises: 7c4d8e2a1b5f
Create Date: 2026-06-03

Per DECISIONS.md#037 (FUP-127): the `reviews` table carried seven denormalized
aggregate-metric columns (`files_examined`, `files_traced_beyond_diff`,
`llm_calls_made`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`,
`wall_clock_seconds`). They were seeded to 0 by the webhook INSERT and never
rolled up — a DEAD denormalized copy. No consumer reads them: the dashboard
computes metrics read-through from the audit stream (`_aggregate_metrics`, per
the FUP-130 work) and replay surfaced the row values verbatim without reading
or verifying them. Drop the columns + their non-negative CHECK constraints; the
audit stream stays the source of truth per #030.

One-way in intent (the columns held no data — all-zero, no reader). The
downgrade re-creates them NOT NULL using a transient `server_default` of 0 to
backfill existing rows, then drops the default so the columns match genesis
exactly (NOT NULL, no server default); replay's `ReconstructedReviewMetadata`
no longer references them, so a downgrade reattaches dead columns, not data.
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
