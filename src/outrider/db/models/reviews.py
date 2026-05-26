"""REVIEWS.

One row per agent review run. Idempotency is enforced by the
`UNIQUE(repo_id, pr_number, head_sha)` constraint per `idempotency-via-db-unique-constraint`
— the webhook handler relies on `IntegrityError` from this constraint as the
dedup signal under near-simultaneous deliveries.

`installation_id` is FK to INSTALLATIONS with `ON DELETE RESTRICT` so the
installation-purge sweep step must explicitly purge reviews before the parent
install hard-deletes (otherwise CASCADE would silently drop content without
writing a per-table `purge_audit` row).

`status` uses the `review_status_enum` PG-native ENUM type. `hitl_request` and
`hitl_decision` are nullable JSONB envelopes; when the graph reaches the HITL
node and gates on a critical/high finding, request lands here and the eventual
decision lands alongside it. The metric columns (`files_examined` etc.) carry
CHECK >= 0 so the migration catches off-by-one bugs at the DB layer.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base, review_status_enum


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (
        UniqueConstraint("repo_id", "pr_number", "head_sha", name="uq_review_natural_key"),
        CheckConstraint("files_examined >= 0", name="ck_reviews_files_examined_nonneg"),
        CheckConstraint(
            "files_traced_beyond_diff >= 0",
            name="ck_reviews_files_traced_beyond_diff_nonneg",
        ),
        CheckConstraint("llm_calls_made >= 0", name="ck_reviews_llm_calls_made_nonneg"),
        CheckConstraint("total_input_tokens >= 0", name="ck_reviews_total_input_tokens_nonneg"),
        CheckConstraint("total_output_tokens >= 0", name="ck_reviews_total_output_tokens_nonneg"),
        CheckConstraint("total_cost_usd >= 0", name="ck_reviews_total_cost_usd_nonneg"),
        CheckConstraint("wall_clock_seconds >= 0", name="ck_reviews_wall_clock_seconds_nonneg"),
        # Retention sweep: rows whose retention_expires_at has passed.
        Index("ix_reviews_retention_expires_at", "retention_expires_at"),
        # Installation purge / scoped query: all reviews for an installation_id.
        Index("ix_reviews_installation_id", "installation_id"),
        # Partial index for the active-review states the sweep job watches
        # (stuck-running detection + HITL-expiry detection per spec §9.9).
        # Partial keeps the index small — terminal-state rows do not bloat it.
        Index(
            "ix_reviews_active_status",
            "status",
            postgresql_where=text("status IN ('running', 'awaiting_approval')"),
        ),
        # HITL-expiry sweep: rows whose HITL approval window has elapsed.
        # Filtered on `status='awaiting_approval'` so the sweep's
        # `WHERE status='awaiting_approval' AND expires_at < NOW()` query
        # walks the index directly. The migration creates the underlying
        # index concurrently; this declaration keeps the SQLAlchemy
        # metadata in sync.
        Index(
            "ix_reviews_awaiting_approval_expires_at",
            "expires_at",
            postgresql_where=text("status = 'awaiting_approval'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("installations.installation_id", ondelete="RESTRICT"),
        nullable=False,
    )
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    head_sha: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(review_status_enum, nullable=False)
    hitl_request: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hitl_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    files_examined: Mapped[int] = mapped_column(Integer, nullable=False)
    files_traced_beyond_diff: Mapped[int] = mapped_column(Integer, nullable=False)
    llm_calls_made: Mapped[int] = mapped_column(Integer, nullable=False)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_cost_usd: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    wall_clock_seconds: Mapped[Decimal] = mapped_column(Numeric, nullable=False)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # HITL approval window cutoff. Mirrors `HITLRequest.expires_at` for
    # sweep-query efficiency (`status='awaiting_approval' AND expires_at <
    # NOW()` per `sweep/hitl_expiry.py`). `None` outside the HITL gate;
    # set by `ReviewStatusSink.mark_awaiting_approval`, left in place by
    # `mark_running` so the forensic record persists past the resume.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
