"""Store a review's status, retained report and pending approval data.

The PR/commit uniqueness constraint prevents duplicate webhook reviews.
Detailed model metrics come from audit events; this row holds the review's
current state and content until expiry.

See DECISIONS.md#014, #016, #037 and idempotency-via-db-unique-constraint.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
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
    # PR title captured at review creation from the webhook payload
    # (`pull_request.title`). Nullable: pre-`f4c8a1d2b9e3` rows have no
    # value, and the value is attacker-controlled webhook data persisted as
    # a parameterized column (never SQL-interpolated) + rendered escaped.
    # Immutable after creation — see the migration docstring.
    pr_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    head_sha: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(review_status_enum, nullable=False)
    hitl_request: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    hitl_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    # Terminal-success timestamp set by `ReviewStatusSink.mark_completed`
    # at the publish node's terminal-success paths (`docs/spec.md` §3.3
    # step 10). `None` while the review is in flight; populated alongside
    # the `status='completed'` flip.
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Approval cutoff for sweep/hitl_expiry.py. NULL until the request is stored;
    # mark_running retains it after resume for forensic visibility. The sweep
    # only selects awaiting_approval rows whose cutoff has elapsed.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
