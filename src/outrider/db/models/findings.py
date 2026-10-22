# See DECISIONS.md#014-audit-events-are-metadata-only-content-purge-targets-reviews-and-findings
"""Retained finding content, one row per persisted finding_id.

Review purge cascades to findings; installation RESTRICT requires explicit purge
with a purge_audit record. Policy-version RESTRICT preserves the severity policy
needed for replay (severity-policy-versioned-for-replay).

Proof columns are nullable here; schemas/review_finding.py enforces each evidence
tier's query_match_id or trace_path requirements. Confidence is computed from
evidence, not stored (confidence-is-computed-not-assigned). Database checks keep
line ranges 1-indexed and ordered. content_hash groups equivalent content; it is
not a write-time uniqueness key.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base


class Finding(Base):
    __tablename__ = "findings"
    __table_args__ = (
        CheckConstraint("line_start >= 1", name="ck_findings_line_start_gte_1"),
        CheckConstraint("line_end >= line_start", name="ck_findings_line_end_gte_line_start"),
        # Retention sweep query.
        Index("ix_findings_retention_expires_at", "retention_expires_at"),
        # Installation-scoped purge query.
        Index("ix_findings_installation_id", "installation_id"),
        # Read-time content grouping per spec §8.5 — find same-content findings
        # by hash (e.g. dashboard collapse of duplicate findings across rounds).
        # The writer keys uniqueness on finding_id (PK), NOT content_hash: one
        # findings row per FindingEvent.finding_id, content_hash is the equality
        # signal for replay + read-time grouping, not a write-time uniqueness key.
        Index("ix_findings_content_hash", "content_hash"),
    )

    finding_id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    review_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False
    )
    installation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("installations.installation_id", ondelete="RESTRICT"),
        nullable=False,
    )
    policy_version: Mapped[str] = mapped_column(
        Text,
        ForeignKey("severity_policies.version", ondelete="RESTRICT"),
        nullable=False,
    )
    finding_type: Mapped[str] = mapped_column(Text, nullable=False)
    dimension: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_tier: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    line_start: Mapped[int] = mapped_column(Integer, nullable=False)
    line_end: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_fix: Mapped[str | None] = mapped_column(Text, nullable=True)
    query_match_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_path: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    original_severity: Mapped[str | None] = mapped_column(Text, nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    overrider_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    publish_destination: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
