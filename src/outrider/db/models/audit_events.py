"""Store permanent review-history metadata in append-only rows.

Payloads contain no review or model content. sequence_number is assigned by
the database. review_id deliberately has no foreign key so history survives
review deletion (DECISIONS.md#014, #026; every-audit-event-has-review-id).
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, Identity, Index, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"
    __table_args__ = (
        # UNIQUE(review_id, sequence_number) produces an implicit unique index
        # which covers replay-traversal queries; no separate Index needed for it.
        UniqueConstraint("review_id", "sequence_number", name="uq_audit_review_sequence"),
        # Dashboard time-range queries: events between X and Y for a review.
        Index("ix_audit_events_review_timestamp", "review_id", "timestamp"),
        # V1.5 forward-compat: parallel-analyze branch grouping per review.
        Index("ix_audit_events_review_phase_key", "review_id", "phase_key"),
    )

    event_id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    # Logical reference to reviews.id; intentionally NOT a foreign key. See
    # docs/schema.md and DECISIONS.md#014 for the no-FK rationale.
    review_id: Mapped[UUID] = mapped_column(Uuid, nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    phase_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    sequence_number: Mapped[int] = mapped_column(BigInteger, Identity(always=False), nullable=False)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
