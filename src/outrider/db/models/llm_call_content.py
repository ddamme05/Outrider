# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
"""Retained model exchanges paired with metadata-only audit events.

The persister inserts the LLMCallEvent and this content row in one transaction
(DECISIONS.md#016; tests/integration/test_llm_content_single_transaction.py).
event_id references the append-only audit parent with NO ACTION. Retention deletes
the content child directly while the audit row survives, leaving metadata-only
replay (DECISIONS.md#014).

Installation RESTRICT requires explicit content purge before installation deletion.
See docs/schema.md, "Content-table foreign-key semantics".
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Index, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base


class LLMCallContent(Base):
    __tablename__ = "llm_call_content"
    __table_args__ = (
        # Retention sweep query.
        Index("ix_llm_call_content_retention_expires_at", "retention_expires_at"),
        # Installation-scoped purge query.
        Index("ix_llm_call_content_installation_id", "installation_id"),
    )

    event_id: Mapped[UUID] = mapped_column(
        Uuid,
        ForeignKey("audit_events.event_id", ondelete="NO ACTION"),
        primary_key=True,
    )
    installation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("installations.installation_id", ondelete="RESTRICT"),
        nullable=False,
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    completion: Mapped[str] = mapped_column(Text, nullable=False)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
