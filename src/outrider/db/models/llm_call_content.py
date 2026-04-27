# See DECISIONS.md#016-llm-exchanges-stored-locally-under-retention-logs-stay-metadata-only
"""LLM_CALL_CONTENT.

Per #016, prompt and completion text for every LLM call land in a separate
content table keyed by `event_id` (matching the corresponding LLMCallEvent audit
row). The audit row stays metadata-only per #014; the content lives here.

`event_id` is BOTH primary key AND foreign key to `audit_events.event_id` with
`NO ACTION` (PostgreSQL default, equivalent to RESTRICT). The FK direction is
child → parent: this content row references the audit row, not vice versa. Since
the parent (`audit_events`) is append-only forever via its trigger, the NO ACTION
behavior never fires in practice — the trigger blocks parent DELETE before any
FK action could trigger. NO ACTION is the right shape because (a) it never fires,
and (b) it documents intent: "if anyone tries to delete a parent audit row, fail
loud." See docs/schema.md "Content-table foreign-key semantics" for the full
analysis of why this is NOT a "loose UUID."

`installation_id` mirrors the FINDINGS pattern: real FK with RESTRICT for
denormalized purge scoping. The retention sweep deletes content rows directly
(safe child-row delete; doesn't touch the parent audit row); the parent stays
forever. Joining audit_events to llm_call_content after a content purge returns
no row for the purged audit events — that's the metadata-only-replay state per
#014 point 4 applied to LLM content.

The single-transaction insert constraint from #016 (LLMCallEvent audit row +
LLM_CALL_CONTENT row in one transaction or neither) is enforced by application
code, not the schema; integration test `test_llm_content_single_transaction.py`
gates this.
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
    is_eval: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
