"""Keep a permanent record of retention deletions.

Each row records the target table and deletion count. These records survive
installation deletion and contain no purged content (DECISIONS.md#012, #014).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, CheckConstraint, Index, Integer, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, Uuid

from outrider.db.models._base import Base


class PurgeAudit(Base):
    __tablename__ = "purge_audit"
    __table_args__ = (
        CheckConstraint("rows_affected >= 0", name="ck_purge_audit_rows_affected_nonneg"),
        # Purge-history queries: "show purges for installation X" or
        # "purges in the last week."
        Index("ix_purge_audit_installation_timestamp", "installation_id", "timestamp"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    # Loose reference; intentionally NOT a foreign key. See docs/schema.md
    # "PURGE_AUDIT.installation_id is a loose reference, not an FK."
    installation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    target_table: Mapped[str] = mapped_column(Text, nullable=False)
    rows_affected: Mapped[int] = mapped_column(Integer, nullable=False)
    purge_role: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
