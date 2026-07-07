# See DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted
"""PURGE_AUDIT.

Append-only forever per #012, alongside `audit_events`. The append-only trigger
on this table is added by migration 0001 (separate DDL); SQLAlchemy doesn't
model triggers in metadata so the trigger is invisible at the ORM layer.

`installation_id` is a plain BIGINT with NO foreign key per docs/schema.md
"PURGE_AUDIT.installation_id is a loose reference, not an FK." The forensic
trail of which content rows were purged on which `installation.deleted` event
must survive the installation hard-delete itself; an FK with CASCADE would
erase the trail, RESTRICT would block the install delete. The loose reference
lets PURGE_AUDIT outlive the installation it references — by design.

Per the schema-layer spec: one PURGE_AUDIT row per target table per sweep run.
The `target_table` column is a single concrete table name — one of the four
content tables ("reviews", "findings", "llm_call_content", "analyze_file_cache")
for a retention/TTL or content purge, OR "installations" for the install-row
hard-delete evidence row that `purge_installation` writes so an uninstall purge
leaves forensic proof even when the install had zero content (#012 — the
`installations` row is the entity delete, deliberately outside the content-table
set). A time-based sweep that purges all four content tables produces four rows;
an install-purge produces one row per content table that had rows PLUS the
"installations" row. Implicit grouping is by `(installation_id, timestamp)`
clustering — the per-table rows commit in the same transaction so their
`timestamp` defaults to a microsecond-precision window. Adding an explicit
`sweep_run_id` column is a future additive change if forensic queries demand
stronger grouping.
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
