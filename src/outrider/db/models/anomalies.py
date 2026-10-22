"""Persistent operations findings emitted through anomaly/persister.py.

Anomalies survive review retention: ON DELETE SET NULL removes their review link
while rule_name retains their meaning. They are not retention-bound audit events.
Their lifecycle uses a separate PostgreSQL enum from review status
(docs/schema.md, "Two distinct PG ENUM types"). Partial unique indexes enforce
per-rule idempotency.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import ForeignKey, Index, Text, desc, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base, anomaly_status_enum


class Anomaly(Base):
    __tablename__ = "anomalies"
    __table_args__ = (
        # Anomaly-queue ordering per spec §9.9: operations dashboard surfaces
        # highest-severity unresolved anomalies first, newest within each tier.
        Index(
            "ix_anomalies_severity_created_at",
            "severity",
            desc("created_at"),
        ),
        # Match emit_anomaly's per-rule ON CONFLICT targets. Migrations create these
        # concurrently; ORM declarations keep autogenerate in sync. See
        # 33f8fe051bec_hitl_node_indexes.py and 7c4d8e2a1b5f_synthesize_node_indexes.py.
        Index(
            "uq_anomalies_hitl_timeout_natural_key",
            "review_id",
            unique=True,
            postgresql_where=text("rule_name = 'hitl_timeout'"),
        ),
        Index(
            "uq_anomalies_cross_round_severity_divergence_natural_key",
            "review_id",
            unique=True,
            postgresql_where=text("rule_name = 'cross_round_severity_divergence'"),
        ),
        Index(
            "uq_anomalies_cost_budget_starvation_natural_key",
            "review_id",
            unique=True,
            postgresql_where=text("rule_name = 'cost_budget_starvation'"),
        ),
        Index(
            "uq_anomalies_gated_findings_over_cap_natural_key",
            "review_id",
            unique=True,
            postgresql_where=text("rule_name = 'gated_findings_over_cap'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    review_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("reviews.id", ondelete="SET NULL"), nullable=True
    )
    rule_name: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(anomaly_status_enum, nullable=False)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
