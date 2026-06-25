"""ANOMALIES.

Forensic-tier table populated by `anomaly/scanner.py` (when written) reading the
audit stream. Anomalies are not retention-bound — they keep their value
indefinitely as the operations record of "did the agent behave correctly?"

`review_id → reviews.id` ON DELETE SET NULL (column is nullable). When a review
purges through retention, anomalies survive as a forensic trail with the FK set
to NULL, marked logically as "associated with a since-purged review." This is
acceptable for anomalies because they don't carry the every-event-has-review-id
invariant that audit_events does — the anomaly is keyed off a `rule_name` which
gives it standalone meaning.

`status` uses the `anomaly_status_enum` PG-native ENUM type (open, acknowledged,
resolved). Distinct from `review_status_enum` per docs/schema.md "Two distinct
PG ENUM types" — both happen to use a column named `status` but the value sets
don't overlap.
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
        # Partial unique indexes — per-rule_name natural-key idempotency for
        # `AnomalyPersister.emit_anomaly` `on_conflict_do_nothing` arbiter
        # inference. Declared in metadata to keep `alembic revision
        # --autogenerate` clean (the migrations actually create them via
        # CREATE INDEX CONCURRENTLY for production-safety — see
        # `33f8fe051bec_hitl_node_indexes.py` and
        # `7c4d8e2a1b5f_synthesize_node_indexes.py`). Per Pass-1 multi-lens
        # audit DB-lens §4. Mirrors `reviews.py` precedent for partial-index
        # metadata declaration.
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
