"""FINDINGS.

One row per analyst finding produced by the agent. Three foreign keys:

  - `review_id → reviews.id` ON DELETE CASCADE — both tables are content-tier
    with retention; cascading review purge to its findings is correct, and
    findings can also purge independently before their review via their own
    `retention_expires_at`. This is the only review-scoped CASCADE in the schema.
  - `installation_id → installations.installation_id` ON DELETE RESTRICT —
    sweep job must purge findings for an install before the install itself
    hard-deletes; RESTRICT prevents a silent CASCADE that would skip the
    per-table `purge_audit` row.
  - `policy_version → severity_policies.version` ON DELETE RESTRICT — severity
    policies are versioned-forever per `severity-policy-versioned-for-replay`;
    a finding's policy version must remain reachable for the lifetime of the
    finding, so deleting a referenced policy is blocked.

Proof-boundary columns (`evidence_tier`, `query_match_id`, `trace_path`) are
shaped here at the DB layer; the application-side `ReviewFinding.enforce_proof_boundary`
validator (in `policy/findings.py` when written) enforces that OBSERVED findings
carry `query_match_id` and INFERRED carry `trace_path`. The DB allows nullable
on both because not every finding has both — the validator enforces the
either-or per-tier rule.

Notably absent: no `confidence` column, per `confidence-is-computed-not-assigned`.
The `tests/unit/test_orm_structural_invariants.py` (next commit's unit test)
asserts the column's absence in code, not just review.

CHECK constraints encode the invariant that `line_start >= 1` (1-indexed per
`coordinates/`) and `line_end >= line_start`.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
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
        CheckConstraint(
            "line_end >= line_start", name="ck_findings_line_end_gte_line_start"
        ),
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
    trace_path: Mapped[list[Any] | None] = mapped_column(JSONB, nullable=True)
    original_severity: Mapped[str | None] = mapped_column(Text, nullable=True)
    override_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    overrider_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)
    publish_destination: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_eval: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
