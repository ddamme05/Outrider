# See specs/2026-06-11-file-hash-analyze-cache.md + DECISIONS.md#014 (retention regime).
"""Retained per-file analysis results, keyed by cache/key.py's composed digest.

These rows contain user-code-derived findings and trace candidates. They may be
deleted; companion audit events remain append-only. The source-review CASCADE,
retention_expires_at, and lookup-time expiry check prevent serving purged content.
Expiry is bounded by cache TTL and source-review retention (DECISIONS.md#014).

Installation RESTRICT requires explicit, audited content purge before hard-delete.
Live keys keep the first writer; expired rows can be refreshed by the store's
conditional upsert. Denormalized recipe and host columns are telemetry only;
the digest is the lookup identity (DECISIONS.md#056, FUP-194).
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import BigInteger, ForeignKey, Index, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import Boolean, DateTime, Uuid

from outrider.db.models._base import Base


class AnalyzeFileCache(Base):
    __tablename__ = "analyze_file_cache"
    __table_args__ = (
        # Retention sweep query.
        Index("ix_analyze_file_cache_retention_expires_at", "retention_expires_at"),
        # Installation-scoped purge query.
        Index("ix_analyze_file_cache_installation_id", "installation_id"),
        # Review-purge CASCADE support + provenance queries.
        Index("ix_analyze_file_cache_source_review_id", "source_review_id"),
    )

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    installation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("installations.installation_id", ondelete="RESTRICT"),
        nullable=False,
    )
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    source_review_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("reviews.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    # Cached outcome: admitted finding content (pre-HITL, policy-stamped)
    # + full trace candidates (incl. `reason` — content-tier, which is
    # WHY it lives here and not on the serve event).
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    # Denormalized key components (observability only; never re-keyed).
    model: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_template_version: Mapped[str] = mapped_column(Text, nullable=False)
    trivial_filter_version: Mapped[str] = mapped_column(Text, nullable=False)
    query_registry_digest: Mapped[str] = mapped_column(Text, nullable=False)
    active_policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    analyze_parser_version: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # Host-identity triad (DECISIONS.md#056) denormalized for group-by-host telemetry
    # (FUP-194): profile_id is the RESOLVED host — "anthropic" for the native host,
    # "baseten" etc. for a registry host. NULL = UNQUALIFIED (no triad supplied: a
    # pre-#056 row or test wiring), NOT the anthropic host — a real anthropic run stamps
    # "anthropic" (key.py: all-None is the unqualified caller, distinct from a real host).
    # The opaque profile_contract_digest stays in the key only — no group-by value.
    profile_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_enabled: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
