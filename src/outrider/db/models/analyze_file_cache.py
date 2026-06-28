# See specs/2026-06-11-file-hash-analyze-cache.md + DECISIONS.md#014 (retention regime).
"""ANALYZE_FILE_CACHE.

One row per cached per-file analyze outcome (cost lever #8). A CACHE,
not audit: rows delete legally (TTL, retention sweep, installation
purge) — the append-only discipline lives on the companion audit events,
never here.

Retention regime (the `findings` / `llm_call_content` sibling): the
payload is user-code-derived content (admitted finding content pre-HITL
+ full trace candidates including their LLM-derived `reason`), so the
table carries `retention_expires_at` (write-time
`min(now + cache TTL, source review retention)`), participates in the
installation-deleted purge, and follows the no-resurrection rule three
ways — `source_review_id` FK CASCADE (review purge takes its cache rows
with it), the row's own `retention_expires_at`, and the store's
lookup-time expiry check (an expired row is a MISS).

Two foreign keys, mirroring `findings`:

  - `source_review_id → reviews.id` ON DELETE CASCADE — content-tier
    coupling: purging the source review must purge cache rows derived
    from it, or a later hit would re-mint legally deleted content.
  - `installation_id → installations.installation_id` ON DELETE
    RESTRICT — the sweep purges cache rows (with `purge_audit`) before
    an installation hard-deletes; RESTRICT prevents a silent CASCADE
    that would skip the purge-audit row.

`cache_key` (the sixteen-field digest — prompt digest + fifteen explicit
components, the last three the host-identity triad per DECISIONS.md#056 —
from `cache/key.py::compute_analyze_cache_key`) is the
primary key — the write path's conflict arbiter for concurrent same-key
reviews (`ON CONFLICT DO UPDATE ... WHERE` the existing row is expired:
live rows keep first-writer-wins; an expired-but-unswept row is
refreshed in place). The key-component columns are denormalized for
observability (stale-rate by version, Stage-B telemetry queries); the
key itself remains the only lookup identity.
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
    is_eval: Mapped[bool] = mapped_column(Boolean, server_default=text("false"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    retention_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
