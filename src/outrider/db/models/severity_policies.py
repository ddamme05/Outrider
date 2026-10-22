"""Stored severity mappings keyed by version for historical replay.

policy/severity.py owns the mapping; policy/versions.py loads and checks stored
versions (severity-policy-versioned-for-replay). Migration 0001 seeds 1.0.0 so
findings can reference it on a fresh database; their RESTRICT foreign key prevents
deleting a referenced policy.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime

from outrider.db.models._base import Base


class SeverityPolicy(Base):
    __tablename__ = "severity_policies"

    version: Mapped[str] = mapped_column(Text, primary_key=True)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
