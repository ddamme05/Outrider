"""SEVERITY_POLICIES.

Versioned mapping from `FindingType` to severity, keyed by `version` PK so a
finding's classification can be replayed under the policy in effect at the time
of review per `severity-policy-versioned-for-replay`. The policy itself lives in
JSONB; its shape is owned by `policy/severity.py` (when it lands), not the DB.

Migration 0001 seeds version "1.0.0" as a non-negotiable migration step — the
FK from FINDINGS.policy_version is RESTRICT, so without this row every finding
insert FK-fails on a fresh DB.
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
