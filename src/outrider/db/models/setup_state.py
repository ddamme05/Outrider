# See DECISIONS.md#070 — the durable singleton setup state machine + nonce store.
"""`SetupState` (singleton) + `SetupNonce` — the App-Manifest onboarding state machine (#070).

`SetupState` is a DB-enforced **singleton** (fixed `id = 1`, `CHECK`): the one row holding the
onboarding status (`UNCONFIGURED → AWAITING_CALLBACK → CONVERTING → CONFIGURED`, + `ORPHANED`), the
attempt binding (`expected_org_login`, `expected_permissions`, `expected_events`,
`manifest_contract_digest`) recorded at Start, and `conversion_started_at` for the stale
timeout. Transitions are guarded by DB compare-and-swap on `status`, not SELECT checks.

`SetupNonce` holds the **hashed** callback nonce (never raw) with `expires_at`; consumption is
atomic delete-on-consume (no periodic sweep — expired rows are deleted by the lazy `POST /setup`
repair).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import CheckConstraint, SmallInteger, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, Uuid

from outrider.db.models._base import Base

__all__ = ["SETUP_STATUSES", "SetupNonce", "SetupState"]

# The setup state-machine vocabulary (DECISIONS.md#070). Text + CHECK rather than a PG ENUM to keep
# the migration simple and the value set greppable; the CHECK is the DB-side guard.
SETUP_STATUSES: tuple[str, ...] = (
    "UNCONFIGURED",
    "AWAITING_CALLBACK",
    "CONVERTING",
    "CONFIGURED",
    "ORPHANED",
)

# The status CHECK, DERIVED from SETUP_STATUSES so the model can't drift from the vocabulary. The
# migration's CHECK is a frozen snapshot of this at authoring time; a test pins the two together.
_STATUS_CHECK = "status IN (" + ", ".join(f"'{s}'" for s in SETUP_STATUSES) + ")"


class SetupState(Base):
    __tablename__ = "setup_state"
    __table_args__ = (
        # Singleton: exactly one row, id pinned to 1.
        CheckConstraint("id = 1", name="ck_setup_state_singleton"),
        CheckConstraint(_STATUS_CHECK, name="ck_setup_state_status"),
    )

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True, autoincrement=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("'UNCONFIGURED'"))
    # Attempt binding (recorded at Start; NULL until an attempt is in flight).
    expected_org_login: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_permissions: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    expected_events: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    manifest_contract_digest: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversion_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )


class SetupNonce(Base):
    __tablename__ = "setup_nonce"

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    nonce_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
