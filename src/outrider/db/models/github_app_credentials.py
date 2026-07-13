# See DECISIONS.md#070 — database-mode GitHub App credential storage.
"""`GitHubAppCredential` — encrypted, versioned GitHub App credentials (`DECISIONS.md#070`).

Under `database` credential mode, the manifest-onboarded App `pem` + `webhook_secret` are stored
encrypted at rest (`github/credential_crypto.py`; ciphertext columns, never plaintext). Exactly one
row is active at a time — a unique partial index on `is_active` enforces first-write-wins so two
racing manifest conversions can never both activate. `app_id` + `slug` + `client_id` are non-secret
metadata. `client_secret` is deliberately NOT stored (OAuth user-tokens are a non-goal; #070).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import BigInteger, Boolean, Index, LargeBinary, Text, text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, Uuid

from outrider.db.models._base import Base

__all__ = ["GitHubAppCredential"]


class GitHubAppCredential(Base):
    __tablename__ = "github_app_credentials"
    __table_args__ = (
        # At most one active credential row (first-write-wins; DECISIONS.md#070): a partial unique
        # index on `is_active` where TRUE — active rows share value `true`, so they collide.
        Index(
            "uq_github_app_credentials_one_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    version: Mapped[int] = mapped_column(BigInteger, nullable=False)
    app_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    slug: Mapped[str] = mapped_column(Text, nullable=False)
    client_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    pem_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    webhook_secret_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
