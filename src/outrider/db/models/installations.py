# See DECISIONS.md#012-data-retention-ttls-configurable-purge-on-installationdeleted
"""INSTALLATIONS + INSTALLATION_REPOSITORIES.

INSTALLATIONS is the per-tenant root in V1's single-tenant deployment (one row per
GitHub App installation). `tombstoned_at` and `purge_after_at` together model the
two-state lifecycle from #012: tombstoned-in-grace (`tombstoned_at IS NOT NULL`,
`purge_after_at` future) and ready-for-hard-delete (`purge_after_at < NOW()`).

INSTALLATION_REPOSITORIES is a join row per repo covered by an install. The
`installation_id → installations.installation_id` FK uses `ON DELETE CASCADE`
because the join table has no retention TTL and no `purge_audit` semantics — its
rows are pure membership state and should follow the install when it hard-deletes.
This is the only join-style cascade in the schema; content tables use RESTRICT
to force the sweep job to delete content explicitly first.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    ColumnElement,
    ColumnExpressionArgument,
    ForeignKey,
    Index,
    LargeBinary,
    Text,
    UniqueConstraint,
    func,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import DateTime, Uuid

from outrider.db.models._base import Base


class Installation(Base):
    __tablename__ = "installations"
    __table_args__ = (
        # Sweep grace-window query: tombstoned_at IS NOT NULL AND purge_after_at < NOW().
        # installation_id is already covered by the column-level unique=True (which
        # produces an implicit UNIQUE INDEX) so it's not declared again here.
        Index("ix_installations_tombstoned_at", "tombstoned_at"),
    )

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    app_slug: Mapped[str] = mapped_column(Text, nullable=False)
    account_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    account_login: Mapped[str] = mapped_column(Text, nullable=False)
    account_type: Mapped[str] = mapped_column(Text, nullable=False)
    permissions_at_install: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("NOW()"), nullable=False
    )
    tombstoned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    purge_after_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Per-installation Slack config (dashboard-in-Slack, commit 6). All nullable —
    # Slack is opt-in per install; an install that never connects Slack leaves them
    # NULL. The bot token is stored ENCRYPTED at rest, never plaintext (see
    # DECISIONS.md#051-slack-bot-tokens-are-encrypted-at-rest; Fernet ciphertext via
    # notify/token_crypto.py); decryption is confined to the Slack notifier boundary.
    # The columns live here (not a side table) so #012's tombstone/purge carries the
    # encrypted token with the install on hard-delete.
    slack_team_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    slack_bot_token_ciphertext: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    slack_channel_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    slack_configured_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    slack_configured_by: Mapped[str | None] = mapped_column(Text, nullable=True)


class InstallationRepository(Base):
    __tablename__ = "installation_repositories"
    __table_args__ = (UniqueConstraint("installation_id", "repo_id", name="uq_installation_repo"),)

    id: Mapped[UUID] = mapped_column(
        Uuid, primary_key=True, server_default=text("gen_random_uuid()")
    )
    installation_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("installations.installation_id", ondelete="CASCADE"),
        nullable=False,
    )
    repo_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    repo_full_name: Mapped[str] = mapped_column(Text, nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


def active_repo_membership(
    installation_id: ColumnExpressionArgument[int] | int,
    repo_id: ColumnExpressionArgument[int] | int,
    repo: type[InstallationRepository] = InstallationRepository,
) -> ColumnElement[bool]:
    """Predicate for the *active* installation_repositories membership of a given
    `(installation_id, repo_id)`: matching coordinates, not soft-removed. The pair
    is unique (`uq_installation_repo`), so at most one row matches; an absent or
    removed membership yields no match — on an outer join the joined
    `repo_full_name` is NULL and the caller falls back to `repo {repo_id}`.

    Single-sources the condition shared by the reviews list/detail repo-name outer
    joins (operands are `Review` columns) and agent_view's scalar repo-name lookup
    (operands are literal ints); `==` is identical for a column or literal RHS.
    `repo` is a parameter, not a global, so an aliased entity can be passed if a
    future query joins the table more than once.
    """
    return (
        (repo.installation_id == installation_id)
        & (repo.repo_id == repo_id)
        & (repo.removed_at.is_(None))
    )


async def set_slack_config(
    session: AsyncSession,
    *,
    installation_id: int,
    team_id: str,
    bot_token_ciphertext: bytes,
    channel_id: str,
    configured_by: str,
) -> bool:
    """Persist per-install Slack config from a completed OAuth exchange (commit 6.3c).

    Updates the five `slack_*` columns on the matching ACTIVE (non-tombstoned)
    `installations` row and returns True; returns False if no active install matched
    — a tombstoned-in-grace or absent install can't be connected (the OAuth callback
    404s). The bot token is stored as Fernet ciphertext (see
    DECISIONS.md#051-slack-bot-tokens-are-encrypted-at-rest); this helper never sees
    or persists plaintext. `slack_configured_at` is set server-side (`func.now()`,
    timestamptz). The caller owns the transaction (wrap in `session.begin()` / commit).
    """
    result = await session.execute(
        update(Installation)
        .where(
            Installation.installation_id == installation_id,
            Installation.tombstoned_at.is_(None),
        )
        .values(
            slack_team_id=team_id,
            slack_bot_token_ciphertext=bot_token_ciphertext,
            slack_channel_id=channel_id,
            slack_configured_at=func.now(),
            slack_configured_by=configured_by,
        )
    )
    # `AsyncSession.execute` is typed `Result` (no `rowcount`); the runtime type for an
    # UPDATE is `CursorResult`, which carries it. `getattr` dodges the base-class typing
    # the same way the webhook/sweep UPDATE paths do.
    rowcount: int = getattr(result, "rowcount", 0) or 0
    return rowcount > 0
