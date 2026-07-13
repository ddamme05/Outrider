"""App-Manifest onboarding tables — github_app_credentials + setup_state + setup_nonce.

Revision ID: b3f9a1c72e04
Revises: a7c3f2b18d40
Create Date: 2026-07-13

DECISIONS.md#070 (self-service onboarding). Three tables for `database` credential mode:

  - github_app_credentials — encrypted (`pem`/`webhook_secret` ciphertext), versioned App creds;
    a unique partial index on `is_active` enforces at most one active row (first-write-wins).
  - setup_state           — the DB-enforced singleton (id=1) state machine + attempt binding.
  - setup_nonce           — hashed, expiring callback nonces (atomic delete-on-consume).

The singleton `setup_state` row is seeded here (id=1, UNCONFIGURED) so the state-machine
compare-and-swap always has a row to UPDATE. Harmless in `env` mode (unused).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "b3f9a1c72e04"
down_revision: str | Sequence[str] | None = "a7c3f2b18d40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "github_app_credentials",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("version", sa.BigInteger(), nullable=False),
        sa.Column("app_id", sa.BigInteger(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=True),
        sa.Column("pem_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("webhook_secret_ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # At most one active credential row (DECISIONS.md#070 first-write-wins).
    op.create_index(
        "uq_github_app_credentials_one_active",
        "github_app_credentials",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "setup_state",
        sa.Column("id", sa.SmallInteger(), autoincrement=False, nullable=False),
        sa.Column("status", sa.Text(), server_default=sa.text("'UNCONFIGURED'"), nullable=False),
        sa.Column("expected_org_login", sa.Text(), nullable=True),
        sa.Column("expected_permissions", postgresql.JSONB(), nullable=True),
        sa.Column("expected_events", postgresql.JSONB(), nullable=True),
        sa.Column("manifest_contract_digest", sa.Text(), nullable=True),
        sa.Column("conversion_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name="ck_setup_state_singleton"),
        sa.CheckConstraint(
            "status IN ('UNCONFIGURED','AWAITING_CALLBACK','CONVERTING','CONFIGURED','ORPHANED')",
            name="ck_setup_state_status",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    # Seed the singleton so state-machine CAS always has a row to UPDATE.
    op.execute("INSERT INTO setup_state (id, status) VALUES (1, 'UNCONFIGURED')")

    op.create_table(
        "setup_nonce",
        sa.Column("id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("nonce_hash", sa.Text(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nonce_hash", name="uq_setup_nonce_hash"),
    )


def downgrade() -> None:
    op.drop_table("setup_nonce")
    op.drop_table("setup_state")
    op.drop_index("uq_github_app_credentials_one_active", table_name="github_app_credentials")
    op.drop_table("github_app_credentials")
