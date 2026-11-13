"""Add severity policy version 1.1.0.

Revision ID: b1d7e4a92c63
Revises: f4c8a1d2b9e3
Create Date: 2026-06-14

The saved mapping includes command injection, unsafe deserialization and
disabled TLS verification (DECISIONS.md#048). It is written here as fixed data,
not copied from the current application policy. Earlier policies are unchanged;
downgrade leaves the append-only policy row in place (DECISIONS.md#021).
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b1d7e4a92c63"
down_revision: str | Sequence[str] | None = "f4c8a1d2b9e3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Seed the 1.1.0 severity policy row (full 15-entry mapping)."""
    op.execute(
        """
        INSERT INTO severity_policies (version, policy)
        VALUES (
            '1.1.0',
            '{
                "sql_injection": "critical",
                "auth_bypass": "critical",
                "hardcoded_secret": "high",
                "xss": "high",
                "path_traversal": "high",
                "missing_input_validation": "medium",
                "n_plus_one_query": "medium",
                "blocking_call_in_async": "medium",
                "missing_error_handling": "low",
                "missing_test": "low",
                "unused_import": "info",
                "deprecated_api": "info",
                "command_injection": "critical",
                "unsafe_deserialization": "high",
                "tls_verify_disabled": "high"
            }'::jsonb
        )
        ON CONFLICT (version) DO NOTHING;
        """
    )


def downgrade() -> None:
    """No-op: `severity_policies` is append-only (the row cannot be deleted),
    and an unreferenced extra policy version is harmless once the
    ACTIVE_POLICY_VERSION code constant reverts to 1.0.0."""
