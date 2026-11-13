"""Add severity policy version 1.2.0.

Revision ID: c7a2f5e1b9d4
Revises: 5786686ef937
Create Date: 2026-06-20

The fixed mapping includes the seven contextual security finding types from
DECISIONS.md#053. Earlier policy rows are unchanged. Downgrade leaves this
append-only row in place so historical reviews can still be reconstructed.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c7a2f5e1b9d4"
down_revision: str | Sequence[str] | None = "5786686ef937"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Seed the 1.2.0 severity policy row (full 22-entry mapping)."""
    op.execute(
        """
        INSERT INTO severity_policies (version, policy)
        VALUES (
            '1.2.0',
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
                "tls_verify_disabled": "high",
                "weak_crypto": "high",
                "weak_password_hash": "critical",
                "insecure_randomness": "high",
                "ssrf": "high",
                "ssrf_metadata": "critical",
                "open_redirect": "medium",
                "open_redirect_authed": "high"
            }'::jsonb
        )
        ON CONFLICT (version) DO NOTHING;
        """
    )


def downgrade() -> None:
    """No-op: `severity_policies` is append-only (the row cannot be deleted),
    and an unreferenced extra policy version is harmless once the
    ACTIVE_POLICY_VERSION code constant reverts to 1.1.0."""
