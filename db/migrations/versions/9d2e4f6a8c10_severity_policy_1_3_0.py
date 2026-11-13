"""Add severity policy version 1.3.0 with an actionable correctness category.

Revision ID: 9d2e4f6a8c10
Revises: b3f9a1c72e04
Create Date: 2026-11-02

Append a new version under DECISIONS.md#021. Historical mappings stay intact;
downgrade leaves the append-only row available for historical review loading.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "9d2e4f6a8c10"
down_revision: str | Sequence[str] | None = "b3f9a1c72e04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Insert the complete 23-entry mapping without changing prior versions."""
    op.execute(
        """
        INSERT INTO severity_policies (version, policy)
        VALUES (
            '1.3.0',
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
                "open_redirect_authed": "high",
                "correctness_bug": "medium"
            }'::jsonb
        )
        ON CONFLICT (version) DO NOTHING;
        """
    )


def downgrade() -> None:
    """Keep this immutable row so historical reviews retain their policy."""
