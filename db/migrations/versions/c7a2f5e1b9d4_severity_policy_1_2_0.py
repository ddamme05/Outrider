"""severity_policies 1.2.0 — contextual security FindingTypes.

Revision ID: c7a2f5e1b9d4
Revises: 5786686ef937
Create Date: 2026-06-20

Seeds policy version 1.2.0, which extends 1.1.0 with seven contextual
(JUDGED-pickable) security types per DECISIONS.md#053: weak_crypto (high),
weak_password_hash (critical), insecure_randomness (high), ssrf (high),
ssrf_metadata (critical), open_redirect (medium), open_redirect_authed
(high). Additive — no existing mapping changes, so 1.0.0 / 1.1.0 reviews
replay untouched per `severity-policy-versioned-for-replay`.

The full 22-entry mapping is inlined verbatim (NOT derived from the live
SEVERITY_POLICY) so the migration is a fixed point-in-time artifact: a
later policy change ships its own version row, never mutating this one.
The lifespan fingerprint (api/lifespan.py Step 1b) binds the live
SEVERITY_POLICY to this row at ACTIVE_POLICY_VERSION=1.2.0.

`severity_policies` is append-only (`trg_severity_policies_append_only`
blocks UPDATE/DELETE), so:
  - upgrade INSERTs with ON CONFLICT (version) DO NOTHING — idempotent
    across down/up cycles; DO NOTHING fires no row UPDATE, so the
    append-only trigger permits it;
  - downgrade is a no-op — the row cannot be deleted under the append-only
    trigger, and an unreferenced extra policy version is harmless once the
    ACTIVE_POLICY_VERSION code constant reverts to 1.1.0.

See:
  - DECISIONS.md#053 (the seven new FindingTypes + their severities)
  - DECISIONS.md#021 (FINDING_TYPE_TO_DIMENSION lockstep)
  - b1d7e4a92c63_severity_policy_1_1_0.py (the 1.1.0 seed this mirrors)
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
