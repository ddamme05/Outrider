"""Validate policy version numbers and protect stored severity policies.

Revision ID: 3d03bca7f2be
Revises: af138edd4b57
Create Date: 2026-05-19

Versions must have three numeric parts. Database triggers reject updates,
deletes and truncation so saved reviews keep their original policy. Publish a
changed policy as a new version instead of editing an existing row.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3d03bca7f2be"
down_revision: str | Sequence[str] | None = "af138edd4b57"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add semver CHECK + append-only trigger to severity_policies."""
    # Tight semver shape: no leading zeros (`01.0.0` rejected), no
    # pre-release/build suffix (`1.0.0-rc1` rejected). Matches the
    # ASCII-only Python guard at `outrider.policy.severity._SEMVER_RE`.
    # Per §0c sharp-edges audit finding #2: the two regexes must agree,
    # so a value accepted by Python matches the DB CHECK.
    #
    # `DROP CONSTRAINT IF EXISTS` first so a partial-failure rerun lands
    # cleanly (§0c data-integrity audit M-2: a CHECK creation that
    # succeeds followed by a trigger creation that fails would otherwise
    # leave the CHECK on a retry and `42710 duplicate_object`).
    op.execute(
        "ALTER TABLE severity_policies "
        "DROP CONSTRAINT IF EXISTS ck_severity_policies_version_semver;"
    )
    op.create_check_constraint(
        "ck_severity_policies_version_semver",
        "severity_policies",
        r"version ~ '^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$'",
    )

    # PostgreSQL CREATE TRIGGER does not support IF NOT EXISTS, so the
    # idempotent pattern is DROP TRIGGER IF EXISTS followed by CREATE
    # TRIGGER. Same shape as af138edd4b57_genesis.py:387-417 for
    # outrider_audit_append_only_guard.
    # Message shape mirrors the genesis `outrider_audit_append_only_guard`
    # at af138edd4b57_genesis.py:387-399 verbatim: `'append-only table %:
    # % not permitted'`. Sharp-edges audit finding #4 — divergent message
    # shapes between the two guards become a footgun for log-monitoring
    # and break-with-rename risk for tests that grep on literals. One
    # canonical message; both tests key on the same regex.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION outrider_severity_policies_append_only_guard()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION
                'append-only table %: % not permitted',
                TG_TABLE_NAME, TG_OP;
        END;
        $$;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_severity_policies_append_only ON severity_policies;
        CREATE TRIGGER trg_severity_policies_append_only
            BEFORE UPDATE OR DELETE ON severity_policies
            FOR EACH ROW
            EXECUTE FUNCTION outrider_severity_policies_append_only_guard();
        """
    )
    # Row-level triggers don't fire for TRUNCATE; a statement-level
    # BEFORE TRUNCATE trigger closes that gap. Without it, an operator
    # or faulty migration with table-owner privileges could `TRUNCATE
    # severity_policies` and erase the genesis-seeded '1.0.0' row,
    # breaking the lifespan fingerprint check at next startup. Audit
    # finding §0c-adv-M1.
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_severity_policies_no_truncate ON severity_policies;
        CREATE TRIGGER trg_severity_policies_no_truncate
            BEFORE TRUNCATE ON severity_policies
            FOR EACH STATEMENT
            EXECUTE FUNCTION outrider_severity_policies_append_only_guard();
        """
    )


def downgrade() -> None:
    """Remove protections in reverse order."""
    op.execute("DROP TRIGGER IF EXISTS trg_severity_policies_no_truncate ON severity_policies;")
    op.execute("DROP TRIGGER IF EXISTS trg_severity_policies_append_only ON severity_policies;")
    op.execute("DROP FUNCTION IF EXISTS outrider_severity_policies_append_only_guard();")
    # IF EXISTS so a partial-failure downgrade retry doesn't raise on an already-dropped
    # constraint — mirrors the upgrade()'s own DROP CONSTRAINT IF EXISTS idempotency.
    op.execute(
        "ALTER TABLE severity_policies "
        "DROP CONSTRAINT IF EXISTS ck_severity_policies_version_semver;"
    )
