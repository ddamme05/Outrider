"""severity_policies protections — semver CHECK + append-only trigger.

Revision ID: 3d03bca7f2be
Revises: af138edd4b57
Create Date: 2026-05-19

Adds the protections genesis didn't have:

  1. CHECK constraint `ck_severity_policies_version_semver` enforcing
     tight bare semver shape on `severity_policies.version`:
     `^(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)$`. Exactly
     three dot-separated numeric components; the literal `0` is allowed
     but otherwise no leading zeros (`01.0.0` rejected); no pre-release
     or build suffix (`1.0.0-rc1` and `1.0.0+build` rejected). The
     existing `version='1.0.0'` row (seeded by af138edd4b57_genesis.py
     at lines 432-449) satisfies the CHECK, so the ALTER applies cleanly
     on a populated DB. Belt-and-suspenders to the Python-side semver
     assertion at `outrider.policy.severity::ACTIVE_POLICY_VERSION`: the
     two regexes agree, so a value the Python guard accepts also passes
     the DB CHECK, and the DB rejects malformed inserts even if a
     future migration sneaks one past the Python guard.

  2. Append-only trigger `trg_severity_policies_append_only` on
     `severity_policies`, mirroring the existing `audit_events`
     append-only discipline at af138edd4b57_genesis.py:389. Closes the
     once-at-lifespan vulnerability: the startup fingerprint check in
     `api/lifespan.py` verifies the DB row matches `dict(SEVERITY_POLICY)`
     ONCE at lifespan; a concurrent process or out-of-band UPDATE on
     `severity_policies` post-startup would tamper with the DB row while
     findings keep being written under the live mapping, causing replay
     divergence. The trigger makes that mutation impossible from any
     non-superuser connection.

Per `severity-policy-versioned-for-replay`: when SEVERITY_POLICY changes
in the future (e.g., upgrading MISSING_INPUT_VALIDATION from MEDIUM to
HIGH), the change ships as a new migration INSERTING the next semver row
(`1.0.1`, etc.) — never as an UPDATE to an existing version row. The
trigger enforces that discipline at the DB boundary.

See:
  - specs/2026-05-19-analyze-foundation.md §0c
  - docs/trust-boundaries.md §2 (severity-set-by-policy)
  - DECISIONS.md#014 (audit append-only contract; mirrored shape)
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
