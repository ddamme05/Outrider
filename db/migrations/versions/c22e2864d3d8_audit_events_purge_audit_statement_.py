"""audit_events + purge_audit statement-level append-only guard (BEFORE TRUNCATE).

Revision ID: c22e2864d3d8
Revises: d6b2f4a8c1e9
Create Date: 2026-06-05

Closes the TRUNCATE gap in the core append-only guarantee (trust-boundary #7).
Genesis (``af138edd4b57``) bound ``BEFORE UPDATE OR DELETE FOR EACH ROW`` triggers
on ``audit_events`` and ``purge_audit``, but ROW-level triggers do NOT fire for
TRUNCATE — so an operator or faulty migration with table-owner privileges could
erase the entire audit log (``audit_events``) or the purge forensic trail
(``purge_audit``), bypassing both row triggers entirely. ``severity_policies``
already closed this same gap for itself (``3d03bca7f2be``, audit finding §0c-adv-M1);
this migration extends the identical STATEMENT-level ``BEFORE TRUNCATE`` guard to the
two tables whose append-only property is the core architectural invariant.

Reuses the existing ``outrider_audit_append_only_guard()`` plpgsql function (genesis
``af138edd4b57``:389), which is table-agnostic — it raises ``'append-only table %: %
not permitted'`` from ``TG_TABLE_NAME`` / ``TG_OP``, so the same function backs the
new statement triggers with no new function needed.

CREATE TRIGGER has no ``IF NOT EXISTS``, so the idempotent pattern is
``DROP TRIGGER IF EXISTS`` then ``CREATE TRIGGER`` — same shape as genesis :401-417
and ``3d03bca7f2be``:118-126.

See:
  - docs/trust-boundaries.md §7 (audit append-only)
  - DECISIONS.md#012 + #014 (append-only contract)
  - db/migrations/versions/3d03bca7f2be_severity_policies_protections.py (mirrored pattern)
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c22e2864d3d8"
down_revision: str | Sequence[str] | None = "d6b2f4a8c1e9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add statement-level BEFORE TRUNCATE guards to audit_events + purge_audit.

    Both reuse the table-agnostic genesis guard function (keyed on TG_TABLE_NAME /
    TG_OP), so the message shape stays canonical with the row-level triggers.
    """
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_audit_events_no_truncate ON audit_events;
        CREATE TRIGGER trg_audit_events_no_truncate
            BEFORE TRUNCATE ON audit_events
            FOR EACH STATEMENT
            EXECUTE FUNCTION outrider_audit_append_only_guard();
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_purge_audit_no_truncate ON purge_audit;
        CREATE TRIGGER trg_purge_audit_no_truncate
            BEFORE TRUNCATE ON purge_audit
            FOR EACH STATEMENT
            EXECUTE FUNCTION outrider_audit_append_only_guard();
        """
    )


def downgrade() -> None:
    """Drop the statement-level TRUNCATE guards (mirror of upgrade)."""
    op.execute("DROP TRIGGER IF EXISTS trg_purge_audit_no_truncate ON purge_audit;")
    op.execute("DROP TRIGGER IF EXISTS trg_audit_events_no_truncate ON audit_events;")
