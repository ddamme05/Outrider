"""reviews.updated_at BEFORE UPDATE trigger (stop it equalling created_at).

Revision ID: 54bb7ed5f51a
Revises: c22e2864d3d8
Create Date: 2026-06-06

`reviews.updated_at` carries `server_default=NOW()` but no `onupdate`, and the
status persister mutates rows via `update(Review).values(status=...)` WITHOUT
setting `updated_at` — so the column was frozen at its insert value (always
equal to `created_at`) for the life of every review. The dashboard exposes
`updated_at` (`api/dashboard/reviews.py`), so a stale value misrepresents the
last-activity time.

Fix at the database level so it holds for every writer (ORM `.values(...)`,
raw SQL, future code) rather than relying on each call site to set the column:
a `BEFORE UPDATE ... FOR EACH ROW` trigger stamps `NEW.updated_at := NOW()` on
every row update. NOW() (= transaction_timestamp) matches the `created_at`
server_default's clock, so the two columns stay on the same time source.

Trigger fires on UPDATE only — reads (the dashboard GET path,
`test_read_is_non_mutating`) and INSERTs (created_at == updated_at at birth)
are unaffected. Touches only `reviews`; no append-only table is involved.

CREATE OR REPLACE FUNCTION is idempotent; CREATE TRIGGER has no IF NOT EXISTS,
so the idempotent pattern is DROP TRIGGER IF EXISTS then CREATE TRIGGER — same
shape as c22e2864d3d8 / 3d03bca7f2be.
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "54bb7ed5f51a"
down_revision: str | Sequence[str] | None = "c22e2864d3d8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the BEFORE UPDATE trigger that stamps reviews.updated_at = NOW()."""
    op.execute(
        """
        CREATE OR REPLACE FUNCTION outrider_reviews_set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at := NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_reviews_set_updated_at ON reviews;
        CREATE TRIGGER trg_reviews_set_updated_at
            BEFORE UPDATE ON reviews
            FOR EACH ROW
            EXECUTE FUNCTION outrider_reviews_set_updated_at();
        """
    )


def downgrade() -> None:
    """Drop the trigger and its function (mirror of upgrade)."""
    op.execute("DROP TRIGGER IF EXISTS trg_reviews_set_updated_at ON reviews;")
    op.execute("DROP FUNCTION IF EXISTS outrider_reviews_set_updated_at();")
