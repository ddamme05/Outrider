"""Check that every audit row requires a review ID without a foreign key.

The ``every-audit-event-has-review-id`` rule requires NOT NULL. A foreign key to
reviews would prevent metadata-only replay after review content is removed;
see docs/schema.md and DECISIONS.md#014."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine


async def test_audit_review_id_is_not_null(migrated_db: str) -> None:
    """INSERT without review_id raises NOT NULL violation."""
    engine = create_async_engine(migrated_db)
    try:
        with pytest.raises(IntegrityError, match="review_id"):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO audit_events (event_type, payload) "
                        "VALUES ('TestEvent', '{}'::jsonb)"
                    )
                )
    finally:
        await engine.dispose()


async def test_audit_events_has_no_foreign_key_constraints(migrated_db: str) -> None:
    """audit_events declares zero foreign keys (review_id is logical-only)."""
    engine = create_async_engine(migrated_db)
    try:
        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT conname FROM pg_constraint "
                    "WHERE conrelid = 'audit_events'::regclass AND contype = 'f'"
                )
            )
            fks = [row[0] for row in result]
            assert fks == [], (
                f"audit_events should carry zero FK constraints; found: {fks}. "
                "review_id must remain a logical reference per docs/schema.md"
            )
    finally:
        await engine.dispose()
