"""reviews.updated_at BEFORE UPDATE trigger stamps NOW() on every row update.

Backs migration 54bb7ed5f51a. Before the trigger, `updated_at` was frozen at
its insert value (== created_at) because the column has no `onupdate` and the
status persister updates via `.values(status=...)` without touching it. The
trigger fires on UPDATE only, so reads and inserts are unaffected.
"""

import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_INSTALLATION_ID = 424242


async def _seed_review(engine: AsyncEngine) -> str:
    """Insert one installation + one running review; return the review id."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"id": _INSTALLATION_ID},
        )
        review_id = (
            await conn.execute(
                text(
                    "INSERT INTO reviews "
                    "(installation_id, repo_id, pr_number, head_sha, status, "
                    " retention_expires_at) "
                    "VALUES (:id, 100, 1, 'sha1', 'running', "
                    " NOW() + INTERVAL '30 days') "
                    "RETURNING id"
                ),
                {"id": _INSTALLATION_ID},
            )
        ).scalar_one()
    return str(review_id)


async def test_updated_at_equals_created_at_on_insert(migrated_db: str) -> None:
    """The trigger fires on UPDATE only — at insert the two timestamps are equal
    (both come from the same transaction's server_default NOW())."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT created_at, updated_at FROM reviews WHERE id = :id"),
                    {"id": review_id},
                )
            ).one()
        assert row.created_at == row.updated_at
    finally:
        await engine.dispose()


async def test_update_advances_updated_at_but_not_created_at(migrated_db: str) -> None:
    """A row UPDATE stamps updated_at = NOW() and leaves created_at untouched."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_review(engine)
        async with engine.connect() as conn:
            before = (
                await conn.execute(
                    text("SELECT created_at, updated_at FROM reviews WHERE id = :id"),
                    {"id": review_id},
                )
            ).one()

        # NOW() is transaction_timestamp; sleep so the update transaction's
        # clock is strictly past the insert transaction's, making the
        # advance assertion deterministic rather than latency-dependent.
        await asyncio.sleep(0.02)

        async with engine.begin() as conn:
            await conn.execute(
                text("UPDATE reviews SET status = 'completed' WHERE id = :id"),
                {"id": review_id},
            )

        async with engine.connect() as conn:
            after = (
                await conn.execute(
                    text("SELECT created_at, updated_at FROM reviews WHERE id = :id"),
                    {"id": review_id},
                )
            ).one()

        assert after.created_at == before.created_at, "created_at must not change on update"
        assert after.updated_at > before.updated_at, "trigger must advance updated_at on update"
        assert after.updated_at > after.created_at, "updated_at must now exceed created_at"
    finally:
        await engine.dispose()
