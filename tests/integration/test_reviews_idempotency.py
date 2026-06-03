"""reviews UNIQUE(repo_id, pr_number, head_sha) blocks duplicate inserts.

Backs ``idempotency-via-db-unique-constraint`` and spec §6.5: the
webhook handler relies on IntegrityError from this constraint as the
dedup signal under near-simultaneous deliveries. The application-level
SELECT is a fast path; the constraint is the real gate.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_INSTALLATION_ID = 12345


async def _seed_installation(engine: AsyncEngine, installation_id: int) -> None:
    """Insert a minimal installation row (FK target for reviews)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, account_type, "
                " permissions_at_install) "
                "VALUES (:installation_id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"installation_id": installation_id},
        )


_INSERT_REVIEW = text(
    "INSERT INTO reviews ("
    "  installation_id, repo_id, pr_number, head_sha, status, "
    "  retention_expires_at"
    ") VALUES ("
    "  :installation_id, :repo_id, :pr_number, :head_sha, 'running', "
    "  NOW() + INTERVAL '180 days'"
    ")"
)

_REVIEW_PARAMS = {
    "installation_id": _INSTALLATION_ID,
    "repo_id": 100,
    "pr_number": 1,
    "head_sha": "abc123def456",
}


async def test_duplicate_reviews_natural_key_raises(migrated_db: str) -> None:
    """Second INSERT with the same (repo_id, pr_number, head_sha) raises."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine, _INSTALLATION_ID)

        async with engine.begin() as conn:
            await conn.execute(_INSERT_REVIEW, _REVIEW_PARAMS)

        with pytest.raises(IntegrityError, match="uq_review_natural_key"):
            async with engine.begin() as conn:
                await conn.execute(_INSERT_REVIEW, _REVIEW_PARAMS)
    finally:
        await engine.dispose()


async def test_reviews_with_distinct_natural_key_succeed(migrated_db: str) -> None:
    """Sanity: changing any one component lets the second INSERT succeed."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine, _INSTALLATION_ID)

        async with engine.begin() as conn:
            await conn.execute(_INSERT_REVIEW, _REVIEW_PARAMS)

        # Different head_sha — should succeed.
        async with engine.begin() as conn:
            await conn.execute(_INSERT_REVIEW, {**_REVIEW_PARAMS, "head_sha": "def456abc789"})

        async with engine.connect() as conn:
            count = await conn.execute(text("SELECT COUNT(*) FROM reviews"))
            assert count.scalar_one() == 2
    finally:
        await engine.dispose()
