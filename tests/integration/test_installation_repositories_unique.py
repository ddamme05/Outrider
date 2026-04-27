"""installation_repositories UNIQUE(installation_id, repo_id) blocks duplicates.

Webhook redelivery on `installation_repositories.added` would otherwise
create duplicate membership rows; the UNIQUE constraint catches the
re-add at the DB layer. On a real re-add of a previously-removed repo,
the application updates the existing row (sets added_at = NOW(), clears
removed_at) rather than inserting a new row — that update path is
application-layer logic, not tested here. This test verifies the DB
gate that backs it.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_INSTALLATION_ID = 12345
_REPO_ID = 100


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, account_type, "
                " permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"id": _INSTALLATION_ID},
        )


_INSERT_MEMBERSHIP = text(
    "INSERT INTO installation_repositories "
    "(installation_id, repo_id, repo_full_name, added_at) "
    "VALUES (:installation_id, :repo_id, 'octocat/test-repo', NOW())"
)


async def test_duplicate_membership_raises(migrated_db: str) -> None:
    """Second INSERT with same (installation_id, repo_id) raises."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine)
        async with engine.begin() as conn:
            await conn.execute(
                _INSERT_MEMBERSHIP,
                {"installation_id": _INSTALLATION_ID, "repo_id": _REPO_ID},
            )

        with pytest.raises(IntegrityError, match="uq_installation_repo"):
            async with engine.begin() as conn:
                await conn.execute(
                    _INSERT_MEMBERSHIP,
                    {"installation_id": _INSTALLATION_ID, "repo_id": _REPO_ID},
                )
    finally:
        await engine.dispose()


async def test_distinct_repo_id_succeeds(migrated_db: str) -> None:
    """Sanity: two memberships under same install but different repos succeed."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_installation(engine)
        async with engine.begin() as conn:
            await conn.execute(
                _INSERT_MEMBERSHIP,
                {"installation_id": _INSTALLATION_ID, "repo_id": _REPO_ID},
            )
            await conn.execute(
                _INSERT_MEMBERSHIP,
                {"installation_id": _INSTALLATION_ID, "repo_id": _REPO_ID + 1},
            )

        async with engine.connect() as conn:
            count = await conn.execute(text("SELECT COUNT(*) FROM installation_repositories"))
            assert count.scalar_one() == 2
    finally:
        await engine.dispose()
