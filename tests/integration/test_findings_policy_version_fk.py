"""findings.policy_version FK to severity_policies (RESTRICT).

Backs ``severity-policy-versioned-for-replay``. A finding's policy
version must remain reachable for the lifetime of the finding so the
classification can be replayed under the version that made it. The FK
is RESTRICT so the parent row cannot be deleted while findings reference
it; INSERT with a non-existent version raises FK violation.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

_INSTALLATION_ID = 12345


async def _seed_installation_and_review(engine: AsyncEngine) -> str:
    """Create installation + review, return the review's UUID."""
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
        review_result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                "  total_input_tokens, total_output_tokens, total_cost_usd, "
                "  wall_clock_seconds, retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha', 'running', 0, 0, 0, 0, 0, 0, 0, "
                "  NOW() + INTERVAL '180 days'"
                ") RETURNING id"
            ),
            {"id": _INSTALLATION_ID},
        )
        return review_result.scalar_one()


_INSERT_FINDING = text(
    "INSERT INTO findings ("
    "  review_id, installation_id, policy_version, finding_type, dimension, "
    "  severity, evidence_tier, file_path, line_start, line_end, title, "
    "  description, evidence, content_hash, retention_expires_at"
    ") VALUES ("
    "  :review_id, :installation_id, :policy_version, 'sql_injection', "
    "  'security', 'critical', 'observed', 'foo.py', 1, 1, 'title', "
    "  'description', 'evidence', 'hash', NOW() + INTERVAL '180 days'"
    ")"
)


async def test_findings_with_unknown_policy_version_raises(migrated_db: str) -> None:
    """INSERT with a policy_version that doesn't exist raises FK violation."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_installation_and_review(engine)

        with pytest.raises(IntegrityError, match="policy_version"):
            async with engine.begin() as conn:
                await conn.execute(
                    _INSERT_FINDING,
                    {
                        "review_id": review_id,
                        "installation_id": _INSTALLATION_ID,
                        "policy_version": "9.9.9-nonexistent",
                    },
                )
    finally:
        await engine.dispose()


async def test_findings_with_existing_policy_version_succeed(migrated_db: str) -> None:
    """Sanity: the seeded v1.0.0 row from the genesis migration is referenceable."""
    engine = create_async_engine(migrated_db)
    try:
        review_id = await _seed_installation_and_review(engine)

        async with engine.begin() as conn:
            await conn.execute(
                _INSERT_FINDING,
                {
                    "review_id": review_id,
                    "installation_id": _INSTALLATION_ID,
                    "policy_version": "1.0.0",
                },
            )

        async with engine.connect() as conn:
            count = await conn.execute(text("SELECT COUNT(*) FROM findings"))
            assert count.scalar_one() == 1
    finally:
        await engine.dispose()
