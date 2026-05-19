"""Retention sweep happy path: expired content rows are purged with audit trail.

Backs the per-table purge_audit contract from the schema-layer spec:
one purge_audit row per target table per sweep run. Verifies all three
retention content tables (reviews, findings, llm_call_content) are
swept correctly and audit_events is untouched (it's append-only forever
per #014; the trigger would block any DELETE attempt and fail the test
loud).
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.sweep.purge_expired import purge_expired

_INSTALLATION_ID = 12345


async def _seed_expired_content(engine: AsyncEngine) -> None:
    """Populate all three retention tables with rows that are already expired."""
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
        review_result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                "  total_input_tokens, total_output_tokens, total_cost_usd, "
                "  wall_clock_seconds, retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'completed', 0, 0, 0, 0, 0, 0, 0, "
                "  NOW() - INTERVAL '1 day'"
                ") RETURNING id"
            ),
            {"id": _INSTALLATION_ID},
        )
        review_id = review_result.scalar_one()
        await conn.execute(
            text(
                "INSERT INTO findings ("
                "  review_id, installation_id, policy_version, finding_type, "
                "  dimension, severity, evidence_tier, file_path, line_start, "
                "  line_end, title, description, evidence, content_hash, "
                "  retention_expires_at"
                ") VALUES ("
                "  :review_id, :installation_id, '1.0.0', 'sql_injection', "
                "  'security', 'critical', 'observed', 'foo.py', 1, 1, 't', 'd', "
                "  'e', 'h', NOW() - INTERVAL '1 day'"
                ")"
            ),
            {"review_id": review_id, "installation_id": _INSTALLATION_ID},
        )
        audit_result = await conn.execute(
            text(
                "INSERT INTO audit_events (review_id, event_type, payload) "
                "VALUES (:review_id, 'LLMCallEvent', '{}'::jsonb) "
                "RETURNING event_id"
            ),
            {"review_id": review_id},
        )
        event_id = audit_result.scalar_one()
        await conn.execute(
            text(
                "INSERT INTO llm_call_content ("
                "  event_id, installation_id, prompt, completion, "
                "  retention_expires_at"
                ") VALUES (:event_id, :id, 'p', 'c', NOW() - INTERVAL '1 day')"
            ),
            {"event_id": event_id, "id": _INSTALLATION_ID},
        )


async def test_purge_expired_deletes_all_three_tables(migrated_db: str) -> None:
    """All three retention tables are swept, audit_events stays intact."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_content(engine)

        # Capture audit_events count BEFORE the sweep. The append-only
        # contract requires unchanged-after-sweep, so a pre/post compare
        # catches both deletes (the obvious bug) and accidental inserts
        # (the subtle bug a hardcoded `== 1` would miss).
        async with engine.connect() as conn:
            audit_count_before = (
                await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            ).scalar_one()

        async with engine.begin() as conn:
            rows_per_table = await purge_expired(conn, purge_role="test")

        assert rows_per_table == {
            "llm_call_content": 1,
            "findings": 1,
            "reviews": 1,
        }

        async with engine.connect() as conn:
            for table in ("llm_call_content", "findings", "reviews"):
                count = await conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                )
                assert count.scalar_one() == 0, f"{table} should be empty post-sweep"

            audit_count_after = (
                await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            ).scalar_one()
            assert audit_count_after == audit_count_before, (
                "audit_events is append-only; sweep must not touch it "
                f"(before={audit_count_before}, after={audit_count_after})"
            )

            purge_rows = await conn.execute(
                text(
                    "SELECT target_table, rows_affected, purge_role "
                    "FROM purge_audit ORDER BY target_table"
                )
            )
            rows = list(purge_rows)
            assert len(rows) == 3
            tables = {row[0] for row in rows}
            assert tables == {"llm_call_content", "findings", "reviews"}
            for _table, rows_affected, purge_role in rows:
                assert rows_affected == 1
                assert purge_role == "test"
    finally:
        await engine.dispose()


async def test_purge_expired_skips_unexpired_rows(migrated_db: str) -> None:
    """Rows with retention_expires_at in the future stay; no purge_audit row written."""
    engine = create_async_engine(migrated_db)
    try:
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
            await conn.execute(
                text(
                    "INSERT INTO reviews ("
                    "  installation_id, repo_id, pr_number, head_sha, status, "
                    "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                    "  total_input_tokens, total_output_tokens, total_cost_usd, "
                    "  wall_clock_seconds, retention_expires_at"
                    ") VALUES ("
                    "  :id, 100, 1, 'sha1', 'running', 0, 0, 0, 0, 0, 0, 0, "
                    "  NOW() + INTERVAL '90 days'"
                    ")"
                ),
                {"id": _INSTALLATION_ID},
            )

        async with engine.begin() as conn:
            rows_per_table = await purge_expired(conn, purge_role="test")

        assert rows_per_table == {}

        async with engine.connect() as conn:
            review_count = await conn.execute(text("SELECT COUNT(*) FROM reviews"))
            assert review_count.scalar_one() == 1
            purge_count = await conn.execute(text("SELECT COUNT(*) FROM purge_audit"))
            assert purge_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_purge_expired_preserves_active_reviews_past_ttl(migrated_db: str) -> None:
    """Reviews in 'running' or 'awaiting_approval' MUST survive the
    time-based sweep even when retention_expires_at has passed.

    Backs the status-filter introduced in commit da994e7 (data-integrity
    audit finding): a HITL-paused review left past TTL would otherwise
    be hard-deleted, breaking HITL resume. A 'running' review past TTL
    would strand the LangGraph checkpoint.

    Seeds two expired reviews:
      - status='running' with retention_expires_at in the past
      - status='awaiting_approval' with retention_expires_at in the past
    Plus one expired 'completed' review as a positive control (must be
    purged so the test fails if the filter accidentally protects all
    reviews).

    Expected: 1 review purged ('completed'); 2 reviews survive
    ('running' + 'awaiting_approval'); 1 purge_audit row.
    """
    engine = create_async_engine(migrated_db)
    try:
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
            # Three reviews, all past-TTL, distinct statuses.
            for pr_number, status in [
                (1, "running"),
                (2, "awaiting_approval"),
                (3, "completed"),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO reviews ("
                        "  installation_id, repo_id, pr_number, head_sha, status, "
                        "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                        "  total_input_tokens, total_output_tokens, total_cost_usd, "
                        "  wall_clock_seconds, retention_expires_at"
                        ") VALUES ("
                        "  :id, 100, :pr_number, :head_sha, :status, 0, 0, 0, 0, 0, 0, 0, "
                        "  NOW() - INTERVAL '1 day'"
                        ")"
                    ),
                    {
                        "id": _INSTALLATION_ID,
                        "pr_number": pr_number,
                        "head_sha": f"sha{pr_number}",
                        "status": status,
                    },
                )

        async with engine.begin() as conn:
            rows_per_table = await purge_expired(conn, purge_role="test")

        # Only the 'completed' review purged; 'running' + 'awaiting_approval' survive.
        assert rows_per_table == {"reviews": 1}, (
            f"Expected exactly 1 review purged (the 'completed' one); "
            f"got {rows_per_table}. The status filter is broken: either "
            f"protecting too much (active filter wider than running + "
            f"awaiting_approval) or protecting too little (active reviews "
            f"being purged)."
        )

        async with engine.connect() as conn:
            surviving = await conn.execute(text("SELECT status FROM reviews ORDER BY pr_number"))
            statuses = [row[0] for row in surviving.fetchall()]
            assert statuses == ["running", "awaiting_approval"], (
                f"Expected running+awaiting_approval to survive; got {statuses}."
            )
    finally:
        await engine.dispose()
