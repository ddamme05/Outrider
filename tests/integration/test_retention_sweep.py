"""Retention sweep happy path: expired content rows are purged with audit trail.

Backs the per-table purge_audit contract from the schema-layer spec:
one purge_audit row per target table per sweep run. Verifies all four
retention content tables (reviews, findings, llm_call_content,
analyze_file_cache — the last per
specs/2026-06-11-file-hash-analyze-cache.md) are swept correctly and
audit_events is untouched (it's append-only forever per #014; the
trigger would block any DELETE attempt and fail the test loud).
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.sweep.purge_expired import purge_expired

_INSTALLATION_ID = 12345


async def _seed_expired_content(engine: AsyncEngine) -> None:
    """Populate all four retention tables with rows that are already expired."""
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
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'completed', "
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
        await conn.execute(
            text(
                "INSERT INTO analyze_file_cache ("
                "  cache_key, installation_id, repo_id, source_review_id, "
                "  file_path, payload, model, prompt_template_version, "
                "  trivial_filter_version, query_registry_digest, "
                "  active_policy_version, analyze_parser_version, prompt_hash, "
                "  retention_expires_at"
                ") VALUES ("
                "  :key, :id, 100, :review_id, 'foo.py', '{}'::jsonb, 'm', "
                "  'pv', 'fv', 'qd', 'apv', 'parser', 'ph', "
                "  NOW() - INTERVAL '1 day'"
                ")"
            ),
            {"key": "a" * 64, "id": _INSTALLATION_ID, "review_id": review_id},
        )


async def test_purge_expired_deletes_all_four_tables(migrated_db: str) -> None:
    """All four retention tables are swept, audit_events stays intact."""
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
            "analyze_file_cache": 1,
            "llm_call_content": 1,
            "findings": 1,
            "reviews": 1,
        }

        async with engine.connect() as conn:
            for table in ("analyze_file_cache", "llm_call_content", "findings", "reviews"):
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
            assert len(rows) == 4
            tables = {row[0] for row in rows}
            assert tables == {"analyze_file_cache", "llm_call_content", "findings", "reviews"}
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
                    "  retention_expires_at"
                    ") VALUES ("
                    "  :id, 100, 1, 'sha1', 'running', "
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
    """Reviews in 'running', 'awaiting_approval', or 'awaiting_approval_expired'
    MUST survive the time-based sweep even when retention_expires_at has passed.

    Backs the status-filter introduced in commit da994e7 (data-integrity
    audit finding): a HITL-paused review left past TTL would otherwise
    be hard-deleted, breaking HITL resume. A 'running' review past TTL
    would strand the LangGraph checkpoint. 'awaiting_approval_expired' is
    the post-timeout REMEDIATION state — still decidable (spec.md: /decide
    accepts expired reviews and publishes immediately); purging it would
    close that human-decision path (whole-repo review fix).

    Seeds three expired active reviews ('running' / 'awaiting_approval' /
    'awaiting_approval_expired') plus one expired 'completed' review as a
    positive control (must be purged so the test fails if the filter
    accidentally protects all reviews).

    Expected: 1 review purged ('completed'); 3 reviews survive; 1 purge_audit row.
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
                (4, "awaiting_approval_expired"),
            ]:
                await conn.execute(
                    text(
                        "INSERT INTO reviews ("
                        "  installation_id, repo_id, pr_number, head_sha, status, "
                        "  retention_expires_at"
                        ") VALUES ("
                        "  :id, 100, :pr_number, :head_sha, :status, "
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

        # Only the 'completed' review purged; the three active states survive.
        assert rows_per_table == {"reviews": 1}, (
            f"Expected exactly 1 review purged (the 'completed' one); "
            f"got {rows_per_table}. The status filter is broken: either "
            f"protecting too much (active filter wider than the three resumable "
            f"states) or too little (an active/resumable review being purged)."
        )

        async with engine.connect() as conn:
            surviving = await conn.execute(text("SELECT status FROM reviews ORDER BY pr_number"))
            statuses = [row[0] for row in surviving.fetchall()]
            assert statuses == [
                "running",
                "awaiting_approval",
                "awaiting_approval_expired",
            ], f"Expected all three resumable states to survive; got {statuses}."

            # purge_audit side effect: one row for the `reviews` table per
            # sweep run that purged any rows. Pins the audit-write contract
            # the per-table loop documents — without this assertion a
            # regression that broke `_write_purge_audit` would still pass.
            audit_count = await conn.scalar(
                text(
                    "SELECT COUNT(*) FROM purge_audit "
                    "WHERE target_table = 'reviews' AND purge_role = 'test'"
                )
            )
            assert audit_count == 1, f"Expected one purge_audit row for reviews; got {audit_count}."
    finally:
        await engine.dispose()
