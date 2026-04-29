"""Installation lifecycle hard-delete: full content purge + cascade.

Per ``DECISIONS.md#012`` end-to-end. After the grace window expires,
purge_installation deletes all content for the installation in strict
order, writes per-table purge_audit rows, then hard-deletes the
installations row. INSTALLATION_REPOSITORIES cascades automatically
via the ON DELETE CASCADE FK declared in migration 0001. PURGE_AUDIT
rows survive the installation hard-delete (loose-reference no-FK).
audit_events stays untouched throughout (append-only forever).
"""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.sweep.purge_expired import purge_installation

_INSTALLATION_ID = 12345


async def _seed_full_installation_state(engine: AsyncEngine) -> None:
    """Set up an installation with content + repositories + audit history."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install, "
                " tombstoned_at, purge_after_at) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb, "
                "        NOW() - INTERVAL '40 days', NOW() - INTERVAL '10 days')"
            ),
            {"id": _INSTALLATION_ID},
        )
        await conn.execute(
            text(
                "INSERT INTO installation_repositories "
                "(installation_id, repo_id, repo_full_name, added_at) "
                "VALUES (:id, 100, 'octocat/repo-1', NOW())"
            ),
            {"id": _INSTALLATION_ID},
        )
        await conn.execute(
            text(
                "INSERT INTO installation_repositories "
                "(installation_id, repo_id, repo_full_name, added_at) "
                "VALUES (:id, 200, 'octocat/repo-2', NOW())"
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
                "  NOW() + INTERVAL '180 days'"
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
                "  'e', 'h', NOW() + INTERVAL '180 days'"
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
                ") VALUES (:event_id, :id, 'p', 'c', NOW() + INTERVAL '90 days')"
            ),
            {"event_id": event_id, "id": _INSTALLATION_ID},
        )


async def test_purge_installation_full_lifecycle(migrated_db: str) -> None:
    """Hard-delete an installation; content + repos cascade; audit + purge_audit survive."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_full_installation_state(engine)

        async with engine.begin() as conn:
            rows_per_table = await purge_installation(
                conn, _INSTALLATION_ID, purge_role="install-purge"
            )

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
                assert count.scalar_one() == 0, (
                    f"{table} content for installation {_INSTALLATION_ID} should be gone"
                )

            install_count = await conn.execute(
                text("SELECT COUNT(*) FROM installations WHERE installation_id = :id"),
                {"id": _INSTALLATION_ID},
            )
            assert install_count.scalar_one() == 0, "installations row should be hard-deleted"

            repo_count = await conn.execute(
                text("SELECT COUNT(*) FROM installation_repositories WHERE installation_id = :id"),
                {"id": _INSTALLATION_ID},
            )
            assert repo_count.scalar_one() == 0, (
                "installation_repositories should cascade-delete with the parent"
            )

            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            assert audit_count.scalar_one() == 1, (
                "audit_events is append-only forever; install purge must not touch it"
            )

            purge_rows = await conn.execute(
                text(
                    "SELECT installation_id, target_table, rows_affected, purge_role "
                    "FROM purge_audit ORDER BY target_table"
                )
            )
            rows = list(purge_rows)
            assert len(rows) == 3
            for installation_id, _table, rows_affected, purge_role in rows:
                assert installation_id == _INSTALLATION_ID, (
                    "purge_audit row should record the scoped installation_id, "
                    "not the global-sweep sentinel"
                )
                assert rows_affected == 1
                assert purge_role == "install-purge"
    finally:
        await engine.dispose()
