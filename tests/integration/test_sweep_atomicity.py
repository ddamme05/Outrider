"""Multi-table purge is atomic — partial failures roll back ALL deletes.

The schema-layer spec requires the sweep job to wrap its multi-table
delete in a single transaction so that a failure during, say, the
findings delete rolls back any llm_call_content delete that already
happened. Without this, a partial sweep would silently lose content
without writing the per-table purge_audit row, breaking the forensic
trail.

Approach: monkeypatch ``_write_purge_audit`` to raise after the second
call. The deletes happen first (all three tables); writes to
purge_audit happen after. By raising mid-purge_audit-write, the test
verifies the whole transaction rolls back — both content rows AND any
purge_audit rows that landed before the failure.
"""

from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from outrider.sweep import purge_expired as sweep_module
from outrider.sweep.purge_expired import purge_expired

_INSTALLATION_ID = 12345


async def _seed_expired_content(engine: AsyncEngine) -> None:
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


async def test_failure_mid_sweep_rolls_back_all_deletes(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RuntimeError raised partway through purge_audit writes rolls back the whole transaction.

    Verifies the multi-table-purge atomicity property: if any step
    fails, no content is deleted and no purge_audit row lands.
    """
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_content(engine)

        # Patch _write_purge_audit to raise on the second call (after
        # the first purge_audit row landed but before the rest do).
        # That confirms even the rows that DID get written get rolled
        # back along with the content deletes.
        real_write = sweep_module._write_purge_audit
        call_count = {"n": 0}

        async def failing_write(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise RuntimeError("simulated mid-sweep failure")
            await real_write(*args, **kwargs)

        monkeypatch.setattr(sweep_module, "_write_purge_audit", failing_write)

        with pytest.raises(RuntimeError, match="simulated mid-sweep failure"):
            async with engine.begin() as conn:
                await purge_expired(conn, purge_role="test")

        # Transaction rolled back: all three content tables retain their
        # rows; no purge_audit rows survive.
        async with engine.connect() as conn:
            for table in ("llm_call_content", "findings", "reviews"):
                count = await conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
                )
                assert count.scalar_one() == 1, f"{table} row should be rolled back, not deleted"

            purge_count = await conn.execute(text("SELECT COUNT(*) FROM purge_audit"))
            assert purge_count.scalar_one() == 0, (
                "any purge_audit row that landed before the failure must roll back too"
            )
    finally:
        await engine.dispose()


async def test_resumed_sweep_after_failure_completes_cleanly(
    migrated_db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the failure condition is resolved, a re-run sweep purges everything."""
    engine = create_async_engine(migrated_db)
    try:
        await _seed_expired_content(engine)

        # Force a failure once...
        real_write = sweep_module._write_purge_audit
        call_count = {"n": 0}

        async def failing_write(*args: Any, **kwargs: Any) -> None:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient failure")
            await real_write(*args, **kwargs)

        monkeypatch.setattr(sweep_module, "_write_purge_audit", failing_write)
        with pytest.raises(RuntimeError):
            async with engine.begin() as conn:
                await purge_expired(conn, purge_role="test")

        # ...then restore the real function and re-run.
        monkeypatch.setattr(sweep_module, "_write_purge_audit", real_write)
        async with engine.begin() as conn:
            rows_per_table = await purge_expired(conn, purge_role="test")

        assert rows_per_table == {
            "llm_call_content": 1,
            "findings": 1,
            "reviews": 1,
        }

        async with engine.connect() as conn:
            purge_count = await conn.execute(text("SELECT COUNT(*) FROM purge_audit"))
            assert purge_count.scalar_one() == 3
    finally:
        await engine.dispose()
