"""Check that a second review can reuse a finding cached by the first review.

Run two distinct PRs with the same changed file against one test database.
The first analysis calls the scripted model and writes the cache; the second
serves the finding without an analysis call. Follow it through publication
and replay the saved history. A derived BASE makes the change body-only;
shared HEADs and scripted findings stay intact. This tests reuse, not a
production cache-mode flip."""

from __future__ import annotations

import json
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import text

from outrider.agent import run_review_persisting
from outrider.agent.nodes.cache_config import CacheMode
from outrider.audit.replay import AuditReplayer
from outrider.cache import AnalyzeCacheStore

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# Two PRs touching profile/client.py IDENTICALLY: the analyze cache key folds
# the prompt digest + (installation_id, repo_id) and version components — NOT
# is_eval and NOT PR identity — so the second PR's analyze computes the same key
# and hits the first PR's cache row. Eval reviews use the cache like production,
# isolated by the lookup's is_eval read-isolation predicate (DECISIONS.md#046).
# Two fixtures (not two drives of one) because `reviews` is UNIQUE on
# (repo_id, pr_number, head_sha) — the same PR head cannot be reviewed twice,
# which is exactly why the cache targets DISTINCT PRs that share a file.
_FIXTURE_COLD = "tests/eval/fixtures/mock_github/missing_error_handling.json"
_FIXTURE_RESUBMIT = "tests/eval/fixtures/mock_github/missing_error_handling_resubmitted.json"


def _write_cache_body_change_fixture(source_path: str, target_path: Path) -> Path:
    """Derive a body-only change while preserving the original HEAD and findings."""
    fixture = json.loads(Path(source_path).read_text(encoding="utf-8"))
    (file,) = fixture["files"]
    file["content_base"] = "import requests\n\n\ndef fetch_profile(user_id):\n    return {}\n"
    diff_lines = tuple(
        unified_diff(
            file["content_base"].splitlines(keepends=True),
            file["content_head"].splitlines(keepends=True),
            fromfile=f"a/{file['path']}",
            tofile=f"b/{file['path']}",
        )
    )
    file["patch"] = "".join(diff_lines)
    fixture["total_additions"] = file["additions"] = sum(
        line.startswith("+") for line in diff_lines[2:]
    )
    fixture["total_deletions"] = file["deletions"] = sum(
        line.startswith("-") for line in diff_lines[2:]
    )
    target_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    return target_path


async def _count_events(
    session_factory: async_sessionmaker[AsyncSession],
    review_id: UUID,
    *,
    event_type: str,
    node_id: str | None = None,
) -> int:
    sql = "SELECT COUNT(*) FROM audit_events WHERE review_id = :rid AND event_type = :etype"
    params: dict[str, object] = {"rid": review_id, "etype": event_type}
    if node_id is not None:
        sql += " AND payload->>'node_id' = :node"
        params["node"] = node_id
    async with session_factory() as session:
        result = await session.execute(text(sql), params)
        return result.scalar_one()


async def test_serve_cache_hit_skips_analyze_and_replays_equivalent(
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """Re-reviewing a shared file across two PRs under SERVE: drive 2 (PR #23)
    serves the analyze finding from drive 1's (PR #22) cache row (zero analyze LLM
    call), flows it to the report through publish, and replays equivalent — alongside
    drive 1, the cold miss that seeded the cache."""
    store = AnalyzeCacheStore(session_factory=eval_db_session_factory)
    cold_fixture = _write_cache_body_change_fixture(_FIXTURE_COLD, tmp_path / "cold.json")
    resubmitted_fixture = _write_cache_body_change_fixture(
        _FIXTURE_RESUBMIT, tmp_path / "resubmitted.json"
    )

    # Eval reviews use the cache (scoped to is_eval rows by the lookup predicate,
    # DECISIONS.md#046); eval_db is an isolated ephemeral DB so the only rows are
    # this scenario's. Drive 1 (PR #22): cache empty → analyze runs the model and
    # writes the row.
    first = await run_review_persisting(
        cold_fixture,
        db_url=eval_db,
        analyze_cache_store=store,
        cache_mode=CacheMode.SERVE,
    )
    # Drive 2 (PR #23, same file, fresh review_id): analyze SERVES from drive 1's
    # row (the lookup's self-hit exclusion keys on the CURRENT review, so the
    # cross-review hit is not suppressed).
    second = await run_review_persisting(
        resubmitted_fixture,
        db_url=eval_db,
        analyze_cache_store=store,
        cache_mode=CacheMode.SERVE,
    )

    assert first.review_id != second.review_id

    # Drive 1 was a cold miss: the model ran for analyze; nothing was served.
    assert (
        await _count_events(
            eval_db_session_factory, first.review_id, event_type="llm_call", node_id="analyze"
        )
        >= 1
    )
    assert (
        await _count_events(eval_db_session_factory, first.review_id, event_type="cache_serve") == 0
    )

    # Drive 2 served: ZERO analyze LLM calls, and a CacheServeEvent recorded.
    assert (
        await _count_events(
            eval_db_session_factory, second.review_id, event_type="llm_call", node_id="analyze"
        )
        == 0
    )
    assert (
        await _count_events(eval_db_session_factory, second.review_id, event_type="cache_serve")
        >= 1
    )

    # The served finding flowed through every downstream gate to publish: the LOW
    # finding never trips HITL, and it reaches the report.
    assert second.hitl_gated is False
    assert len(first.findings) == 1 and len(second.findings) == 1

    # It is the cached finding RE-STAMPED: assert the full CONTENT matches the cold
    # drive's finding, not just finding_type — a one-element-set equality would pass
    # even if the re-stamp corrupted title/evidence/span/severity. review_id +
    # finding_id legitimately differ (re-minted per review); content must not.
    cold, served = first.findings[0], second.findings[0]
    for field in (
        "finding_type",
        "severity",
        "dimension",
        "evidence_tier",
        "file_path",
        "line_start",
        "line_end",
        "title",
        "description",
        "evidence",
        "content_hash",
    ):
        assert getattr(served, field) == getattr(cold, field), field

    # Both reviews replay equivalent from the persisted audit stream — the serve
    # path is reconstructable end-to-end, not just unit-correct.
    replayer = AuditReplayer(session_factory=eval_db_session_factory)
    await replayer.assert_replay_equivalent(first.review_id)
    await replayer.assert_replay_equivalent(second.review_id)
