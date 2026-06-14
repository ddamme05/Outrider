"""Serve-mode eval scenario (FUP-176): re-review a file across TWO PRs; the second
PR's analyze serves the finding from the first PR's cache row and flows it to publish.

Two `CacheMode.SERVE` drives of TWO DISTINCT fixtures (two PRs that touch one file
IDENTICALLY — NOT one fixture twice; `reviews` is UNIQUE on (repo_id, pr_number,
head_sha), so the same PR head cannot be reviewed twice) against ONE eval DB:

- **Drive 1** (PR #22, cache empty): analyze MISSES → calls the model → writes the
  `analyze_file_cache` row (the write-on-miss happens in any cache mode).
- **Drive 2** (PR #23, same file, cache populated by drive 1, a FRESH `review_id`):
  analyze HITS → reconstructs + re-stamps the cached finding (zero analyze LLM call)
  → the served finding flows through synthesize → HITL (LOW finding, no gate) → publish.

This drives the serve path end-to-end through the real seven-node graph — the
coverage FUP-177's unit tests (re-mint determinism, degrade guard) could not give:
the served finding crossing every downstream deterministic gate, AND the full
audit stream replaying equivalent. The fixture is the proven LOW-severity
`missing_error_handling` PR (single file, non-gating, reaches publish).

`run_review_persisting` runs against the caller-owned `eval_db` so the audit
stream + the cache row survive BOTH drives (and the post-run replay). Asserting
drive 2 serves and replays is exactly the readiness signal the `cache_mode=serve`
flip waits on (the production flip itself is telemetry-gated — out of scope here).
"""

from __future__ import annotations

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
) -> None:
    """Re-reviewing a shared file across two PRs under SERVE: drive 2 (PR #23)
    serves the analyze finding from drive 1's (PR #22) cache row (zero analyze LLM
    call), flows it to the report through publish, and replays equivalent — alongside
    drive 1, the cold miss that seeded the cache."""
    store = AnalyzeCacheStore(session_factory=eval_db_session_factory)

    # Eval reviews use the cache (scoped to is_eval rows by the lookup predicate,
    # DECISIONS.md#046); eval_db is an isolated ephemeral DB so the only rows are
    # this scenario's. Drive 1 (PR #22): cache empty → analyze runs the model and
    # writes the row.
    first = await run_review_persisting(
        _FIXTURE_COLD,
        db_url=eval_db,
        analyze_cache_store=store,
        cache_mode=CacheMode.SERVE,
    )
    # Drive 2 (PR #23, same file, fresh review_id): analyze SERVES from drive 1's
    # row (the lookup's self-hit exclusion keys on the CURRENT review, so the
    # cross-review hit is not suppressed).
    second = await run_review_persisting(
        _FIXTURE_RESUBMIT,
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
