"""Parallel fan-out concurrency scenario (increment 6 of
specs/2026-07-05-parallel-analyze.md) — the driver's concurrency-safety
proof, opt-in per fixture.

Three DEEP files run through the REAL graph with `analyze_max_concurrency=4`
and file-keyed scripted responses (`analyze_responses_by_path`, served via
the worker's `LLMRequest.phase_key`) — correct under ANY worker completion
order, which is the property the index-keyed `llm_responses["analyze"]`
list cannot give. Covers: per-file finding attribution under concurrency,
the HITL gate on a concurrent round, interrupt → restart → resume with a
parallel-produced round (reducer idempotence across process boundaries),
replay-equivalence over the persisted interleaved keyed stream, concurrent
same-review cache writes on distinct file keys, and the fail-loud guard on
the unsafe index-keyed + concurrency combination.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from outrider.agent import run_review_persisting, run_review_with_resume
from outrider.agent.eval_driver import EvalDriverError
from outrider.audit.replay import AuditReplayer
from outrider.cache import AnalyzeCacheStore
from outrider.policy.severity import FindingType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_FIXTURE = "tests/eval/fixtures/mock_github/parallel_fanout.json"

# Which file each scripted finding belongs to — the attribution ground truth.
_EXPECTED_FILE_BY_TYPE = {
    FindingType.SQL_INJECTION: "app/alpha.py",
    FindingType.MISSING_INPUT_VALIDATION: "app/beta.py",
    FindingType.MISSING_ERROR_HANDLING: "app/gamma.py",
}

pytestmark = pytest.mark.asyncio


async def test_concurrent_workers_attribute_findings_to_their_own_files(
    eval_db: str,
) -> None:
    """Four-permit concurrency over three files: every scripted finding must
    land on ITS OWN file (cross-attribution is the failure mode file-keyed
    scripting exists to prevent), the CRITICAL finding still gates at HITL,
    and all three files are examined in the single pass-0 round."""
    result = await run_review_persisting(_FIXTURE, db_url=eval_db, analyze_max_concurrency=4)

    assert result.hitl_gated is True  # the alpha CRITICAL gated the run
    assert len(result.findings) == len(_EXPECTED_FILE_BY_TYPE)
    for finding in result.findings:
        assert finding.file_path == _EXPECTED_FILE_BY_TYPE[finding.finding_type], (
            f"{finding.finding_type} landed on {finding.file_path} — cross-attribution "
            f"under concurrency (expected {_EXPECTED_FILE_BY_TYPE[finding.finding_type]})"
        )
    assert result.review_metrics is not None
    assert result.review_metrics.files_examined == 3


async def test_concurrent_round_resumes_through_hitl_and_replays(
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The full increment-6 arc in one run: a parallel-produced round gates
    at HITL, checkpoints, resumes on a FRESH graph (reducer idempotence —
    exactly one round survives rehydration, worker outcomes dedup on their
    slots), publishes, and the persisted interleaved keyed stream passes
    replay-equivalence under the strict hybrid grouping."""
    final = await run_review_with_resume(_FIXTURE, db_url=eval_db, analyze_max_concurrency=4)

    assert final.hitl_gated is True
    assert final.review_status == "completed"
    assert len(final.analysis_rounds) == 1  # one pass-0 round; no double-accumulation
    (round_,) = final.analysis_rounds
    assert set(round_.files_examined) == set(_EXPECTED_FILE_BY_TYPE.values())
    assert len(final.published_comments) >= 1  # publish ran only after the decision

    # Replay-equivalence over the persisted stream: keyed worker phases,
    # aggregate-keyed findings, strict hybrid verification — end to end.
    await AuditReplayer(session_factory=eval_db_session_factory).assert_replay_equivalent(
        final.review_id
    )

    # Sequence-number containment — the load-bearing assumption under the
    # replay grouping: each keyed worker/aggregate phase's per-operation
    # rows carry sequence numbers strictly BETWEEN that phase's start and
    # end markers, on the PERSISTED stream (not just recorder list order).
    reconstruction = await AuditReplayer(session_factory=eval_db_session_factory).reconstruct(
        final.review_id
    )
    keyed_phases = [p for p in reconstruction.phases if p.phase_key is not None]
    assert len(keyed_phases) >= 5  # plan + 3 workers + aggregate
    checked_ops = 0
    for phase in keyed_phases:
        assert phase.start is not None and phase.end is not None
        for event in phase.events:
            assert phase.start.sequence_number < event.sequence_number < phase.end.sequence_number
            checked_ops += 1
    assert checked_ops >= 6  # per-file exams + LLM calls + findings + completed


async def test_concurrent_cache_writes_land_on_distinct_file_keys(
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Concurrent same-review cache writes (SHADOW mode, distinct file keys):
    one row per analyzed file, no collision, and each payload carries its
    OWN file's finding — a cross-write would poison later serves."""
    store = AnalyzeCacheStore(session_factory=eval_db_session_factory)
    result = await run_review_persisting(
        _FIXTURE, db_url=eval_db, analyze_cache_store=store, analyze_max_concurrency=4
    )
    assert result.review_metrics is not None
    assert result.review_metrics.files_examined == 3

    async with eval_db_session_factory() as session:
        rows = (
            await session.execute(text("SELECT file_path, payload FROM analyze_file_cache"))
        ).all()
    assert sorted(row.file_path for row in rows) == sorted(_EXPECTED_FILE_BY_TYPE.values())
    expected_type_by_file = {path: t.value for t, path in _EXPECTED_FILE_BY_TYPE.items()}
    for row in rows:
        payload_findings = row.payload["findings"]
        assert len(payload_findings) == 1
        assert payload_findings[0]["finding_type"] == expected_type_by_file[row.file_path], (
            f"cache row for {row.file_path} carries another file's finding (concurrent cross-write)"
        )


async def test_index_keyed_scripting_with_concurrency_fails_loud(eval_db: str) -> None:
    """The unsafe combination refuses to run: multiple index-keyed analyze
    responses + concurrency > 1 would let completion order decide response
    attribution — a silently misattributed scenario is worse than a refused
    one."""
    index_keyed = "tests/eval/fixtures/mock_github/cost_budget_starvation_rescue.json"
    with pytest.raises(EvalDriverError, match="analyze_responses_by_path"):
        await run_review_persisting(index_keyed, db_url=eval_db, analyze_max_concurrency=4)
