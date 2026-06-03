"""HITL-resume eval scenario: graph interrupt + checkpoint + resume + replay-equivalence.

Per spec §11.2 (the not-optional scenario): process a PR with a guaranteed
CRITICAL finding → the graph interrupts at `hitl` and checkpoints to Postgres →
a fresh graph instance + a fresh checkpointer on the SAME DB resume via
`Command(resume=decision)` → assert `len(analysis_rounds) == 1` (reducer
idempotence) + published comments + `completed` status → run replay-equivalence
over the full audit log.

`len(analysis_rounds) == 1` proves the resume continued the ORIGINAL interrupted
run and its single analysis round survived rehydration — a fresh run (e.g. a
thread_id mismatch) would re-enter analyze, re-hit the gate, and never reach
publish. It pins single-round structure across resume; it does NOT by itself
exercise dedup-vs-concat, because resume re-enters the `hitl` node and analyze does
not re-execute — the reducer's idempotence-under-replay is covered by
`append_with_dedup_by`'s own tests, not inferred from this count.

The boundary invariant it guards (trust boundary #6): the base `run_review` never
approves — it stops at the gate (asserted by the negative-half test below). Here
`run_review_with_resume` approves EXPLICITLY, through the same `Command(resume=...)`
path production uses, and publish runs only after that decision — so the gated
finding's comment is non-empty only post-resume.

Driven by `run_review_with_resume` against
`tests/eval/fixtures/mock_github/hitl_resume_critical.json` (a SQL-injection PR
whose finding policy-maps to CRITICAL). Two sequential `AsyncPostgresSaver`s prove
the suspended state lives in Postgres, not a Python object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import text

from outrider.agent import run_review, run_review_with_resume
from outrider.audit.replay import AuditReplayer

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_FIXTURE = "tests/eval/fixtures/mock_github/hitl_resume_critical.json"

EXPECTED_FINAL_STATE = {
    "analysis_rounds_count": 1,  # one analysis pass, then HITL gate, then resume + publish
    "review_status": "completed",
}


async def test_hitl_resume_idempotent_under_checkpoint_replay(
    eval_db: str,
    eval_db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Seven-step HITL-resume flow ends idempotent + replay-equivalent.

    `eval_db` is the migrated per-test DB URL the driver runs against; the SAME DB
    backs `eval_db_session_factory`, so the replayer reconstructs from exactly the
    stream the run recorded. `run_review_with_resume` is awaited (it is async — the
    replay assertion forces an async test, and a sync `asyncio.run`-based driver
    can't run inside a live event loop).
    """
    # 1-5. Process PR → interrupt at hitl → checkpoint to Postgres → fresh
    #      graph+saver on the same DB → resume via Command(resume=approve-all).
    final = await run_review_with_resume(_FIXTURE, db_url=eval_db)

    # The interrupt fired (the CRITICAL finding gated), and resume drove to publish.
    assert final.hitl_gated is True
    # 6. Proof of resume-of-original: a fresh run would re-enter analyze, re-hit
    #    the gate, and never reach publish. Exactly 1 = the single phase-A round
    #    survived rehydration (analyze does not re-run on resume — it re-enters hitl).
    assert len(final.analysis_rounds) == EXPECTED_FINAL_STATE["analysis_rounds_count"]
    # Boundary #6 positive test: the gated finding reached GitHub ONLY after the
    # explicit decision was supplied through Command(resume=...).
    assert len(final.published_comments) >= 1
    assert final.review_status == EXPECTED_FINAL_STATE["review_status"]

    # 3 (explicit). The interrupt persisted to the Postgres `checkpoints` table for
    # this thread — the durability the two-saver restart depends on.
    async with eval_db_session_factory() as session:
        checkpoint_rows = await session.scalar(
            text("SELECT count(*) FROM checkpoints WHERE thread_id = :tid"),
            {"tid": str(final.review_id)},
        )
    assert checkpoint_rows is not None and checkpoint_rows >= 1

    # 7. Replay-equivalence over the full audit log (reconstruct from the SAME DB).
    #    `assert_replay_equivalent` raises on mismatch, returns None on success.
    await AuditReplayer(session_factory=eval_db_session_factory).assert_replay_equivalent(
        final.review_id
    )


def test_critical_fixture_holds_gate_without_resume() -> None:
    """Boundary #6 negative half: the base `run_review` NEVER approves.

    On the SAME CRITICAL fixture, the single-pass driver gates and nothing reaches
    GitHub — no resume, no decision, no publish. Paired with the resume test above
    (which publishes ONLY after the explicit `Command(resume=...)` decision), this
    is the complete "withheld until approved, then published" proof. `run_review`
    is sync + self-contained (it carves its own ephemeral DB), so no `eval_db`.
    """
    result = run_review(_FIXTURE)
    assert result.hitl_gated is True
    assert result.published_comments == ()  # gate held — nothing posted to GitHub
    assert len(result.findings) >= 1  # synthesize ran before the gate
