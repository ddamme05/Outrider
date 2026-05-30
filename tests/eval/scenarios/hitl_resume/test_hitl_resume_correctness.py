"""HITL-resume eval scenario: graph interrupt + checkpoint + resume + replay-equivalence.

Per spec §11.2 (the not-optional scenario): process a PR with a guaranteed
critical finding → assert graph interrupts at `hitl` → assert state writes
to `checkpoints` table → simulate process restart (new graph instance,
same checkpointer) → resume via `Command(resume=decision)` → assert
`len(final_state.analysis_rounds)` matches expected count → run
replay-equivalence on the full audit log.

The seven-step flow is the load-bearing test for the LangGraph reducer's
checkpoint-resume idempotence claim. A concatenation reducer would
silently double-accumulate `analysis_rounds` on resume; the
`append_with_dedup_by` reducer makes resume idempotent regardless of
LangGraph's internal rehydration behavior.

V1: scaffolded; the scenario definition + fixtures land here.
`agent/nodes/hitl.py` (the HITL node + `interrupt()` mechanics) shipped
2026-05-26, and `audit/replay.py` (the replay reconstructor +
`AuditReplayer.assert_replay_equivalent`) shipped 2026-05-29. Remaining
blockers before this scenario becomes executable:
  - the `run_review_with_resume` resume shim (FUP-105)
  - a mock LLM provider (FUP-106)
  - the `mock_github/hitl_resume_critical.json` fixture (FUP-108)

The skip marker lifts when the remaining blockers ship.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="requires the run_review_with_resume resume shim (FUP-105) + a mock LLM "
    "provider (FUP-106) + the mock_github/hitl_resume_critical.json fixture (FUP-108); "
    "hitl node + audit/replay.py already shipped"
)

EXPECTED_FINAL_STATE = {
    "analysis_rounds_count": 1,  # one analysis pass, then HITL gate, then publish
    "hitl_interrupted": True,
    "replay_equivalent": True,
}


async def test_hitl_resume_idempotent_under_checkpoint_replay(eval_db_session_factory) -> None:  # type: ignore[no-untyped-def]
    """Seven-step HITL-resume flow ends idempotent + replay-equivalent."""
    from outrider.agent import run_review_with_resume  # type: ignore[import-not-found]
    from outrider.audit.replay import AuditReplayer

    # 1. Process PR with guaranteed critical finding
    # 2. Assert graph interrupts at hitl
    # 3. Assert state writes to checkpoints table
    # 4. Simulate process restart (new graph instance, same checkpointer)
    # 5. Resume via Command(resume=decision)
    # 6. Assert len(analysis_rounds) matches
    # 7. Run replay-equivalence on the audit log
    final_state = run_review_with_resume(
        "tests/eval/fixtures/mock_github/hitl_resume_critical.json"
    )

    assert len(final_state.analysis_rounds) == EXPECTED_FINAL_STATE["analysis_rounds_count"]
    # `AuditReplayer.assert_replay_equivalent` is assertion-style: raises on
    # mismatch, returns None on success (per the shipped audit/replay API).
    # The previous shape `assert ... is expected` (where `expected = True`)
    # would always evaluate to `None is True == False`, masking real failures
    # behind a permanently-failing test. Pin the constant as a contract: this
    # test is wired exclusively for the replay-equivalent expectation.
    # Repurposing it for a non-equivalent scenario requires a rename, not a
    # constant flip.
    assert EXPECTED_FINAL_STATE["replay_equivalent"] is True, (
        "this test is wired for the replay-equivalent expectation only; "
        "flipping the constant would silently skip the equivalence check"
    )
    # `assert_replay_equivalent` is a method on `AuditReplayer` (session_factory
    # injected from the eval harness DB). The async + session_factory wiring
    # finalizes with the run_review_with_resume harness (FUP-105); the call
    # shape below is correct against the shipped replay API.
    replayer = AuditReplayer(session_factory=eval_db_session_factory)
    await replayer.assert_replay_equivalent(final_state.review_id)
