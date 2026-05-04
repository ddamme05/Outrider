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

V1: scaffolded; the scenario definition + fixtures land here. Two
dependencies must ship before this scenario becomes executable:
  - `agent/nodes/hitl.py` (the HITL node + `interrupt()` mechanics)
  - `audit/replay.py` (the replay-equivalence assertion harness)

The skip marker lifts only when BOTH ship. Whichever ships first does
not unblock this scenario alone.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires hitl node + audit/replay")

EXPECTED_FINAL_STATE = {
    "analysis_rounds_count": 1,  # one analysis pass, then HITL gate, then publish
    "hitl_interrupted": True,
    "replay_equivalent": True,
}


def test_hitl_resume_idempotent_under_checkpoint_replay() -> None:
    """Seven-step HITL-resume flow ends idempotent + replay-equivalent."""
    from outrider.agent import run_review_with_resume  # type: ignore[import-not-found]
    from outrider.audit.replay import assert_replay_equivalent  # type: ignore[import-not-found]

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
    # `assert_replay_equivalent` is assertion-style: raises on mismatch,
    # returns None on success (per the audit/replay module's eventual
    # API). The previous shape `assert ... is expected` (where
    # `expected = True`) would always evaluate to `None is True == False`,
    # masking real failures behind a permanently-failing test. Pin the
    # constant as a contract: this test is wired exclusively for the
    # replay-equivalent expectation. Repurposing it for a non-equivalent
    # scenario requires a rename, not a constant flip.
    assert EXPECTED_FINAL_STATE["replay_equivalent"] is True, (
        "this test is wired for the replay-equivalent expectation only; "
        "flipping the constant would silently skip the equivalence check"
    )
    assert_replay_equivalent(final_state.review_id)
