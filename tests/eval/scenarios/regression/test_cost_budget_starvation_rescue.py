"""analyze-cost-fairness Stage 1c: the PR #8 starvation, promoted to a permanent
end-to-end guard.

The arc started because a coverage smoke (`ddamme05/outrider-smoke-test` PR #8)
dropped a CRITICAL `command_injection` (`os.system`) — not because triage tiered
it out, but because the analyze cost gate skipped it `COST_BUDGET_EXHAUSTED`
behind benign DEEP files (the budget drained before its turn). The Stage 1
high-risk reserve fixes that. This scenario pins the fix through the REAL 7-node
graph.

Fixture `cost_budget_starvation_rescue.json`: 5 DEEP files (4 benign + a late
`ops_service.py` whose added `os.system(cmd)` line trips `policy.recall`) under a
tight 80k budget. The general pool fits the first 3 benign; benign4 starves;
`ops_service` draws the reserve and is analyzed. Its scripted `command_injection`
finding (line 3, inside `ops_service`'s scope) is admitted → CRITICAL → the HITL
gate holds (single-pass driver does not resume).

LOAD-BEARING (verified): neutralize the reserve (`HIGH_RISK_RESERVE_FRACTION=0.0`)
and `ops_service` starves — benign4 is analyzed in its place, the `command_injection`
response misdirects to benign4's 2-line scope, gets rejected (line 3 outside
scope), and BOTH assertions below fail. So this scenario cannot pass without the
reserve actually rescuing the high-risk file.

Lives in `regression/` (not `structural/`): it drives a real scripted-LLM analyze
pass, so it needs `--is-eval` + the `postgres-test` container (`run_review`
self-manages an ephemeral DB).
"""

from __future__ import annotations

from pathlib import Path

from outrider.agent import run_review

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "mock_github"
    / "cost_budget_starvation_rescue.json"
)


def test_high_risk_file_rescued_from_budget_starvation_end_to_end() -> None:
    """The late CRITICAL command_injection file is rescued by the reserve under
    budget pressure and reaches analysis end-to-end."""
    result = run_review(str(_FIXTURE))

    # 1. The CRITICAL command_injection from ops_service is admitted — proving it
    #    was ANALYZED, not COST_BUDGET_EXHAUSTED-skipped behind benign DEEP files.
    #    This is the load-bearing assertion: without the reserve, ops_service
    #    starves and this finding is absent (verified via HIGH_RISK_RESERVE_FRACTION=0).
    command_injections = [f for f in result.findings if f.finding_type.value == "command_injection"]
    assert len(command_injections) == 1, (
        "expected the ops_service command_injection to be admitted (rescued by the "
        f"high-risk reserve); got {[f.finding_type.value for f in result.findings]}. "
        "Absent means ops_service was budget-starved before analysis — the reserve "
        "regressed (the exact PR #8 failure this guards)."
    )
    finding = command_injections[0]
    assert finding.line_start == 3
    assert finding.severity.value == "critical"

    # 2. A CRITICAL finding trips the HITL gate; the single-pass driver records it
    #    and does not resume.
    assert result.hitl_gated is True

    # 3. Budget pressure was REAL: not all 5 files fit (a benign file starved). If
    #    this is 5, the budget no longer starves anything and assertion 1 would pass
    #    trivially without exercising the reserve. A mismatch here means the analyze
    #    prompt size (the per-file token estimate) drifted — re-tune the fixture's
    #    `total_review_budget_tokens` so exactly one benign file starves.
    assert result.review_metrics is not None
    assert result.review_metrics.files_examined == 4
