"""Check that reserved budget reaches a risky file after ordinary budget runs out.

Drive the real graph with scripted responses and a tight budget. The late
command-injection file must still be analyzed, and its critical finding
must wait for approval. Requires --is-eval and the postgres-test database;
the driver creates and removes its own test database."""

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
