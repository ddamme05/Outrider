"""Driver meta-test: `run_review` drives the real graph from a fixture.

The harness-internal proof for the eval graph driver, analogous to
`tests/integration/test_e2e_smoke.py` but fixture-driven and returning an
`EvalRunResult`. It reuses the smoke test's CI-proven scenario (a modified file
that adds a function; one MEDIUM finding → no HITL gate → publish runs) so that
"does `run_review` reproduce the proven end-to-end outcome" is the only thing
under test, not the scenario shape.

Integration-tier: needs `--is-eval` (sets `OUTRIDER_IS_EVAL=1`) + a running
`postgres-test` container (`run_review` carves its own ephemeral DB off
`TEST_DATABASE_URL`; it does NOT use the `eval_db` fixture). The fail-closed
guard test needs neither and runs anywhere.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from outrider.agent import run_review
from outrider.agent.eval_driver import EvalRunResult
from outrider.eval_support import EvalModeNotEnabledError

if TYPE_CHECKING:
    from pathlib import Path

_FILE = "src/handler.py"
_BASE = "def existing():\n    return 1\n"
_HEAD = "def existing():\n    return 1\n\ndef vulnerable(user_input):\n    return user_input\n"
# Hunks-only — real GitHub /pulls/{n}/files wire shape (coordinates wraps it
# with synthesized headers; post-wrap identical to the smoke test's patch).
_PATCH = (
    "@@ -1,2 +1,5 @@\n"
    " def existing():\n"
    "     return 1\n"
    "+\n"
    "+def vulnerable(user_input):\n"
    "+    return user_input\n"
)
# 1-indexed line of "    return user_input" in _HEAD (the finding target).
_FINDING_LINE = _HEAD[: _HEAD.index("    return user_input")].count("\n") + 1


def _smoke_fixture() -> dict[str, object]:
    return {
        "installation_id": 12345,
        "owner": "acme",
        "repo": "widget",
        "pr_number": 7,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "pr_title": "Add vulnerable handler",
        "author": "someone",
        "total_additions": 3,
        "total_deletions": 0,
        "files": [
            {
                "path": _FILE,
                "status": "modified",
                "additions": 3,
                "deletions": 0,
                "patch": _PATCH,
                "content_base": _BASE,
                "content_head": _HEAD,
            }
        ],
        "llm_responses": {
            "triage": [
                json.dumps(
                    {
                        "file_tiers": {_FILE: "deep"},
                        "overall_risk": "medium",
                        "relevant_dimensions": ["security"],
                        "reasoning": "deep-review the changed handler.",
                    }
                )
            ],
            "analyze": [
                json.dumps(
                    {
                        "findings": [
                            {
                                "finding_type": "missing_input_validation",
                                "evidence_tier": "judged",
                                "query_match_id": None,
                                "trace_path": None,
                                "title": "Unvalidated user input returned directly",
                                "description": "vulnerable() returns user_input unvalidated.",
                                "evidence": "    return user_input",
                                "line_start": _FINDING_LINE,
                                "line_end": _FINDING_LINE,
                                "trace_candidates": [],
                            }
                        ]
                    }
                )
            ],
            "synthesize": ["One input-validation finding on the new function."],
        },
    }


def _write_fixture(tmp_path: Path, payload: dict[str, object]) -> str:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_run_review_requires_eval_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed before any DB work when OUTRIDER_IS_EVAL is unset (no DB needed)."""
    monkeypatch.delenv("OUTRIDER_IS_EVAL", raising=False)
    fixture_path = _write_fixture(tmp_path, _smoke_fixture())
    with pytest.raises(EvalModeNotEnabledError):
        run_review(fixture_path)


def test_run_review_drives_graph_and_returns_result(tmp_path: Path) -> None:
    """run_review drives the real graph to publish and returns a populated result.

    Needs postgres-test (run_review carves its own ephemeral DB). The internal
    is_eval integrity gate runs inside run_review; a violation would raise here.
    """
    fixture_path = _write_fixture(tmp_path, _smoke_fixture())

    result = run_review(fixture_path)

    assert isinstance(result, EvalRunResult)
    # MEDIUM finding -> no HITL gate -> publish ran.
    assert result.hitl_gated is False
    # Finding present + iterable/len contract holds.
    assert len(result) >= 1
    assert any(f.finding_type.value == "missing_input_validation" for f in result.findings)
    # Sub-HIGH finding on a changed line -> routed inline -> published.
    assert len(result.published_comments) >= 1
    assert result.review_id is not None

    # FUP-093 Finding-2 pin: the eval driver emits faithful LLMCallEvent rows, and
    # synthesize populates ReviewMetrics by SUMming them — so the aggregates are
    # NON-ZERO, not the false-zero a non-emitting provider double would produce.
    # Each scripted call carries fixed sentinels (100 in / 50 out), so the totals
    # are exact multiples of the call count: the `== 100 * calls` form cannot pass
    # at zero. This is the guard that the driver can't silently report false-zero
    # metrics (and that synthesize's POST-call query counted synthesize's own call,
    # not just the earlier scripted ones).
    metrics = result.review_metrics
    assert metrics is not None
    assert metrics.llm_calls_made is not None and metrics.llm_calls_made > 0
    assert metrics.total_input_tokens == 100 * metrics.llm_calls_made
    assert metrics.total_output_tokens == 50 * metrics.llm_calls_made
    assert metrics.total_cost_usd is not None and metrics.total_cost_usd > 0
