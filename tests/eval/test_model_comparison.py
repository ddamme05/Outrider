"""Tests for the model-tier comparison runner (tests/eval/model_comparison.py).

Proves the END-TO-END machinery zero-spend: `run_analyze_under_model` runs a real
analyze pass under an injected (scripted) provider, and `compare_models_on_scenario`
grades both models + applies the gate — so a model that MISSES a finding the baseline
caught is provably flagged. The real Sonnet-vs-Haiku run is the same code path with the
`AnthropicProvider` injected, gated behind `OUTRIDER_EVAL_REAL_MODELS=1` (skipped in CI;
that's the opt-in SPEND run that produces the actual flip evidence).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from outrider.llm.base import LLMRequest, LLMResponse
from outrider.policy import FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

from .grading import ExpectedFinding
from .model_comparison import compare_models_on_scenario, run_analyze_under_model

_SIMPLE_PY = (
    "def my_function():\n    return 42\n\ndef another_function(x):\n    y = x + 1\n    return y\n"
)
_PATCH = (
    "--- a/src/example.py\n+++ b/src/example.py\n"
    "@@ -1,1 +1,2 @@\n def my_function():\n+    return 42\n"
)

# The finding the "Sonnet" script emits — sql_injection at lines 1-2 (inside my_function,
# the changed scope unit). sql_injection -> CRITICAL via SEVERITY_POLICY.
_FINDS_RESPONSE = json.dumps(
    {
        "findings": [
            {
                "finding_type": "sql_injection",
                "evidence_tier": "judged",
                "query_match_id": None,
                "trace_path": None,
                "title": "SQL injection",
                "description": "A SQL injection in the changed function.",
                "evidence": "def my_function():\n    return 42",
                "line_start": 1,
                "line_end": 2,
                "trace_candidates": [],
            }
        ]
    }
)
_MISSES_RESPONSE = json.dumps({"findings": []})

# Ground truth: the scenario is KNOWN to contain this finding.
_GROUND_TRUTH = (
    ExpectedFinding(
        file_path="src/example.py",
        line_start=1,
        line_end=2,
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,
    ),
)


class _ScriptedProvider:
    """Returns a fixed canned response (ignores request.model) — stands in for one
    model. Two of these with different responses inject a recall divergence."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.calls: list[LLMRequest] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self.response_text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=10,
        )


def _build_state() -> ReviewState:
    changed_file = ChangedFile(
        path="src/example.py",
        status="modified",
        additions=2,
        deletions=0,
        patch=_PATCH,
        content_base="def my_function():\n    return 0\n",
        content_head=_SIMPLE_PY,
        previous_path=None,
        language="python",
    )
    pr_context = PRContext(
        installation_id=99999,
        owner="o",
        repo="r",
        pr_number=1,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="t",
        pr_body=None,
        author="a",
        total_additions=2,
        total_deletions=0,
        changed_files=(changed_file,),
    )
    triage_result = TriageResult(
        file_tiers={"src/example.py": ReviewTier.STANDARD},  # the tier the flip changes
        overall_risk=RiskLevel.HIGH,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="r",
        policy_version=ACTIVE_POLICY_VERSION,
    )
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=pr_context,
        triage_result=triage_result,
        is_eval=True,
    )


# ---------------------------------------------------------------------------
# run_analyze_under_model — the per-model run produces gradeable findings
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_analyze_under_model_returns_findings() -> None:
    """A finding-emitting scripted run yields a ReviewFinding the grader can match; an
    empty-response run yields none. Pins that the comparison's inputs are real."""
    finds = await run_analyze_under_model(
        _build_state(), provider=_ScriptedProvider(_FINDS_RESPONSE), model="claude-sonnet-4-6"
    )
    assert len(finds) >= 1, "the scripted finding response did not admit a finding"
    assert finds[0].finding_type == FindingType.SQL_INJECTION

    misses = await run_analyze_under_model(
        _build_state(), provider=_ScriptedProvider(_MISSES_RESPONSE), model="claude-haiku-4-5"
    )
    assert misses == ()


# ---------------------------------------------------------------------------
# compare_models_on_scenario — the gate catches a recall regression end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_comparison_catches_recall_regression() -> None:
    """The whole point: a candidate model that MISSES a finding the baseline caught FAILS
    the gate — proven through the real analyze path, not just the pure grader."""
    cmp = await compare_models_on_scenario(
        _build_state(),
        _GROUND_TRUTH,
        baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
        baseline_model="claude-sonnet-4-6",
        candidate_provider=_ScriptedProvider(_MISSES_RESPONSE),
        candidate_model="claude-haiku-4-5",
    )
    assert cmp.baseline.recall.value == 1.0
    assert cmp.candidate.recall.value == 0.0
    assert cmp.recall_held is False
    assert cmp.passes is False


@pytest.mark.asyncio
async def test_comparison_passes_when_candidate_holds_recall() -> None:
    """Both models catch the finding → the gate passes (the green-light case)."""
    cmp = await compare_models_on_scenario(
        _build_state(),
        _GROUND_TRUTH,
        baseline_provider=_ScriptedProvider(_FINDS_RESPONSE),
        baseline_model="claude-sonnet-4-6",
        candidate_provider=_ScriptedProvider(_FINDS_RESPONSE),
        candidate_model="claude-haiku-4-5",
    )
    assert cmp.baseline.recall.value == 1.0
    assert cmp.candidate.recall.value == 1.0
    assert cmp.passes is True


# ---------------------------------------------------------------------------
# Opt-in REAL-model run (SPEND) — the evidence path. Skipped unless explicitly enabled.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model comparison spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_real_model_comparison_wires_anthropic_provider() -> None:
    """OPT-IN, real API spend: run the SAME comparison path with the real
    `AnthropicProvider`, Sonnet (baseline) vs Haiku (candidate). This wires the evidence
    run; point `_build_state()` (or a real scenario fixture) at genuinely vulnerable code
    and read the reported recall/precision before flipping the default. NOT a pass/fail
    quality assertion here — the human reads the comparison; CI never runs this.
    """
    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig

    cfg = ModelConfig()
    provider = AnthropicProvider()  # type: ignore[call-arg]  # real creds from env
    try:
        cmp = await compare_models_on_scenario(
            _build_state(),
            _GROUND_TRUTH,
            baseline_provider=provider,
            baseline_model=cfg.analyze_model,  # Sonnet
            candidate_provider=provider,
            candidate_model="claude-haiku-4-5",  # the proposed STANDARD-tier model
        )
    finally:
        await provider.aclose()
    # Reported for the human's flip decision; no quality assertion on synthetic input.
    print(  # noqa: T201 — opt-in run output for the operator
        f"\nREAL comparison: baseline recall={cmp.baseline.recall.value} "
        f"candidate recall={cmp.candidate.recall.value} passes={cmp.passes}"
    )
    assert cmp.baseline is not None  # the run completed
