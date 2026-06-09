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
from outrider.policy import FindingSeverity, FindingType, lookup_severity
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

from .grading import ExpectedFinding
from .model_comparison import (
    compare_models_on_scenario,
    run_analyze_under_model,
    state_from_eval_fixture,
)

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

# ---------------------------------------------------------------------------
# Real PyGoat true-positive scenarios — the ACTUAL gate inputs (not synthetic).
# Each maps a checked-in mock_github fixture (real vulnerable code + patch, the same
# content the driven scenarios in scenarios/true_positives/ exercise) to its
# ground-truth findings. Severity comes from `lookup_severity` (policy is the source of
# truth, per the scenario convention — hard-coding CRITICAL would drift if the table
# changes). Line numbers are HEAD source lines of the vulnerability, matching the
# fixtures' own scripted analyze responses (which the driven scenarios validate).
# ---------------------------------------------------------------------------
_PYGOAT_SQL_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"
_PYGOAT_AUTH_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_auth_bypass.json"

_GROUND_TRUTH_BY_FIXTURE: dict[str, tuple[ExpectedFinding, ...]] = {
    _PYGOAT_SQL_FIXTURE: (
        ExpectedFinding(
            file_path="pygoat/introduction/views.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        ),
    ),
    _PYGOAT_AUTH_FIXTURE: (
        ExpectedFinding(
            file_path="pygoat/introduction/auth_views.py",
            line_start=7,
            line_end=8,
            finding_type=FindingType.AUTH_BYPASS,
            severity=lookup_severity(FindingType.AUTH_BYPASS),
        ),
    ),
}


def _judged_response_for(expected: ExpectedFinding) -> str:
    """A scripted analyze response emitting one JUDGED finding matching `expected`.

    JUDGED needs no `query_match_id` (the proof boundary's structural requirement is
    OBSERVED/INFERRED only), so this stands in for "the model found the known
    vulnerability" without depending on which tree-sitter queries fire — exactly what
    the zero-spend wiring proof needs. Grading is evidence-tier-agnostic, so a JUDGED
    stand-in matches ground truth the same as the fixture's own (sometimes OBSERVED)
    response would."""
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": expected.finding_type.value,
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": "known vulnerability",
                    "description": "the finding the scenario is known to contain.",
                    "evidence": "e",
                    "line_start": expected.line_start,
                    "line_end": expected.line_end,
                    "trace_candidates": [],
                }
            ]
        }
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


class _NoOpExchangePersister:
    """A no-op `LLMExchangePersister` for the real-model run. The comparison reads
    findings from analyze's RETURN, not the llm_call/audit stream, so the exchange
    persist is discarded — same rationale as `_NullSink`. Required (not `None`) because
    the real `AnthropicProvider.complete()` is fail-closed: it raises
    `LLMPersisterNotWiredError` BEFORE the SDK call when `persister is None` (DECISIONS
    #016), so `persister=None` would crash the opt-in run on its first analyze call."""

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        return None


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
# Real-fixture wiring (ZERO-SPEND) — the gate over ACTUAL scenario content, with the
# model still scripted. Proves state_from_eval_fixture -> analyze -> grade works on the
# real PyGoat code (not synthetic `return 42`); only the provider response is faked.
# ---------------------------------------------------------------------------


def test_state_from_eval_fixture_builds_enriched_standard_state() -> None:
    """The adapter stands in for intake+triage: changed_files populated from the
    fixture's real content, tier pinned STANDARD (the tier the flip evaluates)."""
    state = state_from_eval_fixture(_PYGOAT_SQL_FIXTURE)
    assert state.triage_result is not None
    assert state.pr_context.changed_files  # intake's job, done by the adapter
    cf = state.pr_context.changed_files[0]
    assert cf.path == "pygoat/introduction/views.py"
    assert cf.content_head is not None and "search_users" in cf.content_head
    assert state.triage_result.file_tiers[cf.path] is ReviewTier.STANDARD


@pytest.mark.parametrize("fixture_path", list(_GROUND_TRUTH_BY_FIXTURE))
@pytest.mark.asyncio
async def test_real_fixture_content_through_analyze_catches_regression(fixture_path: str) -> None:
    """END-TO-END zero-spend over EACH real PyGoat fixture: a scripted "Sonnet" that
    returns the known finding scores recall 1.0; a scripted "Haiku" that misses it scores
    0.0 and FAILS the gate. The STATE is the real vulnerable code (built by
    state_from_eval_fixture); only the provider is faked — so the real run differs only by
    swapping in the AnthropicProvider. Parametrized over BOTH fixtures so the SQL path
    (analyze accepting a finding at views.py:5) is verified, not just the auth path the
    opt-in run also depends on. A JUDGED stand-in needs no tree-sitter query to fire."""
    ground_truth = _GROUND_TRUTH_BY_FIXTURE[fixture_path]
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(fixture_path),
        ground_truth,
        baseline_provider=_ScriptedProvider(_judged_response_for(ground_truth[0])),
        baseline_model="claude-sonnet-4-6",
        candidate_provider=_ScriptedProvider(_MISSES_RESPONSE),
        candidate_model="claude-haiku-4-5",
    )
    assert cmp.baseline.recall.value == 1.0  # found the known finding in real code
    assert cmp.candidate.recall.value == 0.0  # missed it
    assert cmp.passes is False  # the gate catches the recall regression


# ---------------------------------------------------------------------------
# Opt-in REAL-model run (SPEND) — the evidence path. Skipped unless explicitly enabled.
# This is the ACTUAL gate: real models over real STANDARD-tier scenarios.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model comparison spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_real_model_comparison_pygoat_true_positives() -> None:
    """OPT-IN, real API spend — the evidence run that gates the STANDARD->Haiku flip.

    For each real PyGoat true-positive fixture (STANDARD tier), run the analyze node
    under Sonnet (baseline, today's STANDARD model) and Haiku (candidate, the flip
    target), grading recall/precision against the known CRITICAL finding, and report the
    gate verdict per scenario. The operator reads whether Haiku held recall before
    flipping the default. Bounded: 2 analyze calls per scenario over small files. NO
    hard quality assertion (model output is nondeterministic — the human adjudicates);
    CI never runs this.

    Read the recall number as TYPE-EXACT recall: a match requires the same
    `finding_type` AND policy severity (plus file + line window), per the structural
    grading contract. So a candidate that genuinely SEES the vulnerability but labels it
    a different type (e.g. SQL injection reported as `missing_input_validation`) scores
    as a miss. That is intended — the gate protects the exact STANDARD-tier findings,
    not "did it notice something here" — but when reading a recall drop, check the
    `missed`/`extra` detail to tell a true regression from a classification disagreement
    before acting on the flip.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the real-model comparison")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.pricing import normalize_to_pricing_key  # noqa: PLC0415

    cfg = ModelConfig()
    # persister MUST be a real LLMExchangePersister, not None: AnthropicProvider.complete()
    # is fail-closed on persister=None (raises before the SDK call). The comparison reads
    # findings from analyze's return, so a no-op persister is the right wiring here.
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )
    # Baseline is `analyze_model` (today's Sonnet top-tier analyze behavior), NOT
    # `standard_analyze_model` — the latter is the very knob the flip changes, so reading it
    # would compare the candidate against ITSELF if the operator already set
    # OUTRIDER_MODEL_STANDARD_ANALYZE_MODEL=claude-haiku-4-5. The methodological question is
    # "does Haiku preserve STANDARD findings vs today's Sonnet analyze?".
    baseline_model = cfg.analyze_model
    candidate_model = "claude-haiku-4-5"  # the model the flip proposes for STANDARD
    # Belt-and-suspenders: fail loudly on a meaningless self-comparison (e.g. analyze_model
    # env-overridden to Haiku). Normalized so a dated pin (…-20251001) can't sneak past.
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the comparison would prove nothing about Sonnet-vs-Haiku. Point "
            "OUTRIDER_MODEL_ANALYZE_MODEL at Sonnet (or unset it) for the evidence run."
        )
    try:
        for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
            cmp = await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=provider,
                baseline_model=baseline_model,
                candidate_provider=provider,
                candidate_model=candidate_model,
            )
            b, c = cmp.baseline, cmp.candidate
            print(  # noqa: T201 — opt-in evidence output for the operator
                f"\n[{fixture_path}]"
                f"\n  baseline ({baseline_model}): "
                f"recall={b.recall.value:.2f} precision={b.precision.value:.2f} "
                f"fp={b.n_false_positives}"
                f"\n  candidate ({candidate_model}): "
                f"recall={c.recall.value:.2f} precision={c.precision.value:.2f} "
                f"fp={c.n_false_positives}"
                f"\n  gate passes={cmp.passes} (baseline_valid={cmp.baseline_valid} "
                f"recall_held={cmp.recall_held} fp_bounded={cmp.fp_bounded})"
            )
            assert cmp.baseline is not None  # the run completed
    finally:
        await provider.aclose()
