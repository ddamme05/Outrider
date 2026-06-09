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

from .grading import ExpectedFinding, ModelComparison
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
# Real STANDARD-tier true-positive scenarios — the ACTUAL gate inputs (not synthetic).
# Each maps a checked-in mock_github fixture (real code + patch, the same content the
# driven scenarios exercise) to its ground-truth findings. The set spans the severity
# range a STANDARD-tier flip actually has to hold — two blatant CRITICALs (SQLi, auth
# bypass) AND two subtler findings a cheaper model is likelier to miss (a LOW
# missing-error-handling and an N+1 query, the kind of thing the easy CRITICALs don't
# probe). Severity comes from `lookup_severity` (policy is the source of truth — hard-
# coding would drift if the table changes). Line numbers are HEAD source lines, matching
# the fixtures' own scripted analyze responses (which the driven scenarios validate).
# ---------------------------------------------------------------------------
_PYGOAT_SQL_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"
_PYGOAT_AUTH_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_auth_bypass.json"
_MISSING_ERROR_HANDLING_FIXTURE = "tests/eval/fixtures/mock_github/missing_error_handling.json"
_N_PLUS_ONE_FIXTURE = "tests/eval/fixtures/mock_github/n_plus_one_query.json"

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
    # Subtler STANDARD-tier findings — the coverage the blatant CRITICALs miss.
    _MISSING_ERROR_HANDLING_FIXTURE: (
        ExpectedFinding(
            file_path="profile/client.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.MISSING_ERROR_HANDLING,
            severity=lookup_severity(FindingType.MISSING_ERROR_HANDLING),
        ),
    ),
    _N_PLUS_ONE_FIXTURE: (
        ExpectedFinding(
            file_path="orders/enrich.py",
            line_start=7,
            line_end=7,
            finding_type=FindingType.N_PLUS_ONE_QUERY,
            severity=lookup_severity(FindingType.N_PLUS_ONE_QUERY),
        ),
    ),
}

# Safe-code scenarios — the PRECISION instrument (distinct from the recall fixtures above).
# These checked-in fixtures contain NO real finding (a pure refactor; eval() on a hardcoded
# literal in a test), so their ground truth is empty and ANY finding a model emits is an
# UNAMBIGUOUS false positive. That's the clean over-flagging signal the known-vulnerability
# fixtures can't give: on a vulnerable file a model's "extra" is often a legitimate second
# finding the single-entry ground truth didn't encode (the real run showed exactly this), so
# precision there is unreliable; on safe code there is nothing legitimate to find.
_SAFE_CODE_FIXTURES: tuple[str, ...] = (
    "tests/eval/fixtures/mock_github/safe_refactor.json",
    "tests/eval/fixtures/mock_github/eval_in_test_fixture.json",
)


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
    fixture's real content, tier pinned STANDARD (the tier the flip evaluates), and
    dimensions/risk read from the fixture's OWN triage (security/high here)."""
    state = state_from_eval_fixture(_PYGOAT_SQL_FIXTURE)
    assert state.triage_result is not None
    assert state.pr_context.changed_files  # intake's job, done by the adapter
    cf = state.pr_context.changed_files[0]
    assert cf.path == "pygoat/introduction/views.py"
    assert cf.content_head is not None and "search_users" in cf.content_head
    assert state.triage_result.file_tiers[cf.path] is ReviewTier.STANDARD
    assert state.triage_result.relevant_dimensions == (ReviewDimension.SECURITY,)
    assert state.triage_result.overall_risk is RiskLevel.HIGH


def test_state_from_eval_fixture_reads_dimensions_from_fixture_triage() -> None:
    """Regression guard: dimensions come from the fixture's own triage, NOT a hard-coded
    SECURITY. A code-quality / performance scenario must carry its real dimension + risk —
    only the tier is overridden — else the adapter's 'triage-faithful' claim is false and a
    dimension-consuming analyze would under-scope exactly the subtler scenarios."""
    eh = state_from_eval_fixture(_MISSING_ERROR_HANDLING_FIXTURE).triage_result
    assert eh is not None
    assert eh.relevant_dimensions == (ReviewDimension.CODE_QUALITY,)
    assert eh.overall_risk is RiskLevel.MEDIUM  # fixture says medium, not hard-coded HIGH
    assert eh.file_tiers["profile/client.py"] is ReviewTier.STANDARD  # tier still overridden

    npo = state_from_eval_fixture(_N_PLUS_ONE_FIXTURE).triage_result
    assert npo is not None
    assert npo.relevant_dimensions == (ReviewDimension.PERFORMANCE,)
    assert npo.overall_risk is RiskLevel.MEDIUM


@pytest.mark.parametrize("fixture_path", _SAFE_CODE_FIXTURES)
def test_state_from_safe_code_fixture_builds(fixture_path: str) -> None:
    """The safe-code (precision) fixtures build an analyzable STANDARD-tier state the same
    way — multi-dimension triage read from the fixture, tier overridden to STANDARD."""
    state = state_from_eval_fixture(fixture_path)
    assert state.pr_context.changed_files
    assert state.triage_result is not None
    path = state.pr_context.changed_files[0].path
    assert state.triage_result.file_tiers[path] is ReviewTier.STANDARD
    assert len(state.triage_result.relevant_dimensions) >= 1


@pytest.mark.parametrize("fixture_path", _SAFE_CODE_FIXTURES)
@pytest.mark.asyncio
async def test_safe_code_clean_under_clean_model_scores_zero_fp(fixture_path: str) -> None:
    """Precision dimension, end-to-end zero-spend: a model that returns NO finding on safe
    code scores 0 false positives against the empty ground truth, so fp_bounded holds. Proves
    state_from_eval_fixture(safe) flows through the real analyze node and grades clean. The
    over-flag FAIL case is covered by grading's precision gate test."""
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(fixture_path),
        (),  # safe code — no real finding, so any finding would be a false positive
        baseline_provider=_ScriptedProvider(_MISSES_RESPONSE),
        baseline_model="claude-sonnet-4-6",
        candidate_provider=_ScriptedProvider(_MISSES_RESPONSE),
        candidate_model="claude-haiku-4-5",
    )
    assert cmp.baseline.n_false_positives == 0
    assert cmp.candidate.n_false_positives == 0
    assert cmp.fp_bounded is True  # neither over-flagged clean code → precision green


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
# Two dimensions: RECALL over known-vulnerability fixtures, PRECISION over safe code.
# ---------------------------------------------------------------------------


def _print_scenario_report(
    fixture_path: str, cmp: ModelComparison, baseline_model: str, candidate_model: str
) -> None:
    """Print one scenario's recall/precision/fp + the raw gate flags + each model's
    extra/missed detail, so a verdict is interpretable: is an extra noise, or a legitimate
    finding the ground truth didn't encode? The caller picks which flag is the gate for the
    scenario's dimension."""
    b, c = cmp.baseline, cmp.candidate
    print(  # noqa: T201 — operator evidence output
        f"\n[{fixture_path}]"
        f"\n  baseline ({baseline_model}): "
        f"recall={b.recall.value:.2f} precision={b.precision.value:.2f} fp={b.n_false_positives}"
        f"\n  candidate ({candidate_model}): "
        f"recall={c.recall.value:.2f} precision={c.precision.value:.2f} fp={c.n_false_positives}"
        f"\n  recall_held={cmp.recall_held} baseline_valid={cmp.baseline_valid} "
        f"fp_bounded={cmp.fp_bounded}"
    )
    for label, g in (("baseline", b), ("candidate", c)):
        for x in g.extra:
            print(  # noqa: T201 — operator diagnostic
                f"    {label} extra (finding not in ground truth): "
                f"{x.finding_type.value} {x.file_path}:{x.line_start} — {x.title}"
            )
        for m in g.missed:
            print(  # noqa: T201 — operator diagnostic
                f"    {label} MISSED: {m.finding_type.value} {m.file_path}:{m.line_start}"
            )


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model comparison spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_real_model_comparison_evidence() -> None:
    """OPT-IN, real API spend — the evidence REPORT for the STANDARD->Haiku flip decision.

    REPORT-ONLY, BY DESIGN: this asserts only that the run COMPLETED. So pytest "passed"
    means "the run executed", NOT "the gate passed" — the per-scenario gate verdict is in
    the printed output and the end summary, and the human adjudicates. It does not hard-fail
    because model output is nondeterministic; a hard pytest fail would red on that noise.

    TWO DIMENSIONS, measured on the fixtures each is valid for:
    - RECALL, over known-vulnerability fixtures (`_GROUND_TRUTH_BY_FIXTURE`): does the
      candidate catch the known finding? Gated on `recall_held` + `baseline_valid`. FP is
      ADVISORY here, NOT gated: a real run showed the "extras" on vulnerable files are
      usually LEGITIMATE second findings the single-entry ground truth didn't encode (an
      unvalidated-input finding alongside the SQLi), so fp on a vulnerable file is an
      unreliable over-flag signal. Read the printed extra detail, don't gate on it.
    - PRECISION, over safe-code fixtures (`_SAFE_CODE_FIXTURES`, no real finding): does the
      candidate over-flag clean code? ANY finding is an unambiguous FP, so gated on
      `fp_bounded` (candidate must not exceed the baseline's FP count). This is the clean
      over-flag signal the vulnerable fixtures can't give.

    Run the analyze node under Sonnet (baseline, today's STANDARD model) and Haiku
    (candidate, the flip target) per scenario; bounded at 2 analyze calls/scenario over
    small files. CI never runs this.

    Recall is TYPE-EXACT: a match requires the same `finding_type` AND policy severity (plus
    file + line window). A candidate that SEES the vulnerability but labels it a different
    type scores as a miss — so on a recall drop, read the `missed`/`extra` detail to tell a
    true regression from a classification disagreement before acting on the flip.
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
    gate_results: list[tuple[str, str, bool]] = []  # (fixture, dimension, verdict)
    try:
        # RECALL dimension — gate on recall_held + baseline_valid; FP advisory (see docstring).
        for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
            cmp = await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=provider,
                baseline_model=baseline_model,
                candidate_provider=provider,
                candidate_model=candidate_model,
            )
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
            gate_results.append((fixture_path, "recall", cmp.recall_held and cmp.baseline_valid))
            assert cmp.baseline is not None  # the run completed
        # PRECISION dimension — safe code, empty ground truth so ANY finding is a real FP;
        # gate on fp_bounded (Haiku must not over-flag clean code more than Sonnet).
        for fixture_path in _SAFE_CODE_FIXTURES:
            cmp = await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                (),
                baseline_provider=provider,
                baseline_model=baseline_model,
                candidate_provider=provider,
                candidate_model=candidate_model,
            )
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
            gate_results.append((fixture_path, "precision", cmp.fp_bounded))
            assert cmp.baseline is not None  # the run completed
        # REPORT-ONLY summary — pytest "passed" means the run completed, NOT the gate verdict.
        failed = [(fx, dim) for fx, dim, ok in gate_results if not ok]
        green = len(gate_results) - len(failed)
        print(  # noqa: T201 — operator gate summary
            "\n"
            + "=" * 72
            + "\nGATE SUMMARY — REPORT ONLY: pytest 'passed' means the run COMPLETED, NOT"
            + "\nthat the gate passed. Adjudicate the per-scenario verdicts above."
            + f"\n  {green}/{len(gate_results)} dimension-verdicts green "
            + "(recall fixtures gate on recall; safe-code fixtures gate on FP)."
            + "".join(f"\n  {dim.upper()} FAILED: {fx}" for fx, dim in failed)
            + "\n"
            + "=" * 72
        )
    finally:
        await provider.aclose()
