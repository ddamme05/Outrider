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

from .grading import ExpectedFinding, GradeResult, ModelComparison
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
# bypass) PLUS four subtler findings a cheaper model is likelier to miss: a HIGH
# path-traversal, two MEDIUMs (an N+1 query and a distinct missing-input-validation), and a
# LOW missing-error-handling — so recall spans CRITICAL -> HIGH -> MEDIUM(x2) -> LOW, not
# just the easy CRITICALs. Severity comes from `lookup_severity` (policy is the source of
# truth — hardcoding would drift if the table changes). Line numbers are HEAD source lines,
# matching the fixtures' own scripted analyze responses (which the driven scenarios validate).
# ---------------------------------------------------------------------------
_PYGOAT_SQL_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"
_PYGOAT_AUTH_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_auth_bypass.json"
_MISSING_ERROR_HANDLING_FIXTURE = "tests/eval/fixtures/mock_github/missing_error_handling.json"
_N_PLUS_ONE_FIXTURE = "tests/eval/fixtures/mock_github/n_plus_one_query.json"
_PATH_TRAVERSAL_FIXTURE = "tests/eval/fixtures/mock_github/path_traversal.json"
_MISSING_INPUT_VALIDATION_FIXTURE = "tests/eval/fixtures/mock_github/missing_input_validation.json"

# Recall HOLD-OUTS — real SQLi in injection forms the analyze-v3 prompt NAMES but never exemplifies
# (f-string / str.format / `+` concatenation, vs the prompt's only-shown %-format). The model MUST
# still flag these; a miss means the remediation over-suppressed real SQLi. NAMED-BUT-UNEXAMPLED — a
# WEAKER generalization signal than the regression safe hold-outs (which the prompt never mentions).
_SQLI_HOLDOUT_FSTRING_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_fstring.json"
_SQLI_HOLDOUT_FORMAT_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_format.json"
_SQLI_HOLDOUT_CONCAT_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_concat.json"

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
    # Severity-breadth fills (de-risk the flip): a HIGH and a second MEDIUM (distinct type),
    # so recall now spans CRITICAL -> HIGH -> MEDIUM(x2) -> LOW rather than the easy CRITICALs.
    _PATH_TRAVERSAL_FIXTURE: (
        ExpectedFinding(
            file_path="reports/views.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.PATH_TRAVERSAL,
            severity=lookup_severity(FindingType.PATH_TRAVERSAL),
        ),
    ),
    _MISSING_INPUT_VALIDATION_FIXTURE: (
        ExpectedFinding(
            file_path="accounts/views.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.MISSING_INPUT_VALIDATION,
            severity=lookup_severity(FindingType.MISSING_INPUT_VALIDATION),
        ),
    ),
    # Recall hold-outs: real SQLi in injection forms the prompt names but does not exemplify.
    # Sonnet must catch them (baseline_valid) and Haiku must still catch them (recall_held) —
    # if Haiku misses, the parameterized-query remediation over-suppressed real SQLi.
    _SQLI_HOLDOUT_FSTRING_FIXTURE: (
        ExpectedFinding(
            file_path="search/views.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        ),
    ),
    _SQLI_HOLDOUT_FORMAT_FIXTURE: (
        ExpectedFinding(
            file_path="lookup/users.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        ),
    ),
    _SQLI_HOLDOUT_CONCAT_FIXTURE: (
        ExpectedFinding(
            file_path="contacts/lookup.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
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

# Regression-track fixtures — safe, correctly-parameterized queries in several idioms (cursor
# `%s` + list/tuple params, named `%(x)s` + dict, Django ORM `raw()`). The tracked caveat
# (DECISIONS#041) is TYPE-SPECIFIC: Haiku can NONDETERMINISTICALLY label a parameterized query
# as sql_injection. So the verdict counts ONLY sql_injection false positives (`_sqli_fp_count`),
# NOT total fp of any type — a baseline that over-flags an UNRELATED dimension (e.g.
# missing_error_handling on a query with no try/except) must not blind the track. Read
# ABSOLUTELY, not the relative `fp_bounded` the general `_SAFE_CODE_FIXTURES` use: baseline
# sql_injection-fp>0 = the over-flag is not Haiku-specific (INCONCLUSIVE); baseline=0 AND
# candidate>0 = the #041 over-flag reproduced; both 0 = clean. (Relative `fp_bounded` is for
# `_SAFE_CODE_FIXTURES` — fine when a shared over-flag like eval_in_test's `eval()` is acceptable.)
_PARAMETERIZED_QUERY_FIXTURE = "tests/eval/fixtures/mock_github/safe_parameterized_query.json"
_PARAMETERIZED_QUERY_PSYCOPG_FIXTURE = (
    "tests/eval/fixtures/mock_github/safe_parameterized_query_psycopg.json"
)
_PARAMETERIZED_QUERY_NAMED_FIXTURE = (
    "tests/eval/fixtures/mock_github/safe_parameterized_query_named.json"
)
_PARAMETERIZED_QUERY_ORM_FIXTURE = (
    "tests/eval/fixtures/mock_github/safe_parameterized_query_orm.json"
)
_REGRESSION_FIXTURES: tuple[str, ...] = (
    _PARAMETERIZED_QUERY_FIXTURE,
    _PARAMETERIZED_QUERY_PSYCOPG_FIXTURE,
    _PARAMETERIZED_QUERY_NAMED_FIXTURE,
    _PARAMETERIZED_QUERY_ORM_FIXTURE,
)

# Regression-track HOLD-OUTS — safe, correctly-parameterized queries in placeholder styles the
# analyze-v3 prompt NEVER MENTIONS (it only shows `%s`/`%(name)s`): SQLAlchemy `text()` + `:name`
# bind, sqlite3 `?` qmark, asyncpg `$1` positional. FULLY UNSEEN — the STRONGEST anti-overfit signal
# (stronger than the recall SQLi hold-outs, which the prompt names but does not exemplify). The
# demonstrated idioms in
# `_REGRESSION_FIXTURES` above have shapes the prompt exemplifies, so a CLEAN verdict there proves
# the model follows guidance — NOT that it generalized. These hold-outs test the generalization: a
# CLEAN verdict here means the model applied the rule (placeholder + separate args = not injection)
# to a shape it was never shown — the real evidence the fix isn't overfit.
_PARAM_HOLDOUT_SQLALCHEMY_FIXTURE = (
    "tests/eval/fixtures/mock_github/safe_param_holdout_sqlalchemy.json"
)
_PARAM_HOLDOUT_SQLITE_FIXTURE = "tests/eval/fixtures/mock_github/safe_param_holdout_sqlite.json"
_PARAM_HOLDOUT_ASYNCPG_FIXTURE = "tests/eval/fixtures/mock_github/safe_param_holdout_asyncpg.json"
_REGRESSION_HOLDOUT_FIXTURES: tuple[str, ...] = (
    _PARAM_HOLDOUT_SQLALCHEMY_FIXTURE,
    _PARAM_HOLDOUT_SQLITE_FIXTURE,
    _PARAM_HOLDOUT_ASYNCPG_FIXTURE,
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


def _judged_response_with(specs: list[tuple[FindingType, int, int]]) -> str:
    """A scripted analyze response emitting several JUDGED findings — one per
    `(finding_type, line_start, line_end)`. Like `_judged_response_for` but multi-finding,
    for the type-scoping tests that need a MIX of sql_injection and non-sql_injection extras
    on one safe fixture (so `_sqli_fp_count` can be shown to isolate the tracked caveat)."""
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": ft.value,
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": f"{ft.value} finding",
                    "description": "scripted finding for the type-scoping test.",
                    "evidence": "e",
                    "line_start": ls,
                    "line_end": le,
                    "trace_candidates": [],
                }
                for ft, ls, le in specs
            ]
        }
    )


def _sqli_fp_count(grade: GradeResult) -> int:
    """Count ONLY `sql_injection` false positives (extras) in a grade. The regression track
    is TYPE-SCOPED because the DECISIONS#041 caveat is type-specific (Haiku mislabeling a
    parameterized query as SQL injection); counting total fp of any type lets an unrelated
    baseline over-flag (e.g. missing_error_handling) wrongly force a NON-DISCRIMINATING verdict."""
    return sum(1 for f in grade.extra if f.finding_type == FindingType.SQL_INJECTION)


def _regression_verdict(baseline_sqli_fp: int, candidate_sqli_fp: int) -> tuple[str, bool, str]:
    """Type-scoped regression verdict over sql_injection FP counts → (verdict_text, ok, label).
    Three states: baseline sql_injection-fp>0 = the baseline itself over-flags a parameterized
    query as SQLi, so the over-flag is NOT Haiku-specific (INCONCLUSIVE, ok=False); baseline=0
    AND candidate>0 = the #041 Haiku over-flag REPRODUCED (ok=False); both 0 = CLEAN (ok=True)."""
    if baseline_sqli_fp > 0:
        return (
            f"NON-DISCRIMINATING — baseline (Sonnet) itself emitted {baseline_sqli_fp} "
            "sql_injection FP on a parameterized query; the over-flag is not Haiku-specific",
            False,
            "INCONCLUSIVE (baseline sql_injection-fp>0)",
        )
    if candidate_sqli_fp > 0:
        return (
            "#041 CAVEAT REPRODUCED — baseline clean of sql_injection FPs, candidate (Haiku) "
            f"emitted {candidate_sqli_fp}",
            False,
            "REPRODUCED (baseline clean, candidate sql_injection-fp>0)",
        )
    return ("CLEAN — neither model emitted a sql_injection FP this run", True, "")


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


@pytest.mark.parametrize(
    "fixture_path", _SAFE_CODE_FIXTURES + _REGRESSION_FIXTURES + _REGRESSION_HOLDOUT_FIXTURES
)
def test_state_from_safe_code_fixture_builds(fixture_path: str) -> None:
    """The safe-code fixtures (precision + regression-track) build an analyzable STANDARD-tier
    state — multi-dimension triage read from the fixture, tier overridden to STANDARD."""
    state = state_from_eval_fixture(fixture_path)
    assert state.pr_context.changed_files
    assert state.triage_result is not None
    path = state.pr_context.changed_files[0].path
    assert state.triage_result.file_tiers[path] is ReviewTier.STANDARD
    assert len(state.triage_result.relevant_dimensions) >= 1


@pytest.mark.parametrize("fixture_path", _SAFE_CODE_FIXTURES)
@pytest.mark.asyncio
async def test_safe_code_clean_under_clean_model_scores_zero_fp(fixture_path: str) -> None:
    """Precision dimension, zero-spend: a model that returns NO finding on safe code scores
    0 false positives against the empty ground truth, so fp_bounded holds — the precision
    PASS case. The `provider.calls` assertions pin that analyze actually RAN on the safe-code
    state (each model's `complete()` was invoked), so this is not a vacuous pass over an empty
    state. The over-flag FAIL case is grading's precision gate test."""
    baseline = _ScriptedProvider(_MISSES_RESPONSE)
    candidate = _ScriptedProvider(_MISSES_RESPONSE)
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(fixture_path),
        (),  # safe code — no real finding, so any finding would be a false positive
        baseline_provider=baseline,
        baseline_model="claude-sonnet-4-6",
        candidate_provider=candidate,
        candidate_model="claude-haiku-4-5",
    )
    assert baseline.calls and candidate.calls  # analyze actually invoked each model
    assert cmp.baseline.n_false_positives == 0
    assert cmp.candidate.n_false_positives == 0
    assert cmp.fp_bounded is True  # neither over-flagged clean code → precision green


@pytest.mark.asyncio
async def test_parameterized_query_candidate_overflag_fails_precision() -> None:
    """Regression track (DECISIONS#041): on the safe parameterized-query fixture, a
    CANDIDATE-ONLY false positive — Haiku flags the correctly-parameterized query as a
    `sql_injection` while Sonnet stays clean — FAILS the precision dimension. This is the
    exact #041 caveat (baseline clean, candidate over-flags), and the gate catches it:
    candidate fp (1) > baseline fp (0). The complementary SHARED-over-flag case (both fp>0,
    which the relative gate would wrongly pass) is handled by the opt-in run's absolute
    baseline-clean guard, not this test."""
    overflag = _judged_response_for(
        ExpectedFinding(
            file_path="directory/users.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        )
    )
    baseline = _ScriptedProvider(_MISSES_RESPONSE)  # Sonnet: clean
    candidate = _ScriptedProvider(overflag)  # Haiku: over-flags the parameterized query
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(_PARAMETERIZED_QUERY_FIXTURE),
        (),  # safe code — the sql_injection is an unambiguous false positive
        baseline_provider=baseline,
        baseline_model="claude-sonnet-4-6",
        candidate_provider=candidate,
        candidate_model="claude-haiku-4-5",
    )
    assert baseline.calls and candidate.calls  # analyze actually invoked each model
    assert cmp.baseline.n_false_positives == 0  # baseline clean — the discriminating precondition
    assert cmp.candidate.n_false_positives == 1  # the over-flag reproduced
    assert cmp.fp_bounded is False  # precision FAILS — the gate catches the candidate-only FP


def test_regression_verdict_is_type_scoped_three_states() -> None:
    """`_regression_verdict` has exactly three states keyed on sql_injection FP counts. The
    discriminating property: it takes ALREADY-type-scoped counts, so a baseline that over-flags
    only an UNRELATED dimension reaches it as 0 (via `_sqli_fp_count`) and yields CLEAN — where
    the old total-fp gate would have read INCONCLUSIVE."""
    _, ok_incon, label_incon = _regression_verdict(1, 0)  # baseline emits a SQLi FP itself
    assert ok_incon is False
    assert label_incon.startswith("INCONCLUSIVE")

    _, ok_repro, label_repro = _regression_verdict(0, 1)  # baseline clean, candidate over-flags
    assert ok_repro is False
    assert label_repro.startswith("REPRODUCED")

    _, ok_clean, label_clean = _regression_verdict(0, 0)  # neither emits a SQLi FP
    assert ok_clean is True
    assert label_clean == ""

    # baseline-first precedence: when BOTH emit a SQLi FP, the baseline check wins (the
    # over-flag is not Haiku-specific) — pins the if-block order against a future reorder.
    _, _, label_both = _regression_verdict(1, 1)
    assert label_both.startswith("INCONCLUSIVE")


@pytest.mark.asyncio
async def test_regression_track_type_scoping_ignores_unrelated_baseline_overflag() -> None:
    """The reason for type-scoping (FUP-159): on the safe parameterized-query fixture, a BASELINE
    that over-flags an UNRELATED dimension (missing_error_handling) while the CANDIDATE adds the
    sql_injection over-flag must read REPRODUCED — not INCONCLUSIVE. Counting TOTAL fp, the
    baseline's fp=1 would force INCONCLUSIVE and hide the candidate's tracked regression. Counting
    only sql_injection FPs isolates the caveat. (get_user spans lines 4-7: line 5 the cursor
    context, line 6 the cursor.execute — both admitted within the changed scope unit.)"""
    baseline_resp = _judged_response_with([(FindingType.MISSING_ERROR_HANDLING, 5, 5)])
    candidate_resp = _judged_response_with(
        [(FindingType.SQL_INJECTION, 6, 6), (FindingType.MISSING_ERROR_HANDLING, 5, 5)]
    )
    baseline = _ScriptedProvider(baseline_resp)
    candidate = _ScriptedProvider(candidate_resp)
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(_PARAMETERIZED_QUERY_FIXTURE),
        (),  # safe code — every finding is a false positive
        baseline_provider=baseline,
        baseline_model="claude-sonnet-4-6",
        candidate_provider=candidate,
        candidate_model="claude-haiku-4-5",
    )
    assert baseline.calls and candidate.calls  # analyze actually invoked each model
    # Total fp shows the noise; the type-scoped count isolates the tracked caveat.
    assert cmp.baseline.n_false_positives == 1  # the unrelated missing_error_handling
    assert cmp.candidate.n_false_positives == 2  # sql_injection + missing_error_handling
    assert _sqli_fp_count(cmp.baseline) == 0  # baseline emitted NO sql_injection FP
    assert _sqli_fp_count(cmp.candidate) == 1  # exactly the tracked caveat
    _, ok, label = _regression_verdict(_sqli_fp_count(cmp.baseline), _sqli_fp_count(cmp.candidate))
    assert ok is False
    assert label.startswith("REPRODUCED")  # baseline noise did NOT force INCONCLUSIVE


def test_holdout_sets_are_registered_and_disjoint() -> None:
    """The anti-overfit hold-outs must be wired in and genuinely held out — guards against one
    silently dropping from the real-model run, which would quietly weaken the generalization claim
    back to "passes on the shapes the prompt demonstrates." Pins: (1) the regression hold-outs
    (unseen placeholder styles) are disjoint from the DEMONSTRATED `_REGRESSION_FIXTURES`; (2) the
    three recall hold-outs are registered as `sql_injection` in the recall ground truth; (3) every
    hold-out fixture file exists on disk."""
    # By property, not headcount (adding a hold-out must not break this; dropping a NAMED one must):
    # the three placeholder-style hold-outs are wired, and the regression hold-out set is disjoint
    # from BOTH the demonstrated set and the safe-code set (landing in either corrupts its verdict).
    for fixture in (
        _PARAM_HOLDOUT_SQLALCHEMY_FIXTURE,
        _PARAM_HOLDOUT_SQLITE_FIXTURE,
        _PARAM_HOLDOUT_ASYNCPG_FIXTURE,
    ):
        assert fixture in _REGRESSION_HOLDOUT_FIXTURES
    assert set(_REGRESSION_HOLDOUT_FIXTURES).isdisjoint(_REGRESSION_FIXTURES)
    assert set(_REGRESSION_HOLDOUT_FIXTURES).isdisjoint(_SAFE_CODE_FIXTURES)

    sqli_holdouts = (
        _SQLI_HOLDOUT_FSTRING_FIXTURE,
        _SQLI_HOLDOUT_FORMAT_FIXTURE,
        _SQLI_HOLDOUT_CONCAT_FIXTURE,
    )
    # SQLi recall hold-outs carry a REAL finding — they must NOT sit in any safe/regression set,
    # where the empty ground truth would flip the real finding into an apparent false positive.
    assert set(sqli_holdouts).isdisjoint(
        _REGRESSION_FIXTURES + _REGRESSION_HOLDOUT_FIXTURES + _SAFE_CODE_FIXTURES
    )
    for fixture in sqli_holdouts:
        ground_truth = _GROUND_TRUTH_BY_FIXTURE[fixture]
        assert len(ground_truth) == 1
        assert ground_truth[0].finding_type is FindingType.SQL_INJECTION

    for fixture in _REGRESSION_HOLDOUT_FIXTURES + sqli_holdouts:
        assert os.path.exists(fixture), fixture


@pytest.mark.parametrize("fixture_path", list(_GROUND_TRUTH_BY_FIXTURE))
@pytest.mark.asyncio
async def test_real_fixture_content_through_analyze_catches_regression(fixture_path: str) -> None:
    """END-TO-END zero-spend over EACH recall fixture (all nine in `_GROUND_TRUTH_BY_FIXTURE` —
    SQLi, auth-bypass, missing-error-handling, N+1, path-traversal, missing-input-validation, plus
    three held-out SQLi forms: f-string / str.format / concatenation):
    a scripted "Sonnet" that returns the known finding scores recall 1.0; a scripted "Haiku"
    that misses it scores 0.0 and FAILS the gate. The STATE is the real vulnerable code (built
    by state_from_eval_fixture); only the provider is faked — so the real run differs only by
    swapping in the AnthropicProvider. Parametrized over every fixture so analyze accepting a
    finding at each known line (e.g. views.py:5) is verified for all of them. A JUDGED stand-in
    needs no tree-sitter query."""
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

    THREE DIMENSIONS, measured on the fixtures each is valid for:
    - RECALL, over known-vulnerability fixtures (`_GROUND_TRUTH_BY_FIXTURE`): does the
      candidate catch the known finding? Gated on `recall_held` + `baseline_valid`. FP is
      ADVISORY here, NOT gated: a real run showed the "extras" on vulnerable files are
      usually LEGITIMATE second findings the single-entry ground truth didn't encode (an
      unvalidated-input finding alongside the SQLi), so fp on a vulnerable file is an
      unreliable over-flag signal. Read the printed extra detail, don't gate on it. The set now
      also carries three HOLD-OUT SQLi forms (f-string / `str.format` / `+` concatenation) the
      analyze-v3 prompt names but never exemplifies — a recall MISS there means the
      parameterized-query remediation over-suppressed real SQLi (generalization, not overfit).
    - PRECISION, over safe-code fixtures (`_SAFE_CODE_FIXTURES`, no real finding): does the
      candidate over-flag clean code MORE than the baseline? Gated on `fp_bounded` (a RELATIVE
      bound — fine where a shared over-flag is acceptable, e.g. eval()-in-test).
    - REGRESSION-TRACK, over `_REGRESSION_FIXTURES` (DEMONSTRATED idioms — shapes the prompt
      exemplifies) AND `_REGRESSION_HOLDOUT_FIXTURES` (HOLD-OUT placeholder styles it never shows:
      SQLAlchemy `:name`, sqlite `?`, asyncpg `$1`), TYPE-SCOPED to sql_injection FPs via
      `_sqli_fp_count` and read with an ABSOLUTE baseline-clean gate: baseline sql_injection-fp>0 is
      non-discriminating; baseline=0 AND candidate>0 is the DECISIONS#041 caveat reproduced. CLEAN
      on the HOLD-OUTS is the anti-overfit evidence — the model applied the rule to a shape it was
      never shown, not just the demonstrated ones. Scoping to sql_injection keeps an unrelated
      baseline over-flag from forcing INCONCLUSIVE; the relative `fp_bounded` gate alone would let a
      SHARED over-flag pass, hence the baseline-clean check.

    Run the analyze node under Sonnet (baseline — the DEEP model + the pre-flip STANDARD
    model) and Haiku (candidate — now the shipped STANDARD default, DECISIONS#041) per
    scenario; bounded at 2 analyze calls/scenario over
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
    # Baseline is `analyze_model` (today's Sonnet top-tier analyze behavior), NOT
    # `standard_analyze_model` — the latter is the very knob the flip changes, so reading it
    # would compare the candidate against ITSELF if the operator already set
    # OUTRIDER_MODEL_STANDARD_ANALYZE_MODEL=claude-haiku-4-5. The methodological question is
    # "does Haiku preserve STANDARD findings vs today's Sonnet analyze?".
    baseline_model = cfg.analyze_model
    candidate_model = "claude-haiku-4-5"  # the shipped STANDARD default (DECISIONS#041)
    # Belt-and-suspenders: fail loudly on a meaningless self-comparison (e.g. analyze_model
    # env-overridden to Haiku). Normalized so a dated pin (…-20251001) can't sneak past.
    # Checked BEFORE constructing the provider so a guard-fire can't leak an unclosed client.
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the comparison would prove nothing about Sonnet-vs-Haiku. Point "
            "OUTRIDER_MODEL_ANALYZE_MODEL at Sonnet (or unset it) for the evidence run."
        )
    # persister MUST be a real LLMExchangePersister, not None: AnthropicProvider.complete()
    # is fail-closed on persister=None (raises before the SDK call). The comparison reads
    # findings from analyze's return, so a no-op persister is the right wiring here.
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )
    # (fixture, dimension, ok, fail_label): fail_label distinguishes WHY a non-ok verdict is
    # non-ok in the end summary — a regression scenario is "INCONCLUSIVE" (the baseline emitted a
    # sql_injection FP itself) vs "REPRODUCED" (only the candidate did); recall/precision use plain
    # "FAILED". Empty label for green verdicts (never printed).
    gate_results: list[tuple[str, str, bool, str]] = []
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
            gate_results.append(
                (fixture_path, "recall", cmp.recall_held and cmp.baseline_valid, "FAILED")
            )
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
            gate_results.append((fixture_path, "precision", cmp.fp_bounded, "FAILED"))
            assert cmp.baseline is not None  # the run completed
        # REGRESSION-TRACK dimension — TYPE-SCOPED to sql_injection FPs (`_sqli_fp_count`), read
        # with an ABSOLUTE baseline-clean gate. The caveat is type-specific (DECISIONS#041: Haiku
        # mislabels a parameterized query as SQLi), so an unrelated baseline over-flag must NOT
        # blind the track. `_regression_verdict` carries the 3-state logic (INCONCLUSIVE /
        # REPRODUCED / clean). Runs over the DEMONSTRATED idioms (`_REGRESSION_FIXTURES`) AND the
        # HOLD-OUTS (`_REGRESSION_HOLDOUT_FIXTURES`, placeholder styles the prompt never shows) —
        # CLEAN on the hold-outs is the anti-overfit evidence. The fixture path names the set.
        for fixture_path in _REGRESSION_FIXTURES + _REGRESSION_HOLDOUT_FIXTURES:
            cmp = await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                (),
                baseline_provider=provider,
                baseline_model=baseline_model,
                candidate_provider=provider,
                candidate_model=candidate_model,
            )
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
            verdict, ok, fail_label = _regression_verdict(
                _sqli_fp_count(cmp.baseline), _sqli_fp_count(cmp.candidate)
            )
            print(f"  REGRESSION-TRACK verdict: {verdict}")  # noqa: T201 — operator diagnostic
            gate_results.append((fixture_path, "regression", ok, fail_label))
            assert cmp.baseline is not None  # the run completed
        # REPORT-ONLY summary — pytest "passed" means the run completed, NOT the gate verdict.
        # Each non-green line carries its own label (recall/precision -> "FAILED"; regression ->
        # "INCONCLUSIVE …" or "REPRODUCED …") so a skimmer can't misread an inconclusive
        # regression as a Haiku failure.
        failed = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
        green = len(gate_results) - len(failed)
        print(  # noqa: T201 — operator gate summary
            "\n"
            + "=" * 72
            + "\nGATE SUMMARY — REPORT ONLY: pytest 'passed' means the run COMPLETED, NOT"
            + "\nthat the gate passed. Adjudicate the per-scenario verdicts above."
            + f"\n  {green}/{len(gate_results)} dimension-verdicts green "
            + "(recall→recall; safe-code→relative FP; "
            + "regression-track→absolute baseline-clean, sql_injection-scoped)."
            + "".join(f"\n  {dim.upper()} {label}: {fx}" for fx, dim, label in failed)
            + "\n"
            + "=" * 72
        )
    finally:
        await provider.aclose()
