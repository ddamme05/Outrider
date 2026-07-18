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
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from outrider.audit.events import AnalyzeCompletedEvent
    from outrider.llm.base import LLMProvider
    from outrider.policy.findings import ReviewFinding

import pytest

from outrider.llm.base import (
    LLMAuthError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
)
from outrider.policy import FindingSeverity, FindingType, lookup_severity
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.triage_result import ReviewDimension, ReviewTier, RiskLevel, TriageResult

from .grading import ExpectedFinding, GradeResult, ModelComparison, grade
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

# JUDGED finding-TYPE coverage (broadening, FUP-196): five vulnerability classes whose
# recall is purely MODEL-DEPENDENT — none triggers an OBSERVED structural query, so a
# miss here is a real model miss, not one a structural backstop would hide. SSRF and
# weak-password-hash have no query at all; command-injection and weak-crypto DO have
# OBSERVED queries, so these fixtures use forms the queries deliberately DON'T match
# (subprocess list-form without `shell=True`; the CAST cipher the broken-cipher query
# excludes by design), keeping them on the JUDGED path. Each verified to yield NOTHING
# from the OBSERVED producer under an empty model (the no-structural-floor check).
_SSRF_FIXTURE = "tests/eval/fixtures/mock_github/ssrf_user_host.json"
_WEAK_PASSWORD_HASH_FIXTURE = "tests/eval/fixtures/mock_github/weak_password_hash_md5.json"  # noqa: S105 (fixture path label, not a password)
_COMMAND_INJECTION_FIXTURE = "tests/eval/fixtures/mock_github/cmd_injection_subprocess.json"
_WEAK_CRYPTO_FIXTURE = "tests/eval/fixtures/mock_github/weak_crypto_cast.json"
_INSECURE_RANDOMNESS_FIXTURE = "tests/eval/fixtures/mock_github/insecure_random_token.json"
# Dangerous-eval command_injection reached INDIRECTLY (getattr(builtins, "eval")), so the
# OBSERVED command_injection_eval_exec query (which pins function: (identifier) to eval/exec)
# does not fire — the model must REASON about the indirection, not lean on the structural
# backstop. A second command_injection fixture, by a distinct mechanism from the subprocess one.
_COMMAND_INJECTION_EVAL_FIXTURE = "tests/eval/fixtures/mock_github/cmd_injection_eval_indirect.json"
# The OBSERVED-free invariant (recall is pure model signal) is asserted directly by
# test_judged_fixture_has_no_observed_floor for each of these — not left to a comment.
_JUDGED_TYPE_FIXTURES = (
    _SSRF_FIXTURE,
    _WEAK_PASSWORD_HASH_FIXTURE,
    _COMMAND_INJECTION_FIXTURE,
    _COMMAND_INJECTION_EVAL_FIXTURE,
    _WEAK_CRYPTO_FIXTURE,
    _INSECURE_RANDOMNESS_FIXTURE,
)

# Recall HOLD-OUTS — real SQLi across injection forms. Hold-out strength is PROMPT-VERSION-
# RELATIVE: under analyze-v3 (the DECISIONS#041 evidence runs) all four were named-but-never-
# exemplified. From analyze-v4 onward, SYSTEM_PROMPT_EXEMPLARS DEMONSTRATES f-string, `+`
# concatenation, and ORM `raw(f"...")` as FLAG examples — for those forms these fixtures now
# test guidance-following, not generalization; `str.format` remains the only never-exemplified
# injection form (the residual generalization signal). The model MUST still flag all of these;
# a miss means the parameterized-query remediation over-suppressed real SQLi.
_SQLI_HOLDOUT_FSTRING_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_fstring.json"
_SQLI_HOLDOUT_FORMAT_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_format.json"
_SQLI_HOLDOUT_CONCAT_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_concat.json"
_SQLI_HOLDOUT_ORM_FSTRING_FIXTURE = "tests/eval/fixtures/mock_github/sqli_holdout_orm_fstring.json"

# These SQLi hold-out forms are ALSO caught by the deterministic OBSERVED
# `python.sql_injection_string_concat` query (f-string interpolation +
# `str.format()`), so a model that misses them still yields PRODUCT recall 1.0 —
# the structural OBSERVED tier backstops the model (Cost Lever 3 / DECISIONS.md#048).
# The concat (variable-on-left) and ORM `.raw()` forms are NOT covered by the
# query, so they still exercise true model-recall regression through the pipeline.
_OBSERVED_BACKSTOPPED_SQLI: frozenset[str] = frozenset(
    {_SQLI_HOLDOUT_FSTRING_FIXTURE, _SQLI_HOLDOUT_FORMAT_FIXTURE}
)

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
    _SQLI_HOLDOUT_ORM_FSTRING_FIXTURE: (
        ExpectedFinding(
            file_path="crm/queries.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        ),
    ),
    # JUDGED finding-TYPE coverage (broadening): five model-dependent classes beyond the
    # SQLi-heavy set above. Each line/type is verified against analyze's admit path (the
    # scripted finding surfaces at exactly this span) and against the OBSERVED producer
    # (empty model → no finding, so recall here is pure model signal).
    _SSRF_FIXTURE: (
        ExpectedFinding(
            file_path="app/fetch.py",
            line_start=7,
            line_end=7,
            finding_type=FindingType.SSRF,
            severity=lookup_severity(FindingType.SSRF),
        ),
    ),
    _WEAK_PASSWORD_HASH_FIXTURE: (
        ExpectedFinding(
            file_path="accounts/auth.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.WEAK_PASSWORD_HASH,
            severity=lookup_severity(FindingType.WEAK_PASSWORD_HASH),
        ),
    ),
    _COMMAND_INJECTION_FIXTURE: (
        ExpectedFinding(
            file_path="ops/net.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.COMMAND_INJECTION,
            severity=lookup_severity(FindingType.COMMAND_INJECTION),
        ),
    ),
    _COMMAND_INJECTION_EVAL_FIXTURE: (
        ExpectedFinding(
            file_path="app/calc.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.COMMAND_INJECTION,
            severity=lookup_severity(FindingType.COMMAND_INJECTION),
        ),
    ),
    _WEAK_CRYPTO_FIXTURE: (
        ExpectedFinding(
            file_path="vault/cipher.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.WEAK_CRYPTO,
            severity=lookup_severity(FindingType.WEAK_CRYPTO),
        ),
    ),
    _INSECURE_RANDOMNESS_FIXTURE: (
        ExpectedFinding(
            file_path="accounts/session.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.INSECURE_RANDOMNESS,
            severity=lookup_severity(FindingType.INSECURE_RANDOMNESS),
        ),
    ),
    # suite-v2 FLAG-side coverage (specs/2026-07-15-exemplar-coverage-fixture-suite-v2.md):
    # one positive per previously-unmeasured EXEMPLARS type. Behaviorally distinct from the
    # prompt fences by authoring rule — same rule boundary, different code/control-flow shape
    # (enforced literally by the fence-copy guard in test_exemplar_baseline.py; structural
    # near-copies are reviewer discipline). Lines are HEAD source lines, generator-computed.
    "tests/eval/fixtures/mock_github/xss_search_echo.json": (
        ExpectedFinding(
            file_path="search/views.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.XSS,
            severity=lookup_severity(FindingType.XSS),
        ),
    ),
    "tests/eval/fixtures/mock_github/hardcoded_secret_release_token.json": (
        ExpectedFinding(
            file_path="tools/release.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.HARDCODED_SECRET,
            severity=lookup_severity(FindingType.HARDCODED_SECRET),
        ),
    ),
    "tests/eval/fixtures/mock_github/blocking_async_export_poll.json": (
        ExpectedFinding(
            file_path="exports/poller.py",
            line_start=13,
            line_end=13,
            finding_type=FindingType.BLOCKING_CALL_IN_ASYNC,
            severity=lookup_severity(FindingType.BLOCKING_CALL_IN_ASYNC),
        ),
    ),
    "tests/eval/fixtures/mock_github/unused_import_added_csv.json": (
        ExpectedFinding(
            file_path="audit/manifest.py",
            line_start=9,
            line_end=9,
            finding_type=FindingType.UNUSED_IMPORT,
            severity=lookup_severity(FindingType.UNUSED_IMPORT),
        ),
    ),
    "tests/eval/fixtures/mock_github/missing_test_shipping_rates.json": (
        ExpectedFinding(
            file_path="pricing/shipping.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.MISSING_TEST,
            severity=lookup_severity(FindingType.MISSING_TEST),
        ),
    ),
    "tests/eval/fixtures/mock_github/deprecated_api_event_loop.json": (
        ExpectedFinding(
            file_path="cli/sync.py",
            line_start=7,
            line_end=7,
            finding_type=FindingType.DEPRECATED_API,
            severity=lookup_severity(FindingType.DEPRECATED_API),
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
# The fixed-host SSRF over-flag trap — named because the GLM scorecard flagged it (the one
# safe-code false positive), so the miss-diagnostic targets it directly.
_SSRF_FIXED_HOST_SAFE_FIXTURE = "tests/eval/fixtures/mock_github/ssrf_fixed_host_safe.json"
_SAFE_CODE_FIXTURES: tuple[str, ...] = (
    "tests/eval/fixtures/mock_github/safe_refactor.json",
    "tests/eval/fixtures/mock_github/eval_in_test_fixture.json",
    # Over-flag TRAPS paired with the new JUDGED recall types: a fixed-host fetch (user
    # value confined to the URL path, not the host) and a fixed-argv subprocess (shell=False,
    # no caller input). Both look superficially like SSRF / command-injection but are safe;
    # any finding is a clean false positive — the precision counterpart to the recall fixtures.
    _SSRF_FIXED_HOST_SAFE_FIXTURE,
    "tests/eval/fixtures/mock_github/safe_subprocess_fixed_argv.json",
    # suite-v2 don't-flag lookalikes (one per newly covered EXEMPLARS type): each targets the
    # suppressive half of its block's contract — escaped echo (xss), env-sourced credential
    # (hardcoded_secret), asyncio.to_thread delegation (blocking_call_in_async), __all__
    # re-exports + TYPE_CHECKING import (unused_import), trivial delegations (missing_test),
    # old-but-stable stdlib (deprecated_api). Any finding is a false positive.
    "tests/eval/fixtures/mock_github/safe_xss_escaped_echo.json",
    "tests/eval/fixtures/mock_github/safe_secret_env_default.json",
    "tests/eval/fixtures/mock_github/safe_async_to_thread_hash.json",
    "tests/eval/fixtures/mock_github/safe_reexport_init_all.json",
    "tests/eval/fixtures/mock_github/safe_trivial_delegations.json",
    "tests/eval/fixtures/mock_github/safe_stable_old_stdlib.json",
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

# Regression-track HOLD-OUTS — safe, correctly-parameterized queries: SQLAlchemy `text()` +
# `:name` bind, sqlite3 `?` qmark, asyncpg `$1` positional. Hold-out strength is PROMPT-VERSION-
# RELATIVE: under analyze-v3 (the DECISIONS#041 evidence runs) these styles were NEVER MENTIONED
# (only `%s`/`%(name)s` shown) — the strongest anti-overfit signal. From analyze-v4 onward the
# exemplars NAME `?`/`:name`/`$1` as binding styles and DEMONSTRATE the `:name` form, so for
# future runs only `?` and `$1` remain demonstration-free (named-not-exemplified), and a CLEAN
# verdict tests guidance-following more than generalization. The demonstrated idioms in
# `_REGRESSION_FIXTURES` above have shapes the prompt exemplifies, so a CLEAN verdict there
# proves the model follows guidance — NOT that it generalized.
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
        from outrider.llm.anthropic_provider import (
            _ANTHROPIC_CONTRACT_DIGEST,
            _ANTHROPIC_PROFILE_ID,
        )

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
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
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


class _CapturingExchangePersister:
    """Like `_NoOpExchangePersister` but RECORDS each call's raw completion text, so a
    diagnostic can SEE what the model actually emitted on a recall miss (the comparison path
    otherwise discards the raw response). Only the REAL providers call `persist`; the scripted
    provider does not, so `completions` populates only on a real run."""

    def __init__(self) -> None:
        self.completions: list[str] = []

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        self.completions.append(getattr(response, "text", ""))


class _TokenRecordingExchangePersister:
    """Like `_NoOpExchangePersister` but RECORDS each real analyze call's token usage
    (model + input/output/cached tokens), tagged with the scenario currently under
    comparison, so a real run can size analyze `MAX_TOKENS` against MEASURED Sonnet 5
    output tokens (FUP-207) rather than the launch-doc's ~30%-denser estimate. Only the
    REAL providers call `persist`; the scripted provider does not, so `records` populates
    only on a real run. `current_fixture` is set by the evidence runner before each scenario
    so output tokens correlate with that scenario's emitted-finding count."""

    def __init__(self) -> None:
        self.current_fixture: str = ""
        # (fixture, model, input_tokens, output_tokens, cached_tokens) per analyze call.
        self.records: list[tuple[str, str, int, int, int]] = []

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        self.records.append(
            (
                self.current_fixture,
                getattr(event, "model", ""),
                getattr(event, "input_tokens", 0),
                getattr(event, "output_tokens", 0),
                getattr(event, "cached_tokens", 0),
            )
        )


def _format_token_sizing(
    records: list[tuple[str, str, int, int, int]],
    emitted: dict[tuple[str, str], int],
    baseline_model: str,
    candidate_model: str,
) -> str:
    """Roll recorded per-call token usage into a per-model OUTPUT-token sizing block for
    FUP-207: max output tokens seen, the per-response ENVELOPE overhead (median output where
    the model emitted 0 findings), and the MARGINAL tokens-per-finding (median over
    findings-bearing scenarios of (output - envelope) / n_findings). Analyze `MAX_TOKENS`
    should hold the target finding count as `envelope + target × marginal`. The harness only
    MEASURES; the sizing edit is the FUP-207 exit. Records are aggregated per (fixture, model)
    so a multi-file scenario's calls sum before the per-finding divide."""
    import statistics  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415

    out_by: dict[tuple[str, str], int] = defaultdict(int)
    for fixture, model, _in, out, _cached in records:
        out_by[(fixture, model)] += out

    lines = ["", "=" * 72, "OUTPUT-TOKEN SIZING (FUP-207) — MEASURED, report-only:"]
    for model in (baseline_model, candidate_model):
        # Only COMPLETED comparisons contribute: a scenario whose provider call errored
        # mid-comparison persists a token record but leaves `emitted` unfilled, and counting
        # that orphan as a 0-finding envelope sample would skew the block (its output tokens
        # aren't a real zero-finding response). Filter to keys present in `emitted`.
        keys = [k for k in out_by if k[1] == model and k in emitted]
        dropped = sum(1 for k in out_by if k[1] == model and k not in emitted)
        if not keys:
            lines.append(f"  {model}: no completed analyze calls")
            continue
        outs = [out_by[k] for k in keys]
        envelope_samples = [out_by[k] for k in keys if emitted[k] == 0]
        envelope = int(statistics.median(envelope_samples)) if envelope_samples else 0
        per_finding = [(out_by[k] - envelope) / emitted[k] for k in keys if emitted[k] > 0]
        marginal = statistics.median(per_finding) if per_finding else 0.0
        scenarios_label = f"scenarios={len(keys)}"
        if dropped:
            scenarios_label += f" ({dropped} errored, excluded)"
        lines.append(
            f"  {model}: {scenarios_label} "
            f"output[min/median/max]={min(outs)}/{int(statistics.median(outs))}/{max(outs)} "
            f"envelope~{envelope} marginal~{marginal:.0f} tok/finding"
        )
    lines.append(
        "  → size analyze MAX_TOKENS = envelope + target_findings × marginal (candidate = "
        "the DEEP model); recompute the docstring's '~N findings' at that rate."
    )
    lines.append("=" * 72)
    return "\n".join(lines)


def test_format_token_sizing_excludes_errored_scenarios() -> None:
    """A scenario whose provider call errored mid-comparison persists a token record but
    leaves `emitted` unfilled (only completed comparisons fill it). That orphan must NOT be
    counted as a 0-finding envelope sample — doing so would skew the block. Reverting the
    `k in emitted` filter turns envelope~8 into envelope~254 here, so this pins the fix."""
    baseline, candidate = "claude-sonnet-4-6", "claude-sonnet-5"
    records = [
        ("fx_a.json", candidate, 100, 600, 0),  # completed, 2 findings
        ("fx_b.json", candidate, 100, 8, 0),  # completed, 0 findings -> the real envelope
        ("fx_err.json", candidate, 100, 500, 0),  # ORPHAN -- errored, no `emitted` entry
    ]
    emitted = {("fx_a.json", candidate): 2, ("fx_b.json", candidate): 0}  # fx_err absent
    block = _format_token_sizing(records, emitted, baseline, candidate)

    assert "scenarios=2 (1 errored, excluded)" in block  # orphan excluded + surfaced
    assert "envelope~8" in block  # NOT the 500-tok orphan (would be ~254 unfiltered)


async def diagnose_scenario_stability(
    fixture_path: str,
    *,
    expected: tuple[ExpectedFinding, ...],
    provider: LLMProvider,
    model: str,
    capturing: _CapturingExchangePersister,
    runs: int,
) -> dict[str, object]:
    """Rerun ONE scenario `runs` times under `provider`, capturing the raw model response on
    each FAILING run so the failure mode is inspectable, and return the pass count + a verdict.
    A run FAILS differently by fixture kind: a recall fixture (non-empty `expected`) fails on a
    MISS (recall < 1.0); a safe fixture (empty `expected`) fails on a FALSE POSITIVE (any
    finding emitted). This separates a STOCHASTIC failure (k of N runs) from a SYSTEMATIC one
    (every run), and the captured raw response tells WHY (emitted nothing / wrong type / wrong
    span / over-flag). `capturing` is the provider's persister; only a real provider populates
    it (the scripted test double does not persist)."""
    is_safe = not expected
    passes = 0
    failures_raw: list[str] = []
    for _ in range(runs):
        captured_before = len(capturing.completions)
        findings, _n_rejected = await run_analyze_under_model(
            state_from_eval_fixture(fixture_path), provider=provider, model=model
        )
        graded = grade(findings, expected)
        failed = graded.n_false_positives > 0 if is_safe else graded.recall.value < 1.0
        if failed:
            this_run = capturing.completions[captured_before:]
            failures_raw.append(" | ".join(this_run) if this_run else "<no captured response>")
        else:
            passes += 1
    return {
        "fixture": fixture_path,
        "kind": "safe" if is_safe else "recall",
        "runs": runs,
        "passes": passes,
        "n_failures": len(failures_raw),
        "verdict": (
            "systematic" if passes == 0 else "stochastic" if passes < runs else "clean-all"
        ),
        "failures_raw": failures_raw,
    }


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
    empty-response run yields none. Pins that the comparison's inputs are real.
    `n_rejected` is 0 for both (the structured output parsed)."""
    finds, finds_n_rejected = await run_analyze_under_model(
        _build_state(), provider=_ScriptedProvider(_FINDS_RESPONSE), model="claude-sonnet-4-6"
    )
    assert len(finds) >= 1, "the scripted finding response did not admit a finding"
    assert finds[0].finding_type == FindingType.SQL_INJECTION
    assert finds_n_rejected == 0

    misses, misses_n_rejected = await run_analyze_under_model(
        _build_state(), provider=_ScriptedProvider(_MISSES_RESPONSE), model="claude-haiku-4-5"
    )
    assert misses == ()
    assert misses_n_rejected == 0  # valid-empty, NOT a rejection


@pytest.mark.asyncio
async def test_run_analyze_under_model_flags_rejected_structured_output() -> None:
    """FUP-196 yield signal: a malformed (unparseable) response is REJECTED — the run
    reports n_rejected=1 (the single file's response failed to parse) with zero findings,
    distinct from a valid-empty response (n_rejected=0). This is the structured-output
    conformance signal the GLM scorecard needs (a FORMAT miss, where the model emitted
    garbage, vs a CAPABILITY miss, where it validly found nothing)."""
    finds, n_rejected = await run_analyze_under_model(
        _build_state(),
        provider=_ScriptedProvider("not JSON — the model failed to emit structured output"),
        model="claude-haiku-4-5",  # any priced model; the rejection is about the RESPONSE
    )
    assert finds == ()
    assert n_rejected == 1  # single-file scenario → exactly one rejected response


# ---------------------------------------------------------------------------
# JUDGED-only fixture invariants — recall is pure model signal (FUP-196)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _JUDGED_TYPE_FIXTURES)
@pytest.mark.asyncio
async def test_judged_fixture_has_no_observed_floor(fixture_path: str) -> None:
    """Each model-dependent recall fixture yields NOTHING from the OBSERVED structural producer
    under an EMPTY model — so its scorecard recall is pure model signal, not a structural
    backstop. Locks the JUDGED-only property the fixtures were authored for (forms the .scm
    queries deliberately don't match): if a future query broadens to match one, this fails by
    NAME, instead of surfacing as a confusing recall==0 in a type-named regression test."""
    findings, n_rejected = await run_analyze_under_model(
        state_from_eval_fixture(fixture_path),
        provider=_ScriptedProvider(_MISSES_RESPONSE),
        model="claude-haiku-4-5",
    )
    assert findings == (), (
        f"{fixture_path} produced an OBSERVED finding under an empty model — it is no longer "
        "JUDGED-only, so its scorecard recall would be structural rather than model signal"
    )
    assert n_rejected == 0  # a valid-empty response, not a rejection


@pytest.mark.parametrize("fixture_path", _JUDGED_TYPE_FIXTURES)
def test_judged_fixture_scripted_analyze_matches_ground_truth(fixture_path: str) -> None:
    """Each fixture's scripted analyze finding (documentation the scorecard itself does not
    read — it runs the real provider) agrees with its `_GROUND_TRUTH_BY_FIXTURE` entry on type
    + line, so the inert block can't silently drift from the ground truth a maintainer reads."""
    with open(fixture_path, encoding="utf-8") as fh:
        scripted = json.loads(json.load(fh)["llm_responses"]["analyze"][0])["findings"]
    assert len(scripted) == 1, f"{fixture_path}: expected exactly one scripted analyze finding"
    (expected,) = _GROUND_TRUTH_BY_FIXTURE[fixture_path]
    assert scripted[0]["finding_type"] == expected.finding_type.value
    assert scripted[0]["line_start"] == expected.line_start
    assert scripted[0]["line_end"] == expected.line_end


# ---------------------------------------------------------------------------
# diagnose_scenario_stability — separate a stochastic failure from a systematic one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capturing_persister_records_completion() -> None:
    """_CapturingExchangePersister records each call's raw completion text — the mechanism the
    miss-diagnostic uses to show WHAT the model emitted on a failed row."""

    class _FakeResponse:
        text = "<<raw model output>>"

    persister = _CapturingExchangePersister()
    await persister.persist(object(), object(), _FakeResponse())
    assert persister.completions == ["<<raw model output>>"]


@pytest.mark.asyncio
async def test_diagnose_scenario_stability_recall_clean() -> None:
    """A recall scenario whose scripted model catches the known finding every run reports
    passes==runs and verdict 'clean-all' with no recorded failures."""
    expected = _GROUND_TRUTH_BY_FIXTURE[_SSRF_FIXTURE]
    result = await diagnose_scenario_stability(
        _SSRF_FIXTURE,
        expected=expected,
        provider=_ScriptedProvider(_judged_response_for(expected[0])),
        model="claude-haiku-4-5",
        capturing=_CapturingExchangePersister(),
        runs=3,
    )
    assert result["kind"] == "recall"
    assert result["passes"] == 3
    assert result["verdict"] == "clean-all"
    assert result["failures_raw"] == []


@pytest.mark.asyncio
async def test_diagnose_scenario_stability_recall_systematic_miss() -> None:
    """A recall scenario whose scripted model emits nothing every run reports passes==0 and
    verdict 'systematic'; the raw slot reads the no-capture sentinel because _ScriptedProvider
    does not persist (only a real provider populates the capturing persister)."""
    result = await diagnose_scenario_stability(
        _SSRF_FIXTURE,
        expected=_GROUND_TRUTH_BY_FIXTURE[_SSRF_FIXTURE],
        provider=_ScriptedProvider(_MISSES_RESPONSE),
        model="claude-haiku-4-5",
        capturing=_CapturingExchangePersister(),
        runs=3,
    )
    assert result["passes"] == 0
    assert result["verdict"] == "systematic"
    assert result["failures_raw"] == ["<no captured response>"] * 3


@pytest.mark.asyncio
async def test_diagnose_scenario_stability_safe_overflag() -> None:
    """A SAFE scenario (empty expected) whose scripted model emits a finding every run FAILS on
    the false positive each run → passes==0, verdict 'systematic', kind 'safe' — the over-flag
    counterpart to a recall miss."""
    overflag = _judged_response_for(
        ExpectedFinding(
            file_path="app/profile.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SSRF,
            severity=lookup_severity(FindingType.SSRF),
        )
    )
    result = await diagnose_scenario_stability(
        _SSRF_FIXED_HOST_SAFE_FIXTURE,
        expected=(),
        provider=_ScriptedProvider(overflag),
        model="claude-haiku-4-5",
        capturing=_CapturingExchangePersister(),
        runs=3,
    )
    assert result["kind"] == "safe"
    assert result["passes"] == 0
    assert result["verdict"] == "systematic"
    assert result["n_failures"] == 3


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
    """Regression track (DECISIONS#041): a CANDIDATE-ONLY sql_injection false positive
    near the parameterized query FAILS the precision dimension while the baseline stays
    clean — candidate fp (1) > baseline fp (0). Since FUP-162, an over-flag ON the
    execute call itself (line 6) is absorbed deterministically at admission (pinned by
    `test_parameterized_overflag_on_the_call_is_vetoed_deterministically` below), so
    this test scripts the FP at line 5 — the cursor-context line the veto deliberately
    does NOT cover — pinning the harness's ability to catch the FPs that survive
    admission. The complementary SHARED-over-flag case (both fp>0, which the relative
    gate would wrongly pass) is handled by the opt-in run's absolute baseline-clean
    guard, not this test."""
    overflag = _judged_response_for(
        ExpectedFinding(
            file_path="directory/users.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        )
    )
    baseline = _ScriptedProvider(_MISSES_RESPONSE)  # Sonnet: clean
    candidate = _ScriptedProvider(overflag)  # Haiku: over-flags near the parameterized query
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


@pytest.mark.asyncio
async def test_parameterized_overflag_on_the_call_is_vetoed_deterministically() -> None:
    """FUP-162 (specs/2026-06-12-sqli-parameterized-call-veto.md), end-to-end through
    the eval driver: the EXACT #041 caveat — a JUDGED sql_injection ON the provably-
    parameterized execute call (line 6) — cannot reach a graded finding anymore,
    regardless of prompt wording or model version. The parser's deterministic veto
    rejects it at admission (`sql_injection_on_parameterized_call` on the audit
    stream; the rejection mechanics are pinned at the parser unit tier), so the
    candidate grades CLEAN and the precision gate passes WITHOUT prompt-layer help."""
    overflag = _judged_response_for(
        ExpectedFinding(
            file_path="directory/users.py",
            line_start=6,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
            severity=lookup_severity(FindingType.SQL_INJECTION),
        )
    )
    baseline = _ScriptedProvider(_MISSES_RESPONSE)  # clean
    candidate = _ScriptedProvider(overflag)  # the on-call over-flag
    cmp = await compare_models_on_scenario(
        state_from_eval_fixture(_PARAMETERIZED_QUERY_FIXTURE),
        (),
        baseline_provider=baseline,
        baseline_model="claude-sonnet-4-6",
        candidate_provider=candidate,
        candidate_model="claude-haiku-4-5",
    )
    assert baseline.calls and candidate.calls
    assert cmp.candidate.n_false_positives == 0  # the veto absorbed the over-flag
    assert cmp.fp_bounded is True  # precision holds without the prompt layer's help


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
    only sql_injection FPs isolates the caveat. (get_user spans lines 4-7; the sql_injection FP
    scripts at line 5, the cursor-context line — since FUP-162, line 6's on-call over-flag is
    deterministically vetoed at admission and can no longer reach a graded finding.)"""
    baseline_resp = _judged_response_with([(FindingType.MISSING_ERROR_HANDLING, 5, 5)])
    candidate_resp = _judged_response_with(
        [(FindingType.SQL_INJECTION, 5, 5), (FindingType.MISSING_ERROR_HANDLING, 5, 5)]
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
        _SQLI_HOLDOUT_ORM_FSTRING_FIXTURE,
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
    """END-TO-END zero-spend over EACH recall fixture (all sixteen in `_GROUND_TRUTH_BY_FIXTURE` —
    SQLi, auth-bypass, missing-error-handling, N+1, path-traversal, missing-input-validation,
    four held-out SQLi forms: f-string / str.format / concatenation / ORM raw() f-string, plus
    six JUDGED-only rows: ssrf / weak-password-hash / command-injection (subprocess + indirect
    eval) / weak-crypto / insecure-randomness):
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
    if fixture_path in _OBSERVED_BACKSTOPPED_SQLI:
        # Cost Lever 3: the OBSERVED sql_injection query catches these SQLi
        # forms deterministically, so the candidate's PRODUCT recall is 1.0
        # even though the model returned nothing — the structural tier backstops
        # the model. The recall-regression gate is intentionally blind to model
        # regressions on OBSERVED-covered findings (the query masks the miss);
        # a model-only recall would still be 0.0 (the diagnostic split the eval
        # harness should keep). DECISIONS.md#048 + the observed-query-library spec.
        assert cmp.candidate.recall.value == 1.0
        assert cmp.passes is True
    else:
        assert cmp.candidate.recall.value == 0.0  # model missed it; no OBSERVED backstop
        assert cmp.passes is False  # the gate catches the recall regression


@pytest.mark.parametrize("fixture_path", _SAFE_CODE_FIXTURES)
@pytest.mark.asyncio
async def test_safe_fixture_reaches_the_model_and_yields_no_findings(fixture_path: str) -> None:
    """The FP instrument is only meaningful if the safe code is ACTUALLY REVIEWED: a fixture
    analyze skips (NO_CHANGED_SCOPE_UNITS, skim/skip tier) produces zero findings VACUOUSLY and
    would "prove" the don't-flag discriminator without exercising it — the FP counterpart of the
    positive admission test above. Three assertions per safe fixture: the model was called at
    least once (a skipped file makes no LLM call), the scripted clean response parsed
    (a rejected response would also zero-out findings vacuously), and no findings survived."""
    provider = _ScriptedProvider(_MISSES_RESPONSE)
    findings, n_rejected = await run_analyze_under_model(
        state_from_eval_fixture(fixture_path), provider=provider, model="claude-sonnet-4-6"
    )
    assert provider.calls, (
        f"{fixture_path}: analyze never called the model — the file was skipped, so its "
        "zero-findings result is vacuous, not evidence of the don't-flag discriminator"
    )
    assert n_rejected == 0  # the clean response actually parsed; rejection would mask findings
    assert findings == ()  # and the genuine outcome on safe code is zero findings


# ---------------------------------------------------------------------------
# Opt-in REAL-model run (SPEND) — the evidence path. Skipped unless explicitly enabled.
# Two dimensions: RECALL over known-vulnerability fixtures, PRECISION over safe code.
# ---------------------------------------------------------------------------


def _print_scenario_report(
    fixture_path: str, cmp: ModelComparison, baseline_model: str, candidate_model: str
) -> str:
    """Print one scenario's recall/precision/fp + the raw gate flags + each model's
    extra/missed detail, so a verdict is interpretable: is an extra noise, or a legitimate
    finding the ground truth didn't encode? The caller picks which flag is the gate for the
    scenario's dimension. Returns the same text it prints so a caller can ALSO persist the
    evidence to a report file (stdout is otherwise lost to pytest capture — the same reason
    `_compute_aggregate_metrics` exists on the scorecard side)."""
    b, c = cmp.baseline, cmp.candidate
    lines = [
        f"\n[{fixture_path}]",
        f"  baseline ({baseline_model}): "
        f"recall={b.recall.value:.2f} precision={b.precision.value:.2f} "
        f"sev={b.severity_accuracy.value:.2f} fp={b.n_false_positives} "
        f"yield={'REJECTED' if cmp.baseline_n_rejected else 'parsed'}",
        f"  candidate ({candidate_model}): "
        f"recall={c.recall.value:.2f} precision={c.precision.value:.2f} "
        f"sev={c.severity_accuracy.value:.2f} fp={c.n_false_positives} "
        f"yield={'REJECTED' if cmp.candidate_n_rejected else 'parsed'}",
        f"  recall_held={cmp.recall_held} baseline_valid={cmp.baseline_valid} "
        f"fp_bounded={cmp.fp_bounded}",
    ]
    for label, g in (("baseline", b), ("candidate", c)):
        for x in g.extra:
            lines.append(
                f"    {label} extra (finding not in ground truth): "
                f"{x.finding_type.value} {x.file_path}:{x.line_start} — {x.title}"
            )
        for m in g.missed:
            lines.append(f"    {label} MISSED: {m.finding_type.value} {m.file_path}:{m.line_start}")
    report = "\n".join(lines)
    print(report)  # noqa: T201 — operator evidence output
    return report


def _compute_aggregate_metrics(
    results: list[tuple[str, str, ModelComparison]],
    ground_truth_by_fixture: dict[str, tuple[ExpectedFinding, ...]],
    baseline_model: str,
    candidate_model: str,
) -> dict[str, object]:
    """Pure compute behind _print_aggregate_metrics: roll the per-scenario comparisons into a
    JSON-serializable per-model metric structure (FUP-196 + the best-metrics set) so the
    headline numbers can be BOTH printed for the operator AND persisted into the scorecard
    artifact (they are otherwise stdout-only and lost to pytest capture).

    The two headline axes are measured on DISJOINT row populations and are intentionally NOT
    collapsed into one score: recall over the 'recall' rows (non-empty ground truth), over-
    flagging over the 'precision'/safe rows (empty ground truth → every finding an unambiguous
    FP). Precision-as-a-ratio and F1 are deliberately NOT reported — on vulnerable fixtures the
    single-entry ground truth under-specifies (a legit second finding scores as an FP, see
    _SAFE_CODE_FIXTURES and test_real_model_comparison_evidence), and on safe rows n_matched is
    structurally 0 so the ratio is degenerate; the all-rows extras tally survives only as a
    labeled diagnostic count. (mean_recall is PRODUCT recall — model output plus the OBSERVED
    structural backstop per DECISIONS#048 — not model-only recall.)"""
    import statistics  # noqa: PLC0415
    from collections import Counter  # noqa: PLC0415

    recall_rows = [(fx, cmp) for fx, dim, cmp in results if dim == "recall"]
    # Safe rows are the 'precision' dimension EXACTLY (not `dim != "recall"`): the GLM caller
    # feeds only recall+precision, but a future caller adding a third dimension (e.g.
    # 'regression') must NOT have those rows silently bucketed as safe — their false positives
    # would corrupt the headline over-flag instrument.
    safe_rows = [(fx, cmp) for fx, dim, cmp in results if dim == "precision"]
    n_total = len(results)
    # Ground-truth expected-by-type is MODEL-INVARIANT — compute once, outside the per-model fn.
    expected_by_type: Counter[str] = Counter()
    for fx, _cmp in recall_rows:
        for ef in ground_truth_by_fixture.get(fx, ()):
            expected_by_type[ef.finding_type.value] += 1

    def _per_model(
        grade_of: Callable[[ModelComparison], GradeResult],
        n_rejected_of: Callable[[ModelComparison], int],
    ) -> dict[str, object]:
        # Per-file rejection count is preserved (a multi-file scenario distinguishes one
        # rejected file from all); yield_rate stays SCENARIO-level (fraction with no rejection).
        n_rejected_files = sum(n_rejected_of(cmp) for _, _, cmp in results)
        n_rejected_scenarios = sum(1 for _, _, cmp in results if n_rejected_of(cmp) > 0)
        yield_rate = (n_total - n_rejected_scenarios) / n_total
        recall_grades = [grade_of(cmp) for _, cmp in recall_rows]
        mean_recall = (
            statistics.fmean(g.recall.value for g in recall_grades) if recall_grades else 0.0
        )
        mean_sev = (
            statistics.fmean(g.severity_accuracy.value for g in recall_grades)
            if recall_grades
            else 0.0
        )
        # HEADLINE over-flag instrument — safe rows only (empty ground truth → every finding an
        # unambiguous FP), a mean-FP-per-scenario RATE (>=0, not a [0,1] ratio: a safe-row
        # precision ratio is degenerate since n_matched is structurally 0).
        safe_fp = sum(grade_of(cmp).n_false_positives for _, cmp in safe_rows)
        fp_per_safe = safe_fp / len(safe_rows) if safe_rows else 0.0
        # All-rows over-flag VOLUME — a labeled diagnostic COUNT only (never a precision ratio):
        # on vulnerable fixtures a legit extra scores as an FP, so this is not the verdict.
        total_findings = sum(grade_of(cmp).precision.denominator for _, _, cmp in results)
        total_fp = sum(grade_of(cmp).n_false_positives for _, _, cmp in results)
        # Per-type recall: clamp the numerator to >=0 and iterate the UNION of expected+missed
        # types, so a desync between the passed ground truth and the graded comparisons surfaces
        # (a missed type with no ground-truth expectation reads None) instead of printing a
        # negative recall or silently dropping the miss.
        missed_by_type: Counter[str] = Counter()
        for _, cmp in recall_rows:
            for m in grade_of(cmp).missed:
                missed_by_type[m.finding_type.value] += 1
        per_type_recall: dict[str, float | None] = {}
        for t in sorted(set(expected_by_type) | set(missed_by_type)):
            expected = expected_by_type[t]
            per_type_recall[t] = (
                None if expected == 0 else max(0, expected - missed_by_type[t]) / expected
            )
        # Per-safe-fixture over-flag DETAIL (FP findings) — persisted so the artifact can name
        # WHICH fixtures a model flagged + WHAT, for BOTH models (the per-row Scorecard records
        # only the candidate's FP detail + a baseline COUNT, so the baseline's safe FPs were
        # otherwise stdout-only). The honest precision instrument is the safe rows.
        safe_overflags = [
            {
                "fixture": fx,
                "findings": [
                    {
                        "finding_type": f.finding_type.value,
                        "line_start": f.line_start,
                        "title": f.title,
                    }
                    for f in g.extra
                ],
            }
            for fx, cmp in safe_rows
            if (g := grade_of(cmp)).extra
        ]
        return {
            "yield_rate": yield_rate,
            "n_rejected_scenarios": n_rejected_scenarios,
            "n_rejected_files": n_rejected_files,
            "mean_recall": mean_recall,
            "mean_severity_acc": mean_sev,
            "fp_per_safe_scenario": fp_per_safe,
            "safe_fp": safe_fp,
            "safe_overflags": safe_overflags,
            "all_row_extras": total_fp,
            "all_row_findings": total_findings,
            "per_type_recall": per_type_recall,
        }

    return {
        "scenarios": {"total": n_total, "recall": len(recall_rows), "safe": len(safe_rows)},
        "baseline": {
            "model": baseline_model,
            **_per_model(lambda c: c.baseline, lambda c: c.baseline_n_rejected),
        },
        "candidate": {
            "model": candidate_model,
            **_per_model(lambda c: c.candidate, lambda c: c.candidate_n_rejected),
        },
    }


def _print_aggregate_metrics(
    results: list[tuple[str, str, ModelComparison]],
    ground_truth_by_fixture: dict[str, tuple[ExpectedFinding, ...]],
    baseline_model: str,
    candidate_model: str,
) -> dict[str, object] | None:
    """Print the cross-scenario aggregate block for the operator AND return the structured
    metrics so the caller can PERSIST them into the scorecard artifact (otherwise the headline
    numbers are stdout-only and lost to pytest capture). Returns None on an empty run. See
    _compute_aggregate_metrics for the methodology (disjoint recall/over-flag populations;
    precision-ratio and F1 deliberately not reported)."""
    if not results:
        return None
    metrics = _compute_aggregate_metrics(
        results, ground_truth_by_fixture, baseline_model, candidate_model
    )
    scenarios = metrics["scenarios"]
    assert isinstance(scenarios, dict)
    print("\n" + "=" * 72)  # noqa: T201 — operator aggregate metric block
    for label in ("baseline", "candidate"):
        m = metrics[label]
        assert isinstance(m, dict)
        per_type = m["per_type_recall"]
        assert isinstance(per_type, dict)
        per_type_str = ", ".join(
            f"{t}={'n/a(desync)' if v is None else format(v, '.2f')}" for t, v in per_type.items()
        )
        print(  # noqa: T201 — operator aggregate metric block
            f"AGGREGATE — {label} ({m['model']}): {scenarios['total']} scenarios "
            f"({scenarios['recall']} recall / {scenarios['safe']} safe)"
            f"\n  yield_rate={m['yield_rate']:.2f} "
            f"({scenarios['total'] - m['n_rejected_scenarios']}/{scenarios['total']} scenarios "
            f"parsed; {m['n_rejected_files']} file-level rejection(s))"
            f"\n  mean_recall={m['mean_recall']:.2f}   "
            f"mean_severity_acc={m['mean_severity_acc']:.2f}   [recall rows only]"
            f"\n  fp_per_safe_scenario={m['fp_per_safe_scenario']:.2f} "
            f"({m['safe_fp']} fp over {scenarios['safe']} safe)"
            f"   [HEADLINE over-flag instrument — safe rows, empty ground truth]"
            f"\n  diagnostic (all rows; precision unreliable on vuln fixtures — single-entry GT "
            f"undercounts, see _SAFE_CODE_FIXTURES — NOT the over-flag verdict): "
            f"all_row_extras={m['all_row_extras']}/{m['all_row_findings']} findings"
            f"\n  per-type recall: {per_type_str or '(none)'}"
        )
    print("=" * 72)  # noqa: T201 — operator aggregate metric block
    return metrics


async def _run_scenario_isolating_transients(
    fixture_path: str,
    dimension: str,
    gate_results: list[tuple[str, str, bool, str]],
    compare_call: Callable[[], Awaitable[ModelComparison]],
) -> ModelComparison | None:
    """Run one evidence scenario, isolating TRANSIENT provider failures.

    ERRORED-and-continue applies only to the taxonomy's retry-eligible set
    (`retry_at_layer="node"`: timeout/429/409/5xx) — the scenario records
    "ERRORED — rerun" in `gate_results` and returns None so the paid run
    continues. Terminal classes (auth, config, persister) recur on every
    scenario — a revoked key would mark all ~28 scenarios ERRORED and
    "complete" a run with zero verdicts — so they re-raise and abort on
    first occurrence. Module-level (not nested in the opt-in test) so the
    zero-spend pins below exercise both paths in the normal eval gate.
    """
    try:
        return await compare_call()
    except LLMProviderError as exc:
        if exc.retry_at_layer != "node":
            raise
        print(  # noqa: T201 — operator diagnostic
            f"\n[{fixture_path}]\n  ERRORED ({type(exc).__name__}) — scenario not "
            "measured; rerun for this verdict"
        )
        gate_results.append(
            (fixture_path, dimension, False, f"ERRORED ({type(exc).__name__}) — rerun")
        )
        return None


async def test_scenario_isolation_transient_failure_records_errored_and_continues() -> None:
    """Zero-spend pin for the evidence runner's transient path: a
    retry_at_layer="node" failure (here LLMTimeoutError — the class that
    killed the 2026-06-10 run) records ERRORED — rerun and returns None so
    the run continues, instead of propagating."""
    gate: list[tuple[str, str, bool, str]] = []

    async def _boom() -> ModelComparison:
        raise LLMTimeoutError()

    result = await _run_scenario_isolating_transients("fx.json", "recall", gate, _boom)
    assert result is None
    assert gate == [("fx.json", "recall", False, "ERRORED (LLMTimeoutError) — rerun")]


async def test_scenario_isolation_terminal_failure_reraises() -> None:
    """Zero-spend pin for the terminal path: retry_at_layer="none" classes
    (here LLMAuthError — the revoked-key shape) re-raise on first occurrence
    and record nothing, aborting the run instead of burning the remaining
    scenarios on a dead configuration."""
    gate: list[tuple[str, str, bool, str]] = []

    async def _boom() -> ModelComparison:
        raise LLMAuthError()

    with pytest.raises(LLMAuthError):
        await _run_scenario_isolating_transients("fx.json", "recall", gate, _boom)
    assert gate == []


async def _run_real_analyze_evidence(
    *,
    provider: LLMProvider,
    baseline_model: str,
    candidate_model: str,
    token_recorder: _TokenRecordingExchangePersister | None = None,
) -> None:
    """Drive the analyze node under `baseline_model` and `candidate_model` over the recall /
    precision / regression-track fixture corpus, printing a per-scenario report and a
    REPORT-ONLY gate summary. Shared by the two opt-in real-model evidence tests (the
    STANDARD->Haiku flip run and the Sonnet 5 migration run); the mechanics are
    model-agnostic, so each caller supplies its own baseline/candidate and owns the
    rationale in its docstring. Does NOT close the provider — the caller owns provider
    lifecycle (wrap the call in `try: ... finally: await provider.aclose()`).

    When `token_recorder` is passed (it MUST be the same object as the provider's
    `persister`), the run appends a MEASURED per-model OUTPUT-token sizing block (FUP-207)
    to the report — otherwise the token half is skipped and behavior is unchanged.

    (fixture, dimension, ok, fail_label): fail_label distinguishes WHY a non-ok verdict is
    non-ok in the end summary — a regression scenario is "INCONCLUSIVE" (the baseline emitted a
    sql_injection FP itself) vs "REPRODUCED" (only the candidate did); recall/precision use plain
    "FAILED". Empty label for green verdicts (never printed). A scenario whose provider call
    dies (timeout, rate limit, 5xx) records "ERRORED — rerun" and the run CONTINUES — one
    transient API failure must not discard the rest of a paid evidence run (a 30s TTFT spike
    cost a full run on 2026-06-10)."""
    gate_results: list[tuple[str, str, bool, str]] = []
    report_chunks: list[str] = []  # per-scenario texts + summary, persisted to reports/ below
    emitted: dict[tuple[str, str], int] = {}  # (fixture, model) -> emitted-finding count (FUP-207)

    async def _compare_or_errored(
        fixture_path: str, ground_truth: tuple[ExpectedFinding, ...], dimension: str
    ) -> ModelComparison | None:
        if token_recorder is not None:
            token_recorder.current_fixture = fixture_path  # tag this compare's records

        async def _compare() -> ModelComparison:
            return await compare_models_on_scenario(
                state_from_eval_fixture(fixture_path),
                ground_truth,
                baseline_provider=provider,
                baseline_model=baseline_model,
                candidate_provider=provider,
                candidate_model=candidate_model,
            )

        cmp = await _run_scenario_isolating_transients(
            fixture_path, dimension, gate_results, _compare
        )
        if token_recorder is not None and cmp is not None:
            emitted[(fixture_path, baseline_model)] = cmp.baseline.n_matched + len(
                cmp.baseline.extra
            )
            emitted[(fixture_path, candidate_model)] = cmp.candidate.n_matched + len(
                cmp.candidate.extra
            )
        return cmp

    # RECALL dimension — gate on recall_held + baseline_valid; FP advisory (see caller docstring).
    for fixture_path, ground_truth in _GROUND_TRUTH_BY_FIXTURE.items():
        cmp = await _compare_or_errored(fixture_path, ground_truth, "recall")
        if cmp is None:
            continue
        report_chunks.append(
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
        )
        gate_results.append(
            (fixture_path, "recall", cmp.recall_held and cmp.baseline_valid, "FAILED")
        )
        assert cmp.baseline is not None  # the run completed
    # PRECISION dimension — safe code, empty ground truth so ANY finding is a real FP;
    # gate on fp_bounded (candidate must not over-flag clean code more than baseline).
    for fixture_path in _SAFE_CODE_FIXTURES:
        cmp = await _compare_or_errored(fixture_path, (), "precision")
        if cmp is None:
            continue
        report_chunks.append(
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
        )
        gate_results.append((fixture_path, "precision", cmp.fp_bounded, "FAILED"))
        assert cmp.baseline is not None  # the run completed
    # REGRESSION-TRACK dimension — TYPE-SCOPED to sql_injection FPs (`_sqli_fp_count`), read
    # with an ABSOLUTE baseline-clean gate. The caveat is type-specific (DECISIONS#041: a model
    # mislabels a parameterized query as SQLi), so an unrelated baseline over-flag must NOT
    # blind the track. `_regression_verdict` carries the 3-state logic (INCONCLUSIVE /
    # REPRODUCED / clean). Runs over the DEMONSTRATED idioms (`_REGRESSION_FIXTURES`) AND the
    # HOLD-OUTS (`_REGRESSION_HOLDOUT_FIXTURES`, placeholder styles the prompt never shows) —
    # CLEAN on the hold-outs is the anti-overfit evidence. The fixture path names the set.
    for fixture_path in _REGRESSION_FIXTURES + _REGRESSION_HOLDOUT_FIXTURES:
        cmp = await _compare_or_errored(fixture_path, (), "regression")
        if cmp is None:
            continue
        report_chunks.append(
            _print_scenario_report(fixture_path, cmp, baseline_model, candidate_model)
        )
        verdict, ok, fail_label = _regression_verdict(
            _sqli_fp_count(cmp.baseline), _sqli_fp_count(cmp.candidate)
        )
        verdict_line = f"  REGRESSION-TRACK verdict: {verdict}"
        print(verdict_line)  # noqa: T201 — operator diagnostic
        report_chunks.append(verdict_line)
        gate_results.append((fixture_path, "regression", ok, fail_label))
        assert cmp.baseline is not None  # the run completed
    # REPORT-ONLY summary — pytest "passed" means the run completed, NOT the gate verdict.
    # Each non-green line carries its own label (recall/precision -> "FAILED"; regression ->
    # "INCONCLUSIVE …" or "REPRODUCED …"; transient provider failures -> "ERRORED (…) —
    # rerun", which is an unmeasured scenario, not a gate verdict) so a skimmer can't
    # misread an inconclusive regression or an errored scenario as a candidate failure.
    failed = [(fx, dim, label) for fx, dim, ok, label in gate_results if not ok]
    green = len(gate_results) - len(failed)
    summary = (
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
    print(summary)  # noqa: T201 — operator gate summary
    report_chunks.append(summary)
    if token_recorder is not None:
        sizing = _format_token_sizing(
            token_recorder.records, emitted, baseline_model, candidate_model
        )
        print(sizing)  # noqa: T201 — operator sizing evidence
        report_chunks.append(sizing)
    # Persist the operator evidence like the scorecard does (`reports/` is gitignored) — stdout is
    # otherwise lost to pytest capture. Overwrites the latest run per model pair; a pinned run can
    # be copied to a dated reports/model_comparison/*.md.
    report_body = "\n".join(report_chunks).lstrip("\n")
    out_dir = Path("reports") / "model_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = f"{baseline_model}_vs_{candidate_model}".replace("/", "-")
    out_path = out_dir / f"{slug}.md"
    out_path.write_text(
        f"# Real-model analyze evidence — {candidate_model} (candidate) "
        f"vs {baseline_model} (baseline)\n\n"
        "Report-only, analyze-node-only. Regenerated each run.\n\n"
        f"```\n{report_body}\n```\n",
        encoding="utf-8",
    )
    print(  # noqa: T201 — operator artifact pointer
        f"\nEVIDENCE — REPORT ONLY: wrote {out_path} "
        f"(baseline={baseline_model}, candidate={candidate_model})"
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
      also carries three HOLD-OUT SQLi forms (f-string / `str.format` / `+` concatenation;
      named-but-unexemplified under analyze-v3, with f-string and `+` now DEMONSTRATED
      by the exemplars from analyze-v4 onward — see the recall hold-out comment) — a
      recall MISS there means the parameterized-query remediation over-suppressed real SQLi.
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
    # findings from analyze's return, so a token-recording persister (observe-only,
    # FUP-207) is the right wiring here.
    token_recorder = _TokenRecordingExchangePersister()
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=token_recorder
    )
    try:
        await _run_real_analyze_evidence(
            provider=provider,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            token_recorder=token_recorder,
        )
    finally:
        await provider.aclose()


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="real-model comparison spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
@pytest.mark.asyncio
async def test_real_sonnet5_migration_evidence() -> None:
    """OPT-IN, real API spend — the evidence REPORT for the Sonnet 4.6 -> Sonnet 5 DEEP-tier
    analyze migration (feat/sonnet-5-migration): did the new default HOLD or IMPROVE analyze
    quality versus the prior default?

    REPORT-ONLY and ANALYZE-NODE-ONLY, same contract as `test_real_model_comparison_evidence`:
    pytest "passed" means the run COMPLETED, not that the gate passed; the per-scenario verdicts
    print and the human adjudicates. This drives ONLY the analyze node over the curated fixture
    corpus — it does NOT exercise triage/trace/synthesize/hitl/publish on the real model, nor the
    full 7-node graph. It is a migration-risk probe, not a full-graph quality gate.

    baseline = claude-sonnet-4-6 (the prior DEEP default); candidate = ModelConfig.analyze_model
    (the shipped default, now claude-sonnet-5). Both are priced in `pricing.py` (v5 retains the
    4.6 rate). Same three dimensions and deterministic scoring as the Haiku-flip evidence test
    (recall / safe-code precision / sql_injection regression-track). Read recall, safe-code FP,
    severity accuracy, and per-type regression to decide the merge:
    - matches/beats 4.6 on recall AND no new safe-code FPs -> the default flip is validated;
    - loses only on a narrow stochastic row -> diagnose that row before blocking;
    - regresses a core finding type OR over-flags safe code -> keep the plumbing, hold the flip.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the real-model comparison")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.pricing import normalize_to_pricing_key  # noqa: PLC0415

    cfg = ModelConfig()
    # baseline = the PRIOR DEEP default (Sonnet 4.6, explicit pre-migration string); candidate =
    # the shipped default (`cfg.analyze_model`, now Sonnet 5). Reading candidate from cfg keeps
    # the probe honest if the default is ever re-pinned.
    baseline_model = "claude-sonnet-4-6"
    candidate_model = cfg.analyze_model
    # Belt-and-suspenders: a meaningless self-comparison (e.g. analyze_model env-overridden back
    # to 4.6) proves nothing. Normalized so a dated pin (…-20251001) can't sneak past. Checked
    # BEFORE constructing the provider so a guard-fire can't leak an unclosed client.
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the migration comparison would prove nothing about Sonnet-5-vs-4.6. "
            "Unset OUTRIDER_MODEL_ANALYZE_MODEL (or point it at Sonnet 5) for the evidence run."
        )
    # persister MUST be a real LLMExchangePersister, not None: AnthropicProvider.complete() is
    # fail-closed on persister=None. The comparison reads findings from analyze's return, so a
    # token-recording persister (observe-only, FUP-207) is the right wiring here.
    token_recorder = _TokenRecordingExchangePersister()
    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=token_recorder
    )
    try:
        await _run_real_analyze_evidence(
            provider=provider,
            baseline_model=baseline_model,
            candidate_model=candidate_model,
            token_recorder=token_recorder,
        )
    finally:
        await provider.aclose()


# ---------------------------------------------------------------------------
# openai-host caller-level regression (openai-native-host spec, Codex round 4):
# the analyze aggregate recomputation must survive an openai-context response —
# and must FAIL exactly when the context is omitted (the revert-the-fold twin).
# ---------------------------------------------------------------------------


class _OpenAIContextScriptedProvider(_ScriptedProvider):
    """`_ScriptedProvider` stamping the openai triad + FULL pricing context —
    the shape the real OpenAICompatibleProvider returns for a default-tier
    5.6 call. `include_pricing_context=False` reproduces the pre-sweep bug
    shape (context-less openai response) for the negative twin."""

    def __init__(self, response_text: str, *, include_pricing_context: bool = True) -> None:
        super().__init__(response_text)
        self._include_pricing_context = include_pricing_context

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.llm.host_profiles import OPENAI_PROFILE

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
            profile_id=OPENAI_PROFILE.host_id,
            reasoning_enabled=False,
            profile_contract_digest=OPENAI_PROFILE.profile_contract_digest,
            billed_prompt_tokens=100 if self._include_pricing_context else None,
            service_tier_actual="default" if self._include_pricing_context else None,
        )


@pytest.mark.asyncio
async def test_analyze_recomputation_survives_openai_context_response() -> None:
    """The REAL analyze caller (run_analyze_under_model drives analyze →
    analyze_file → analyze_aggregate, whose cost recomputation is
    analyze.py's compute_cost_usd site) prices an openai default-tier
    response without raising."""
    finds, n_rejected = await run_analyze_under_model(
        _build_state(),
        provider=_OpenAIContextScriptedProvider(_FINDS_RESPONSE),
        model="gpt-5.6-sol",
    )
    assert len(finds) >= 1
    assert n_rejected == 0


@pytest.mark.asyncio
async def test_analyze_recomputation_rejects_contextless_openai_response() -> None:
    """Revert-the-fold twin: a context-less openai response (the exact
    pre-sweep provider shape) makes the analyze recomputation classify
    absent_tier and raise — proving THIS caller consumes the context, so the
    round-3 omission cannot silently recur while pricing-level tests stay
    green."""
    with pytest.raises(ValueError, match="absent_tier"):
        await run_analyze_under_model(
            _build_state(),
            provider=_OpenAIContextScriptedProvider(_FINDS_RESPONSE, include_pricing_context=False),
            model="gpt-5.6-sol",
        )


async def _run_analyze_capture_round(
    state: ReviewState, *, provider: LLMProvider, model: str
) -> tuple[tuple[ReviewFinding, ...], AnalyzeCompletedEvent | None]:
    """Test-local mirror of `run_analyze_under_model` that ALSO returns the
    emitted `AnalyzeCompletedEvent` (via a capture sink) — the round-5 closure
    needs the aggregate cost, which lives on that event (`AnalysisRound`
    carries no cost field) and which the shared runner deliberately
    discards."""
    from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS, analyze, analyze_file
    from outrider.agent.nodes.analyze_aggregate import analyze_aggregate

    from .model_comparison import _NoOpImportPathResolver, _NullSink

    class _CompletedCapturingSink(_NullSink):
        def __init__(self) -> None:
            super().__init__()
            self.completed_events: list = []

        async def emit_analyze_completed(self, event) -> None:  # type: ignore[no-untyped-def]
            self.completed_events.append(event)

    sink = _CompletedCapturingSink()
    resolver = _NoOpImportPathResolver()
    cmd = await analyze(
        state,
        provider=provider,
        analyze_model=model,
        standard_analyze_model=model,
        phase_event_sink=sink,
        file_examination_sink=sink,
        analyze_event_sink=sink,
        anomaly_sink=sink,
        import_path_resolver=resolver,
    )
    outcomes = []
    for send in cmd.goto if isinstance(cmd.goto, list) else []:
        worker_update = await analyze_file(
            send.arg,
            provider=provider,
            analyze_model=model,
            standard_analyze_model=model,
            import_path_resolver=resolver,
            phase_event_sink=sink,
            file_examination_sink=sink,
            analyze_event_sink=sink,
        )
        outcomes.extend(worker_update["analyze_worker_outcomes"])
    state_after = state.model_copy(
        update={
            "analyze_worker_outcomes": [*state.analyze_worker_outcomes, *outcomes],
            "analyze_pass_started_at": (cmd.update or {})["analyze_pass_started_at"],
        }
    )
    result = await analyze_aggregate(
        state_after,
        analyze_event_sink=sink,
        phase_event_sink=sink,
        anomaly_sink=sink,
        analyze_model=model,
        standard_analyze_model=model,
        total_review_budget_tokens=DEFAULT_REVIEW_BUDGET_TOKENS,
    )
    rounds = result["analysis_rounds"]
    completed = sink.completed_events[-1] if sink.completed_events else None
    return (tuple(rounds[0].findings) if rounds else (), completed)


@pytest.mark.asyncio
async def test_openai_analyze_aggregate_cost_and_context_spy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-5 closures 1+2: the ordinary analyze pricing call receives BOTH
    context arguments (spied at the call site — omitting billed alone would
    otherwise still price flat and stay green), and the emitted
    AnalyzeCompletedEvent.total_cost_usd equals an INDEPENDENTLY computed
    figure from the raw RATE_TABLE rates."""
    import outrider.agent.nodes.analyze as analyze_mod
    from outrider.llm.pricing import RATE_TABLE

    recorded: list[tuple[str, str, dict[str, object]]] = []
    real_compute = analyze_mod.compute_cost_usd

    def spy(profile_id: str, model: str, **kwargs: object):  # type: ignore[no-untyped-def]
        recorded.append((profile_id, model, kwargs))
        return real_compute(profile_id, model, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analyze_mod, "compute_cost_usd", spy)
    finds, completed_event = await _run_analyze_capture_round(
        _build_state(),
        provider=_OpenAIContextScriptedProvider(_FINDS_RESPONSE),
        model="gpt-5.6-sol",
    )
    assert len(finds) >= 1
    assert completed_event is not None

    openai_calls = [(p, m, k) for (p, m, k) in recorded if p == "openai"]
    assert len(openai_calls) == 1
    _, _, kwargs = openai_calls[0]
    assert kwargs["billed_prompt_tokens"] == 100
    assert kwargs["service_tier"] == "default"

    rates = RATE_TABLE[("openai", "gpt-5.6-sol")]
    independent = float(rates.in_per_token * 100 + rates.out_per_token * 50)
    assert completed_event.total_cost_usd == pytest.approx(independent)


@pytest.mark.asyncio
async def test_openai_trace_expansion_recomputation_passes_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-5 closure 3: the trace-expansion recomputation
    (`_process_one_trace_fetched_file`, analyze.py's SECOND compute_cost_usd
    site) forwards the complete pricing context — spied at the call site while
    driving the real pass-1 function."""
    from uuid import uuid4 as _uuid4

    import outrider.agent.nodes.analyze as analyze_mod
    from outrider.agent.nodes.analyze import _process_one_trace_fetched_file
    from outrider.schemas.trace_fetched_file import TraceFetchedFile

    from .model_comparison import _NoOpImportPathResolver, _NullSink

    # A source finding from a real scripted pass-0 run (no hand-built proof fields).
    finds, _ = await _run_analyze_capture_round(
        _build_state(),
        provider=_OpenAIContextScriptedProvider(_FINDS_RESPONSE),
        model="gpt-5.6-sol",
    )
    assert finds, "pass-0 fixture must admit a source finding"

    recorded: list[tuple[str, str, dict[str, object]]] = []
    real_compute = analyze_mod.compute_cost_usd

    def spy(profile_id: str, model: str, **kwargs: object):  # type: ignore[no-untyped-def]
        recorded.append((profile_id, model, kwargs))
        return real_compute(profile_id, model, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(analyze_mod, "compute_cost_usd", spy)
    sink = _NullSink()
    outcome = await _process_one_trace_fetched_file(
        fetched_file=TraceFetchedFile(
            path="src/traced.py",
            content_head="def traced():\n    return 1\n",
            source_finding_id=finds[0].finding_id,
        ),
        source_finding=finds[0],
        review_id=_uuid4(),
        installation_id=1,
        is_eval=True,
        provider=_OpenAIContextScriptedProvider(_MISSES_RESPONSE),
        analyze_model="gpt-5.6-sol",
        import_path_resolver=_NoOpImportPathResolver(),
        file_examination_sink=sink,
        analyze_event_sink=sink,
        active_policy_version=ACTIVE_POLICY_VERSION,
        pass_index=1,
        per_file_cap_tokens=100_000,
        remaining_budget_tokens=100_000,
    )
    assert outcome is not None
    openai_calls = [(p, m, k) for (p, m, k) in recorded if p == "openai"]
    assert len(openai_calls) == 1, "the trace-expansion site must price exactly one call"
    _, _, kwargs = openai_calls[0]
    assert kwargs["billed_prompt_tokens"] == 100
    assert kwargs["service_tier"] == "default"
