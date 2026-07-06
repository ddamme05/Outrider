# Worker-outcome wiring pins per specs/2026-07-05-parallel-analyze.md (3b-2c-1).
"""Real-node wiring: outcomes constructed from actual branch data, and the
FOLD-vs-SEQUENTIAL parity that justifies the fan-out cutover.

These tests drive the REAL analyze node (the same harness as
test_analyze_node.py) and assert (a) every pass-0 file yields a worker
outcome whose source matches its sequential branch, (b) producer origin
comes from real `produce_observed_findings` output through the real #054
merge site (the reviewer's origin-truth acceptance gate, end-to-end), and
(c) folding the state outcomes reproduces the sequential round and
accounting — the cutover's parity contract.
"""

# ruff: noqa: F811  — the imported deps fixture is intentionally shadowed by test params
from __future__ import annotations

import json
from types import MappingProxyType
from typing import Any

import pytest

# Reuse the node harness: fixtures, builders, and the scripted deps.
from test_analyze_node import (  # noqa: F401  (deps is a fixture)
    _build_changed_file,
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _ConfigurableTokensStubProvider,
    _StubLLMProvider,
    deps,
    run_analyze_pass,
)

import outrider.queries.registry as query_registry
from outrider.agent.nodes.analyze_aggregate import fold_worker_outcomes
from outrider.ast_facts.models import SkipReason
from outrider.policy import EvidenceTier
from outrider.queries.observed import QueryClass
from outrider.schemas.triage_result import ReviewTier

_KILL_SWITCH_JS = b'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
_KILL_SWITCH_PATCH = (
    "--- a/src/index.js\n+++ b/src/index.js\n"
    '@@ -0,0 +1,1 @@\n+process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n'
)


@pytest.mark.asyncio
async def test_ride_out_produces_observed_skip_outcome_with_real_producer_origin(
    deps: dict[str, Any],
) -> None:
    """The ride-out path with REAL producer output: the state outcome must
    be observed_skip with the producer's finding listed by hash. (The
    origin-truth gate proper — model-cited exclusion through the parser's
    #054 merge — is the dedicated test below.)"""
    cf = _build_changed_file(
        path="src/index.js",
        content=_KILL_SWITCH_JS,
        patch=_KILL_SWITCH_PATCH,
        content_base="",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/index.js": ReviewTier.DEEP}),
    )
    deps["total_review_budget_tokens"] = 100
    result = await run_analyze_pass(state, deps)

    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "observed_skip"
    assert outcome.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED
    (finding,) = outcome.admitted_findings
    assert finding.evidence_tier is EvidenceTier.OBSERVED
    assert finding.query_match_id == "javascript.tls_env_verify_disabled"
    assert outcome.producer_observed_hashes == (finding.content_hash,)
    # Non-aliasing: the state outcome's finding is a clone, and the round's
    # finding is a further clone — no shared object identity.
    (round_finding,) = result["analysis_rounds"][0].findings
    assert round_finding is not finding
    assert round_finding.content_hash == finding.content_hash


# Deliberately vulnerable FIXTURE CONTENT (never executed): os.system is
# the trigger for the python.command_injection_os_system producer query.
_OS_SYSTEM_PY = b"""\
import os

def run(cmd):
    os.system(cmd)
"""
_OS_SYSTEM_PATCH = (
    "--- a/src/runner.py\n+++ b/src/runner.py\n"
    "@@ -0,0 +1,4 @@\n+import os\n+\n+def run(cmd):\n+    os.system(cmd)\n"
)


@pytest.mark.asyncio
async def test_model_cited_observed_is_excluded_from_producer_list_end_to_end(
    deps: dict[str, Any],
) -> None:
    """ORIGIN TRUTH through the REAL parser and #054 merge: the scripted
    model cites a structural id (`python.function_definition` fires on
    this file, so the citation admits as OBSERVED) while the real producer
    fires `python.command_injection_os_system` on the same file. The
    outcome's producer list must carry EXACTLY the producer's finding —
    the model-cited OBSERVED finding is admitted but stays out, because
    origin derives from the merge's object placement, never from tier."""
    response_json = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "sql_injection",
                    "evidence_tier": "observed",
                    "query_match_id": "python.function_definition",
                    "trace_path": None,
                    "title": "Model-cited structural OBSERVED",
                    "description": "A legitimate proposal citing a fired structural query.",
                    "evidence": "def run(cmd):\n    os.system(cmd)",
                    "line_start": 3,
                    "line_end": 4,
                    "trace_candidates": [],
                }
            ]
        }
    )
    deps["provider"] = _StubLLMProvider(response_json)
    cf = _build_changed_file(
        path="src/runner.py",
        content=_OS_SYSTEM_PY,
        patch=_OS_SYSTEM_PATCH,
        content_base="",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/runner.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)

    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"
    by_query = {f.query_match_id: f for f in outcome.admitted_findings}
    producer = by_query["python.command_injection_os_system"]
    cited = by_query["python.function_definition"]
    assert cited.evidence_tier is EvidenceTier.OBSERVED  # admitted as OBSERVED...
    assert outcome.producer_observed_hashes == (producer.content_hash,)
    assert cited.content_hash not in outcome.producer_observed_hashes  # ...but not producer-listed


def _assert_fold_parity(result: dict[str, Any], deps: dict[str, Any]) -> Any:
    """Fold the wired state outcomes and assert FULL parity with the
    sequential round + emitted AnalyzeCompletedEvent — every fold-owned
    counter the event carries, none omitted. Returns the fold so callers
    can add scenario-specific (e.g. non-zero) assertions on top.

    ROUND IDENTITY IS EQUAL: the sequential loop iterates tier-descending
    (budget-pressure order) and the fold canonicalizes to sorted-path
    order, so the state-visible TUPLES may differ in order — but
    compute_round_id sorts its hashed inputs internally, so both
    implementations produce the SAME round_id and collapse as one round
    on the dedup reducer. Content assertions stay order-insensitive
    because tuple order is the one permitted delta."""
    outcomes = tuple(result["analyze_worker_outcomes"])
    sequential_round = result["analysis_rounds"][0]
    fold = fold_worker_outcomes(
        outcomes,
        pass_index=0,
        started_at=sequential_round.started_at,
        ended_at=sequential_round.ended_at,
    )
    assert fold.round.round_id == sequential_round.round_id
    assert sorted(f.content_hash for f in fold.round.findings) == sorted(
        f.content_hash for f in sequential_round.findings
    )
    assert set(fold.round.files_examined) == set(sequential_round.files_examined)
    assert set(fold.round.files_skipped) == set(sequential_round.files_skipped)

    (event,) = [e for e in deps["analyze_event_sink"].completed if e.pass_index == 0]
    assert fold.n_llm_calls == event.n_llm_calls
    assert fold.n_proposals_seen == event.n_proposals_seen
    assert fold.n_findings_emitted == event.n_findings_emitted
    assert fold.n_findings_served == event.n_findings_served
    assert fold.n_findings_observed == event.n_findings_observed
    assert fold.n_proposals_rejected == event.n_proposals_rejected
    assert fold.n_responses_rejected == event.n_responses_rejected
    assert fold.n_proposals_superseded_by_observed == event.n_proposals_superseded_by_observed
    assert fold.n_proposals_dropped == event.n_proposals_dropped
    assert fold.n_findings_dropped_over_cap == event.n_findings_dropped_over_cap
    assert fold.n_trace_candidates_emitted == event.n_trace_candidates_emitted
    assert fold.n_trace_candidates_dropped_malformed == event.n_trace_candidates_dropped_malformed
    assert sorted(m.model_dump_json() for m in fold.subsumed_matches) == sorted(
        m.model_dump_json() for m in event.subsumed_matches
    )
    assert fold.n_files_analyzed == event.n_files_analyzed
    assert fold.n_files_skipped == event.n_files_skipped
    assert fold.total_input_tokens == event.total_input_tokens
    assert fold.total_output_tokens == event.total_output_tokens
    assert fold.total_cache_read_tokens == event.total_cache_read_tokens
    assert fold.total_cache_write_tokens == event.total_cache_write_tokens
    assert float(fold.total_cost) == event.total_cost_usd
    return fold


@pytest.mark.asyncio
async def test_fold_over_state_outcomes_reproduces_the_sequential_round(
    deps: dict[str, Any],
) -> None:
    """THE PARITY CONTRACT: folding the wired outcomes reproduces the
    sequential accumulation — same kept findings (by content hash), same
    files examined/skipped, same accounting counters as the emitted
    AnalyzeCompletedEvent. This equality is what licenses the fan-out
    cutover to replace the sequential loop with the fold."""
    kill_switch = _build_changed_file(
        path="src/index.js",
        content=_KILL_SWITCH_JS,
        patch=_KILL_SWITCH_PATCH,
        content_base="",
    )
    plain = _build_changed_file(path="src/example.py")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(kill_switch, plain)),
        triage_result=_build_triage_result(
            file_tiers={
                "src/index.js": ReviewTier.DEEP,
                "src/example.py": ReviewTier.DEEP,
            }
        ),
    )
    result = await run_analyze_pass(state, deps)

    assert len(result["analyze_worker_outcomes"]) == 2
    _assert_fold_parity(result, deps)


@pytest.mark.asyncio
async def test_every_pass_zero_file_yields_exactly_one_outcome(
    deps: dict[str, Any],
) -> None:
    """One outcome per kept file, source matching the sequential branch."""
    plain = _build_changed_file(path="src/example.py")
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(plain,)),
        triage_result=_build_triage_result(file_tiers={"src/example.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)
    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"
    assert outcome.path == "src/example.py"
    assert outcome.pass_index == 0


# Deliberately vulnerable FIXTURE CONTENT (never executed): os.system fires
# the command-injection producer at line 5; DES.new fires the weak-crypto
# producer at line 8 — the two anchors the transcription scenario needs.
_MIXED_VULN_PY = b"""\
import os
from Crypto.Cipher import DES

def run(cmd):
    os.system(cmd)

def encrypt(key, data):
    cipher = DES.new(key, DES.MODE_ECB)
    return cipher.encrypt(data)
"""
_MIXED_VULN_PATCH = (
    "--- a/src/mixed.py\n+++ b/src/mixed.py\n"
    "@@ -0,0 +1,9 @@\n"
    "+import os\n+from Crypto.Cipher import DES\n+\n+def run(cmd):\n"
    "+    os.system(cmd)\n+\n+def encrypt(key, data):\n"
    "+    cipher = DES.new(key, DES.MODE_ECB)\n+    return cipher.encrypt(data)\n"
)


def _mixed_vuln_response() -> str:
    """Four scripted proposals engineered to make every previously-zero
    transcription channel non-zero through the REAL node: a #054
    supersession, a #055 subsumption, a same-content-hash duplicate
    (FUP-180 collapse → n_proposals_dropped), and a malformed trace
    candidate (path separator → dropped_malformed)."""

    def proposal(
        finding_type: str,
        line_start: int,
        line_end: int,
        title: str,
        description: str,
        trace_candidates: list[dict[str, str]],
    ) -> dict[str, Any]:
        return {
            "finding_type": finding_type,
            "evidence_tier": "judged",
            "query_match_id": None,
            "trace_path": None,
            "title": title,
            "description": description,
            "evidence": "e",
            "line_start": line_start,
            "line_end": line_end,
            "trace_candidates": trace_candidates,
        }

    return json.dumps(
        {
            "findings": [
                # Same (path, span, type) as the producer's command_injection
                # → #054 prefer-OBSERVED evicts it → superseded = 1.
                proposal("command_injection", 5, 5, "Shell exec", "Model saw it too.", []),
                # More-specific type at the producer weak_crypto's EXACT span
                # → #055 subsumption drops the OBSERVED, retains the record.
                # Also carries the malformed trace candidate (path separator)
                # → n_trace_candidates_dropped_malformed = 1.
                proposal(
                    "weak_password_hash",
                    8,
                    8,
                    "DES for passwords",
                    "Password-specific weak crypto.",
                    [{"import_string_raw": "bad/../import", "reason": "r"}],
                ),
                # Two same-content-hash variants (same span + type, different
                # prose → different proposal_hash) → FUP-180 collapse drops
                # one → n_proposals_dropped = 1.
                proposal("sql_injection", 4, 5, "SQLi one", "First phrasing.", []),
                proposal("sql_injection", 4, 5, "SQLi two", "Second phrasing.", []),
            ]
        }
    )


@pytest.mark.asyncio
async def test_parity_with_every_transcription_channel_non_zero(
    deps: dict[str, Any],
) -> None:
    """ANTI-VACUITY parity: the base parity scenario leaves supersession,
    drops, malformed candidates, cache tokens, and #055 records all at
    zero, so a live transcription bug in any of them would still pass.
    This scenario drives every one of those channels NON-ZERO through the
    real node, asserts each is non-zero (so the scenario can't silently
    decay), then requires full fold-vs-event parity."""
    deps["provider"] = _ConfigurableTokensStubProvider(
        response_text=_mixed_vuln_response(),
        tokens_per_call={
            "input_tokens": 120,
            "output_tokens": 60,
            "cache_read_tokens": 32,
            "cache_write_tokens": 16,
        },
    )
    cf = _build_changed_file(
        path="src/mixed.py",
        content=_MIXED_VULN_PY,
        patch=_MIXED_VULN_PATCH,
        content_base="",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/mixed.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)

    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"
    # Worker-level transcription: each channel non-zero and verbatim.
    assert outcome.n_proposals_superseded_by_observed == 1
    assert outcome.n_trace_candidates_dropped_malformed == 1
    assert outcome.cache_read_tokens == 32
    assert outcome.cache_write_tokens == 16
    assert len(outcome.subsumed_matches) == 1

    fold = _assert_fold_parity(result, deps)
    # Fold-level non-vacuity: the previously-zero channels are live here.
    assert fold.n_proposals_superseded_by_observed == 1
    assert fold.n_proposals_dropped == 1
    assert fold.n_trace_candidates_dropped_malformed == 1
    assert fold.total_cache_read_tokens == 32
    assert fold.total_cache_write_tokens == 16
    assert len(fold.subsumed_matches) == 1


_PLAIN_FUNCS_PY = b"""\
def alpha():
    return 1

def beta():
    return 2
"""
_PLAIN_FUNCS_PATCH = (
    "--- a/src/funcs.py\n+++ b/src/funcs.py\n"
    "@@ -0,0 +1,5 @@\n+def alpha():\n+    return 1\n+\n+def beta():\n+    return 2\n"
)


@pytest.mark.asyncio
async def test_cap_drop_parity_with_low_cap(
    deps: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one channel the non-zero scenario leaves at 0 == 0: cap drops.
    Forcing 200+ admitted findings through the harness is not the point —
    the cap SEMANTICS are; so the soft cap is patched low in BOTH the node
    and the fold (same call-time module-global read), four distinct
    non-gated findings admit, the cap keeps two, and the drop count must
    be EXACTLY two on both sides before full parity runs. Non-gated type
    only: the cap never drops CRITICAL/HIGH (FUP-180)."""
    import outrider.agent.nodes.analyze as analyze_mod
    import outrider.agent.nodes.analyze_aggregate as aggregate_mod

    monkeypatch.setattr(analyze_mod, "MAX_FINDINGS_PER_ROUND", 2)
    monkeypatch.setattr(aggregate_mod, "MAX_FINDINGS_PER_ROUND", 2)
    response_json = json.dumps(
        {
            "findings": [
                {
                    "finding_type": "missing_error_handling",  # LOW — never gated
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": f"Unhandled failure {line}",
                    "description": "d",
                    "evidence": "e",
                    "line_start": line,
                    "line_end": line,
                    "trace_candidates": [],
                }
                for line in (1, 2, 4, 5)  # distinct spans → distinct content hashes
            ]
        }
    )
    deps["provider"] = _StubLLMProvider(response_json)
    cf = _build_changed_file(
        path="src/funcs.py",
        content=_PLAIN_FUNCS_PY,
        patch=_PLAIN_FUNCS_PATCH,
        content_base="",
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/funcs.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)

    (outcome,) = result["analyze_worker_outcomes"]
    assert len(outcome.admitted_findings) == 4  # the outcome is PRE-cap
    assert len(result["analysis_rounds"][0].findings) == 2  # the round is capped

    fold = _assert_fold_parity(result, deps)
    assert fold.n_findings_dropped_over_cap == 2  # exact, non-zero, both sides


# Single-changed-line fixture for the #049 ENFORCED coverage skip: line 6
# (the shell=True call, dead code — never executed fixture content) is the
# only added line, fully covered by the promoted skip_safe query. Mirrors
# tests/eval/scenarios/observed_skip_safe/'s promotion fixture.
_COVERED_PY = (
    b"import subprocess\n\n\ndef run_it(cmd):\n"
    b"    return cmd\n    subprocess.run(cmd, shell=True)\n"
)
_COVERED_PATCH = (
    "--- a/src/vuln.py\n+++ b/src/vuln.py\n"
    "@@ -1,5 +1,6 @@\n import subprocess\n \n \n def run_it(cmd):\n"
    "     return cmd\n+    subprocess.run(cmd, shell=True)\n"
)
_COVERED_BASE = "import subprocess\n\n\ndef run_it(cmd):\n    return cmd\n"


@pytest.mark.asyncio
async def test_enforced_coverage_skip_maps_to_observed_coverage_live(
    deps: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The #049 ENFORCED branch through the REAL node: with a test-local
    skip_safe promotion + analyze_observed_skip_enforced=True, the only
    changed line is fully covered, the LLM never runs, and the wired
    outcome must be observed_coverage — clean status, no SkipReason,
    producer-origin findings, examined-not-skipped — with fold parity."""
    # Test-local promotion (the eval scenario's pattern): swap the module
    # attribute for a COPY; monkeypatch restores at teardown.
    promoted = dict(query_registry.OBSERVED_QUERIES)
    promote_id = "python.command_injection_subprocess_shell"
    promoted[promote_id] = promoted[promote_id].model_copy(
        update={"query_class": QueryClass.SKIP_SAFE}
    )
    monkeypatch.setattr(query_registry, "OBSERVED_QUERIES", MappingProxyType(promoted))
    deps["analyze_observed_skip_enforced"] = True

    cf = _build_changed_file(
        path="src/vuln.py",
        content=_COVERED_PY,
        patch=_COVERED_PATCH,
        content_base=_COVERED_BASE,
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/vuln.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)

    assert deps["provider"].calls == []  # the LLM never ran
    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "observed_coverage"
    assert outcome.parse_status == "clean"
    assert outcome.skip_reason is None
    (finding,) = outcome.admitted_findings
    assert finding.query_match_id == promote_id
    assert outcome.producer_observed_hashes == (finding.content_hash,)

    fold = _assert_fold_parity(result, deps)
    assert fold.n_files_analyzed == 1  # examined, not skipped
    assert fold.n_files_skipped == 0
    assert fold.n_llm_calls == 0
