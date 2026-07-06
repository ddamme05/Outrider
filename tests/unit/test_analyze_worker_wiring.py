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
from typing import Any

import pytest

# Reuse the node harness: fixtures, builders, and the scripted deps.
from test_analyze_node import (  # noqa: F401  (deps is a fixture)
    _build_changed_file,
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    analyze,
    deps,
)

from outrider.agent.nodes.analyze_aggregate import fold_worker_outcomes
from outrider.ast_facts.models import SkipReason
from outrider.policy import EvidenceTier
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
    result = await analyze(state, **deps)

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
    result = await analyze(state, **deps)

    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"
    by_query = {f.query_match_id: f for f in outcome.admitted_findings}
    producer = by_query["python.command_injection_os_system"]
    cited = by_query["python.function_definition"]
    assert cited.evidence_tier is EvidenceTier.OBSERVED  # admitted as OBSERVED...
    assert outcome.producer_observed_hashes == (producer.content_hash,)
    assert cited.content_hash not in outcome.producer_observed_hashes  # ...but not producer-listed


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
    result = await analyze(state, **deps)

    outcomes = tuple(result["analyze_worker_outcomes"])
    assert len(outcomes) == 2
    sequential_round = result["analysis_rounds"][0]
    fold = fold_worker_outcomes(
        outcomes,
        pass_index=0,
        started_at=sequential_round.started_at,
        ended_at=sequential_round.ended_at,
    )
    # ROUND IDENTITY IS EQUAL: the sequential loop iterates
    # tier-descending (budget-pressure order) and the fold canonicalizes
    # to sorted-path order, so the state-visible TUPLES may differ in
    # order — but compute_round_id sorts its hashed inputs internally,
    # so both implementations produce the SAME round_id and collapse as
    # one round on the dedup reducer. Content assertions stay
    # order-insensitive because tuple order is the one permitted delta.
    assert fold.round.round_id == sequential_round.round_id
    assert sorted(f.content_hash for f in fold.round.findings) == sorted(
        f.content_hash for f in sequential_round.findings
    )
    assert set(fold.round.files_examined) == set(sequential_round.files_examined)
    assert set(fold.round.files_skipped) == set(sequential_round.files_skipped)

    # Accounting parity against the emitted event — EVERY fold-owned
    # counter the event carries, none omitted.
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
    result = await analyze(state, **deps)
    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"
    assert outcome.path == "src/example.py"
    assert outcome.pass_index == 0
