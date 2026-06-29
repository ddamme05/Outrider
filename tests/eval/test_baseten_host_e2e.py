"""Graph execution under OUTRIDER_LLM_HOST=baseten (GLM-5.2): the non-analyze nodes run on a
Baseten-built graph the GLM scorecard never reaches.

The GLM scorecard proves analyze-QUALITY on real Baseten calls, but it runs analyze in
ISOLATION (no triage/trace/synthesize/hitl/publish). These two tests drive the REAL graph
via the eval drivers under host="baseten", so the non-analyze nodes run on a graph built
with the non-anthropic host identity (DECISIONS.md#056) — the coverage FUP-194 flagged as
missing.

Split across two tests because `run_review`'s single pass STOPS at the HITL gate on a
CRITICAL finding (boundary #6: nothing reaches GitHub without an explicit decision), so
publish is unreachable from it. Together they cover intake→triage→analyze→synthesize→
hitl→resume→publish on the Baseten-built graph:

  * test_graph_gates_at_hitl_under_baseten_host — single-pass: intake→triage→analyze→
    synthesize→hitl(GATED). Proves the triad flows through analyze + synthesize (synthesize
    sums the baseten-stamped LLMCallEvents into ReviewMetrics) and the HITL gate holds
    identically under a non-anthropic host. (`_analyze_router` skips the trace node — see below.)
  * test_resume_reaches_publish_under_baseten_host — resume: the explicit
    Command(resume=...) drives hitl→publish, so PUBLISH runs on the Baseten-built graph.

Why this proves host-qualification: every LLM call stamps the BASETEN triad on its
`LLMCallEvent`, and analyze + synthesize additionally stamp it on their
`AnalyzeCompletedEvent` / `SynthesizeCompletedEvent` (DECISIONS.md#056). The audit persister
ENFORCES host-qualification on those fresh writes (`_assert_fresh_triad_qualified` raises
`AuditPersisterUnqualifiedFreshWriteError` on an unqualified triad-bearing event), so if the
triad failed to flow through analyze or synthesize the run would RAISE — a completed run is
the proof it cleared the guard there. Publish is NOT a triad-bearing node: it makes no LLM
calls and emits no triad event (publish.py:10), so a posted comment proves only that resume
drove the BASETEN-BUILT graph through to publish — host-qualification is already proven
upstream by the single-pass test, not re-proven at publish.

Not covered (intentionally): the trace NODE (and so the analyze⇄trace loop). Both fixtures
emit `"trace_candidates": []`, so `_analyze_router` (graph.py) routes analyze→synthesize
directly — the trace node is SKIPPED, not run-and-empty. Covering it under baseten needs a
fixture whose analyze output carries trace candidates; once it runs, trace's ranking call
goes through the SAME host-stamping provider path as analyze, so it carries no
host-qualification risk beyond what these tests already prove (deferred — FUP-194).

Driver-backed: needs `--is-eval` + the postgres-test DB (same as every run_review / resume test).
"""

from outrider.agent import run_review, run_review_with_resume
from outrider.policy import EvidenceTier, FindingType

# A CRITICAL OBSERVED SQL-injection finding: the single pass gates at hitl on it.
_GATED_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"
# The resume scenario's fixture — a CRITICAL finding proven to reach publish-with-comments
# once the gate is approved (tests/eval/scenarios/hitl_resume/).
_RESUME_FIXTURE = "tests/eval/fixtures/mock_github/hitl_resume_critical.json"


def test_graph_gates_at_hitl_under_baseten_host() -> None:
    """Single-pass under host='baseten' produces the same review the anthropic path does
    and gates at HITL (publish unreached); completion proves the triad flowed through
    analyze + synthesize past the persister's host-qualification guard."""
    baseten = run_review(_GATED_FIXTURE, host="baseten")
    anthropic = run_review(_GATED_FIXTURE)  # default host -> anthropic, byte-identical to before

    # The scripted model output is host-independent, so the review is identical: the same
    # finding types in the same order. (Only the host triad + pricing differ underneath.)
    assert [f.finding_type for f in baseten.findings] == [
        f.finding_type for f in anthropic.findings
    ]
    sql = [f for f in baseten.findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql) >= 1
    assert sql[0].evidence_tier == EvidenceTier.OBSERVED

    # The CRITICAL finding gated: the single pass stopped at hitl and posted nothing
    # (boundary #6 — no auto-publish of a critical finding without a human decision).
    assert baseten.hitl_gated is True
    assert baseten.published_comments == ()

    # synthesize SUMmed the baseten-stamped LLMCallEvent rows into ReviewMetrics: its
    # presence proves the completion events flowed past the persister's host-qualification
    # guard under the baseten triad (an unqualified fresh write would have raised).
    assert baseten.review_metrics is not None


async def test_resume_reaches_publish_under_baseten_host(eval_db: str) -> None:
    """Resume past the HITL gate so PUBLISH runs on a graph both legs built with host='baseten'.

    The publish + resume nodes are exactly what `run_review`'s single pass cannot reach
    (it stops at the gate). The explicit `Command(resume=...)` supplies the approval, and
    publish runs only after it (boundary #6). A posted comment proves resume drove the
    BASETEN-BUILT graph through to publish; publish itself makes no LLM calls and emits no
    triad event (publish.py:10), so this asserts reachability, not that publish cleared the
    host-qualification guard — that guard is proven upstream by the single-pass test."""
    resumed = await run_review_with_resume(_RESUME_FIXTURE, db_url=eval_db, host="baseten")

    assert resumed.hitl_gated is True  # the CRITICAL finding gated; resume approved it
    # Boundary #6 positive half: the gated finding reached GitHub ONLY after the resume
    # decision — publish RAN on the Baseten-built graph and posted its comment(s).
    assert len(resumed.published_comments) >= 1
    assert resumed.review_status == "completed"
