"""Graph execution under OUTRIDER_LLM_HOST=baseten (GLM-5.2): the host-qualified triad
flows through the non-analyze nodes the GLM scorecard never runs.

The GLM scorecard proves analyze-QUALITY on real Baseten calls, but it runs analyze in
ISOLATION (no triage/trace/synthesize/hitl/publish). These two tests drive the REAL graph
via the eval drivers under host="baseten", so the non-analyze nodes run under a
non-anthropic identity triad (DECISIONS.md#056) — the coverage FUP-194 flagged as missing.

Split across two tests because `run_review`'s single pass STOPS at the HITL gate on a
CRITICAL finding (boundary #6: nothing reaches GitHub without an explicit decision), so
publish is unreachable from it. Together they cover intake→triage→analyze→synthesize→
hitl→resume→publish under the baseten triad:

  * test_graph_gates_at_hitl_under_baseten_host — single-pass: intake→triage→analyze→
    (trace, no loop)→synthesize→hitl(GATED). Proves the triad flows through analyze +
    synthesize (synthesize sums the baseten-stamped LLMCallEvents into ReviewMetrics) and
    the HITL gate holds identically under a non-anthropic host.
  * test_resume_reaches_publish_under_baseten_host — resume: the explicit
    Command(resume=...) drives hitl→publish, so PUBLISH runs under the baseten triad.

Why completion is the proof: the scripted provider + every per-node completion event stamp
the BASETEN triad, and the audit persister ENFORCES host-qualification on fresh writes
(`_assert_fresh_triad_qualified` raises `AuditPersisterUnqualifiedFreshWriteError` on an
unqualified event). So if the triad failed to flow through any covered node, the run would
RAISE — a completed run / a posted comment is the proof the triad reached that node.

Not covered (intentionally): the analyze⇄trace LOOP. The trace NODE runs in both tests (it
examines the findings and proceeds with no candidates), but no fixture here has trace
candidates, so analyze never re-enters. The trace node's ranking call goes through the SAME
host-stamping provider path as analyze, so it carries no host-qualification risk beyond what
these tests already prove (FUP-194).

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
    """Resume past the HITL gate so PUBLISH runs under host='baseten'.

    The publish + resume nodes are exactly what `run_review`'s single pass cannot reach
    (it stops at the gate). The explicit `Command(resume=...)` supplies the approval, and
    publish runs only after it (boundary #6) — under the baseten triad. A posted comment is
    the proof publish executed and its completion events cleared host-qualification."""
    resumed = await run_review_with_resume(_RESUME_FIXTURE, db_url=eval_db, host="baseten")

    assert resumed.hitl_gated is True  # the CRITICAL finding gated; resume approved it
    # Boundary #6 positive half: the gated finding reached GitHub ONLY after the resume
    # decision — publish RAN under the baseten triad and posted its comment(s).
    assert len(resumed.published_comments) >= 1
    assert resumed.review_status == "completed"
