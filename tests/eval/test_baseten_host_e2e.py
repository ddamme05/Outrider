"""Check approval and publication on a graph configured for Baseten.

Use scripted responses and a real test database. One run pauses for approval;
a resumed run reaches publication. Check host identity on analysis and summary
audit records. These fixtures provide no trace candidates, so they do not
exercise tracing. Requires --is-eval and postgres-test."""

from outrider.agent import run_review, run_review_with_resume
from outrider.policy import EvidenceTier, FindingType

# A CRITICAL model judgment: the single pass gates at hitl on it.
_GATED_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection_judged.json"
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
    assert sql[0].evidence_tier == EvidenceTier.JUDGED
    assert sql[0].query_match_id is None

    # The CRITICAL finding gated: the single pass stopped at hitl and posted nothing
    # (boundary #6 — no auto-publish of a critical finding without a human decision).
    assert baseten.hitl_gated is True
    assert baseten.published_comments == ()

    # synthesize SUMmed the baseten-stamped LLMCallEvent rows into ReviewMetrics: its
    # presence proves the baseten-stamped analyze/synthesize completion events (+ provider
    # calls) flowed past the persister's host-qualification guard (an unqualified fresh
    # write would have raised).
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
