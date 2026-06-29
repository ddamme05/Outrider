"""Full-graph end-to-end under OUTRIDER_LLM_HOST=baseten (GLM-5.2).

The GLM scorecard proves *analyze-quality* on real Baseten calls, but it runs analyze
in ISOLATION (no trace/synthesize/hitl/publish). This drives the REAL 7-node graph via
`run_review(host="baseten")` so the non-analyze nodes run under a non-anthropic host —
the coverage the audit (FUP-194) flagged as missing.

What a completed baseten run proves end-to-end: the scripted provider + every per-node
completion event stamp the BASETEN identity triad (DECISIONS.md#056), and the audit
persister ENFORCES host-qualification on fresh writes (`_assert_fresh_triad_qualified`
raises `AuditPersisterUnqualifiedFreshWriteError` on an unqualified event). So if the
baseten triad failed to flow through analyze/synthesize, the run would RAISE — completion
is the proof. The findings match the anthropic path (the scripted model output is
host-independent); only the host metadata + pricing differ.

Driver-backed: needs `--is-eval` + the postgres-test DB (same as every run_review test).
"""

from outrider.policy import EvidenceTier, FindingType

_FIXTURE = "tests/eval/fixtures/mock_github/pygoat_sql_injection.json"


def test_full_graph_runs_end_to_end_under_baseten_host() -> None:
    """The full 7-node graph completes under host='baseten', produces the same review the
    anthropic path does, and its completion events are accepted (host-qualified)."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    baseten = run_review(_FIXTURE, host="baseten")
    anthropic = run_review(_FIXTURE)  # default host -> anthropic, byte-identical to before

    # The scripted model output is host-independent, so the review is identical: the same
    # finding types in the same order. (Only the host triad + pricing differ underneath.)
    assert [f.finding_type for f in baseten.findings] == [
        f.finding_type for f in anthropic.findings
    ]
    sql = [f for f in baseten.findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql) >= 1
    assert sql[0].evidence_tier == EvidenceTier.OBSERVED

    # synthesize SUMmed the baseten-stamped LLMCallEvent rows into ReviewMetrics: its
    # presence proves the completion events flowed past the persister's host-qualification
    # guard under the baseten triad (an unqualified fresh write would have raised).
    assert baseten.review_metrics is not None
