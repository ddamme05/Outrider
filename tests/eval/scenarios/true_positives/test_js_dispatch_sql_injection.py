"""True-positive eval scenario: JS file through the full graph (dispatch spec).

The full-graph proof the adapters spec deferred: a `.js` PR drives the
real 7-node graph via `run_review` — the analyze registry dispatch
parses it with the JavaScript adapter, the scope-aware prompt renders,
and the finding lands as JUDGED with severity from policy. Since the
JS/TS OBSERVED catalog, `judged` is the correct tier for THIS fixture
specifically because its sink is deliberately outside the catalog
(`db.raw(...)` with the concat at an assignment, not at a `.query`/
`.execute` call site): OBSERVED is per-language-conditional now —
impossible on a file with no matching query, possible on one that
matches (see the structural JS catalog scenario for the positive case).
"""

from outrider.policy import EvidenceTier, FindingType, lookup_severity


def test_js_pr_drives_the_full_graph_with_judged_finding() -> None:
    """A JS PR produces SQL_INJECTION + JUDGED + policy severity end to end."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/js_dispatch_sql_injection.json")
    sql_findings = [f for f in findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql_findings) >= 1
    finding = sql_findings[0]
    # JUDGED by design FOR THIS FIXTURE: its sink (`db.raw`, concat built
    # at an assignment) matches no JS catalog query, so the deterministic
    # producer stays silent and the model's contextual call is the only
    # source (proof boundary; per-language-conditional since the JS/TS
    # OBSERVED catalog).
    assert finding.evidence_tier == EvidenceTier.JUDGED
    assert finding.query_match_id is None
    assert finding.severity == lookup_severity(FindingType.SQL_INJECTION)


def test_js_pr_with_no_matching_query_emits_no_observed_findings() -> None:
    """No-matching-query proof at graph level: a JS PR whose content fires
    no catalog query yields zero OBSERVED findings — the model cannot claim
    the tier (empty structural admission set for JS/TS) and the producer
    has nothing to emit. OBSERVED on a JS file is possible ONLY via a real
    catalog match (the positive case lives in the structural JS catalog
    scenario); this pins the no-match side of the conditional."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/js_dispatch_sql_injection.json")
    assert findings, "fixture must produce at least one finding"
    assert all(f.evidence_tier != EvidenceTier.OBSERVED for f in findings)
