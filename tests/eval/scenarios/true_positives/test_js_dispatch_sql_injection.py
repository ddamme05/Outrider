"""Check JavaScript analysis through the real graph with a scripted SQL finding.

This fixture is outside the structural query catalog, so its finding is model
judgment with severity assigned by policy."""

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
