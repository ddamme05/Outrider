"""True-positive eval scenario: JS file through the full graph (dispatch spec).

The full-graph proof the adapters spec deferred: a `.js` PR drives the
real 7-node graph via `run_review` — the analyze registry dispatch
parses it with the JavaScript adapter, the scope-aware prompt renders,
and the finding lands as JUDGED (the Python-only OBSERVED machinery is
gated off for JS/TS, so `judged` is the CORRECT tier here, not a
degradation) with severity from policy.
"""

from outrider.policy import EvidenceTier, FindingType, lookup_severity


def test_js_pr_drives_the_full_graph_with_judged_finding() -> None:
    """A JS PR produces SQL_INJECTION + JUDGED + policy severity end to end."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/js_dispatch_sql_injection.json")
    sql_findings = [f for f in findings if f.finding_type == FindingType.SQL_INJECTION]
    assert len(sql_findings) >= 1
    finding = sql_findings[0]
    # JUDGED by design: queries/ is Python-only and the OBSERVED
    # producer is language-gated (proof boundary, dispatch spec).
    assert finding.evidence_tier == EvidenceTier.JUDGED
    assert finding.query_match_id is None
    assert finding.severity == lookup_severity(FindingType.SQL_INJECTION)


def test_js_pr_emits_no_observed_findings() -> None:
    """Zero-OBSERVED proof at graph level: no finding from a JS-only PR
    may carry the observed tier (there is no JS query registry to cite)."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/js_dispatch_sql_injection.json")
    assert findings, "fixture must produce at least one finding"
    assert all(f.evidence_tier != EvidenceTier.OBSERVED for f in findings)
