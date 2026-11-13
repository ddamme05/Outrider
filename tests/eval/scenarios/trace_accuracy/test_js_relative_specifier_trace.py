"""Resolve a JavaScript import and review its target outside the changed files.

The real graph uses fake GitHub and scripted responses. Check the inferred
finding, unresolved candidate and rejection of fabricated structural proof.
Singleton candidate groups do not require a ranking model call."""


def test_js_relative_specifier_trace_resolves_and_grounds_inferred() -> None:
    from outrider.agent import run_review  # type: ignore[import-not-found]

    review_state = run_review("tests/eval/fixtures/mock_github/js_relative_specifier_trace.json")

    # --- trace decisions: one per finding bucket, matched by proposal ---
    decisions = {d.proposed_import_strings: d for d in review_state.trace_decisions}
    assert set(decisions) == {("./db",), ("../../evil",)}

    resolved = decisions[("./db",)]
    assert resolved.resolution_status == "resolved"
    assert resolved.target_file == "src/routes/db.js"
    assert resolved.resolved_candidate_paths == ("src/routes/db.js",)

    unresolved = decisions[("../../evil",)]
    assert unresolved.resolution_status == "unresolved"
    assert unresolved.target_file is None
    assert unresolved.resolved_candidate_paths == ()

    # --- pass 1: the grounded INFERRED finding admits on the FETCHED
    # beyond-diff file (its file_path proves the phase-2 fetch + pass-1
    # analysis happened); the fabricated observed claim does not admit ---
    inferred_findings = [f for f in review_state.findings if f.evidence_tier == "inferred"]
    assert len(inferred_findings) == 1
    inferred = inferred_findings[0]
    assert inferred.file_path == "src/routes/db.js"
    assert inferred.trace_path == ("runQuery",)
    assert inferred.finding_type == "sql_injection"

    # No observed finding may survive THIS run: the pass-1 response CLAIMED
    # observed with a fabricated Python query id (model claims always reject
    # on JS/TS — the structural admission set is empty by registration), and
    # the deterministic producer is silent here by construction — the pass-0
    # file fires no catalog query, and the producer never runs on
    # trace-fetched files. OBSERVED on a catalog-MATCHING JS file is the
    # structural JS catalog scenario's positive case.
    assert all(f.evidence_tier != "observed" for f in review_state.findings)
    # Exactly the two scripted pass-0 JUDGED findings + the traced
    # INFERRED one made it through.
    assert len(review_state.findings) == 3
