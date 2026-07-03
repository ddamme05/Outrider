"""Trace-accuracy eval scenario: JS relative-specifier resolution end-to-end.

Per `specs/2026-07-03-js-ts-trace-resolver.md`: a JS PR's scripted pass-0
finding proposes the relative specifier `'./db'`; trace fans out the
pragmatic-six probe set against fake GitHub, resolves the single real path
(`src/routes/db.js`, served beyond-diff in `repository_contents_head`),
pass 1 analyzes the fetched file, and a grounded INFERRED finding admits
with an admissible `trace_path` — the reachability this feature exists to
provide. The same run scripts two negative properties as explicit attacks:

- a `'../../evil'` candidate (shape-valid, repo-root-contained after two
  pops from `src/routes/`) probes its fan-out, misses everything, and
  lands an `unresolved` TraceDecision — the designed degradation;
- a pass-1 proposal CLAIMING `observed` with a fabricated Python query id
  on the fetched JS file, which admission must reject (the dispatch
  feature's language-gated query-id set keeps OBSERVED impossible for
  JS/TS) — asserted as only-the-INFERRED-finding admitting in round 2.

Two singleton candidate buckets skip the Haiku ranking call, so no trace
LLM response is scripted.
"""


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

    # OBSERVED stays impossible for JS/TS: the pass-1 response CLAIMED an
    # observed finding with a fabricated Python query id — nothing
    # observed may survive admission anywhere in the run.
    assert all(f.evidence_tier != "observed" for f in review_state.findings)
    # Exactly the two scripted pass-0 JUDGED findings + the traced
    # INFERRED one made it through.
    assert len(review_state.findings) == 3
