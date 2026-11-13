"""Follow a handler's direct import to a model file outside the PR diff.

The real graph resolves and fetches the related file from fake GitHub at the
head revision. A single candidate needs no ranking call. Findings and audit
events carry the trace path; TraceDecision identifies the resolved target."""

EXPECTED_DECISION = {
    "resolution_status": "resolved",
    # The model file imported by the handler, served beyond-diff in
    # repository_contents_head; trace resolves the `app.models` import to it.
    "target_file": "app/models.py",
}


def test_handler_to_model_trace_resolves_via_direct_import() -> None:
    """Trace node walks handler→model_definition import + records resolved decision.

    Verifies `TraceDecision` shape per spec §7.2 + DECISIONS#017 only.
    `trace_path` verification is deferred to the future companion test
    that walks `ReviewFinding.trace_path` (the proof-boundary location)
    or the `TraceDecisionEvent.trace_path` audit-event field.
    """
    from outrider.agent import run_review  # type: ignore[import-not-found]

    review_state = run_review("tests/eval/fixtures/mock_github/handler_to_model_trace.json")
    trace_decisions = review_state.trace_decisions
    assert len(trace_decisions) >= 1
    decision = trace_decisions[0]
    assert decision.resolution_status == EXPECTED_DECISION["resolution_status"]
    assert decision.target_file == EXPECTED_DECISION["target_file"]
    # Per DECISIONS.md#017 × #024: a resolved target_file must equal the
    # single resolved_candidate_paths entry. The schema-level validator
    # enforces it on construction; this assertion documents the contract
    # at the test surface for clarity.
    assert len(decision.resolved_candidate_paths) == 1
    assert decision.target_file == decision.resolved_candidate_paths[0]
