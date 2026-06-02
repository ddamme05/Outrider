"""Trace-accuracy eval scenario: handler-to-model resolution via direct import.

Per spec §11.2: PR modifies a request handler where the vulnerability is
resolvable via simple direct import to a model definition file. Expected:
`TraceDecision.resolution_status="resolved"` + `target_file` pointing at
the model file + `resolved_candidate_paths` containing exactly one path
equal to `target_file` (per DECISIONS.md#017 × #024: the resolved
selection equals the single `resolved_candidate_paths` entry).

`trace_path` is NOT a field on `TraceDecision` (`docs/spec.md` §7.2). It
lives on:
- `ReviewFinding.trace_path` — for INFERRED findings, per the proof
  boundary. The corresponding finding is identified by
  `decision.source_finding_id`.
- `TraceDecisionEvent.trace_path` (audit event) — same data, on the
  audit-log side.

Driven by the eval graph driver (`run_review`) against
`tests/eval/fixtures/mock_github/handler_to_model_trace.json`. The changed
handler (`app/handlers.py`) imports `app.models`; the model file
(`app/models.py`) lives ONLY in the fixture's `repository_contents_head`
(beyond the diff — NOT in `files`), so trace's two-phase probe fetches it at
`head_sha` and resolves. That beyond-diff fetch is the whole point of this
scenario. A single trace candidate skips the Haiku ranking call, so no
`trace` LLM response is scripted.
"""

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
