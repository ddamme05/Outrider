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

V1: scaffolded; assertions wire up when the eval graph driver lands
(trace node + import resolver shipped) per §15.3.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver (mock LLM provider + run_review shim + "
    "mock_github fixtures); trace node + import resolver already shipped"
)

EXPECTED_DECISION = {
    "resolution_status": "resolved",
    # target_file: pinned at flip time; the test fixture defines the
    # canonical handler→model relationship.
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
    assert decision.target_file is not None
    # Per DECISIONS.md#017 × #024: a resolved target_file must equal the
    # single resolved_candidate_paths entry. The schema-level validator
    # enforces it on construction; this assertion documents the contract
    # at the test surface for clarity.
    assert len(decision.resolved_candidate_paths) == 1
    assert decision.target_file == decision.resolved_candidate_paths[0]
