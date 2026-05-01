"""Trace-accuracy eval scenario: handler-to-model resolution via direct import.

Per spec §11.2: PR modifies a request handler where the vulnerability is
resolvable via simple direct import to a model definition file. Expected:
`TraceDecisionEvent.resolution_status="resolved"` + `target_file`
pointing at the model file + `trace_path` listing the import-walked
scopes.

V1: scaffolded; assertions wire up when the trace node lands per §15.3.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires trace node + ast_facts import resolver")

EXPECTED_DECISION = {
    "resolution_status": "resolved",
    # target_file: pinned at flip time; the test fixture defines the
    # canonical handler→model relationship.
}


def test_handler_to_model_trace_resolves_via_direct_import() -> None:
    """Trace node walks handler→model_definition import + records resolved decision."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    review_state = run_review("tests/eval/fixtures/mock_github/handler_to_model_trace.json")
    trace_decisions = review_state.trace_decisions
    assert len(trace_decisions) >= 1
    decision = trace_decisions[0]
    assert decision.resolution_status == EXPECTED_DECISION["resolution_status"]
    assert decision.target_file is not None
    assert decision.trace_path is not None
    assert len(decision.trace_path) > 0
