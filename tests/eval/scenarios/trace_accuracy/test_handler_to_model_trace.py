"""Trace-accuracy eval scenario: handler-to-model resolution via direct import.

Per spec §11.2: PR modifies a request handler where the vulnerability is
resolvable via simple direct import to a model definition file. Expected:
`TraceDecision.resolution_status="resolved"` + `target_file` pointing at
the model file + `target_file in candidates_considered` (per
DECISIONS.md#017 — resolved selection must be a member of the LLM-proposed
candidate set).

`trace_path` is NOT a field on `TraceDecision` (`docs/spec.md` §7.2). It
lives on:
- `ReviewFinding.trace_path` — for INFERRED findings, per the proof
  boundary. The corresponding finding is identified by
  `decision.source_finding_id`.
- `TraceDecisionEvent.trace_path` (audit event) — same data, on the
  audit-log side.

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
    # Per DECISIONS.md#017: resolved target_file must be a member of
    # candidates_considered. This is the canonical "resolved" contract;
    # the schema-level validator enforces it on construction, but this
    # assertion documents the contract at the test surface for clarity.
    assert decision.target_file in decision.candidates_considered
