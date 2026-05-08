# Skeletal LangGraph state object per docs/spec.md §7.1 (V1 foundation slice)
"""ReviewState: the LangGraph state envelope, V1 skeletal slice.

This file ships ONLY the slots populated by the intake and triage nodes;
the slots populated by analyze, trace, synthesize, hitl, and publish are
deferred to their respective node specs. The deferred slots and their
dedup-keyed reducers (per spec §7.1) carry a replay-equivalence rationale
that belongs with the node that owns them, not in this schema-foundation
arc. Adding them here would force the spec to introduce reducers it does
not yet exercise, and a partial-reducer surface would silently regress
when the analyze/trace specs land.

Deferred slots — landing with their respective node specs:
- `analysis_rounds: list[AnalysisRound]` (analyze-node spec) — append-with-
  dedup-by(round_id) reducer, idempotent under checkpoint replay.
- `trace_decisions: list[TraceDecision]` (trace-node spec) — append-with-
  dedup-by(source_finding_id) reducer per DECISIONS.md#017.
- `review_report: ReviewReport | None` (synthesize-node spec).
- `hitl_request: HITLRequest | None` (hitl-node spec).
- `hitl_decision: HITLDecision | None` (hitl-node spec; reviewer-set,
  consumed by publish).
- `publish_result: PublishResult | None` (publish-node spec).

ReviewState is NOT frozen: nodes return partial-update dicts that LangGraph
merges through reducers (per docs/conventions.md "LangGraph specifics").
A frozen state would force every node to construct a fully-formed instance
on every return, defeating the reducer contract.

`validate_assignment=True` matches the project precedent established by
`ReviewFinding` (review_finding.py module docstring): when `frozen=False`,
construction-time validators (AwareDatetime, typed enums, nested-model
construction) are bypassable via direct attribute assignment unless every
assignment re-runs the validator chain. Without this flag,
`state.received_at = datetime(...)  # naive` would silently admit, and
`state.pr_context = some_dict` would coerce-or-bypass typed validation
quietly. With it, the assignment raises. Note: under normal LangGraph
operation nodes return dicts and the framework constructs a fresh
`model_validate`d instance per super-step (which validates anyway); the
flag's misuse-resistance value is the secondary defense against any code
path that does `state.field = value` directly (which the conventions doc
forbids but the type system doesn't structurally prevent).

Per spec §7.1: the state object round-trips through Postgres JSON via
langgraph-checkpoint-postgres. All field types must be JSON-serializable;
runtime dependencies (DB sessions, HTTP clients) are NOT in state — they
are injected at build_graph(...) time and closed over in node functions.
"""

from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict

from outrider.schemas.pr_context import PRContext
from outrider.schemas.triage_result import TriageResult


class ReviewState(BaseModel):
    """LangGraph state object — V1 skeletal slice.

    Currently carries only the intake- and triage-populated slots. See
    module docstring for the deferred slots and which spec each lands in.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True)

    # Populated at webhook receipt
    review_id: UUID
    pr_context: PRContext
    received_at: AwareDatetime

    # Populated by triage node (separate spec)
    triage_result: TriageResult | None = None


__all__ = [
    "ReviewState",
]
