# Skeletal LangGraph state object per docs/spec.md §7.1 (V1 foundation slice)
"""ReviewState: the LangGraph state envelope, V1 skeletal slice.

This file ships ONLY the slots populated at webhook seed time
(`review_id`, `pr_context`, `received_at` per `DECISIONS.md#020`) plus the
slot populated by the triage node (`triage_result`). Per `DECISIONS.md#020`,
the webhook receiver constructs the seed `ReviewState` (with seed
`PRContext` carrying `installation_id` + repo coords + PR identity + author
+ totals + empty `changed_files=()`) and the dispatcher carries that state
to the graph; intake then enriches `pr_context` in place by fetching the
file list + per-file content and returning a fresh `PRContext` with the
populated `changed_files` tuple via `{"pr_context": new_pr_context}`.

The slots populated by analyze, trace, synthesize, hitl, and publish are
deferred to their respective node specs. The deferred slots and their
dedup-keyed reducers (per spec §7.1) carry a replay-equivalence rationale
that belongs with the node that owns them, not in this schema-foundation
arc. Adding them here would force the spec to introduce reducers it does
not yet exercise, and a partial-reducer surface would silently regress
when the analyze/trace specs land.

Slots populated by analyze/trace/synthesize/hitl/publish are landing
with their respective node specs as those land. The current set
includes `analysis_rounds` + `trace_candidates` per §3 of
`specs/2026-05-19-analyze-foundation.md` — both consume the
`append_with_dedup_by` reducer from `outrider.agent.reducers`, which is
idempotent under LangGraph checkpoint replay.

Still deferred:
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
assignment re-runs the validator chain. With the flag, invalid values
raise on assignment — `state.received_at = datetime(...)  # naive` raises
because the AwareDatetime validator fires; `state.pr_context = "string"`
raises because PRContext is not str-coercible; `state.pr_context = {...}`
with an INVALID/INCOMPLETE dict shape raises because PRContext's own
construction validators fire on the dict. Important Pydantic 2.12 nuance
a VALID dict-shaped
payload that matches PRContext's schema IS validated and constructed into
a fresh PRContext instance on assignment — that's reconstructive
validation, not bypass. Tests pin the rejection cases (string-rejection,
incomplete-dict-rejection, naive-datetime-rejection), and the well-typed
escape-hatch test pins that a fresh PRContext instance assignment also
succeeds. Per LangGraph 1.1.6 (`narrative/use-graph-api.md` "Use Pydantic
models for graph state"), Pydantic structural validation runs ONLY on
the first node's input — subsequent nodes' outputs and reducer-merged
state are NOT auto-validated, and the graph output is not a Pydantic
instance. So `validate_assignment=True` is the primary structural defense
across the post-first-input lifetime (catches direct attribute mutation
post-merge); callers needing typed-validated post-reducer state must
construct via `ReviewState.model_validate({**state.model_dump(), **delta})`
rather than bare `model_copy(update=...)` (Pydantic 2 docs explicitly say
`model_copy` does NOT revalidate the update payload).
An earlier docstring claim that the framework constructs a fresh
`model_validate`d instance per super-step — the local LangGraph docs
confirm that's false.

Per spec §7.1: the state object round-trips through Postgres JSON via
langgraph-checkpoint-postgres. All field types must be JSON-serializable;
runtime dependencies (DB sessions, HTTP clients) are NOT in state — they
are injected at build_graph(...) time and closed over in node functions.
"""

from typing import Annotated
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from outrider.agent.reducers import append_with_dedup_by
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.pr_context import PRContext
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.triage_result import TriageResult


class ReviewState(BaseModel):
    """LangGraph state object — V1 skeletal slice.

    Currently carries only the slots that exist at webhook seed time per
    DECISIONS.md#020 (`review_id`, `pr_context`, `received_at`) plus the
    slot populated by triage (`triage_result`). Intake enriches
    `pr_context.changed_files` in place by returning a fresh PRContext via
    `{"pr_context": new_pr_context}`; it does not populate a new top-level
    state slot. See module docstring for the deferred slots and which
    spec each lands in.
    """

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True)

    # Populated at webhook receipt
    review_id: UUID
    pr_context: PRContext
    received_at: AwareDatetime

    # Eval-isolation flag per `docs/testing.md` "Eval isolation end-to-end" +
    # `docs/schema.md` (five tables carry `is_eval`). The webhook receiver /
    # dispatcher sets this on the seed `ReviewState`; nodes thread it into
    # their `LLMRequest.is_eval` so audit rows produced during eval runs
    # are correctly tagged. Default `False` means "real production review"
    # — eval-harness factories construct seeds with `is_eval=True`.
    is_eval: bool = False

    # Populated by triage node (separate spec)
    triage_result: TriageResult | None = None

    # Populated by analyze ⇄ trace loop iterations per §3 of
    # `specs/2026-05-19-analyze-foundation.md`. Sister analyze-
    # implementation spec wires the producer; this foundation spec
    # provides the slot + reducer so checkpoint replay is idempotent
    # the moment producers begin emitting. Annotated with
    # `append_with_dedup_by(key_fn)` per the canonical reducer shape in
    # `docs/spec.md` §7.1 — same pattern downstream slots will adopt.
    # Merge key is content-derived (`round_id` is SHA-256 hex over the
    # round's content via `compute_identity_hash`), so re-emission of
    # the same logical round collapses on replay.
    analysis_rounds: Annotated[
        list[AnalysisRound],
        append_with_dedup_by(lambda r: r.round_id),
    ] = Field(default_factory=list)

    # Analyze's deterministic request channel for the trace node. Same
    # reducer shape; merge key is `candidate_id`. Trace consumes the
    # accumulated list. Per §3.
    trace_candidates: Annotated[
        list[TraceCandidate],
        append_with_dedup_by(lambda c: c.candidate_id),
    ] = Field(default_factory=list)


__all__ = [
    "ReviewState",
]
