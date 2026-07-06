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

Now landed beyond the analyze-foundation set:
- `publish_result: PublishResult | None` — populated by the publish
  node per the publish-node spec; single-writer field with default
  overwrite reducer (publish is a graph terminal).
- `trace_decisions: list[TraceDecision]` — populated by the trace node
  per `specs/2026-05-23-trace-node.md` Q3 + Q4 + #017 × #024 amendment.
  Reducer: `append_with_dedup_by(source_finding_id)`. One decision per
  source finding across the review (per #017 amended point 1); replay
  re-application collapses on the source_finding_id key.
- `trace_fetched_files: list[TraceFetchedFile]` — populated by the trace
  node per Q3. Reducer: `append_with_dedup_by(path)`. First-write-wins
  on path collision; multi-cause provenance recovers via
  `query state.trace_decisions where target_file == self.path` (M2).

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

from outrider.agent.reducers import (
    append_with_dedup_by,
    append_with_slot_guard,
    semantic_digest,
)
from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.analyze_worker import (
    WORKER_OUTCOME_EXCLUDE_PATHS,
    AnalyzeWorkerOutcome,
    worker_outcome_slot,
)
from outrider.schemas.hitl import HITLDecision, HITLRequest
from outrider.schemas.pr_context import PRContext
from outrider.schemas.publish import PublishResult
from outrider.schemas.review_report import ReviewReport
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.trace_decision import TraceDecision
from outrider.schemas.trace_fetched_file import TraceFetchedFile
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
    # `docs/schema.md` (six tables carry `is_eval`). The webhook receiver /
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

    # Per-(file, pass) worker outcomes for the parallel-analyze fan-out
    # (specs/2026-07-05-parallel-analyze.md) — populated at pass 0 by the
    # analyze node's per-file branches; the Send workers own the writes
    # after the fan-out cutover. Slot key is POSITIONAL, so the merge is the
    # #063 slot guard, not first-wins dedup: identical-digest retries are
    # replay no-ops, divergent same-slot content raises (state must not
    # fork from audit). The digest recomputes over the canonical dump at
    # merge time (generated finding_ids excluded); state is never mutated
    # post-insertion, and a mutation that DID happen fails loud as
    # divergence rather than passing silently as an identical retry.
    # NON-ALIASING OBLIGATION (3b-2): recompute safety additionally
    # requires that no live object mutated elsewhere is ALIASED into an
    # outcome — workers clone findings into outcomes (model_validate
    # round-trip, the validator-safe clone) and an aggregate-level test
    # asserts no object identity is shared between
    # analyze_worker_outcomes and analysis_rounds findings.
    analyze_worker_outcomes: Annotated[
        list[AnalyzeWorkerOutcome],
        append_with_slot_guard(
            worker_outcome_slot,
            lambda o: semantic_digest(o, exclude_paths=WORKER_OUTCOME_EXCLUDE_PATHS),
        ),
    ] = Field(default_factory=list)

    # Wall-clock start of the in-flight analyze pass, written by the analyze
    # planner step and read by the aggregate step to stamp the folded
    # `AnalysisRound.started_at` (the pass spans multiple graph vertices
    # under the fan-out, so the anchor must ride state — a monotonic anchor
    # cannot cross vertices/processes; the aggregate clamps `ended_at` to
    # `max(started_at, now)` instead, preserving the round's ordering
    # invariant under clock jumps). Last-write-wins per pass; None until
    # the first analyze pass starts.
    analyze_pass_started_at: AwareDatetime | None = None

    # Analyze's deterministic request channel for the trace node. Same
    # reducer shape; merge key is `candidate_id`. Trace consumes the
    # accumulated list. Per §3.
    trace_candidates: Annotated[
        list[TraceCandidate],
        append_with_dedup_by(lambda c: c.candidate_id),
    ] = Field(default_factory=list)

    # Populated by the publish node (`agent/nodes/publish.py`) per
    # specs/2026-05-21-publish-node.md. Single-writer field with the
    # default overwrite reducer (no dedup-key needed — only the publish
    # node assigns it, and the node is a graph terminal). Carries the
    # publish outcome shape per `PublishResult` (success / empty /
    # idempotently_skipped / idempotently_skipped_external_record).
    # `None` until publish runs.
    publish_result: PublishResult | None = None

    # Populated by the trace node (`agent/nodes/trace.py`) per
    # `specs/2026-05-23-trace-node.md` + `DECISIONS.md#017` × `#024`
    # amendment. Reducer: `append_with_dedup_by(source_finding_id)` per
    # #017 amended point 1 — one decision per source finding across the
    # review; explicit rejection of the `(source_finding_id, target_file)`
    # key that would collapse unresolved/ambiguous rows on (id, None).
    # Trace's emission gate (#025 point 5 + M7 audit-first contract)
    # consults this list to enforce once-per-finding semantics.
    trace_decisions: Annotated[
        list[TraceDecision],
        append_with_dedup_by(lambda d: d.source_finding_id),
    ] = Field(default_factory=list)

    # Populated by the trace node per Q3 resolution
    # (`specs/2026-05-23-trace-node.md`). Reducer:
    # `append_with_dedup_by(path)`. First-write-wins on path collision
    # (per M2 audit-fold) — multi-cause provenance recovers via
    # `query state.trace_decisions where target_file == self.path`.
    # Trace's fetched file content (head-side); analyze pass-2 consumes
    # it alongside `pr_context.changed_files` for the post-trace pass.
    trace_fetched_files: Annotated[
        list[TraceFetchedFile],
        append_with_dedup_by(lambda f: f.path),
    ] = Field(default_factory=list)

    # Per-invocation delta count: how many NEW trace-fetched files the
    # most recent `trace()` call added to state (NOT the cumulative
    # `len(trace_fetched_files)`). Default-overwrite reducer (LangGraph's
    # default for un-annotated fields) means each trace() invocation's
    # value REPLACES the previous, preserving per-invocation delta
    # semantics across checkpoint replay.
    #
    # `_trace_router` reads this to decide whether to re-enter analyze.
    # Checking cumulative `trace_fetched_files` would route to analyze
    # even when the latest trace() call yielded nothing new (replay
    # path or trace producing zero new fetches after an earlier
    # successful pass). With MAX_ANALYSIS_ROUNDS=2 the depth gate
    # masks this today, but the contract is "re-enter analyze iff
    # trace JUST yielded new fetches" — the per-invocation delta is
    # the source of truth.
    #
    # `ge=0` because the field models a count; a negative value is
    # meaningless AND silently violates router invariants (`> 0` would
    # be false for `-N` even if trace did emit N fetches). Fail fast
    # at the schema boundary rather than at the router.
    last_trace_pass_fetched_count: int = Field(default=0, ge=0)

    # Populated by the synthesize node (`agent/nodes/synthesize.py`) per
    # specs/2026-05-28-synthesize-node.md. Scalar overwrite reducer
    # (LangGraph default for un-annotated fields). Aggregates findings
    # from `analysis_rounds` (deduped by `content_hash`, severity-sorted),
    # computes deterministic `ReviewMetrics` from audit events, and
    # carries the LLM-generated summary prose. HITL and publish nodes
    # consume `review_report.findings` instead of walking
    # `analysis_rounds[*].findings` directly. `None` until synthesize runs.
    review_report: ReviewReport | None = None

    # Populated by the HITL node (`agent/nodes/hitl.py`) per the
    # hitl-node spec. Both fields are scalar (overwrite reducer is
    # correct — LangGraph's default `last-write-wins` per `replay-equivalent
    # reducers handle scalars natively). Re-emission of identical
    # values on body re-runs (resume + replay) is safe because the
    # values are deterministic per `compute_phase_id`-style derivation
    # (HITLRequest from `state.received_at`; HITLDecision from the
    # resume value passed to `Command(resume=...)`).
    #
    # `hitl_request` is set at interrupt time (step 5 of node body
    # per spec): the partition of `findings_requiring_approval` vs
    # `auto_post_findings` + the deterministic `expires_at` derivation.
    # `hitl_decision` is set on resume completion (step 13 of node
    # body): the reviewer-submitted decision constructed by the
    # endpoint with server-set `reviewer_id` + server-derived
    # `original_severity` per the spec's Q1 + Fix 2 contracts.
    hitl_request: HITLRequest | None = None
    hitl_decision: HITLDecision | None = None


__all__ = [
    "ReviewState",
]
