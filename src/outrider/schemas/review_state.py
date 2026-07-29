# LangGraph state object per docs/spec.md §7.1
"""`ReviewState`: the LangGraph state envelope.

Review execution starts with `review_id`, `pr_context`, `received_at`, and
`is_eval`; every other slot is written by its owning node. Intake replaces
`pr_context` with an enriched `PRContext` rather than adding another top-level slot.

State round-trips through Postgres JSON via langgraph-checkpoint-postgres, so
every field type must be JSON-serializable and runtime dependencies (DB sessions,
HTTP clients) are never stored here — they are injected at `build_graph(...)` and
closed over in the node callables.

NOT frozen, deliberately: nodes return partial-update dicts that LangGraph merges
through reducers, and a frozen model would force every node to construct a
fully-formed instance on return.

`validate_assignment=True` because `frozen=False` otherwise lets direct attribute
assignment bypass the construction validators. Two nuances worth knowing: a valid
dict-shaped payload IS validated and reconstructed into a real model instance on
assignment (that is reconstructive validation, not a bypass), and LangGraph runs
Pydantic validation only on the FIRST node's input — reducer-merged state and the
graph output are not revalidated. So callers needing typed post-merge state must
use `ReviewState.model_validate({**state.model_dump(), **delta})`;
`model_copy(update=...)` does not revalidate.
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
    """The state every node receives and returns partial updates against."""

    model_config = ConfigDict(frozen=False, extra="forbid", validate_assignment=True)

    review_id: UUID
    pr_context: PRContext
    received_at: AwareDatetime

    # Threaded into LLMRequest.is_eval so audit rows from eval runs are tagged;
    # False means a real review.
    is_eval: bool = False

    triage_result: TriageResult | None = None

    # `round_id` is content-derived, so re-emitting the same logical round on
    # replay collapses rather than duplicating.
    analysis_rounds: Annotated[
        list[AnalysisRound],
        append_with_dedup_by(lambda r: r.round_id),
    ] = Field(default_factory=list)

    # The slot key is POSITIONAL, so this guards rather than dedupes: an
    # identical-digest retry is a replay no-op, but divergent content in the same
    # slot RAISES — state must not fork from audit (DECISIONS.md#063).
    #
    # Because the digest recomputes at merge time, callers must not ALIAS a live
    # object into an outcome. Workers clone findings via a `model_validate`
    # round-trip, and a test asserts no object identity is shared with
    # `analysis_rounds` findings.
    analyze_worker_outcomes: Annotated[
        list[AnalyzeWorkerOutcome],
        append_with_slot_guard(
            worker_outcome_slot,
            lambda o: semantic_digest(o, exclude_paths=WORKER_OUTCOME_EXCLUDE_PATHS),
        ),
    ] = Field(default_factory=list)

    # Must ride state rather than a monotonic clock: the pass spans several graph
    # vertices under the fan-out, and a monotonic anchor cannot cross processes.
    # The aggregate clamps `ended_at` to `max(started_at, now)` so a clock jump
    # cannot invert the round's ordering.
    analyze_pass_started_at: AwareDatetime | None = None

    # Analyze's request channel to trace, which consumes the accumulated list.
    trace_candidates: Annotated[
        list[TraceCandidate],
        append_with_dedup_by(lambda c: c.candidate_id),
    ] = Field(default_factory=list)

    # Overwrite reducer is safe: publish is the only writer and a graph terminal.
    publish_result: PublishResult | None = None

    # Keyed on `source_finding_id` ALONE — one decision per source finding across
    # the review. A `(source_finding_id, target_file)` key was rejected because it
    # collapses every unresolved row onto `(id, None)` (DECISIONS.md#017).
    trace_decisions: Annotated[
        list[TraceDecision],
        append_with_dedup_by(lambda d: d.source_finding_id),
    ] = Field(default_factory=list)

    # First-write-wins on a path collision, which drops the second cause. Full
    # provenance is recoverable from `trace_decisions` where `target_file` matches.
    trace_fetched_files: Annotated[
        list[TraceFetchedFile],
        append_with_dedup_by(lambda f: f.path),
    ] = Field(default_factory=list)

    # A PER-INVOCATION delta, not `len(trace_fetched_files)`. `_trace_router`
    # re-enters analyze iff trace JUST fetched something new; reading the
    # cumulative list would re-enter even when the latest pass yielded nothing.
    # The depth gate masks that today, but the contract is the delta.
    last_trace_pass_fetched_count: int = Field(default=0, ge=0)

    # HITL and publish read `review_report.findings` rather than walking
    # `analysis_rounds[*].findings`, so this is the deduplicated, sorted view.
    review_report: ReviewReport | None = None

    # Overwrite is safe on both despite the body re-running on every resume,
    # because both values are derived deterministically: the request from
    # `state.received_at`, the decision from the `Command(resume=...)` value.
    hitl_request: HITLRequest | None = None
    hitl_decision: HITLDecision | None = None


__all__ = [
    "ReviewState",
]
