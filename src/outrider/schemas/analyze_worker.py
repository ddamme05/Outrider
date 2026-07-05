# Worker-outcome state model per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""Per-(file, pass) worker outcome for the parallel-analyze fan-out.

`AnalyzeWorkerOutcome` is what an `analyze_file` worker returns into state:
the aggregate step folds these into ONE `AnalysisRound` per pass plus the
`AnalyzeCompletedEvent` accounting (per `DECISIONS.md#063`, workers never
emit rounds — round count is the trace depth counter). The field set is a
faithful port of what the sequential main loop consumes from its in-memory
`_FileOutcome`: the accounting counters, the worker-locally-admitted
findings, trace candidates, token/cost tallies, and the skip/tier facts.

State discipline (`state-is-pure-data`): every field JSON-round-trips —
`cost` is a `Decimal` (Pydantic serializes it as an exact string), there
are NO generated identities or timestamps on the outcome itself (worker
latency rides `LLMCallEvent.latency_ms`, not state — a per-retry timestamp
would make every retry digest-divergent and defeat the #063 replay no-op).
The only generated identities in the tree are the nested findings'
`finding_id`s, excluded from the semantic digest via
`WORKER_OUTCOME_EXCLUDE_PATHS` — a frozen, POSITIONAL path list pinned
one-for-one against the models' generated fields.
"""

from decimal import Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.ast_facts.models import SkipReason
from outrider.coordinates import validate_diff_path
from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.triage_result import ReviewTier

__all__ = [
    "WORKER_OUTCOME_EXCLUDE_PATHS",
    "AnalyzeWorkerCounters",
    "AnalyzeWorkerOutcome",
    "worker_outcome_slot",
]

# Positional exclusion paths for the #063 semantic digest (dot-separated,
# `[]` = list traversal). The ONLY generated identities in an outcome's
# tree are the nested findings' uuid4 `finding_id`s; the outcome itself
# carries none. Pinned one-for-one against the models' default_factory
# fields by `test_analyze_worker_outcome.py` — a generated field this list
# misses would make identical retries digest-divergent (fail-loud crash on
# legitimate replay), and a semantic field wrongly listed would be
# silently ignored; the pin catches both directions.
WORKER_OUTCOME_EXCLUDE_PATHS: Final[frozenset[str]] = frozenset({"admitted_findings.[].finding_id"})


class AnalyzeWorkerCounters(BaseModel):
    """The per-file accounting terms the aggregate sums into
    `AnalyzeCompletedEvent` — mirrors the sequential main loop's
    accumulators, all defaulting to zero so skip outcomes construct bare.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_proposals_seen: int = Field(default=0, ge=0)
    n_findings_emitted: int = Field(default=0, ge=0)
    n_findings_observed: int = Field(default=0, ge=0)
    n_findings_served: int = Field(default=0, ge=0)
    n_proposals_superseded_by_observed: int = Field(default=0, ge=0)
    n_proposals_rejected: int = Field(default=0, ge=0)
    n_responses_rejected: int = Field(default=0, ge=0)
    n_trace_candidates_emitted: int = Field(default=0, ge=0)
    n_trace_candidates_dropped_malformed: int = Field(default=0, ge=0)


class AnalyzeWorkerOutcome(BaseModel):
    """One worker's verdict for one (file, pass) slot.

    Merged into `ReviewState.analyze_worker_outcomes` by the slot-guard
    reducer: identical-digest retries are replay no-ops, divergent
    same-slot content fails loud (`DECISIONS.md#063`). The aggregate step
    is the ONLY consumer — it folds outcomes (sorted by path, so worker
    completion order never changes the round) into the pass's
    `AnalysisRound` + accounting event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Annotated[str, Field(max_length=1024)]
    pass_index: int = Field(ge=0)
    parse_status: Literal["clean", "failed", "degraded", "skipped"]
    skip_reason: SkipReason | None = None
    review_tier: ReviewTier
    llm_called: bool
    counters: AnalyzeWorkerCounters
    admitted_findings: tuple[ReviewFinding, ...] = ()
    trace_candidates: tuple[TraceCandidate, ...] = ()
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    # Exact per-file Decimal (JSON-serializes as a string, round-trips
    # exactly); the aggregate sums Decimals and casts float ONCE at the
    # event — the sequential loop's FP-drift discipline, preserved.
    cost: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    # Cross-type subsumption proof-retention records (DECISIONS.md#055) —
    # the aggregate forwards them onto AnalyzeCompletedEvent; dropping
    # them here would vanish the dropped-OBSERVED structural proof.
    subsumed_matches: tuple[ObservedSubsumedMatch, ...] = ()
    # Content hashes of CACHE-SERVED findings (identity, not just count):
    # the post-cap accounting recomputes how many KEPT findings were
    # served vs proposal-born AFTER aggregate dedup and capping — a count
    # alone cannot survive that recomputation.
    served_content_hashes: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def _enforce_canonical_path(cls, path: str) -> str:
        """Same canonical-path rule as `AnalysisRound.files_examined` — the
        slot key is positional, so two spellings of one path would occupy
        two slots and the duplicate-path planner gate would never see it."""
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_skip_coherence(self) -> Self:
        """`skip_reason` non-None iff `parse_status == "skipped"` (the #018
        iff contract), and a skipped file never made an LLM call. Findings
        on a skip are legitimate (the module-arm ride-out emits OBSERVED
        findings on budget/trivial skips) — deliberately not constrained."""
        skipped = self.parse_status == "skipped"
        if skipped != (self.skip_reason is not None):
            raise ValueError(
                "AnalyzeWorkerOutcome: skip_reason must be non-None iff "
                f"parse_status='skipped'; got parse_status={self.parse_status!r}, "
                f"skip_reason={self.skip_reason!r}"
            )
        if skipped and self.llm_called:
            raise ValueError("AnalyzeWorkerOutcome: a skipped file never makes an LLM call")
        return self

    @model_validator(mode="after")
    def _enforce_no_llm_means_no_spend(self) -> Self:
        """`llm_called=False` (skips, cache serves, ride-outs) constructs with
        zero provider tokens and zero cost everywhere in the node — nonzero
        spend without a call would let the aggregate's accounting contradict
        the LLMCallEvent stream."""
        if not self.llm_called and (
            self.input_tokens
            or self.output_tokens
            or self.cache_read_tokens
            or self.cache_write_tokens
            or self.cost
        ):
            raise ValueError(
                "AnalyzeWorkerOutcome: llm_called=False requires zero provider tokens and zero cost"
            )
        return self


def worker_outcome_slot(outcome: AnalyzeWorkerOutcome) -> tuple[str, int]:
    """The positional `(file, pass)` slot key (`DECISIONS.md#063`)."""
    return (outcome.path, outcome.pass_index)
