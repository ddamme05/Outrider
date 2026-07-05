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

import re
from decimal import Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.ast_facts.models import SkipReason
from outrider.coordinates import validate_diff_path
from outrider.policy import EvidenceTier
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

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")


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
    # alone cannot survive that recomputation. Canonical form is enforced
    # below: SHA-256 hex, sorted, unique (an order-of-emission encoding
    # would make identical retries digest-divergent), a subset of the
    # admitted findings' hashes, and length-coupled to
    # counters.n_findings_served.
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
    def _enforce_served_hashes_canonical(self) -> Self:
        """Sorted-unique SHA-256 hex, subset of admitted hashes, coupled to
        the served counter. Sorted-unique is the digest-stability half (an
        emission-order encoding would falsely diverge identical retries);
        the subset + counter coupling is the accounting half (a served hash
        naming a finding this worker never admitted, or a count the hash
        set contradicts, would corrupt the post-cap kept_served split)."""
        for h in self.served_content_hashes:
            if not _SHA256_HEX_RE.fullmatch(h):
                raise ValueError(
                    f"AnalyzeWorkerOutcome: served_content_hashes entry is not SHA-256 hex: {h!r}"
                )
        if list(self.served_content_hashes) != sorted(set(self.served_content_hashes)):
            raise ValueError(
                "AnalyzeWorkerOutcome: served_content_hashes must be sorted and unique"
            )
        admitted_hashes = {f.content_hash for f in self.admitted_findings}
        if not set(self.served_content_hashes) <= admitted_hashes:
            raise ValueError(
                "AnalyzeWorkerOutcome: served_content_hashes must be a subset of "
                "the admitted findings' content hashes"
            )
        if len(self.served_content_hashes) != self.counters.n_findings_served:
            raise ValueError(
                f"AnalyzeWorkerOutcome: {len(self.served_content_hashes)} served "
                f"hashes but counters.n_findings_served="
                f"{self.counters.n_findings_served}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_single_file_attribution(self) -> Self:
        """Every nested finding and subsumption record names THIS worker's
        file. The sequential per-file loop makes cross-file attribution
        impossible; a worker outcome must not be able to express it."""
        for finding in self.admitted_findings:
            if finding.file_path != self.path:
                raise ValueError(
                    f"AnalyzeWorkerOutcome: finding names {finding.file_path!r} "
                    f"but the worker's file is {self.path!r}"
                )
        for match in self.subsumed_matches:
            if match.file_path != self.path:
                raise ValueError(
                    f"AnalyzeWorkerOutcome: subsumed match names "
                    f"{match.file_path!r} but the worker's file is {self.path!r}"
                )
        return self

    @model_validator(mode="after")
    def _enforce_skip_findings_are_observed_only(self) -> Self:
        """Every real findings-on-skip path is the deterministic producer
        (enforced coverage skip; budget/trivial ride-outs) — a skipped
        outcome carrying a JUDGED or INFERRED finding is a shape no
        sequential path can produce, and the proof boundary must not gain
        one through the fan-out."""
        if self.parse_status == "skipped":
            for finding in self.admitted_findings:
                if finding.evidence_tier is not EvidenceTier.OBSERVED:
                    raise ValueError(
                        f"AnalyzeWorkerOutcome: a skipped outcome carries only "
                        f"OBSERVED findings; got {finding.evidence_tier} for "
                        f"{finding.content_hash}"
                    )
        return self

    @model_validator(mode="after")
    def _enforce_branch_union(self) -> Self:
        """The sequential branches are a union: parser XOR cache-serve XOR
        observed-skip. Served findings coexisting with an LLM call — or any
        proposal-lifecycle counter on a no-LLM outcome — is unrepresentable
        there and stays unrepresentable here."""
        if self.llm_called and self.counters.n_findings_served:
            raise ValueError(
                "AnalyzeWorkerOutcome: cache-served findings cannot coexist "
                "with llm_called=True (sequential branch union)"
            )
        if not self.llm_called and (
            self.counters.n_proposals_seen
            or self.counters.n_proposals_rejected
            or self.counters.n_responses_rejected
            or self.counters.n_proposals_superseded_by_observed
        ):
            raise ValueError(
                "AnalyzeWorkerOutcome: proposal-lifecycle counters require an "
                "LLM call (no parser ran)"
            )
        return self

    @model_validator(mode="after")
    def _enforce_worker_accounting(self) -> Self:
        """Per-worker halves of AnalyzeCompletedEvent's proposal equation, so
        errors cannot cancel ACROSS workers at the aggregate: (a) emitted
        findings are exactly the admitted set (worker-local, pre-aggregate
        dedup/cap); (b) `seen == (emitted - served - observed) + rejected +
        superseded` — the canonical equation WITHOUT the drop term, because
        `n_proposals_dropped` is computed at the aggregate over the
        post-dedup kept set, never per worker."""
        c = self.counters
        if c.n_findings_emitted != len(self.admitted_findings):
            raise ValueError(
                f"AnalyzeWorkerOutcome: n_findings_emitted={c.n_findings_emitted} "
                f"but {len(self.admitted_findings)} findings admitted"
            )
        lhs = c.n_proposals_seen
        rhs = (
            c.n_findings_emitted
            - c.n_findings_served
            - c.n_findings_observed
            + c.n_proposals_rejected
            + c.n_proposals_superseded_by_observed
        )
        if lhs != rhs:
            raise ValueError(
                f"AnalyzeWorkerOutcome: proposal accounting violated: "
                f"n_proposals_seen={lhs} != (emitted - served - observed) + "
                f"rejected + superseded = {rhs}"
            )
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
