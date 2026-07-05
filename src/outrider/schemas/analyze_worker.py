# Worker-outcome state model per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""Per-(file, pass) worker outcome for the parallel-analyze fan-out.

`AnalyzeWorkerOutcome` is what an `analyze_file` worker returns into state:
the aggregate step folds these into ONE `AnalysisRound` per pass plus the
`AnalyzeCompletedEvent` accounting (per `DECISIONS.md#063`, workers never
emit rounds — round count is the trace depth counter). The field set is a
faithful port of what the sequential main loop consumes from its in-memory
`_FileOutcome`; the DISCRIMINATED `source` field makes the sequential
branch union explicit (parser XOR cache-serve XOR observed-skip XOR
plain-skip), and every coherence rule hangs off it — inferring the union
from counters proved unreliable across three review rounds.

State discipline (`state-is-pure-data`): every field JSON-round-trips —
`cost` is a `Decimal` (Pydantic serializes it as an exact string), there
are NO generated identities or timestamps on the outcome itself (worker
latency rides `LLMCallEvent.latency_ms`, not state — a per-retry timestamp
would make every retry digest-divergent and defeat the #063 replay no-op).

The #063 merge digest is SNAPSHOTTED at construction into
`semantic_snapshot`: nested `ReviewFinding`s are deliberately NOT frozen
(multi-stage lifecycle), so a recompute-at-merge digest could change after
a nested mutation and falsely trigger `SlotDivergenceError` on a
legitimate retry. The snapshot is immutable-at-birth; the slot-guard
reducer compares snapshots, never recomputes.
"""

import re
from decimal import Decimal
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.agent.reducers import semantic_digest
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
    "WorkerSource",
    "worker_outcome_slot",
]

# Positional exclusion paths for the #063 semantic digest (dot-separated,
# `[]` = list traversal). Two classes of exclusion: the nested findings'
# uuid4 `finding_id`s (the only generated identities in the tree), and the
# digest's own storage field (self-referential — a digest cannot cover
# itself). Pinned by `test_analyze_worker_outcome.py`: the generated
# entries one-for-one against the models' default_factory fields, the
# self-reference explicitly.
WORKER_OUTCOME_EXCLUDE_PATHS: Final[frozenset[str]] = frozenset(
    {"admitted_findings.[].finding_id", "semantic_snapshot"}
)

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

WorkerSource = Literal["parser", "cache_serve", "observed_skip", "plain_skip"]
"""The sequential branch union, made explicit. Exactly one of: the LLM
parser ran (`parser`); a cache hit served reconstructed findings with no
LLM call (`cache_serve`); an enforced/ride-out skip emitted deterministic
OBSERVED findings (`observed_skip`); a skip emitted nothing
(`plain_skip`)."""


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
    reducer over the stored `semantic_snapshot`: identical-snapshot retries
    are replay no-ops, divergent same-slot content fails loud
    (`DECISIONS.md#063`). The aggregate step is the ONLY consumer — it
    folds outcomes (sorted by path, so worker completion order never
    changes the round) into the pass's `AnalysisRound` + accounting event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Annotated[str, Field(max_length=1024)]
    pass_index: int = Field(ge=0)
    source: WorkerSource
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
    # served vs proposal-born AFTER aggregate dedup and capping. Canonical
    # form enforced below: SHA-256 hex, sorted, unique, and (per-source)
    # exactly the admitted set on `cache_serve`, empty everywhere else.
    served_content_hashes: tuple[str, ...] = ()
    # The #063 merge digest, snapshotted at construction (empty input →
    # computed; non-empty → verified). Nested findings are mutable by
    # design, so the merge contract binds to the AT-CONSTRUCTION content;
    # the slot-guard reducer compares this field and never recomputes.
    semantic_snapshot: str = ""

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
        iff contract), and skipped iff a skip source."""
        skipped = self.parse_status == "skipped"
        if skipped != (self.skip_reason is not None):
            raise ValueError(
                "AnalyzeWorkerOutcome: skip_reason must be non-None iff "
                f"parse_status='skipped'; got parse_status={self.parse_status!r}, "
                f"skip_reason={self.skip_reason!r}"
            )
        if skipped != (self.source in ("observed_skip", "plain_skip")):
            raise ValueError(
                f"AnalyzeWorkerOutcome: parse_status={self.parse_status!r} does "
                f"not match source={self.source!r}"
            )
        return self

    @model_validator(mode="after")
    def _enforce_served_hashes_canonical(self) -> Self:
        """Sorted-unique SHA-256 hex, subset of admitted hashes, coupled to
        the served counter (per-source equality is enforced above)."""
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
    def _enforce_source_coherence(self) -> Self:
        """Per-source rules — the sequential branch shapes, verified against
        the main loop rather than inferred: `llm_called` iff `parser`;
        observed/served counters DERIVED from the findings they describe;
        trace counters tied to the candidates tuple; the response-rejection
        all-zero exclusive shape; skips carrying nothing but (for
        `observed_skip`) the deterministic producer's findings."""
        c = self.counters
        if self.llm_called != (self.source == "parser"):
            raise ValueError(
                f"AnalyzeWorkerOutcome: llm_called={self.llm_called} contradicts "
                f"source={self.source!r}"
            )
        observed_count = sum(
            1 for f in self.admitted_findings if f.evidence_tier is EvidenceTier.OBSERVED
        )
        if self.source == "parser":
            if self.served_content_hashes or c.n_findings_served:
                raise ValueError(
                    "AnalyzeWorkerOutcome: cache-served findings cannot coexist "
                    "with llm_called=True (sequential branch union)"
                )
            # This pass's OBSERVED findings are exactly the producer-merged
            # OBSERVED-tier admitted set — a JUDGED finding counted as
            # observed (or vice versa) breaks the accounting subtraction.
            if c.n_findings_observed != observed_count:
                raise ValueError(
                    f"AnalyzeWorkerOutcome: n_findings_observed="
                    f"{c.n_findings_observed} but {observed_count} OBSERVED-tier "
                    f"findings admitted"
                )
            if c.n_trace_candidates_emitted != len(self.trace_candidates):
                raise ValueError(
                    f"AnalyzeWorkerOutcome: n_trace_candidates_emitted="
                    f"{c.n_trace_candidates_emitted} but "
                    f"{len(self.trace_candidates)} candidates carried"
                )
            if c.n_responses_rejected and (
                c.n_responses_rejected != 1
                or c.n_proposals_seen
                or c.n_findings_emitted
                or self.admitted_findings
                or self.trace_candidates
            ):
                # Exclusive shape per the parser contract: a rejected
                # response yields zero proposals, findings, and candidates.
                raise ValueError(
                    "AnalyzeWorkerOutcome: a response-level rejection is "
                    "exclusive — counters all zero except "
                    "n_responses_rejected=1, no findings, no candidates"
                )
        else:
            # No parser ran: no proposal lifecycle, no trace emission.
            if (
                c.n_proposals_seen
                or c.n_proposals_rejected
                or c.n_responses_rejected
                or c.n_proposals_superseded_by_observed
                or c.n_trace_candidates_emitted
                or c.n_trace_candidates_dropped_malformed
            ):
                raise ValueError(
                    "AnalyzeWorkerOutcome: proposal/trace counters require an "
                    "LLM call (no parser ran)"
                )
        if self.source == "cache_serve":
            if self.parse_status != "clean":
                raise ValueError("AnalyzeWorkerOutcome: cache_serve requires parse_status='clean'")
            # Served findings ride the served counter regardless of tier
            # (they were produced in a PRIOR pass); nothing here is
            # this-pass observed.
            if c.n_findings_observed:
                raise ValueError(
                    "AnalyzeWorkerOutcome: cache_serve never counts this-pass observed findings"
                )
            expected = tuple(sorted({f.content_hash for f in self.admitted_findings}))
            if self.served_content_hashes != expected:
                raise ValueError(
                    "AnalyzeWorkerOutcome: cache_serve served_content_hashes must "
                    "be exactly the admitted findings' hashes"
                )
            if c.n_findings_served != len(self.admitted_findings):
                raise ValueError(
                    f"AnalyzeWorkerOutcome: cache_serve n_findings_served="
                    f"{c.n_findings_served} but {len(self.admitted_findings)} "
                    f"findings admitted"
                )
        if self.source == "observed_skip":
            if not self.admitted_findings:
                raise ValueError(
                    "AnalyzeWorkerOutcome: observed_skip carries the producer's "
                    "findings; an empty skip is plain_skip"
                )
            if self.served_content_hashes or c.n_findings_served:
                raise ValueError("AnalyzeWorkerOutcome: observed_skip never serves from cache")
            if c.n_findings_observed != len(self.admitted_findings):
                raise ValueError(
                    "AnalyzeWorkerOutcome: observed_skip findings are all this-pass observed"
                )
            if self.trace_candidates:
                raise ValueError("AnalyzeWorkerOutcome: a skipped file emits no trace candidates")
        if self.source == "plain_skip" and (
            self.admitted_findings
            or self.trace_candidates
            or self.served_content_hashes
            or c.n_findings_emitted
            or c.n_findings_observed
            or c.n_findings_served
        ):
            raise ValueError("AnalyzeWorkerOutcome: plain_skip carries nothing")
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

    @model_validator(mode="after")
    def _snapshot_semantic_digest(self) -> Self:
        """LAST validator: snapshot (or verify) the #063 merge digest.

        Nested `ReviewFinding`s are mutable by design, so a
        recompute-at-merge digest could change after insertion and falsely
        diverge a legitimate retry. Computed here over the fully-validated
        content (the field excludes itself via
        `WORKER_OUTCOME_EXCLUDE_PATHS`); non-empty input (a JSON round-trip
        or a hand-built value) is VERIFIED, not trusted.
        `object.__setattr__` writes through the frozen config — the
        standard Pydantic pattern for validator-computed fields.
        """
        computed = semantic_digest(self, exclude_paths=WORKER_OUTCOME_EXCLUDE_PATHS)
        if not self.semantic_snapshot:
            object.__setattr__(self, "semantic_snapshot", computed)
        elif self.semantic_snapshot != computed:
            raise ValueError(
                "AnalyzeWorkerOutcome: semantic_snapshot does not match the outcome's content"
            )
        return self


def worker_outcome_slot(outcome: AnalyzeWorkerOutcome) -> tuple[str, int]:
    """The positional `(file, pass)` slot key (`DECISIONS.md#063`)."""
    return (outcome.path, outcome.pass_index)
