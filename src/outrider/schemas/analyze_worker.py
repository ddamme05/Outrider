# Worker-outcome state model per specs/2026-07-05-parallel-analyze.md + DECISIONS.md#063.
"""Per-(file, pass) worker outcome for the parallel-analyze fan-out.

`AnalyzeWorkerOutcome` is what an `analyze_file` worker returns into state:
the aggregate step folds these into ONE `AnalysisRound` per pass plus the
`AnalyzeCompletedEvent` accounting (per `DECISIONS.md#063`, workers never
emit rounds — round count is the trace depth counter).

Design rule, learned across five review rounds: the outcome carries
**facts and identities, never re-derived judgments.** Early drafts made
workers report derived counters (emitted/observed/served) and validators
re-inferred branch semantics from them — every such inference was a guess
about the node, and several proved subtly wrong (evidence tier is NOT
producer origin; response rejection does NOT preclude producer
augmentation). The shipped shape instead records what only the worker
knows, as explicit identity:

- `source` — the sequential branch, discriminated (parser / cache_serve /
  observed_skip / plain_skip), never inferred.
- `producer_observed_hashes` — WHICH findings the deterministic OBSERVED
  producer made this pass. Origin is identity, not tier: a model-cited
  OBSERVED proposal is a legitimate proposal and stays out of this list.
- `served_content_hashes` — WHICH findings a cache hit reconstructed.
- Parser tallies copied VERBATIM from `ParserCounters` (no derivation).

Everything derivable is derived by the AGGREGATE from these identities
(the event-level emitted/served/observed counters are recomputed over the
post-cap kept set anyway), and the canonical proposal equation is
enforced where it always was — `AnalyzeCompletedEvent`'s accounting
validator — not re-approximated per worker.

State discipline (`state-is-pure-data`): every field JSON-round-trips;
`cost` is an exact `Decimal`; NO generated identities or timestamps on
the outcome itself. The #063 merge digest is recomputed by the state
reducer over the outcome's canonical dump (generated finding_ids
excluded) — the `AnalysisRound.round_id` precedent: a content-derived
key over findings that are nested-mutable by lifecycle design, safe
because state is never mutated post-insertion, and a mutation that DID
happen should fail loud as divergence, never pass silently as an
identical retry.
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
    "AnalyzeWorkerOutcome",
    "WorkerSource",
    "worker_outcome_slot",
]

# Positional exclusion paths for the #063 semantic digest (dot-separated,
# `[]` = list traversal): exactly the model tree's generated identities —
# the nested findings' uuid4 `finding_id`s. Pinned one-for-one against
# the models' default_factory fields.
WORKER_OUTCOME_EXCLUDE_PATHS: Final[frozenset[str]] = frozenset({"admitted_findings.[].finding_id"})

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")

WorkerSource = Literal["parser", "cache_serve", "observed_skip", "plain_skip"]
"""The sequential branch union, made explicit. Exactly one of: the LLM
parser ran (`parser`); a cache hit served reconstructed findings with no
LLM call (`cache_serve`); an enforced/ride-out skip emitted deterministic
OBSERVED findings (`observed_skip`); a skip emitted nothing
(`plain_skip`)."""


def _canonical_hash_tuple(name: str, hashes: tuple[str, ...]) -> None:
    """SHA-256 hex, sorted, unique — the canonical identity-list form (an
    emission-order encoding would make identical retries digest-divergent)."""
    for h in hashes:
        if not _SHA256_HEX_RE.fullmatch(h):
            raise ValueError(f"AnalyzeWorkerOutcome: {name} entry is not SHA-256 hex: {h!r}")
    if list(hashes) != sorted(set(hashes)):
        raise ValueError(f"AnalyzeWorkerOutcome: {name} must be sorted and unique")


class AnalyzeWorkerOutcome(BaseModel):
    """One worker's verdict for one (file, pass) slot.

    Merged into `ReviewState.analyze_worker_outcomes` by the slot-guard
    reducer (`DECISIONS.md#063`): identical-digest retries are replay
    no-ops, divergent same-slot content fails loud. The aggregate step is
    the ONLY consumer — it folds outcomes (sorted by path, so worker
    completion order never changes the round) into the pass's
    `AnalysisRound` + accounting event, deriving the emitted/served/
    observed counters from the identity lists here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Annotated[str, Field(max_length=1024)]
    pass_index: int = Field(ge=0)
    source: WorkerSource
    parse_status: Literal["clean", "failed", "degraded", "skipped"]
    skip_reason: SkipReason | None = None
    review_tier: ReviewTier
    admitted_findings: tuple[ReviewFinding, ...] = ()
    trace_candidates: tuple[TraceCandidate, ...] = ()
    # Cross-type subsumption proof-retention records (DECISIONS.md#055) —
    # the aggregate forwards them onto AnalyzeCompletedEvent. A subsumption
    # needs a JUDGED subsumer, so these exist only where a parser ran.
    subsumed_matches: tuple[ObservedSubsumedMatch, ...] = ()
    # ORIGIN IDENTITY, recorded by the producer that knows it (never
    # inferred from evidence tier): content hashes of the findings the
    # deterministic OBSERVED producer made THIS pass. A model-cited
    # OBSERVED proposal is a proposal — admitted, but not listed here.
    # The schema verifies tier + subset; ORIGIN TRUTHFULNESS cannot be
    # schema-verified (a model-cited structural finding is
    # indistinguishable by shape) — it is the worker's write obligation,
    # pinned at the wiring increment where the list is constructed
    # directly from produce_observed_findings output, the only code path
    # that knows.
    producer_observed_hashes: tuple[str, ...] = ()
    # Content hashes of CACHE-SERVED findings (produced in a PRIOR pass,
    # reconstructed without an LLM call). Identity, not count: the
    # post-cap accounting recomputes served-vs-proposal origins after
    # aggregate dedup and capping.
    served_content_hashes: tuple[str, ...] = ()
    # Parser tallies, copied VERBATIM from ParserCounters — no worker-side
    # derivation. Zero whenever no parser ran.
    n_proposals_seen: int = Field(default=0, ge=0)
    n_proposals_rejected: int = Field(default=0, ge=0)
    n_responses_rejected: int = Field(default=0, ge=0)
    n_proposals_superseded_by_observed: int = Field(default=0, ge=0)
    n_trace_candidates_dropped_malformed: int = Field(default=0, ge=0)
    # Provider spend (zero on every no-LLM source) + the planner-facing
    # estimate. `cost` is an exact Decimal; the aggregate sums Decimals
    # and casts float ONCE at the event (the sequential FP discipline).
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)
    cost: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_tokens: int = Field(default=0, ge=0)

    @field_validator("path")
    @classmethod
    def _enforce_canonical_path(cls, path: str) -> str:
        """Same canonical-path rule as `AnalysisRound.files_examined` — the
        slot key is positional, so two spellings of one path would occupy
        two slots and the duplicate-path planner gate would never see it."""
        return validate_diff_path(path)

    @model_validator(mode="after")
    def _enforce_fact_coherence(self) -> Self:
        """The fact-anchored rules only — each verified against the node,
        none inferred from counters:

        - `skip_reason` non-None iff skipped (the #018 iff contract), and
          skipped iff a skip source.
        - Parser tallies, provider spend, trace candidates, and #055
          subsumption records (a subsumer is JUDGED) require a parser.
          `n_responses_rejected > 0` additionally forces
          `n_proposals_seen == 0` — the parser's verified rejected-response
          contract — but does NOT preclude producer-OBSERVED augmentation.
        - Identity lists are canonical (hex, sorted, unique), subsets of
          the admitted set, and per-source: producer hashes cover EXACTLY
          the admitted set on `observed_skip`, served hashes EXACTLY the
          admitted set on `cache_serve`, both empty on `plain_skip`;
          served is cache_serve-exclusive; every producer-listed finding
          is OBSERVED-tier (the producer makes nothing else).
        - Cross-file attribution is unrepresentable (the sequential
          per-file loop cannot express it).
        """
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

        if self.source != "parser":
            if (
                self.n_proposals_seen
                or self.n_proposals_rejected
                or self.n_responses_rejected
                or self.n_proposals_superseded_by_observed
                or self.n_trace_candidates_dropped_malformed
            ):
                raise ValueError("AnalyzeWorkerOutcome: parser tallies require source='parser'")
            if (
                self.input_tokens
                or self.output_tokens
                or self.cache_read_tokens
                or self.cache_write_tokens
                or self.cost
            ):
                raise ValueError("AnalyzeWorkerOutcome: provider spend requires source='parser'")
            if self.subsumed_matches:
                raise ValueError(
                    "AnalyzeWorkerOutcome: subsumption records require a parser "
                    "(a #055 subsumer is JUDGED)"
                )
            # Cache hits RESTORE prior-pass candidates into state (the
            # sequential serve branch extends state without counting them
            # as newly emitted — the aggregate derives this-pass emission
            # from parser-source outcomes only); skips carry none.
            if self.trace_candidates and self.source != "cache_serve":
                raise ValueError(
                    "AnalyzeWorkerOutcome: trace candidates exist only on "
                    "parser and cache_serve sources"
                )
        # One worker = one LLM call, so the rejected-response shape is
        # binary and total: 0/1, and a rejection produced NOTHING from the
        # response — no proposals, no trace candidates (well-formed or
        # malformed), no subsumption records (a subsumer is a response
        # proposal). Producer-OBSERVED augmentation is unaffected; the
        # identity equation below covers the proposal terms.
        if self.n_responses_rejected > 1:
            raise ValueError(
                "AnalyzeWorkerOutcome: one worker makes one LLM call; "
                "n_responses_rejected is 0 or 1"
            )
        if self.n_responses_rejected and (
            self.n_proposals_seen
            or self.trace_candidates
            or self.n_trace_candidates_dropped_malformed
            or self.subsumed_matches
        ):
            raise ValueError(
                "AnalyzeWorkerOutcome: a rejected response yields zero proposals, "
                "trace candidates, and subsumption records (parser contract); "
                "producer-OBSERVED augmentation is unaffected"
            )

        _canonical_hash_tuple("producer_observed_hashes", self.producer_observed_hashes)
        _canonical_hash_tuple("served_content_hashes", self.served_content_hashes)
        admitted = {f.content_hash for f in self.admitted_findings}
        producer = set(self.producer_observed_hashes)
        served = set(self.served_content_hashes)
        if not producer <= admitted:
            raise ValueError(
                "AnalyzeWorkerOutcome: producer_observed_hashes must be a subset "
                "of the admitted findings' hashes"
            )
        if not served <= admitted:
            raise ValueError(
                "AnalyzeWorkerOutcome: served_content_hashes must be a subset of "
                "the admitted findings' hashes"
            )
        if served and self.source != "cache_serve":
            raise ValueError(
                "AnalyzeWorkerOutcome: served findings exist only on "
                "source='cache_serve' (sequential branch union)"
            )
        if self.source == "cache_serve":
            if self.parse_status != "clean":
                raise ValueError("AnalyzeWorkerOutcome: cache_serve requires parse_status='clean'")
            if served != admitted or producer:
                raise ValueError(
                    "AnalyzeWorkerOutcome: cache_serve admits exactly the served "
                    "set (nothing was produced this pass)"
                )
        if self.source == "observed_skip":
            if not self.admitted_findings:
                raise ValueError(
                    "AnalyzeWorkerOutcome: observed_skip carries the producer's "
                    "findings; an empty skip is plain_skip"
                )
            if producer != admitted:
                raise ValueError(
                    "AnalyzeWorkerOutcome: observed_skip findings are all "
                    "producer-origin (producer_observed_hashes covers the "
                    "admitted set)"
                )
        if self.source == "plain_skip" and (
            self.admitted_findings or self.producer_observed_hashes
        ):
            raise ValueError("AnalyzeWorkerOutcome: plain_skip carries nothing")

        for finding in self.admitted_findings:
            if (
                finding.content_hash in producer
                and finding.evidence_tier is not EvidenceTier.OBSERVED
            ):
                raise ValueError(
                    "AnalyzeWorkerOutcome: producer-listed findings are "
                    "OBSERVED-tier (the deterministic producer makes nothing else)"
                )
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
        if self.source == "parser":
            # The exact per-worker proposal equation, every term an identity
            # count (no derived counters to trust): worker-locally, emitted
            # is the admitted set, observed is the producer list, served is
            # zero — so seen == (admitted - producer) + rejected +
            # superseded. Aggregate-only accounting would let invalid
            # tallies cancel across files; this pins each worker.
            lhs = self.n_proposals_seen
            rhs = (
                len(self.admitted_findings)
                - len(self.producer_observed_hashes)
                + self.n_proposals_rejected
                + self.n_proposals_superseded_by_observed
            )
            if lhs != rhs:
                raise ValueError(
                    f"AnalyzeWorkerOutcome: proposal accounting violated: "
                    f"n_proposals_seen={lhs} != (admitted - producer) + "
                    f"rejected + superseded = {rhs}"
                )
        return self


def worker_outcome_slot(outcome: AnalyzeWorkerOutcome) -> tuple[str, int]:
    """The positional `(file, pass)` slot key (`DECISIONS.md#063`)."""
    return (outcome.path, outcome.pass_index)
