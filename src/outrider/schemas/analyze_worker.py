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
  observed_coverage / observed_skip / plain_skip), never inferred.
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

WorkerSource = Literal["parser", "cache_serve", "observed_coverage", "observed_skip", "plain_skip"]
"""The sequential branch union, made explicit. Exactly one of: the LLM
parser ran (`parser`); a cache hit served reconstructed findings with no
LLM call (`cache_serve`); the #049 ENFORCED coverage skip — every changed
scope skip_safe-covered, LLM not called, producer findings emitted, but
the file is EXAMINED (clean FileExaminationEvent, no SkipReason)
(`observed_coverage`); a budget/trivial ride-out skip carrying producer
findings (`observed_skip`); a skip that emitted nothing (`plain_skip`)."""


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
    # the aggregate forwards them onto AnalyzeCompletedEvent. Present on
    # BOTH non-skip paths: the parser merge assembles them, and a cache
    # serve RECONSTRUCTS them from the cached payload (proof retention
    # survives the serve). Skip sources carry none.
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
        """Run the fact-anchored checks in order; each is verified against the
        node, none inferred from counters.

        Order is behaviour: when an outcome violates more than one rule, the
        caller sees the first, and the groups run cheapest-and-most-fundamental
        first.
        """
        _check_skip_coherence(self)
        _check_source_capabilities(self)
        _check_identity_lists(self)
        _check_proposal_equation(self)
        return self


def _check_skip_coherence(outcome: AnalyzeWorkerOutcome) -> None:
    """`skip_reason` is non-None iff skipped (#018), and skipped iff a skip source."""
    skipped = outcome.parse_status == "skipped"
    if skipped != (outcome.skip_reason is not None):
        raise ValueError(
            "AnalyzeWorkerOutcome: skip_reason must be non-None iff "
            f"parse_status='skipped'; got parse_status={outcome.parse_status!r}, "
            f"skip_reason={outcome.skip_reason!r}"
        )
    if skipped != (outcome.source in ("observed_skip", "plain_skip")):  # coverage is EXAMINED
        raise ValueError(
            f"AnalyzeWorkerOutcome: parse_status={outcome.parse_status!r} does "
            f"not match source={outcome.source!r}"
        )


def _check_source_capabilities(outcome: AnalyzeWorkerOutcome) -> None:
    """What each `source` is allowed to carry.

    Parser tallies and provider spend require a parser. Trace candidates and #055
    subsumption records exist on parser AND `cache_serve` — a serve restores
    candidates and reconstructs subsumption records from the cached payload — and
    on neither skip source. One worker makes one LLM call, so a rejected response
    is 0 or 1 and yields nothing from that response; producer-OBSERVED
    augmentation is unaffected.
    """
    if outcome.source != "parser":
        if (
            outcome.n_proposals_seen
            or outcome.n_proposals_rejected
            or outcome.n_responses_rejected
            or outcome.n_proposals_superseded_by_observed
            or outcome.n_trace_candidates_dropped_malformed
        ):
            raise ValueError("AnalyzeWorkerOutcome: parser tallies require source='parser'")
        if (
            outcome.input_tokens
            or outcome.output_tokens
            or outcome.cache_read_tokens
            or outcome.cache_write_tokens
            or outcome.cost
        ):
            raise ValueError("AnalyzeWorkerOutcome: provider spend requires source='parser'")
        # #055 records exist on BOTH non-skip paths: the parser merge
        # assembles them, and a cache serve RECONSTRUCTS them from the
        # cached payload (proof retention must survive the serve).
        # Only skip sources have nothing to retain.
        if outcome.subsumed_matches and outcome.source != "cache_serve":
            raise ValueError(
                "AnalyzeWorkerOutcome: subsumption records exist only on "
                "parser and cache_serve sources (#055 proof retention)"
            )
        # Cache hits RESTORE prior-pass candidates into state (the
        # sequential serve branch extends state without counting them
        # as newly emitted — the aggregate derives this-pass emission
        # from parser-source outcomes only); skips carry none.
        if outcome.trace_candidates and outcome.source != "cache_serve":
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
    if outcome.n_responses_rejected > 1:
        raise ValueError(
            "AnalyzeWorkerOutcome: one worker makes one LLM call; n_responses_rejected is 0 or 1"
        )
    if outcome.n_responses_rejected and (
        outcome.n_proposals_seen
        or outcome.trace_candidates
        or outcome.n_trace_candidates_dropped_malformed
        or outcome.subsumed_matches
    ):
        raise ValueError(
            "AnalyzeWorkerOutcome: a rejected response yields zero proposals, "
            "trace candidates, and subsumption records (parser contract); "
            "producer-OBSERVED augmentation is unaffected"
        )
    # Candidates are processed only inside the parser's proposal loop
    # (well-formed AND malformed alike; #025's rejected-parent
    # candidates are still candidates of SEEN proposals) — so any
    # trace-candidate evidence implies at least one proposal was seen.
    if (
        outcome.source == "parser"
        and (outcome.trace_candidates or outcome.n_trace_candidates_dropped_malformed)
        and not outcome.n_proposals_seen
    ):
        raise ValueError(
            "AnalyzeWorkerOutcome: trace candidates (or malformed drops) "
            "imply n_proposals_seen > 0 — candidates exist only inside "
            "the proposal loop"
        )


def _check_identity_lists(outcome: AnalyzeWorkerOutcome) -> None:
    """Hash lists are canonical, are subsets of the admitted set, and match their
    source.

    Producer hashes cover exactly the admitted set on `observed_coverage` (never
    empty — coverage requires matches) and on `observed_skip`; served hashes cover
    exactly it on `cache_serve` and are exclusive to that source; both are empty on
    `plain_skip`. Every producer-listed finding is OBSERVED-tier, and no finding or
    subsumed match may name a file other than this worker's — cross-file
    attribution is unrepresentable in the per-file loop.
    """
    _canonical_hash_tuple("producer_observed_hashes", outcome.producer_observed_hashes)
    _canonical_hash_tuple("served_content_hashes", outcome.served_content_hashes)
    admitted = {f.content_hash for f in outcome.admitted_findings}
    producer = set(outcome.producer_observed_hashes)
    served = set(outcome.served_content_hashes)
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
    if served and outcome.source != "cache_serve":
        raise ValueError(
            "AnalyzeWorkerOutcome: served findings exist only on "
            "source='cache_serve' (sequential branch union)"
        )
    if outcome.source == "cache_serve":
        if outcome.parse_status != "clean":
            raise ValueError("AnalyzeWorkerOutcome: cache_serve requires parse_status='clean'")
        if served != admitted or producer:
            raise ValueError(
                "AnalyzeWorkerOutcome: cache_serve admits exactly the served "
                "set (nothing was produced this pass)"
            )
    if outcome.source == "observed_coverage":
        # #049 enforced coverage: clean status, producer-only findings,
        # no LLM, no candidates, no #054/#055 merge (the LLM never ran).
        if outcome.parse_status != "clean":
            raise ValueError(
                "AnalyzeWorkerOutcome: observed_coverage requires "
                "parse_status='clean' (the file is examined, not skipped)"
            )
        if not outcome.admitted_findings or producer != admitted:
            raise ValueError(
                "AnalyzeWorkerOutcome: observed_coverage findings are all "
                "producer-origin (an empty coverage skip is impossible — "
                "coverage requires matches)"
            )
        if outcome.trace_candidates or outcome.subsumed_matches or served:
            raise ValueError(
                "AnalyzeWorkerOutcome: observed_coverage carries no "
                "candidates, subsumption records, or served findings "
                "(the LLM never ran; no merge happened)"
            )
    if outcome.source == "observed_skip":
        if not outcome.admitted_findings:
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
    if outcome.source == "plain_skip" and (
        outcome.admitted_findings or outcome.producer_observed_hashes
    ):
        raise ValueError("AnalyzeWorkerOutcome: plain_skip carries nothing")

    for finding in outcome.admitted_findings:
        if finding.content_hash in producer and finding.evidence_tier is not EvidenceTier.OBSERVED:
            raise ValueError(
                "AnalyzeWorkerOutcome: producer-listed findings are "
                "OBSERVED-tier (the deterministic producer makes nothing else)"
            )
        if finding.file_path != outcome.path:
            raise ValueError(
                f"AnalyzeWorkerOutcome: finding names {finding.file_path!r} "
                f"but the worker's file is {outcome.path!r}"
            )
    for match in outcome.subsumed_matches:
        if match.file_path != outcome.path:
            raise ValueError(
                f"AnalyzeWorkerOutcome: subsumed match names "
                f"{match.file_path!r} but the worker's file is {outcome.path!r}"
            )


def _check_proposal_equation(outcome: AnalyzeWorkerOutcome) -> None:
    """The per-worker proposal equation, every term an identity count.

    Worker-locally emitted is the admitted set, observed is the producer list and
    served is zero, so seen == (admitted - producer) + rejected + superseded.
    Aggregate-only accounting would let invalid tallies cancel across files; this
    pins each worker. Parser source only.
    """
    if outcome.source == "parser":
        # The exact per-worker proposal equation, every term an identity
        # count (no derived counters to trust): worker-locally, emitted
        # is the admitted set, observed is the producer list, served is
        # zero — so seen == (admitted - producer) + rejected +
        # superseded. Aggregate-only accounting would let invalid
        # tallies cancel across files; this pins each worker.
        lhs = outcome.n_proposals_seen
        rhs = (
            len(outcome.admitted_findings)
            - len(outcome.producer_observed_hashes)
            + outcome.n_proposals_rejected
            + outcome.n_proposals_superseded_by_observed
        )
        if lhs != rhs:
            raise ValueError(
                f"AnalyzeWorkerOutcome: proposal accounting violated: "
                f"n_proposals_seen={lhs} != (admitted - producer) + "
                f"rejected + superseded = {rhs}"
            )


def worker_outcome_slot(outcome: AnalyzeWorkerOutcome) -> tuple[str, int]:
    """The positional `(file, pass)` slot key (`DECISIONS.md#063`)."""
    return (outcome.path, outcome.pass_index)
