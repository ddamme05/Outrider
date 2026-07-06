# Worker-outcome builders per specs/2026-07-05-parallel-analyze.md (3b-2b).
"""Construct `AnalyzeWorkerOutcome` records at the node's five branch sites.

One builder per `WorkerSource`, called by the `analyze_file` worker at the
exact site that knows the branch's facts — these builders are how the two
3b-2 acceptance gates are met:

- **Origin truth**: `worker_outcome_from_parser` derives
  `producer_observed_hashes` from the #054 merge's OWN OBJECT PLACEMENT —
  a producer finding that survived the merge is the same object in the
  admitted list (eviction places it; append places it; a drop against a
  non-JUDGED incumbent leaves it out). Evidence tier is never consulted,
  so a model-cited OBSERVED proposal can never be classified as producer
  output.
- **Non-aliasing**: every builder CLONES findings into the outcome via
  `model_validate` round-trip (the validator-safe clone — `model_copy`
  skips validators per the `ReviewFinding` docstring), so no live parser
  or producer object is aliased into state.

Import-light (schemas only): parser tallies arrive as scalars — the
wiring reads them off `ParserCounters`; this module never imports the
parser.
"""

from decimal import Decimal
from typing import Literal

from outrider.ast_facts.models import SkipReason
from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.triage_result import ReviewTier

__all__ = [
    "worker_outcome_from_observed_coverage",
    "worker_outcome_from_observed_skip",
    "worker_outcome_from_parser",
    "worker_outcome_from_plain_skip",
    "worker_outcome_from_serve",
]


def _clone_all(findings: tuple[ReviewFinding, ...]) -> tuple[ReviewFinding, ...]:
    return tuple(f.validated_clone() for f in findings)


def _sorted_hashes(findings: tuple[ReviewFinding, ...]) -> tuple[str, ...]:
    return tuple(sorted({f.content_hash for f in findings}))


def worker_outcome_from_parser(
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
    parse_status: Literal["clean", "failed", "degraded"],
    admitted_findings: tuple[ReviewFinding, ...],
    producer_findings: tuple[ReviewFinding, ...],
    trace_candidates: tuple[TraceCandidate, ...],
    subsumed_matches: tuple[ObservedSubsumedMatch, ...],
    n_proposals_seen: int,
    n_proposals_rejected: int,
    n_responses_rejected: int,
    n_proposals_superseded_by_observed: int,
    n_trace_candidates_dropped_malformed: int,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    cost: Decimal,
    estimated_tokens: int,
) -> AnalyzeWorkerOutcome:
    """The LLM branch. `admitted_findings` is the POST-#054-merge list;
    `producer_findings` is `produce_observed_findings`' raw output. Origin
    derives from object placement: a producer finding whose OBJECT is in
    the admitted list survived the merge (evicted a JUDGED collision or
    appended fresh); one dropped against a non-JUDGED incumbent is absent
    and stays unlisted."""
    admitted_ids = {id(f) for f in admitted_findings}
    producer_hashes = tuple(
        sorted({f.content_hash for f in producer_findings if id(f) in admitted_ids})
    )
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="parser",
        parse_status=parse_status,
        review_tier=review_tier,
        admitted_findings=_clone_all(admitted_findings),
        trace_candidates=trace_candidates,
        subsumed_matches=subsumed_matches,
        producer_observed_hashes=producer_hashes,
        n_proposals_seen=n_proposals_seen,
        n_proposals_rejected=n_proposals_rejected,
        n_responses_rejected=n_responses_rejected,
        n_proposals_superseded_by_observed=n_proposals_superseded_by_observed,
        n_trace_candidates_dropped_malformed=n_trace_candidates_dropped_malformed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        cost=cost,
        estimated_tokens=estimated_tokens,
    )


def worker_outcome_from_serve(
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
    served_findings: tuple[ReviewFinding, ...],
    trace_candidates: tuple[TraceCandidate, ...],
    subsumed_matches: tuple[ObservedSubsumedMatch, ...],
    estimated_tokens: int,
) -> AnalyzeWorkerOutcome:
    """The cache-serve branch: everything admitted was reconstructed from
    the cached payload (served hashes are EXACTLY the admitted set), the
    candidates are prior-pass restorations, and the #055 records are the
    payload's retained proof."""
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="cache_serve",
        parse_status="clean",
        review_tier=review_tier,
        admitted_findings=_clone_all(served_findings),
        trace_candidates=trace_candidates,
        subsumed_matches=subsumed_matches,
        served_content_hashes=_sorted_hashes(served_findings),
        estimated_tokens=estimated_tokens,
    )


def worker_outcome_from_observed_skip(
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
    skip_reason: SkipReason,
    producer_findings: tuple[ReviewFinding, ...],
) -> AnalyzeWorkerOutcome:
    """The enforced/ride-out skip: every finding is the deterministic
    producer's output by construction, so the producer list is the whole
    admitted set."""
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="observed_skip",
        parse_status="skipped",
        skip_reason=skip_reason,
        review_tier=review_tier,
        admitted_findings=_clone_all(producer_findings),
        producer_observed_hashes=_sorted_hashes(producer_findings),
    )


def worker_outcome_from_observed_coverage(
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
    producer_findings: tuple[ReviewFinding, ...],
    estimated_tokens: int,
) -> AnalyzeWorkerOutcome:
    """The #049 ENFORCED coverage skip: every changed scope was
    skip_safe-covered, the LLM was not called, and the file counts as
    EXAMINED (clean status, no SkipReason). Every finding is producer
    output by construction."""
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="observed_coverage",
        parse_status="clean",
        review_tier=review_tier,
        admitted_findings=_clone_all(producer_findings),
        producer_observed_hashes=_sorted_hashes(producer_findings),
        estimated_tokens=estimated_tokens,
    )


def worker_outcome_from_plain_skip(
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
    skip_reason: SkipReason,
) -> AnalyzeWorkerOutcome:
    """A skip that emitted nothing."""
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="plain_skip",
        parse_status="skipped",
        skip_reason=skip_reason,
        review_tier=review_tier,
    )
