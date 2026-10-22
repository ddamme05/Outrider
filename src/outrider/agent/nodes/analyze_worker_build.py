# Worker-outcome builders per specs/2026-07-05-parallel-analyze.md.
"""Build per-file analysis outcomes from parser, cache, and skip results.

Builders use `ReviewFinding.validated_clone()` to avoid sharing findings
with callers and to rerun validation. Counters arrive as values, so this
module does not import the parser.
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
    return tuple(finding.validated_clone() for finding in findings)


def _sorted_hashes(findings: tuple[ReviewFinding, ...]) -> tuple[str, ...]:
    return tuple(sorted({finding.content_hash for finding in findings}))


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
    """Build an LLM outcome from findings after the producer merge.

    Check producer object identity before cloning: only producer findings
    that survive in `admitted_findings` count as producer output. Evidence
    tier alone would also count model proposals citing structural queries.
    """
    admitted_object_ids = {id(finding) for finding in admitted_findings}
    surviving_producer_hashes = tuple(
        sorted(
            {
                finding.content_hash
                for finding in producer_findings
                if id(finding) in admitted_object_ids
            }
        )
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
        producer_observed_hashes=surviving_producer_hashes,
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
    """Restore cached findings, trace candidates, and subsumption proof.

    Served hashes cover exactly the admitted findings. Cached candidates
    are restored without counting as new model output in this pass.
    """
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
    """Keep deterministic findings when LLM analysis is skipped.

    Every finding is producer output; the outcome retains skipped status
    and its reason.
    """
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
    """Record deterministic coverage without an LLM call.

    The caller proved that added lines in the included scopes have skip-safe
    query coverage; removed lines would block skipping. Every finding is
    producer output. The file counts as examined: clean status, no skip reason.
    """
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
    """Record a skip with no findings."""
    return AnalyzeWorkerOutcome(
        path=path,
        pass_index=pass_index,
        source="plain_skip",
        parse_status="skipped",
        skip_reason=skip_reason,
        review_tier=review_tier,
    )
