# Aggregate fold for the parallel-analyze fan-out per specs/2026-07-05-parallel-analyze.md.
"""Combine worker outcomes into one `AnalysisRound` and its event inputs.

`fold_worker_outcomes` is pure; `analyze_aggregate` emits the resulting
events and anomalies. Workers do not create rounds (DECISIONS.md#063).

Sorted path order makes duplicate winners and output order independent
of worker completion order. Admission deduplicates by `(content_hash,
proposal_hash)`, then keeps the first finding per content hash before
applying the severity cap. Round identity sorts the examined/skipped paths
and retained content hashes, so reordering the same inputs keeps the same ID.

Counters classify findings by recorded producer and cache provenance,
not evidence tier. Kept findings use `ReviewFinding.validated_clone()`
to rerun validation and avoid sharing objects with worker outcomes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from outrider.agent.nodes.finding_cap import (
    _collapse_by_content_hash,
    admit_with_pair_dedup,
    cap_findings_by_severity,
)
from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.ast_facts.models import SkipReason
from outrider.audit.events import AnalyzeCompletedEvent, ReviewPhaseEvent
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.canonical import compute_phase_id, compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas.analysis_round import (
    MAX_FINDINGS_HARD_CAP,
    MAX_FINDINGS_PER_ROUND,
    AnalysisRound,
)
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from pydantic import AwareDatetime

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.sinks import AnalyzeEventSink, PhaseEventSink
    from outrider.schemas import ReviewState
    from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
    from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.trace_candidate import TraceCandidate

logger = logging.getLogger(__name__)

__all__ = ["AggregateFold", "FoldInputError", "analyze_aggregate", "fold_worker_outcomes"]


class FoldInputError(ValueError):
    """Worker outcomes contain mixed pass indices or duplicate file paths."""


@dataclass(frozen=True, slots=True)
class AggregateFold:
    """Round, counters, and anomaly inputs for the aggregate node.

    Counter fields populate `AnalyzeCompletedEvent`. `budget_skip_count`
    and `gated_overflow` drive anomalies; `standard_tier_llm_used` selects
    the configured model name without passing model strings into the fold.
    """

    round: AnalysisRound
    trace_candidates: tuple[TraceCandidate, ...]
    subsumed_matches: tuple[ObservedSubsumedMatch, ...]
    n_files_analyzed: int
    n_files_skipped: int
    n_llm_calls: int
    n_proposals_seen: int
    n_findings_emitted: int
    n_findings_served: int
    n_findings_observed: int
    n_proposals_superseded_by_observed: int
    n_proposals_dropped: int
    n_findings_dropped_over_cap: int
    n_proposals_rejected: int
    n_responses_rejected: int
    n_trace_candidates_emitted: int
    n_trace_candidates_dropped_malformed: int
    total_input_tokens: int
    total_output_tokens: int
    total_cache_read_tokens: int
    total_cache_write_tokens: int
    total_cost: Decimal
    budget_skip_count: int
    gated_overflow: bool
    standard_tier_llm_used: bool


def _clone(finding: ReviewFinding) -> ReviewFinding:
    """Clone through `validated_clone()` to preserve validation and isolation."""
    return finding.validated_clone()


def fold_worker_outcomes(
    outcomes: tuple[AnalyzeWorkerOutcome, ...],
    *,
    pass_index: int,
    started_at: AwareDatetime,
    ended_at: AwareDatetime,
) -> AggregateFold:
    """Fold one pass in sorted path order, regardless of completion order.

    Reject mixed pass indices and duplicate paths. Zero outcomes produces
    an empty round for the planner's zero-worker route.
    """
    if any(outcome.pass_index != pass_index for outcome in outcomes):
        raise FoldInputError(
            f"fold_worker_outcomes: outcomes span pass indices "
            f"{sorted({outcome.pass_index for outcome in outcomes})}; expected only {pass_index}"
        )
    ordered_outcomes = tuple(sorted(outcomes, key=lambda outcome: outcome.path))
    outcome_paths = [outcome.path for outcome in ordered_outcomes]
    if len(set(outcome_paths)) != len(outcome_paths):
        raise FoldInputError("fold_worker_outcomes: duplicate paths in one pass")

    files_examined: list[str] = []
    files_skipped: list[str] = []
    admitted_findings: list[ReviewFinding] = []
    admitted_finding_keys: set[tuple[str, str]] = set()
    trace_candidates: list[TraceCandidate] = []
    subsumed_matches: list[ObservedSubsumedMatch] = []
    served_content_hashes: set[str] = set()
    producer_content_hashes: set[str] = set()
    n_llm_calls = 0
    n_proposals_seen = 0
    n_proposals_rejected = 0
    n_responses_rejected = 0
    n_proposals_superseded_by_observed = 0
    n_trace_candidates_dropped_malformed = 0
    n_trace_candidates_emitted = 0
    admitted_count_before_dedup = 0
    served_count_before_dedup = 0
    producer_count_before_dedup = 0
    input_tokens = output_tokens = cache_read_tokens = cache_write_tokens = 0
    total_cost = Decimal("0")
    budget_skip_count = 0
    standard_tier_llm_used = False

    for outcome in ordered_outcomes:
        if outcome.parse_status == "skipped":
            files_skipped.append(outcome.path)
            if outcome.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED:
                budget_skip_count += 1
        else:
            files_examined.append(outcome.path)
        if outcome.source == "parser":
            n_llm_calls += 1
            if outcome.review_tier is ReviewTier.STANDARD:
                standard_tier_llm_used = True
            # Only parser candidates count as new output; cached candidates are restored.
            n_trace_candidates_emitted += len(outcome.trace_candidates)
        n_proposals_seen += outcome.n_proposals_seen
        n_proposals_rejected += outcome.n_proposals_rejected
        n_responses_rejected += outcome.n_responses_rejected
        n_proposals_superseded_by_observed += outcome.n_proposals_superseded_by_observed
        n_trace_candidates_dropped_malformed += outcome.n_trace_candidates_dropped_malformed
        admitted_count_before_dedup += len(outcome.admitted_findings)
        served_count_before_dedup += len(outcome.served_content_hashes)
        producer_count_before_dedup += len(outcome.producer_observed_hashes)
        served_content_hashes.update(outcome.served_content_hashes)
        producer_content_hashes.update(outcome.producer_observed_hashes)
        input_tokens += outcome.input_tokens
        output_tokens += outcome.output_tokens
        cache_read_tokens += outcome.cache_read_tokens
        cache_write_tokens += outcome.cache_write_tokens
        total_cost += outcome.cost
        # Share the admission key with sequential pass-1 accumulation.
        admit_with_pair_dedup(outcome.admitted_findings, admitted_findings, admitted_finding_keys)
        trace_candidates.extend(outcome.trace_candidates)
        subsumed_matches.extend(outcome.subsumed_matches)

    # Different proposals can share a content hash; AnalysisRound requires
    # unique content hashes, so keep the first finding in path order.
    unique_findings = _collapse_by_content_hash(admitted_findings)

    # Keep gated findings beyond the soft cap (hitl-gates-high-severity).
    # More gated findings than the hard ceiling raises before round creation.
    kept_findings, dropped_findings = cap_findings_by_severity(
        unique_findings, soft_cap=MAX_FINDINGS_PER_ROUND, hard_cap=MAX_FINDINGS_HARD_CAP
    )

    # Recount kept findings by provenance: served, then producer-observed,
    # then model proposals. Served findings take precedence over producer hashes.
    kept_served_count = sum(
        1 for finding in kept_findings if finding.content_hash in served_content_hashes
    )
    kept_producer_count = sum(
        1
        for finding in kept_findings
        if finding.content_hash not in served_content_hashes
        and finding.content_hash in producer_content_hashes
    )
    kept_proposal_count = len(kept_findings) - kept_served_count - kept_producer_count
    admitted_proposal_count_before_dedup = (
        admitted_count_before_dedup - served_count_before_dedup - producer_count_before_dedup
    )
    n_proposals_dropped = admitted_proposal_count_before_dedup - kept_proposal_count

    round_findings = tuple(_clone(finding) for finding in kept_findings)
    round_id = compute_round_id(
        pass_index=pass_index,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        finding_content_hashes=tuple(finding.content_hash for finding in round_findings),
    )
    analysis_round = AnalysisRound(
        round_id=round_id,
        pass_index=pass_index,
        findings=round_findings,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        started_at=started_at,
        ended_at=ended_at,
    )
    return AggregateFold(
        round=analysis_round,
        trace_candidates=tuple(trace_candidates),
        subsumed_matches=tuple(subsumed_matches),
        n_files_analyzed=len(files_examined),
        n_files_skipped=len(files_skipped),
        n_llm_calls=n_llm_calls,
        n_proposals_seen=n_proposals_seen,
        n_findings_emitted=len(kept_findings),
        n_findings_served=kept_served_count,
        n_findings_observed=kept_producer_count,
        n_proposals_superseded_by_observed=n_proposals_superseded_by_observed,
        n_proposals_dropped=n_proposals_dropped,
        n_findings_dropped_over_cap=len(dropped_findings),
        n_proposals_rejected=n_proposals_rejected,
        n_responses_rejected=n_responses_rejected,
        n_trace_candidates_emitted=n_trace_candidates_emitted,
        n_trace_candidates_dropped_malformed=n_trace_candidates_dropped_malformed,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cache_read_tokens=cache_read_tokens,
        total_cache_write_tokens=cache_write_tokens,
        total_cost=total_cost,
        budget_skip_count=budget_skip_count,
        gated_overflow=len(kept_findings) > MAX_FINDINGS_PER_ROUND,
        standard_tier_llm_used=standard_tier_llm_used,
    )


async def analyze_aggregate(
    state: ReviewState,
    *,
    analyze_event_sink: AnalyzeEventSink,
    phase_event_sink: PhaseEventSink,
    anomaly_sink: AnomalySink,
    analyze_model: str,
    standard_analyze_model: str,
    total_review_budget_tokens: int,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
) -> dict[str, object]:
    """Fold pass-0 outcomes after all workers complete, including an empty pass.

    Emit starvation and gated-overflow anomalies, kept findings, the completion
    event, then the aggregate phase end, in that order. Planner, worker, and
    aggregate phases share the logical `analyze` node (DECISIONS.md#064).

    Clamp `ended_at` to `started_at` if the clock moves backwards. A pass can
    resume in another process, so a process-local monotonic clock cannot
    measure its full duration.
    """
    pass_index = len(state.analysis_rounds)
    # Use the phase key unchanged for retry-stable identity under the logical
    # analyze node (DECISIONS.md#064).
    phase_key = f"aggregate#{pass_index}"
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="analyze",
        attempt_key=phase_key,
    )
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="start",
            is_eval=state.is_eval,
            phase_key=phase_key,
        )
    )
    outcomes = tuple(
        outcome for outcome in state.analyze_worker_outcomes if outcome.pass_index == pass_index
    )
    started_at = state.analyze_pass_started_at
    if started_at is None:
        raise RuntimeError(
            "analyze_aggregate: state.analyze_pass_started_at is unset — the "
            "planner step writes it before any Send; reaching the aggregate "
            "without it is a graph-wiring bug"
        )
    ended_at = max(started_at, datetime.now(UTC))

    fold = fold_worker_outcomes(
        outcomes,
        pass_index=pass_index,
        started_at=started_at,
        ended_at=ended_at,
    )

    # Validate accounting before anomalies and findings can be emitted.
    completed_event = AnalyzeCompletedEvent(
        review_id=state.review_id,
        is_eval=state.is_eval,
        pass_index=pass_index,
        n_files_analyzed=fold.n_files_analyzed,
        n_files_skipped=fold.n_files_skipped,
        n_llm_calls=fold.n_llm_calls,
        n_proposals_seen=fold.n_proposals_seen,
        n_findings_emitted=fold.n_findings_emitted,
        n_findings_served=fold.n_findings_served,
        n_findings_observed=fold.n_findings_observed,
        n_proposals_superseded_by_observed=fold.n_proposals_superseded_by_observed,
        n_proposals_dropped=fold.n_proposals_dropped,
        n_findings_dropped_over_cap=fold.n_findings_dropped_over_cap,
        subsumed_matches=fold.subsumed_matches,
        n_proposals_rejected=fold.n_proposals_rejected,
        n_responses_rejected=fold.n_responses_rejected,
        n_trace_candidates_emitted=fold.n_trace_candidates_emitted,
        n_trace_candidates_dropped_malformed=fold.n_trace_candidates_dropped_malformed,
        total_input_tokens=fold.total_input_tokens,
        total_cache_read_tokens=fold.total_cache_read_tokens,
        total_cache_write_tokens=fold.total_cache_write_tokens,
        total_output_tokens=fold.total_output_tokens,
        # Sum costs as Decimal; convert to float only at the event boundary.
        total_cost_usd=float(fold.total_cost),
        pricing_version=PRICING_VERSION,
        policy_version=active_policy_version,
        analyze_model=analyze_model,
        standard_analyze_model=(standard_analyze_model if fold.standard_tier_llm_used else None),
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        # Aggregate-keyed: pass-level accounting is aggregate work.
        phase_key=phase_key,
    )

    # Keep this import local so importing the pure fold does not load the analyze node.
    from outrider.agent.nodes.analyze import COST_BUDGET_STARVATION_THRESHOLD  # noqa: PLC0415

    # Log ordinary anomaly failures and continue; cancellation still propagates.
    if fold.budget_skip_count >= COST_BUDGET_STARVATION_THRESHOLD:
        try:
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.COST_BUDGET_STARVATION,
                severity=AnomalySeverity.MEDIUM,
                details={
                    "budget_skipped_count": fold.budget_skip_count,
                    "total_review_budget_tokens": total_review_budget_tokens,
                    "pass_index": pass_index,
                },
                is_eval=state.is_eval,
            )
        except Exception:
            logger.exception("analyze_cost_budget_starvation_anomaly_emit_failed")
    if fold.gated_overflow:
        try:
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.GATED_FINDINGS_OVER_CAP,
                severity=AnomalySeverity.HIGH,
                details={
                    "n_kept": len(fold.round.findings),
                    "soft_cap": MAX_FINDINGS_PER_ROUND,
                    "pass_index": pass_index,
                },
                is_eval=state.is_eval,
            )
        except Exception:
            logger.exception("analyze_gated_findings_over_cap_anomaly_emit_failed")

    # Emit the round's kept findings under the aggregate phase; each event
    # retains its own file path for attribution.
    for finding in fold.round.findings:
        await analyze_event_sink.emit_finding(finding, is_eval=state.is_eval, phase_key=phase_key)

    await analyze_event_sink.emit_analyze_completed(completed_event)

    # Close the aggregate phase after findings and completion accounting.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="end",
            is_eval=state.is_eval,
            phase_key=phase_key,
        )
    )

    return {
        "analysis_rounds": [fold.round],
        "trace_candidates": list(fold.trace_candidates),
    }
