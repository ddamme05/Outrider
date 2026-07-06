# Aggregate fold for the parallel-analyze fan-out per specs/2026-07-05-parallel-analyze.md.
"""Fold per-(file, pass) worker outcomes into one `AnalysisRound` per pass.

Pure and import-light (no LLM, prompt, or graph machinery — the
`decide_degradation` / `analyze_budget` precedent): the aggregate node
calls `fold_worker_outcomes` once per pass and turns the returned
`AggregateFold` into side effects (the `AnalyzeCompletedEvent`, the
FindingEvent emissions, the anomaly signals). Per `DECISIONS.md#063`
workers never emit rounds; this fold is the ONE place worker results
become a round.

Fidelity contract: this mirrors the sequential main loop's accumulation
and post-processing exactly — cross-source `(content_hash,
proposal_hash)` admission dedup, content-hash collapse (first-wins),
the gated-aware severity cap, and the post-cap counter recompute — with
outcomes folded in SORTED PATH ORDER so worker completion order can
never change the round (a completion-order-dependent fold would break
replay idempotence). The sequential loop iterates tier-descending (a
budget-pressure ordering that is a planner concern, not round
identity), so the round's state-visible tuples (`files_examined`,
`findings`) come out in a different ORDER than the sequential round's —
but `compute_round_id` sorts its hashed inputs internally, so both
orderings produce the SAME `round_id` and collapse as one round on the
dedup reducer. Ordering is not an identity divergence.

The ONE deliberate, documented divergence: the sequential post-cap recompute
classifies producer-OBSERVED findings by a HEURISTIC (tier + registry
membership), which miscounts a model-cited OBSERVED proposal as producer
output. The fold classifies by ORIGIN IDENTITY
(`producer_observed_hashes`, recorded by the code that produced them) —
strictly more accurate, and the accounting the event docstring always
claimed ("findings the analyze node PRODUCED this pass").

Non-aliasing (the 3b-2 acceptance gate): the fold CLONES every kept
finding via `model_validate` round-trip (the validator-safe clone —
`model_copy` skips validators per the `ReviewFinding` docstring), so no
live object is shared between `analyze_worker_outcomes` and the round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from outrider.agent.nodes.finding_cap import cap_findings_by_severity
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
from outrider.schemas.review_finding import ReviewFinding
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from pydantic import AwareDatetime

    from outrider.anomaly.sinks import AnomalySink
    from outrider.audit.sinks import AnalyzeEventSink, PhaseEventSink
    from outrider.schemas import ReviewState
    from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
    from outrider.schemas.observed_subsumption import ObservedSubsumedMatch
    from outrider.schemas.trace_candidate import TraceCandidate

logger = logging.getLogger(__name__)

__all__ = ["AggregateFold", "FoldInputError", "analyze_aggregate", "fold_worker_outcomes"]


class FoldInputError(ValueError):
    """The outcome set is not a valid single pass (mixed pass indices or
    duplicate paths). Both are producer bugs the slot-guard reducer and
    planner gate should have made impossible — fail loud, never fold."""


@dataclass(frozen=True, slots=True)
class AggregateFold:
    """Everything the aggregate node needs to emit the pass's side effects.

    `round` carries CLONED findings (non-aliasing gate). The counter
    fields map one-to-one onto `AnalyzeCompletedEvent`'s producer-side
    fields; `budget_skip_count` and `gated_overflow` feed the two anomaly
    signals; `standard_tier_llm_used` lets the node record the STANDARD
    model name (config-owned — the fold never sees model strings).
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
    """Validator-safe deep clone (non-aliasing gate): `model_validate` over
    the dump re-runs the full validator chain; `model_copy` would skip it."""
    return ReviewFinding.model_validate(finding.model_dump())


def fold_worker_outcomes(
    outcomes: tuple[AnalyzeWorkerOutcome, ...],
    *,
    pass_index: int,
    started_at: AwareDatetime,
    ended_at: AwareDatetime,
) -> AggregateFold:
    """Fold one pass's worker outcomes into the round + event inputs.

    Deterministic: outcomes fold in sorted-path order regardless of the
    tuple's (completion) order. Zero outcomes is the valid empty pass —
    one empty round still folds (the zero-worker planner→aggregate route).
    """
    if any(o.pass_index != pass_index for o in outcomes):
        raise FoldInputError(
            f"fold_worker_outcomes: outcomes span pass indices "
            f"{sorted({o.pass_index for o in outcomes})}; expected only {pass_index}"
        )
    ordered = tuple(sorted(outcomes, key=lambda o: o.path))
    paths = [o.path for o in ordered]
    if len(set(paths)) != len(paths):
        raise FoldInputError("fold_worker_outcomes: duplicate paths in one pass")

    files_examined: list[str] = []
    files_skipped: list[str] = []
    admitted: list[ReviewFinding] = []
    admitted_keys: set[tuple[str, str]] = set()
    trace_candidates: list[TraceCandidate] = []
    subsumed: list[ObservedSubsumedMatch] = []
    served_hashes: set[str] = set()
    producer_hashes: set[str] = set()
    n_llm_calls = 0
    n_proposals_seen = 0
    n_proposals_rejected = 0
    n_responses_rejected = 0
    n_superseded = 0
    n_dropped_malformed = 0
    n_trace_emitted = 0
    emitted_pre = 0
    served_pre = 0
    observed_pre = 0
    input_tokens = output_tokens = cache_read = cache_write = 0
    total_cost = Decimal("0")
    budget_skip_count = 0
    standard_tier_llm_used = False

    for o in ordered:
        if o.parse_status == "skipped":
            files_skipped.append(o.path)
            if o.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED:
                budget_skip_count += 1
        else:
            files_examined.append(o.path)
        if o.source == "parser":
            n_llm_calls += 1
            if o.review_tier is ReviewTier.STANDARD:
                standard_tier_llm_used = True
            # Only parser candidates count as emitted THIS pass;
            # cache_serve candidates are prior-pass restorations.
            n_trace_emitted += len(o.trace_candidates)
        n_proposals_seen += o.n_proposals_seen
        n_proposals_rejected += o.n_proposals_rejected
        n_responses_rejected += o.n_responses_rejected
        n_superseded += o.n_proposals_superseded_by_observed
        n_dropped_malformed += o.n_trace_candidates_dropped_malformed
        emitted_pre += len(o.admitted_findings)
        served_pre += len(o.served_content_hashes)
        observed_pre += len(o.producer_observed_hashes)
        served_hashes.update(o.served_content_hashes)
        producer_hashes.update(o.producer_observed_hashes)
        input_tokens += o.input_tokens
        output_tokens += o.output_tokens
        cache_read += o.cache_read_tokens
        cache_write += o.cache_write_tokens
        total_cost += o.cost
        # Cross-source admission dedup on the (content_hash, proposal_hash)
        # pair — the sequential `_admit_with_dedup` contract (FUP-178).
        for finding in o.admitted_findings:
            key = (finding.content_hash, finding.proposal_hash)
            if key not in admitted_keys:
                admitted_keys.add(key)
                admitted.append(finding)
        trace_candidates.extend(o.trace_candidates)
        subsumed.extend(o.subsumed_matches)

    # Content-hash collapse, first-wins (the sequential FUP-180 finding-A
    # collapse: two findings may share content_hash with differing
    # proposal_hash; AnalysisRound enforces content_hash uniqueness).
    collapsed: list[ReviewFinding] = []
    seen_hashes: set[str] = set()
    for finding in admitted:
        if finding.content_hash not in seen_hashes:
            seen_hashes.add(finding.content_hash)
            collapsed.append(finding)

    # Gated-aware severity cap: non-gated drop to the soft cap; gated are
    # never dropped (hitl-gates-high-severity); the hard ceiling fails loud.
    kept, dropped = cap_findings_by_severity(
        collapsed, soft_cap=MAX_FINDINGS_PER_ROUND, hard_cap=MAX_FINDINGS_HARD_CAP
    )

    # Post-cap recompute over the KEPT set, classified by ORIGIN IDENTITY:
    # served (hash in the served union), producer-observed (hash in the
    # producer union, not served), else a surviving model proposal.
    kept_served = sum(1 for f in kept if f.content_hash in served_hashes)
    kept_observed = sum(
        1 for f in kept if f.content_hash not in served_hashes and f.content_hash in producer_hashes
    )
    kept_proposals = len(kept) - kept_served - kept_observed
    parser_proposal_emitted = emitted_pre - served_pre - observed_pre
    n_proposals_dropped = parser_proposal_emitted - kept_proposals

    kept_clones = tuple(_clone(f) for f in kept)  # non-aliasing gate
    round_id = compute_round_id(
        pass_index=pass_index,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        finding_content_hashes=tuple(f.content_hash for f in kept_clones),
    )
    new_round = AnalysisRound(
        round_id=round_id,
        pass_index=pass_index,
        findings=kept_clones,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        started_at=started_at,
        ended_at=ended_at,
    )
    return AggregateFold(
        round=new_round,
        trace_candidates=tuple(trace_candidates),
        subsumed_matches=tuple(subsumed),
        n_files_analyzed=len(files_examined),
        n_files_skipped=len(files_skipped),
        n_llm_calls=n_llm_calls,
        n_proposals_seen=n_proposals_seen,
        n_findings_emitted=len(kept),
        n_findings_served=kept_served,
        n_findings_observed=kept_observed,
        n_proposals_superseded_by_observed=n_superseded,
        n_proposals_dropped=n_proposals_dropped,
        n_findings_dropped_over_cap=len(dropped),
        n_proposals_rejected=n_proposals_rejected,
        n_responses_rejected=n_responses_rejected,
        n_trace_candidates_emitted=n_trace_emitted,
        n_trace_candidates_dropped_malformed=n_dropped_malformed,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
        total_cache_read_tokens=cache_read,
        total_cache_write_tokens=cache_write,
        total_cost=total_cost,
        budget_skip_count=budget_skip_count,
        gated_overflow=len(kept) > MAX_FINDINGS_PER_ROUND,
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
    """The pass-0 aggregate vertex: fold worker outcomes into the round.

    Runs once per pass-0 superstep, after every `analyze_file` worker
    completes (or directly from the planner on the zero-worker route).
    Folds this pass's slot-guarded outcomes and turns the pure
    `AggregateFold` into the pass's side effects, mirroring the
    sequential tail's ORDER exactly: starvation anomaly → gated-overflow
    anomaly → one `FindingEvent` per kept finding → the accounting-
    validated `AnalyzeCompletedEvent` → the phase END marker (the same
    `phase_id` recipe the planner's start marker used — the pass's
    envelope spans the three physical vertices under one logical
    `node_id="analyze"`, per `DECISIONS.md#064`).

    `ended_at` is clamped to `max(started_at, now)`: the pass spans
    vertices (and possibly processes, across a checkpoint resume), so
    the sequential loop's single-process monotonic anchor (FUP-141)
    cannot apply — the clamp preserves the round's ordering invariant
    under a backwards clock jump, trading exact duration for it.
    """
    pass_index = len(state.analysis_rounds)
    # The aggregate's own keyed phase pair (increment 4): attempt_key =
    # phase_key VERBATIM, so phase_id inherits the recipe's retry
    # stability; node_id stays the logical "analyze" (DECISIONS.md#064).
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
    outcomes = tuple(o for o in state.analyze_worker_outcomes if o.pass_index == pass_index)
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

    # Construct the accounting-validated event BEFORE any side effect
    # (the sequential strand-prevention contract: a validator raise
    # crashes cleanly with nothing emitted).
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
        # Decimal-summed across outcomes, cast to float ONCE (the
        # sequential FP discipline).
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

    # Side effects, in the sequential tail's order. Both anomalies are
    # best-effort — observability must never fail the review. The
    # threshold import is function-scoped so the FOLD stays import-light
    # for direct structural testing (this node only runs in graph
    # context, where analyze is imported anyway).
    from outrider.agent.nodes.analyze import COST_BUDGET_STARVATION_THRESHOLD  # noqa: PLC0415

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

    # One FindingEvent per kept finding — the emitted set equals the
    # round by construction (fold output IS the round). Aggregate-keyed
    # (admission is aggregate work); per-file attribution rides the
    # event's own file_path field.
    for finding in fold.round.findings:
        await analyze_event_sink.emit_finding(finding, is_eval=state.is_eval, phase_key=phase_key)

    await analyze_event_sink.emit_analyze_completed(completed_event)

    # Close the aggregate's own phase pair (the planner and each worker
    # closed theirs; three keyed envelopes per pass replace the
    # sequential era's single analyze-pass envelope).
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
