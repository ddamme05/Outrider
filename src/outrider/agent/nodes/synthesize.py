# Synthesize-node body per specs/2026-05-28-synthesize-node.md
"""Synthesize node — aggregate findings into a `ReviewReport`.

Per spec gate #7 (fail-loud on cross-round severity divergence): when
two findings share a `content_hash` across analysis rounds but disagree
on `severity`, the `severity-set-by-policy` invariant is violated by
construction (the hash is keyed over `finding_type` and severity is
SEVERITY_POLICY[finding_type], so same content_hash → same severity
must hold). Divergence is corruption, not variance. Synthesize emits
an `AnomalyRuleName.CROSS_ROUND_SEVERITY_DIVERGENCE` anomaly via the
injected `AnomalySink` THEN raises `SynthesizeAggregationError`. The
anomaly row commits before the raise so ops sees the corruption signal
in the queue while the review parks for triage.

Per pre-spec gate #1: `SynthesizeCompletedEvent` uses event_id-PK
idempotency (NOT natural-key). The natural-key state-lockstep gate
fails because `ReviewReport.summary` lives in `llm_call_content`, not
the audit-row payload.

Per pre-spec gate #6 (option c): summary text persists in BOTH
`llm_call_content` AND LangGraph checkpoint payloads with independent
retention authorities. Replay-equivalence is retention-conditional —
within the LLM-content TTL window, audit_events + llm_call_content
reconstruct the full prose; outside it, metadata-only replay via
`summary_content_hash` is the canonical claim.

Per the `AnomalySink` two-caller-class contract: synthesize is a GRAPH
caller, NOT a sweep caller — it does NOT acquire `SWEEP_LOCK_ID`.
Rationale is DB-layer idempotency, NOT serialization. The per-rule
partial unique index + `postgresql_insert(...).on_conflict_do_nothing(...)`
makes re-emission a clean no-op regardless of concurrent-ainvoke
ordering (which DECISIONS.md#027 line 946 says is NOT guaranteed).

Order of operations (failure-path-significant):
  1. Capture monotonic clock for wall_clock_seconds metric.
  2. Emit ReviewPhaseEvent(marker=start).
  3. Flatten findings across all analysis_rounds + validate severity
     consistency on duplicate content_hashes. On divergence: emit
     anomaly, then raise SynthesizeAggregationError.
  4. Dedup by content_hash (validator on ReviewReport.findings also
     enforces uniqueness as defense-in-depth).
  5. Compute deterministic metrics (files_examined, files_traced
     beyond_diff, wall_clock_seconds — others are V1 placeholders).
  6. Build LLMRequest for the Sonnet summary call.
  7. Call provider.complete() (raises LLMProviderError subclasses
     on transport failure).
  8. Parse summary via strip_outer_json_fence (Anthropic occasionally
     wraps despite prompt instruction — vendor-payloads-normalized-
     at-boundary).
  9. Compute summary_content_hash.
  10. Construct ReviewReport (schema validators run: max_length=2000
      on summary, dedup-and-sort on findings, ge=0 / le bounds on
      metrics).
  11. Emit SynthesizeCompletedEvent (event_id-PK idempotent via
      _persist_non_phase_event).
  12. Emit ReviewPhaseEvent(marker=end).
  13. Return state delta {"review_report": ...}.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING

from outrider.anomaly import AnomalyRuleName, AnomalySeverity
from outrider.audit.events import ReviewPhaseEvent, SynthesizeCompletedEvent
from outrider.llm.base import LLMRequest
from outrider.llm.parsing import strip_outer_json_fence
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.canonical import compute_phase_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import synthesize as synthesize_prompt
from outrider.schemas.review_report import ReviewMetrics, ReviewReport

if TYPE_CHECKING:
    from outrider.anomaly import AnomalySink
    from outrider.audit.sinks import PhaseEventSink, SynthesizeEventSink
    from outrider.llm.base import LLMProvider
    from outrider.policy.severity import FindingSeverity
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_state import ReviewState


class SynthesizeAggregationError(RuntimeError):
    """Raised when cross-round severity divergence is detected.

    `compute_finding_content_hash` is keyed over `(file_path,
    line_start, line_end, finding_type)`, and
    `ReviewFinding._verify_baseline_severity` requires severity =
    SEVERITY_POLICY[finding_type]. Same content_hash within a single
    review (single policy_version) MUST have identical severity by
    construction — divergence indicates corruption (validator bypass,
    hash-recipe drift, mid-review policy-version change), NOT model
    variance.

    Carries the diverging content_hash + the severity-set + round
    indices for the anomaly payload + ops triage. The emit-then-raise
    contract in the node body commits the anomaly row before this
    exception propagates, so ops sees the signal in the queue while
    the review parks unfinished.
    """

    def __init__(
        self,
        *,
        content_hash: str,
        severities: tuple[FindingSeverity, ...],
        round_indices: tuple[int, ...],
    ) -> None:
        self.content_hash = content_hash
        self.severities = severities
        self.round_indices = round_indices
        super().__init__(
            f"Cross-round severity divergence for content_hash={content_hash!r}: "
            f"severities={[s.value for s in severities]!r} across "
            f"round_indices={list(round_indices)!r}. This is corruption per the "
            f"severity-set-by-policy invariant — same content_hash MUST have "
            f"same severity by construction. Anomaly emitted; review parked."
        )


def _compute_summary_content_hash(text: str) -> str:
    """SHA-256 hex of the summary text (UTF-8 bytes).

    Identity check for retention-conditional replay per pre-spec gate
    #6 option (c). Within the LLM-content TTL window, an audit reader
    can join on this hash to fetch the prose from llm_call_content;
    outside it, the hash is the only proof of which summary was
    produced.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _detect_and_report_divergence(
    *,
    state: ReviewState,
    anomaly_sink: AnomalySink,
) -> dict[str, ReviewFinding]:
    """Walk analysis_rounds, group by content_hash, detect severity
    divergence. On first divergence: emit anomaly + raise.

    Returns a `content_hash → kept_finding` mapping for the dedup step.
    Tie-breaks on (round_index ASC, finding_id ASC) within a group of
    same-content_hash findings; per the severity-set-by-policy invariant
    all findings in a group MUST share severity, so any representative
    suffices.
    """
    by_hash: dict[str, list[tuple[int, ReviewFinding]]] = {}
    for round_index, analysis_round in enumerate(state.analysis_rounds):
        for finding in analysis_round.findings:
            by_hash.setdefault(finding.content_hash, []).append((round_index, finding))

    kept: dict[str, ReviewFinding] = {}
    for content_hash, entries in by_hash.items():
        severities = {entry[1].severity for entry in entries}
        if len(severities) > 1:
            severity_tuple = tuple(sorted({e[1].severity for e in entries}, key=lambda s: s.value))
            round_indices_tuple = tuple(sorted({e[0] for e in entries}))
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.CROSS_ROUND_SEVERITY_DIVERGENCE,
                severity=AnomalySeverity.HIGH,
                details={
                    "content_hash": content_hash,
                    "severities": [s.value for s in severity_tuple],
                    "round_indices": list(round_indices_tuple),
                },
                is_eval=state.is_eval,
            )
            raise SynthesizeAggregationError(
                content_hash=content_hash,
                severities=severity_tuple,
                round_indices=round_indices_tuple,
            )
        # No divergence — pick deterministic representative (lowest
        # round_index first; finding_id sort as final tie-break).
        entries.sort(key=lambda pair: (pair[0], str(pair[1].finding_id)))
        kept[content_hash] = entries[0][1]

    return kept


def _compute_files_traced_beyond_diff(state: ReviewState) -> int:
    """Count of files trace fetched that weren't already in the PR
    diff. Reads `state.trace_decisions` for resolution_status='resolved'
    entries whose `target_file` was outside the original
    `pr_context.changed_files` set.
    """
    diff_paths = {cf.path for cf in state.pr_context.changed_files}
    traced: set[str] = set()
    for decision in state.trace_decisions:
        if decision.target_file is not None and decision.target_file not in diff_paths:
            traced.add(decision.target_file)
    return len(traced)


def _compute_metrics(
    *,
    state: ReviewState,
    wall_clock_seconds: float,
) -> ReviewMetrics:
    """Build ReviewMetrics from state + wall-clock measurement.

    V1 caveat: LLM-aggregate metrics (llm_calls_made, total_*_tokens,
    total_cost_usd) are placeholder zeros — the deterministic
    derivation requires querying audit_events for this review_id and
    summing LLMCallEvent rows. Tracked as FUP: the dashboard reads
    audit truth, not these denormalized fields, so V1 ships with
    placeholders + the FUP for the audit-query helper. Adding the
    helper changes the values but does not change the schema or any
    downstream contract.
    """
    files_examined: set[str] = set()
    for analysis_round in state.analysis_rounds:
        files_examined.update(analysis_round.files_examined)

    return ReviewMetrics(
        files_examined=len(files_examined),
        files_traced_beyond_diff=_compute_files_traced_beyond_diff(state),
        # V1 placeholders — FUP for audit-query-derived aggregates.
        llm_calls_made=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_cost_usd=0.0,
        wall_clock_seconds=wall_clock_seconds,
    )


async def synthesize(  # noqa: PLR0913 — closure-injected deps + node-body orchestration
    state: ReviewState,
    *,
    provider: LLMProvider,
    synthesize_model: str,
    phase_event_sink: PhaseEventSink,
    synthesize_event_sink: SynthesizeEventSink,
    anomaly_sink: AnomalySink,
) -> dict[str, ReviewReport]:
    """Run the synthesize aggregation pass.

    Returns `{"review_report": ReviewReport(...)}` for LangGraph's
    reducer to merge into state. Default reducer is overwrite —
    appropriate here because `review_report` is a singleton field.

    Closure-injected dependencies per `nodes-receive-deps-via-closure`:
    `provider` for the Sonnet call, `synthesize_model` from config per
    `model-strings-from-config-not-hardcoded`, the four sinks for the
    four audit/anomaly surfaces this node touches.

    Raises:
        SynthesizeAggregationError: cross-round severity divergence
            detected (corruption per severity-set-by-policy). An
            anomaly row commits before the raise.
        LLMProviderError: provider transport/parsing failure on the
            Sonnet summary call.
        pydantic.ValidationError: ReviewReport/ReviewMetrics validators
            reject the constructed values (oversize summary, mutated
            findings tuple, out-of-range metrics).
    """
    # Step 1: capture monotonic clock for the wall-clock metric.
    t0 = time.monotonic()

    # Step 2: emit start phase event.
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="synthesize",
        attempt_key="synthesize",
    )
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="synthesize",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Step 3-4: flatten + dedup with severity-divergence detection.
    # Raises SynthesizeAggregationError on corruption (anomaly emitted
    # before the raise; review parks).
    kept_by_hash = await _detect_and_report_divergence(
        state=state,
        anomaly_sink=anomaly_sink,
    )
    # `ReviewReport._canonicalize_findings` re-sorts by severity; we
    # pass arbitrary order here (the schema canonicalizes).
    deduplicated_findings = tuple(kept_by_hash.values())

    # Step 5: compute metrics (wall-clock not yet final — assembled
    # after the LLM call).

    # Step 6: build LLMRequest for the Sonnet summary call.
    # `overall_risk` is required upstream (triage produces it).
    if state.triage_result is None:
        msg = (
            "synthesize requires state.triage_result to be set "
            "(triage node must have run before synthesize)"
        )
        raise RuntimeError(msg)
    overall_risk = state.triage_result.overall_risk

    # Pre-compute the user prompt with PLACEHOLDER metrics — the
    # final wall_clock_seconds is set after the LLM call, but the
    # prompt's metrics_summary is content the model sees BEFORE its
    # own call lands. Using a pre-call snapshot is correct for the
    # prompt (the model summarizes the review state at synthesize
    # entry, not the synthesize call's own cost contribution).
    pre_call_metrics = _compute_metrics(
        state=state,
        wall_clock_seconds=time.monotonic() - t0,
    )
    parts = synthesize_prompt.render(
        overall_risk=overall_risk,
        findings=deduplicated_findings,
        metrics=pre_call_metrics,
    )

    request = LLMRequest(
        model=synthesize_model,
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        max_tokens=synthesize_prompt.MAX_TOKENS,
        temperature=synthesize_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="synthesize",
        is_eval=state.is_eval,
        prompt_template_version=synthesize_prompt.VERSION,
        degraded_mode=False,
    )

    # Step 7: provider call. Internal persister emits LLMCallEvent +
    # llm_call_content rows BEFORE returning per LLMProvider contract.
    response = await provider.complete(request)

    # Step 8: normalize Sonnet envelope (sometimes wraps in ```json```
    # despite the prompt instruction — vendor-payloads-normalized-at-
    # boundary). For prose output this is harmless when no fence is
    # present and removes the wrapper when one is.
    summary_text = strip_outer_json_fence(response.text).strip()

    # Step 9: compute the canonical summary content hash.
    summary_content_hash = _compute_summary_content_hash(summary_text)

    # Step 10: construct the final ReviewReport. Schema validators
    # run: Field(max_length=2000) on summary, _canonicalize_findings
    # on the findings tuple, ge=0/le bounds on metrics.
    final_metrics = _compute_metrics(
        state=state,
        wall_clock_seconds=time.monotonic() - t0,
    )
    review_report = ReviewReport(
        summary=summary_text,
        overall_risk=overall_risk,
        findings=deduplicated_findings,
        metrics=final_metrics,
    )

    # Step 11: emit the per-review completion event.
    await synthesize_event_sink.emit_synthesize_completed(
        SynthesizeCompletedEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            summary_content_hash=summary_content_hash,
            overall_risk=overall_risk,
            n_findings=len(deduplicated_findings),
            files_examined=final_metrics.files_examined,
            files_traced_beyond_diff=final_metrics.files_traced_beyond_diff,
            llm_calls_made=final_metrics.llm_calls_made,
            total_input_tokens=final_metrics.total_input_tokens,
            total_output_tokens=final_metrics.total_output_tokens,
            total_cost_usd=final_metrics.total_cost_usd,
            wall_clock_seconds=final_metrics.wall_clock_seconds,
            pricing_version=PRICING_VERSION,
            policy_version=ACTIVE_POLICY_VERSION,
            synthesize_model=synthesize_model,
        )
    )

    # Step 12: emit end phase event.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="synthesize",
            marker="end",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Step 13: return state delta. LangGraph's default overwrite reducer
    # applies (scalar slot).
    return {"review_report": review_report}


__all__ = [
    "SynthesizeAggregationError",
    "synthesize",
]
