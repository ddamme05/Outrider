"""Build a review report from findings, suggested patches, and a summary.

Validate the triage policy snapshot, reject conflicting duplicate findings,
and cap the report before patch and summary calls. Prompt metrics exclude
the summary call; final metrics include it.

Display text is normalized, while the audit hash binds to the raw response.
Completion uses event-id idempotency. Full prose replay requires retained
LLM content; the hash supports metadata-only replay after content expires.
See specs/2026-05-28-synthesize-node.md for the persistence contract.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from typing import TYPE_CHECKING

from outrider.agent.nodes.finding_cap import cap_findings_by_severity
from outrider.agent.nodes.patch_generation import generate_patches
from outrider.anomaly import AnomalyRuleName, AnomalySeverity
from outrider.audit.events import ReviewPhaseEvent, SynthesizeCompletedEvent
from outrider.llm.base import LLMRequest
from outrider.llm.parsing import strip_outer_json_fence
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.canonical import compute_phase_id
from outrider.prompts import synthesize as synthesize_prompt
from outrider.schemas.analysis_round import MAX_FINDINGS_HARD_CAP
from outrider.schemas.review_report import (
    MAX_FINDINGS_PER_REPORT,
    ReviewMetrics,
    ReviewReport,
)

if TYPE_CHECKING:
    from outrider.anomaly import AnomalySink
    from outrider.audit.aggregates import ReviewLLMAggregates
    from outrider.audit.sinks import PhaseEventSink, SynthesizeEventSink
    from outrider.llm.base import LLMProvider
    from outrider.policy.severity import FindingSeverity
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_state import ReviewState


class FindingForgeryDetectedError(RuntimeError):
    """A finding disagrees with the triage policy snapshot or carries an early override.

    The snapshot is the trusted policy version, including on historical replay.
    original_severity is reserved for HITL, which has not run at node entry.
    """


class SynthesizeAggregationError(RuntimeError):
    """One finding content hash carries conflicting severities or policy versions.

    The exception includes both axes and the participating rounds for triage.
    Anomaly reporting is best-effort; this exception always stops the review.
    """

    def __init__(
        self,
        *,
        content_hash: str,
        severities: tuple[FindingSeverity, ...],
        policy_versions: tuple[str, ...],
        round_indices: tuple[int, ...],
    ) -> None:
        self.content_hash = content_hash
        self.severities = severities
        self.policy_versions = policy_versions
        self.round_indices = round_indices
        super().__init__(
            f"Cross-round divergence for content_hash={content_hash!r}: "
            f"severities={[s.value for s in severities]!r} "
            f"policy_versions={list(policy_versions)!r} across "
            f"round_indices={list(round_indices)!r}. Indicates corruption "
            f"(severity and/or policy_version drift for one content_hash) "
            f"per severity-set-by-policy + severity-policy-versioned-for-"
            f"replay invariants. Anomaly emission is best-effort; review "
            f"parked regardless of emit outcome."
        )


def _compute_summary_content_hash(text: str) -> str:
    """Hash the raw response bytes to match persisted llm_call_content.completion.

    The displayed summary can have its outer fence stripped. Hashing that
    normalized text would break replay's identity check against retained content.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _enforce_synthesize_input_invariants(state: ReviewState) -> None:
    """Require findings to match the triage snapshot and have no HITL override yet.

    Compare against the review's snapshot, rather than the current deployment's
    policy constant, so historical replay remains valid (DECISIONS.md#028).
    """
    # Triage must supply the policy snapshot before findings can be trusted.
    if state.triage_result is None:
        msg = (
            "synthesize requires state.triage_result to be set as "
            "the policy_version snapshot anchor (triage node must "
            "have run before synthesize). Aborting before any audit "
            "row lands."
        )
        raise FindingForgeryDetectedError(msg)
    expected_policy_version = state.triage_result.policy_version
    for round_index, analysis_round in enumerate(state.analysis_rounds):
        for finding in analysis_round.findings:
            if finding.policy_version != expected_policy_version:
                raise FindingForgeryDetectedError(
                    f"synthesize rejected finding with "
                    f"policy_version={finding.policy_version!r} "
                    f"differing from the triage snapshot "
                    f"({expected_policy_version!r}) at "
                    f"round_index={round_index}, "
                    f"content_hash={finding.content_hash!r}. "
                    f"`ReviewFinding._enforce_severity_matches_policy` "
                    f"short-circuits on non-active policy_version; "
                    f"a finding carrying a different version than the "
                    f"triage snapshot indicates a forge attempt. "
                    f"Aborting before audit row lands."
                )
            if finding.original_severity is not None:
                raise FindingForgeryDetectedError(
                    f"synthesize rejected finding with non-None "
                    f"original_severity={finding.original_severity!r} "
                    f"at round_index={round_index}, "
                    f"content_hash={finding.content_hash!r}. HITL has "
                    f"not run at synthesize entry — original_severity "
                    f"is set ONLY after a reviewer override at HITL. "
                    f"A finding carrying original_severity here "
                    f"indicates a forge attempt to bypass the gated "
                    f"set. Aborting before audit row lands."
                )


async def _detect_and_report_divergence(
    *,
    state: ReviewState,
    anomaly_sink: AnomalySink,
) -> dict[str, ReviewFinding]:
    """Group findings by content hash, reject conflicts, and keep one per group.

    Severity or policy-version disagreement is corruption. Try to report an
    anomaly, then raise even if the anomaly sink or logger fails. Keep the
    lowest round index, then lowest finding ID, as the duplicate representative.
    """
    by_hash: dict[str, list[tuple[int, ReviewFinding]]] = {}
    for round_index, analysis_round in enumerate(state.analysis_rounds):
        for finding in analysis_round.findings:
            by_hash.setdefault(finding.content_hash, []).append((round_index, finding))

    kept: dict[str, ReviewFinding] = {}
    for content_hash, entries in by_hash.items():
        severities = {entry[1].severity for entry in entries}
        policy_versions = {entry[1].policy_version for entry in entries}
        # Keep both checks: alternate producers can call this helper without the entry gate.
        if len(severities) > 1 or len(policy_versions) > 1:
            severity_tuple = tuple(sorted({e[1].severity for e in entries}, key=lambda s: s.value))
            policy_version_tuple = tuple(sorted(policy_versions))
            round_indices_tuple = tuple(sorted({e[0] for e in entries}))
            try:
                await anomaly_sink.emit_anomaly(
                    review_id=state.review_id,
                    rule_name=AnomalyRuleName.CROSS_ROUND_SEVERITY_DIVERGENCE,
                    severity=AnomalySeverity.HIGH,
                    details={
                        "content_hash": content_hash,
                        "severities": [s.value for s in severity_tuple],
                        "policy_versions": list(policy_version_tuple),
                        "round_indices": list(round_indices_tuple),
                    },
                    is_eval=state.is_eval,
                )
            except Exception as emit_exc:
                # Even a broken logger must not mask the corruption exception below.
                with contextlib.suppress(Exception):
                    logging.getLogger(__name__).exception(
                        "synthesize_anomaly_emit_failed_during_divergence",
                        extra={
                            "review_id": str(state.review_id),
                            "content_hash": content_hash,
                            "emit_exception_type": type(emit_exc).__name__,
                        },
                    )
            raise SynthesizeAggregationError(
                content_hash=content_hash,
                severities=severity_tuple,
                policy_versions=policy_version_tuple,
                round_indices=round_indices_tuple,
            )
        entries.sort(key=lambda pair: (pair[0], str(pair[1].finding_id)))
        kept[content_hash] = entries[0][1]

    return kept


def _compute_files_traced_beyond_diff(state: ReviewState) -> int:
    """Count distinct paths referenced by trace outside the changed-file set.

    Include decision targets, resolved candidate paths (including ambiguous
    resolutions), and fetched paths. A reference counts even without a fetch
    (DECISIONS.md#030).
    """
    diff_paths = {cf.path for cf in state.pr_context.changed_files}
    referenced: set[str] = set()
    for decision in state.trace_decisions:
        if decision.target_file is not None:
            referenced.add(decision.target_file)
        referenced.update(decision.resolved_candidate_paths)
    for fetched in state.trace_fetched_files:
        referenced.add(fetched.path)
    return len(referenced - diff_paths)


def _compute_metrics(
    *,
    state: ReviewState,
    wall_clock_seconds: float,
    llm_aggregates: ReviewLLMAggregates,
) -> ReviewMetrics:
    """Build metrics from examined files, trace references, elapsed time, and LLM totals.

    New reports carry persisted totals. Nullable fields retain compatibility
    with older audit rows.
    """
    files_examined: set[str] = set()
    for analysis_round in state.analysis_rounds:
        files_examined.update(analysis_round.files_examined)

    return ReviewMetrics(
        files_examined=len(files_examined),
        files_traced_beyond_diff=_compute_files_traced_beyond_diff(state),
        llm_calls_made=llm_aggregates.llm_calls_made,
        total_input_tokens=llm_aggregates.total_input_tokens,
        total_output_tokens=llm_aggregates.total_output_tokens,
        total_cost_usd=llm_aggregates.total_cost_usd,
        wall_clock_seconds=wall_clock_seconds,
    )


async def _snapshot_review_metrics(
    *,
    state: ReviewState,
    synthesize_event_sink: SynthesizeEventSink,
    started_at: float,
) -> ReviewMetrics:
    """Read persisted LLM totals, then capture elapsed time for one metrics snapshot."""
    llm_aggregates = await synthesize_event_sink.query_review_llm_aggregates(
        review_id=state.review_id, is_eval=state.is_eval
    )
    return _compute_metrics(
        state=state,
        wall_clock_seconds=time.monotonic() - started_at,
        llm_aggregates=llm_aggregates,
    )


async def synthesize(  # noqa: PLR0913 — closure-injected deps + node-body orchestration
    state: ReviewState,
    *,
    provider: LLMProvider,
    synthesize_model: str,
    patch_model: str,
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
    patches_enabled: bool,
    max_suggestions: int,
    phase_event_sink: PhaseEventSink,
    synthesize_event_sink: SynthesizeEventSink,
    anomaly_sink: AnomalySink,
) -> dict[str, ReviewReport]:
    """Validate and merge findings, add patches, summarize, and return a review report.

    Cap findings before patch and summary calls. Query metrics before the
    summary for its prompt and after it for the final report. Anomaly reporting
    is best-effort; divergent findings always stop the review.
    """
    started_at = time.monotonic()

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

    _enforce_synthesize_input_invariants(state)
    kept_by_hash = await _detect_and_report_divergence(
        state=state,
        anomaly_sink=anomaly_sink,
    )
    # The report schema canonicalizes finding order.
    report_findings = tuple(kept_by_hash.values())

    # Never drop gated findings. Apply the report cap before patches and summary.
    capped_findings, report_dropped_findings = cap_findings_by_severity(
        list(report_findings),
        soft_cap=MAX_FINDINGS_PER_REPORT,
        hard_cap=MAX_FINDINGS_HARD_CAP,
    )
    n_findings_dropped_over_cap = len(report_dropped_findings)
    # Report gated overflow only after ReviewReport validates.
    report_gated_overflow = len(capped_findings) > MAX_FINDINGS_PER_REPORT
    report_findings = tuple(capped_findings)

    # Include patch work in the prompt metrics and completion snapshot.
    report_findings = await generate_patches(
        report_findings,
        state,
        provider=provider,
        patch_model=patch_model,
        patches_enabled=patches_enabled,
        max_suggestions=max_suggestions,
    )

    # The entry gate guarantees this; retain the check for type narrowing.
    if state.triage_result is None:
        raise FindingForgeryDetectedError(
            "synthesize: triage_result missing past invariant gate "
            "(_enforce_synthesize_input_invariants should have raised "
            "at step 3 — step ordering is broken)"
        )
    overall_risk = state.triage_result.overall_risk

    # Prompt metrics include patch work and exclude this summary call.
    prompt_metrics = await _snapshot_review_metrics(
        state=state,
        synthesize_event_sink=synthesize_event_sink,
        started_at=started_at,
    )
    prompt_parts = synthesize_prompt.render(
        overall_risk=overall_risk,
        findings=report_findings,
        metrics=prompt_metrics,
    )

    request = LLMRequest(
        model=synthesize_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=synthesize_prompt.MAX_TOKENS,
        temperature=synthesize_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="synthesize",
        is_eval=state.is_eval,
        prompt_template_version=synthesize_prompt.VERSION,
        degraded_mode=False,
    )

    response = await provider.complete(request)

    summary_text = strip_outer_json_fence(response.text).strip()

    # Hash raw text so audit identity matches persisted content.
    summary_content_hash = _compute_summary_content_hash(response.text)

    # The final snapshot includes the summary call, already persisted by the provider.
    final_metrics = await _snapshot_review_metrics(
        state=state,
        synthesize_event_sink=synthesize_event_sink,
        started_at=started_at,
    )
    review_report = ReviewReport(
        summary=summary_text,
        overall_risk=overall_risk,
        findings=report_findings,
        metrics=final_metrics,
    )

    # Emit overflow only after the report validates; observability is best-effort.
    if report_gated_overflow:
        try:
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.GATED_FINDINGS_OVER_CAP,
                severity=AnomalySeverity.HIGH,
                details={"n_kept": len(report_findings), "soft_cap": MAX_FINDINGS_PER_REPORT},
                is_eval=state.is_eval,
            )
        except Exception:
            logging.getLogger(__name__).exception(
                "synthesize_gated_findings_over_cap_anomaly_emit_failed"
            )

    await synthesize_event_sink.emit_synthesize_completed(
        SynthesizeCompletedEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            summary_content_hash=summary_content_hash,
            overall_risk=overall_risk,
            n_findings=len(report_findings),
            n_findings_dropped_over_cap=n_findings_dropped_over_cap,
            files_examined=final_metrics.files_examined,
            files_traced_beyond_diff=final_metrics.files_traced_beyond_diff,
            llm_calls_made=final_metrics.llm_calls_made,
            total_input_tokens=final_metrics.total_input_tokens,
            total_output_tokens=final_metrics.total_output_tokens,
            total_cost_usd=final_metrics.total_cost_usd,
            wall_clock_seconds=final_metrics.wall_clock_seconds,
            pricing_version=PRICING_VERSION,
            # Preserve the review-start policy snapshot for replay.
            policy_version=state.triage_result.policy_version,
            synthesize_model=synthesize_model,
            profile_id=profile_id,
            reasoning_enabled=reasoning_enabled,
            profile_contract_digest=profile_contract_digest,
        )
    )

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

    return {"review_report": review_report}


__all__ = [
    "FindingForgeryDetectedError",
    "SynthesizeAggregationError",
    "synthesize",
]
