# Synthesize-node body per specs/2026-05-28-synthesize-node.md
"""Synthesize node — aggregate findings into a `ReviewReport`.

Per spec gate #7 (fail-loud on cross-round divergence): when two
findings share a `content_hash` across analysis rounds but disagree
on EITHER `severity` OR `policy_version`, the
`severity-set-by-policy` + `severity-policy-versioned-for-replay`
invariants are violated by construction (the hash is keyed over
`finding_type` + the review runs under one triage-anchored policy
snapshot, so same content_hash → same severity AND same
policy_version must hold). Divergence on either axis is corruption,
not variance. Synthesize emits an
`AnomalyRuleName.CROSS_ROUND_SEVERITY_DIVERGENCE` anomaly via the
injected `AnomalySink` THEN raises `SynthesizeAggregationError`. The
anomaly emit is best-effort observability shadow; the raise is the
authoritative signal regardless of emit outcome (per
`_detect_and_report_divergence` docstring contract). When emit
succeeds, ops sees the corruption signal
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

import contextlib
import hashlib
import logging
import time
from typing import TYPE_CHECKING

from outrider.anomaly import AnomalyRuleName, AnomalySeverity
from outrider.audit.events import ReviewPhaseEvent, SynthesizeCompletedEvent
from outrider.llm.base import LLMRequest
from outrider.llm.parsing import strip_outer_json_fence
from outrider.llm.pricing import PRICING_VERSION
from outrider.policy.canonical import compute_phase_id

# ACTIVE_POLICY_VERSION used only in docstring references for context;
# the runtime snapshot anchor is `state.triage_result.policy_version`.
from outrider.prompts import synthesize as synthesize_prompt
from outrider.schemas.review_report import ReviewMetrics, ReviewReport

if TYPE_CHECKING:
    from outrider.anomaly import AnomalySink
    from outrider.audit.sinks import PhaseEventSink, SynthesizeEventSink
    from outrider.llm.base import LLMProvider
    from outrider.policy.severity import FindingSeverity
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_state import ReviewState


class FindingForgeryDetectedError(RuntimeError):
    """Raised when synthesize detects a forge-class invariant violation
    at entry — distinct from cross-round severity divergence.

    Two surfaces fire this:

    - A finding's `policy_version` diverges from the per-review
      triage snapshot (`state.triage_result.policy_version`, captured
      at triage entry by the Rule (d) gate). The trust root is the
      snapshot, NOT the live `ACTIVE_POLICY_VERSION` — replay paths
      legitimately carry historical values that match a historical
      snapshot. `ReviewFinding._enforce_severity_matches_policy`
      short-circuits when `policy_version != ACTIVE_POLICY_VERSION`,
      so a finding whose version diverges from the snapshot would
      bypass severity validation and survive into the audit row +
      HITL partition. Synthesize rejects at entry.

    - A finding's `original_severity` is not None at synthesize entry.
      `original_severity` is set only by HITL after a reviewer
      override; finding it set BEFORE HITL means the producer
      forged the override triplet to bypass the gated set.

    Same operational handling as `SynthesizeAggregationError`: the
    review parks, ops triages the forge attempt as a corruption signal.
    A future enhancement may emit an anomaly here too (separate
    rule_name) for surfacing in the anomaly queue; V1 fail-loud is
    sufficient.
    """


class SynthesizeAggregationError(RuntimeError):
    """Raised when cross-round divergence is detected on EITHER axis
    (severity OR policy_version) for the same finding `content_hash`.

    `compute_finding_content_hash` is keyed over `(file_path,
    line_start, line_end, finding_type)`, and
    `ReviewFinding._verify_baseline_severity` requires severity =
    SEVERITY_POLICY[finding_type]. Same content_hash within a single
    review (single policy_version snapshot) MUST have identical
    severity by construction — divergence on EITHER axis indicates
    corruption (validator bypass, hash-recipe drift, mid-review
    policy-version change), NOT model variance.

    Carries the diverging content_hash + the severity-set + the
    policy_version-set + round indices. Anomaly emission is
    best-effort observability per `_detect_and_report_divergence`'s
    docstring contract; when emit fails this exception is the only
    diagnostic, so all axes worth investigating travel on the
    exception payload.
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
    """SHA-256 hex of the RAW LLM response text (UTF-8 bytes).

    Identity check for retention-conditional replay per pre-spec gate
    #6 option (c). The hash MUST bind to the same canonical text the
    LLM provider persists to `llm_call_content.completion` (raw
    `response.text` per `audit/persister.py::_persist_llm_call_event`),
    NOT the post-`strip_outer_json_fence` display text. Within the
    LLM-content TTL window, an audit reader recomputes this hash over
    the stored `completion` row to prove identity; if the inputs
    differ (e.g., hash over stripped, completion stores raw), the
    binding breaks the moment Anthropic wraps a response in a fence.

    The displayed summary on `ReviewReport.summary` IS the
    `strip_outer_json_fence(response.text).strip()` form — clean for
    consumption. The hash is the audit-event identity proof and
    intentionally diverges from the display.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _enforce_synthesize_input_invariants(state: ReviewState) -> None:
    """Reject forged findings at synthesize entry.

    See DECISIONS.md#028-per-review-policy-version-snapshot-anchor-on-triageresult
    for the trust-root rationale (triage-captured snapshot, producer-side
    Rule (d) gate, V1 scope limitation).

    `ReviewFinding._enforce_severity_matches_policy` short-circuits when
    `policy_version != ACTIVE_POLICY_VERSION` — the historical replay
    path. An attacker (or a buggy upstream) can smuggle in a finding
    with arbitrary severity by setting `policy_version` to a
    non-active version string. Synthesize compares
    every finding's `policy_version` to the triage-captured snapshot
    (`state.triage_result.policy_version`, set upstream of analyze) and
    emits the SAME snapshot as `SynthesizeCompletedEvent.policy_version`
    so the audit row records the snapshot under which findings were
    classified — survives mid-deploy bumps and replay correctly.
    Reject any divergence at node entry.

    `ReviewFinding.original_severity` is the pre-override baseline used
    by HITL `_resolve_effective_severity`. At synthesize entry HITL has
    NOT run yet — every finding must have `original_severity is None`.
    A finding with `original_severity != None` indicates the producer
    forged a HITL-override triplet to bypass the gated set. Reject the
    original_severity smuggle at node entry.

    Both surfaces raise `FindingForgeryDetectedError`; ops triage
    routes the same way as cross-round severity divergence.
    """
    # Snapshot anchor: `state.triage_result.policy_version`. Triage
    # runs FIRST in the graph and captures `ACTIVE_POLICY_VERSION` at
    # its own node-entry time; that capture is upstream of any
    # attacker-controllable analyze output. Comparing each finding's
    # `policy_version` against the triage snapshot:
    #   - Catches single-finding-poisoning DoS (attacker plants ONE
    #     forged finding with a different version; trusted triage
    #     snapshot makes it visible).
    #   - Survives mid-deploy hot-reload bumps (legitimate findings
    #     and triage all share the version captured at review START,
    #     regardless of where the live `ACTIVE_POLICY_VERSION`
    #     constant has moved by synthesize time).
    #   - Catches mid-batch divergence (mixed legit + forged batches).
    # The residual gap (full triage + analyze + state compromise with
    # coherent fake triage_result snapshot) requires multiple-node
    # graph compromise; the persister-side known-version check
    # (matching `severity_policies` row) is the next defense layer
    # and is tracked as a future snapshot-fortification FUP.
    if state.triage_result is None:
        # Triage MUST have run before synthesize per the canonical
        # graph topology. The orchestration error is itself a
        # corruption signal (graph routed past triage somehow).
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
    """Walk analysis_rounds, group by content_hash, detect severity/
    policy_version divergence. On first divergence: best-effort emit
    anomaly, then UNCONDITIONALLY raise SynthesizeAggregationError.

    Contract — emit-vs-raise dependency direction:
      * Anomaly emission is a best-effort observability shadow.
      * The SynthesizeAggregationError raise is the authoritative
        signal. It propagates regardless of emit outcome.
      * A transient anomaly-DB outage MUST NOT delay or mask the
        authoritative raise path. The wrong direction — coupling
        corruption detection to anomaly-table availability — would
        let an anomaly-sink outage compromise the dependency direction
        the raise depends on. Defended by the broad `except Exception`
        around emit + nested `contextlib.suppress` around the logger;
        both preserve the unconditional raise on the way out.

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
        policy_versions = {entry[1].policy_version for entry in entries}
        # Cross-round divergence detection: severity OR policy_version
        # mismatch within a content_hash group both indicate corruption
        # per `severity-set-by-policy` + `severity-policy-versioned-
        # for-replay`. Either axis triggers the same
        # CROSS_ROUND_SEVERITY_DIVERGENCE anomaly + fail-loud raise
        # because the recovery action is identical: stop the review,
        # investigate the upstream policy-resolution layer.
        #
        # Defense-in-depth: the policy_version axis is structurally
        # unreachable on the canonical path because
        # `_enforce_synthesize_input_invariants` (step 3) already
        # raises FindingForgeryDetectedError on any
        # `finding.policy_version != state.triage_result.policy_version`.
        # The check here catches future producer bypasses (direct
        # ReviewState construction in tests, alternate dispatchers,
        # ad-hoc tooling) that skip the upstream gate — strict
        # subset/superset relationship: any path triggering this
        # branch on policy_version axis is a producer-side bug.
        # **DO NOT remove the policy_versions check even when the
        # upstream gate is the only canonical-path enforcer.** The
        # subset/superset property makes this branch a structural
        # safety net for non-canonical producers; deleting it would
        # silently admit forged-policy_version findings on any future
        # bypass path. Pinned by
        # `test_policy_version_axis_divergence_emits_anomaly_and_raises`.
        if len(severities) > 1 or len(policy_versions) > 1:
            severity_tuple = tuple(sorted({e[1].severity for e in entries}, key=lambda s: s.value))
            policy_version_tuple = tuple(sorted(policy_versions))
            round_indices_tuple = tuple(sorted({e[0] for e in entries}))
            # See docstring: anomaly emit is best-effort; the raise below
            # is authoritative (broad `except Exception` preserves the
            # dependency direction on emit/transport failure).
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
                # See docstring: emit is best-effort; the raise below is
                # authoritative. Nested contextlib.suppress defends
                # against a broken-logger configuration raising during
                # `logging.exception()` — preserving the unconditional
                # divergence raise even when observability is degraded.
                with contextlib.suppress(Exception):
                    logging.getLogger(__name__).exception(
                        "synthesize_anomaly_emit_failed_during_divergence",
                        extra={
                            "review_id": str(state.review_id),
                            "content_hash": content_hash,
                            "emit_exception_type": type(emit_exc).__name__,
                        },
                    )
                # Fall through to the SynthesizeAggregationError raise.
            raise SynthesizeAggregationError(
                content_hash=content_hash,
                severities=severity_tuple,
                policy_versions=policy_version_tuple,
                round_indices=round_indices_tuple,
            )
        # No divergence — pick deterministic representative (lowest
        # round_index first; finding_id sort as final tie-break).
        entries.sort(key=lambda pair: (pair[0], str(pair[1].finding_id)))
        kept[content_hash] = entries[0][1]

    return kept


def _compute_files_traced_beyond_diff(state: ReviewState) -> int:
    """Count of distinct file paths that trace REFERENCED outside the
    original PR diff — `(target_file ∪ resolved_candidate_paths across
    all TraceDecision rows) ∪ (trace_fetched_files.path)` minus
    `pr_context.changed_files` paths.

    See DECISIONS.md#030-reviewreport-tuple-not-list-findings-field
    for the canonical-record anchor on the union recipe semantic.
    Executable contract pin:
    `tests/unit/test_synthesize_files_traced_metric.py`.

    "Beyond diff" here means "outside the PR's changed-files set" —
    NOT "Phase-2-fetched" specifically. Per the trace-node spec,
    paths can land in three states relative to the metric:

    - **target_file on a resolved decision** — the canonical Phase-2
      target. Counted.
    - **resolved_candidate_paths on an ambiguous decision** — multiple
      candidates were resolved by `ast_facts/` but trace declined to
      auto-pick a Phase-2 fetch target; no fetch occurred, but those
      paths WERE referenced by trace's resolution work. Counted.
    - **trace_fetched_files.path** — files actually fetched by trace
      Phase-2. On the canonical path this is a subset of
      target_files, but reducer-replay or trace-side filtering paths
      can land entries here without a matching `target_file` on the
      decision (per `trace_fetched_files`' state-vs-event divergence
      notes in `schemas/review_state.py`). Counted via union, so
      either side surfacing the path admits it once.

    Unresolved decisions contribute nothing — their
    `resolved_candidate_paths` is empty by schema (see
    `schemas/trace_decision.py`: `unresolved → len(...) == 0`).

    Per CodeRabbit 2026-05-28 catch (narrowed by Codex): the prior
    implementation counted only `target_file` on resolved decisions,
    missing the ambiguous-resolution + fetched-without-decision-target
    paths.
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
) -> ReviewMetrics:
    """Build ReviewMetrics from state + wall-clock measurement.

    V1: LLM-aggregate metrics (llm_calls_made, total_*_tokens,
    total_cost_usd) ship as `None` — honest "unknown" semantics rather
    than false zeros. The deterministic derivation requires querying
    `audit_events` for this review_id and summing `LLMCallEvent` rows;
    that audit-query helper is a FUP. Dashboard reads audit truth
    (joining LLMCallEvent by review_id), not these denormalized fields,
    so V1 ships nullable; downstream consumers that need the aggregate
    today must query audit directly.

    files_examined, files_traced_beyond_diff, and wall_clock_seconds
    are computed deterministically and ship as real values.
    """
    files_examined: set[str] = set()
    for analysis_round in state.analysis_rounds:
        files_examined.update(analysis_round.files_examined)

    return ReviewMetrics(
        files_examined=len(files_examined),
        files_traced_beyond_diff=_compute_files_traced_beyond_diff(state),
        # V1 placeholders — None semantics, not zero. FUP for
        # audit-query-derived aggregates. See ReviewMetrics docstring.
        llm_calls_made=None,
        total_input_tokens=None,
        total_output_tokens=None,
        total_cost_usd=None,
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

    # Step 3a: forge-class invariants. Reject findings carrying
    # non-active policy_version OR pre-set original_severity BEFORE
    # the divergence loop sees them — both are smuggle paths that
    # would otherwise survive into the audit row + HITL partition.
    _enforce_synthesize_input_invariants(state)
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

    # Step 6: build LLMRequest for the Sonnet summary call.
    # (Step 5's metrics computation is interleaved: a pre-call snapshot
    # below feeds the prompt; the final snapshot at step 10 captures the
    # post-call wall-clock for the audit row.)
    # `_enforce_synthesize_input_invariants` (step 3) already raised
    # `FindingForgeryDetectedError` if `triage_result` was None. The
    # runtime check below is structurally dead — kept for type
    # narrowing (mypy can't prove the invariant) and as defense in
    # depth if step ordering is ever broken.
    if state.triage_result is None:
        raise FindingForgeryDetectedError(
            "synthesize: triage_result missing past invariant gate "
            "(_enforce_synthesize_input_invariants should have raised "
            "at step 3 — step ordering is broken)"
        )
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

    # Step 9: compute the canonical summary content hash over the RAW
    # `response.text` (matches what the LLM provider persists into
    # `llm_call_content.completion` — see _compute_summary_content_hash
    # docstring). Hashing the stripped text would break identity-binding
    # the moment Anthropic wraps a response in ```json``` fences.
    summary_content_hash = _compute_summary_content_hash(response.text)

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
            # Use the triage snapshot (captured at review start) for
            # replay-correctness, NOT the live `ACTIVE_POLICY_VERSION`.
            # state.triage_result is guaranteed non-None at this point
            # (checked at `_enforce_synthesize_input_invariants` entry).
            policy_version=state.triage_result.policy_version,
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
    "FindingForgeryDetectedError",
    "SynthesizeAggregationError",
    "synthesize",
]
