# Analyze node body per specs/2026-05-19-analyze-node.md §7.
"""Analyze node body — orchestration around the proof-boundary parser.

The node body is deliberately boring per the user-direction memo
(2026-05-20, post-§6 audit): assemble inputs, enforce triage gating,
call provider, hand raw response to the parser, lift parser rejection
payloads into audit events, return state deltas. Admission logic lives
in `analyze_parser.py`; the node body does NOT replicate it.

**Convention (matches `agent/nodes/triage.py`):** `async def analyze(...)`
with kwarg-bound deps wired at graph-construction time via
`functools.partial`. No `make_analyze_node` factory; the closure
happens at the partial-application call site. Spec §7 originally
described a `make_analyze_node(...)` factory shape; shipped follows
the triage convention. Documented in the spec's Actual Outcome.

**Provider-failure policy.** `LLMProviderError` subclasses propagate
out of `provider.complete()` without a try/except wrapper — same as
triage. If the provider fails on file N of M, files 0..N-1's audit
events have already landed (FileExaminationEvent, FindingEvent,
ProposalRejection events); the start `ReviewPhaseEvent` is dangling
without a matching end. That dangling-start signals "this pass was
interrupted" in audit replay. Broad `try/except` around the per-file
loop would silently mask transport failures as some made-up
skip/degrade outcome — the spec's emission-ordering invariant
(parser-decision counters from local bookkeeping, not re-read audit
stream) is what makes the partial-state landing auditable.

**Event ordering / counter source-of-truth.** Local accumulators
(`admitted_findings`, `proposal_rejections`, counters) are populated
during the per-file iteration from `ParserResult.counters` returned
by the parser. `AnalyzeCompletedEvent` at step 5 reads from these
accumulators — never from re-reading the audit stream. The shipped
`_enforce_proposal_accounting` validator backstops drift; the
producer-side accounting is required to be correct, not just
catch-on-construction.

**Commit-7 scope (locked).** This first landing handles two outcomes:

- `clean+full_llm` — file parses cleanly, has scope units in the
  changed regions, cost gate passes; full LLM call + parser invocation.
- `skipped+COST_BUDGET_EXHAUSTED` — file parses cleanly, has scope
  units, but cost gate's per-file ceiling OR remaining-budget check
  fails; no LLM call.

The other outcomes documented in spec §7 step 3a — `failed+degraded_llm`,
`degraded+degraded_llm`, `skipped+NO_REVIEWABLE_CONTEXT`,
`skipped+NO_CHANGED_SCOPE_UNITS` — raise `NotImplementedError` with
a stable message naming the deferred outcome. Subsequent commits
land them incrementally.

**Per-file context simplifications (commit-7 only, deferred to commit-8):**

- `included_scope_units` = all scope units from the parse result (no
  changed-region intersection). Over-broad but admission-correct.
- `query_match_id_set` = `frozenset()` (no registry queries fired).
  Every OBSERVED proposal will reject at producer admission;
  `JUDGED` is the V1 supported tier.
- Scope-unit context block + query-match-id list = empty strings (the
  prompt's structural placeholders fill cleanly but carry no per-
  scope-unit detail). The model receives only the file path + pass
  index + raw patch as user-prompt content.
- Diff hunks = `cf.patch or ""` (no scope-unit clipping; full patch
  content if available).

Each simplification has a TODO marker at its insertion site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from outrider.agent.nodes.analyze_parser import (
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    parse_analyze_response,
)
from outrider.ast_facts.models import SkipReason
from outrider.ast_facts.python_adapter import parse_python
from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingEvent,
    FindingProposalRejectedEvent,
    ReviewPhaseEvent,
)
from outrider.llm.base import LLMRequest
from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd
from outrider.policy.canonical import compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
from outrider.schemas import AnalysisRound
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from uuid import UUID

    from outrider.ast_facts.base import ImportPathResolver
    from outrider.audit.sinks import (
        AnalyzeEventSink,
        FileExaminationSink,
        PhaseEventSink,
    )
    from outrider.llm.base import LLMProvider, LLMResponse
    from outrider.schemas import ReviewFinding, ReviewState, TraceCandidate
    from outrider.schemas.pr_context import ChangedFile


# Spec §7 step 3d / FUP-044 V1 guard: one file can starve at most four
# others. Per-file ceiling = total_review_budget * 0.25. Richer fairness
# policy (iteration ordering, per-installation budgets) is FUP-044 V1.5
# scope.
PER_FILE_CAP_FRACTION: Final[float] = 0.25

# Default per-review token budget. Caller can override via the
# `total_review_budget_tokens` kwarg. The default is intentionally
# generous (200K tokens / review ≈ several Sonnet calls of bounded
# prompts); production wires a tighter value from settings.
DEFAULT_REVIEW_BUDGET_TOKENS: Final[int] = 200_000


def _estimate_tokens(text: str) -> int:
    """Rough token estimate for the cost gate. `len(text) // 4` is the
    canonical Anthropic-shaped heuristic. Tighter estimators (e.g.,
    tiktoken on Sonnet's BPE) are a measured optimization for later;
    the cost gate's job is to catch order-of-magnitude blowup, and
    `//4` is accurate enough for that. The first per-file landing
    refines this; commit-7 uses the heuristic."""
    return len(text) // 4


async def analyze(
    state: ReviewState,
    *,
    provider: LLMProvider,
    analyze_model: str,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    import_path_resolver: ImportPathResolver,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
) -> dict[str, object]:
    """Run one analyze pass over the triage-classified PR.

    Returns `{"analysis_rounds": [round], "trace_candidates": [...]}`
    for LangGraph's reducer to merge into state. Per
    `reducers-dedup-not-concat`, both fields use
    `append_with_dedup_by` with content-derived stable keys.

    Step order (failure-path-significant):
      1. Emit start phase event.
      2. Triage-gate filter over `state.pr_context.changed_files`.
      3. Per kept file: parse + outcome + cost gate +
         FileExaminationEvent + provider call + parser + lift
         rejection payloads + collect admitted findings + trace
         candidates.
      4. Build `AnalysisRound`.
      5. Emit `AnalyzeCompletedEvent` with aggregate counters.
      6. Emit end phase event.
      7. Return state delta.

    Counter source-of-truth: per-file local bookkeeping accumulators
    summed at step 5. NEVER re-read from the audit stream.
    """
    # V1 single-pass: trace ⇄ analyze loop is post-V1 work.
    pass_index = 0
    phase_id = str(uuid4())
    started_at = datetime.now(UTC)
    per_file_cap_tokens = int(total_review_budget_tokens * PER_FILE_CAP_FRACTION)

    # Step 1: start phase event. If this raises (audit infra outage),
    # the node fails before any work — no dangling start.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="start",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Local accumulators. These are the SINGLE source of truth for the
    # AnalyzeCompletedEvent counters at step 5. Reading them back from
    # the audit stream would couple counter correctness to emission
    # ordering and break the `_enforce_proposal_accounting` equation
    # under future concurrent-emit refactors.
    admitted_findings: list[ReviewFinding] = []
    trace_candidates: list[TraceCandidate] = []
    files_examined: list[str] = []
    files_skipped: list[str] = []
    n_proposals_seen = 0
    n_findings_emitted = 0
    n_proposals_rejected = 0
    n_responses_rejected = 0
    n_trace_candidates_emitted = 0
    n_llm_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0
    total_cost_usd = 0.0
    remaining_budget_tokens = total_review_budget_tokens

    # Step 2: triage-gate filter. SKIM/SKIP excluded by construction;
    # files absent from the tier map are treated as SKIP (defensive
    # against tier-map gaps; per spec §7 step 2). No FileExaminationEvent
    # fires for excluded files — they never enter the per-file
    # iteration scope.
    triage_result = state.triage_result
    for changed_file in state.pr_context.changed_files:
        tier = (
            triage_result.file_tiers.get(changed_file.path, ReviewTier.SKIP)
            if triage_result is not None
            else ReviewTier.SKIP
        )
        if tier not in (ReviewTier.DEEP, ReviewTier.STANDARD):
            continue

        # Step 3: per-file processing.
        file_outcome = await _process_one_file(
            changed_file=changed_file,
            review_id=state.review_id,
            installation_id=state.pr_context.installation_id,
            is_eval=state.is_eval,
            provider=provider,
            analyze_model=analyze_model,
            import_path_resolver=import_path_resolver,
            file_examination_sink=file_examination_sink,
            analyze_event_sink=analyze_event_sink,
            active_policy_version=active_policy_version,
            pass_index=pass_index,
            per_file_cap_tokens=per_file_cap_tokens,
            remaining_budget_tokens=remaining_budget_tokens,
        )

        if file_outcome.parser_result is not None:
            # LLM call was made; parser ran.
            n_llm_calls += 1
            n_proposals_seen += file_outcome.parser_result.counters.n_proposals_seen
            n_findings_emitted += file_outcome.parser_result.counters.n_findings_emitted
            n_proposals_rejected += file_outcome.parser_result.counters.n_proposals_rejected
            n_responses_rejected += file_outcome.parser_result.counters.n_responses_rejected
            n_trace_candidates_emitted += (
                file_outcome.parser_result.counters.n_trace_candidates_emitted
            )
            admitted_findings.extend(file_outcome.parser_result.admitted_findings)
            trace_candidates.extend(file_outcome.parser_result.trace_candidates)

        total_input_tokens += file_outcome.input_tokens
        total_output_tokens += file_outcome.output_tokens
        total_cached_tokens += file_outcome.cached_tokens
        total_cost_usd += file_outcome.cost_usd
        remaining_budget_tokens -= file_outcome.estimated_tokens

        if file_outcome.parse_status == "skipped":
            files_skipped.append(changed_file.path)
        else:
            files_examined.append(changed_file.path)

    ended_at = datetime.now(UTC)

    # Step 4: build AnalysisRound. `round_id` is content-derived from
    # pass_index + file lists + finding content_hashes per the canonical
    # recipe so re-emission of the same logical round produces the same
    # id (idempotent under checkpoint replay).
    round_id = compute_round_id(
        pass_index=pass_index,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        finding_content_hashes=tuple(f.content_hash for f in admitted_findings),
    )
    new_round = AnalysisRound(
        round_id=round_id,
        pass_index=pass_index,
        findings=tuple(admitted_findings),
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        started_at=started_at,
        ended_at=ended_at,
    )

    # Step 5: AnalyzeCompletedEvent. Counters from local accumulators
    # — the producer-side source of truth per spec §7 step 5.
    await analyze_event_sink.emit_analyze_completed(
        AnalyzeCompletedEvent(
            review_id=state.review_id,
            is_eval=state.is_eval,
            pass_index=pass_index,
            n_files_analyzed=len(files_examined),
            n_files_skipped=len(files_skipped),
            n_llm_calls=n_llm_calls,
            n_proposals_seen=n_proposals_seen,
            n_findings_emitted=n_findings_emitted,
            n_proposals_rejected=n_proposals_rejected,
            n_responses_rejected=n_responses_rejected,
            n_trace_candidates_emitted=n_trace_candidates_emitted,
            total_input_tokens=total_input_tokens,
            total_cached_tokens=total_cached_tokens,
            total_output_tokens=total_output_tokens,
            total_cost_usd=total_cost_usd,
            pricing_version=PRICING_VERSION,
            policy_version=active_policy_version,
            analyze_model=analyze_model,
        )
    )

    # Step 6: end phase event. Same phase_id as the start event.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="end",
            is_eval=state.is_eval,
            phase_key=None,
        )
    )

    # Step 7: state delta. list shape (not tuple) per canonical
    # docs/spec.md §7.1 — the `append_with_dedup_by` reducer expects
    # list-of-T.
    return {
        "analysis_rounds": [new_round],
        "trace_candidates": list(trace_candidates),
    }


class _FileOutcome:
    """Per-file processing result. Populated by `_process_one_file` and
    consumed by the main loop's accumulators."""

    __slots__ = (
        "cached_tokens",
        "cost_usd",
        "estimated_tokens",
        "input_tokens",
        "output_tokens",
        "parse_status",
        "parser_result",
    )

    def __init__(
        self,
        *,
        parse_status: str,
        parser_result: ParserResult | None,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int,
        cost_usd: float,
        estimated_tokens: int,
    ) -> None:
        self.parse_status = parse_status
        self.parser_result = parser_result
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cached_tokens = cached_tokens
        self.cost_usd = cost_usd
        self.estimated_tokens = estimated_tokens


async def _process_one_file(  # noqa: PLR0913 — explicit kwargs at the orchestration boundary
    *,
    changed_file: ChangedFile,
    review_id: UUID,
    installation_id: int,
    is_eval: bool,
    provider: LLMProvider,
    analyze_model: str,
    import_path_resolver: ImportPathResolver,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    active_policy_version: str,
    pass_index: int,
    per_file_cap_tokens: int,
    remaining_budget_tokens: int,
) -> _FileOutcome:
    """Process one triage-kept file through parse → cost gate → LLM
    call → parser → audit events. Returns a `_FileOutcome` carrying
    the per-file counters the main loop sums.

    Commit-7 scope: clean+full_llm and skipped+COST_BUDGET_EXHAUSTED
    only. Other outcomes raise `NotImplementedError` with a stable
    message naming the deferred outcome.
    """
    # Step 3a: parse + outcome determination.
    content = changed_file.content_head or changed_file.content_base
    if content is None:
        raise NotImplementedError(
            f"analyze: skipped+NO_REVIEWABLE_CONTEXT outcome not yet implemented "
            f"(file_path={changed_file.path!r} has neither content_head nor content_base)"
        )
    file_byte_length = len(content.encode("utf-8"))

    # parse_python returns a ParseResult; raises on adapter-level
    # failure but tolerates source-level errors via `has_error`.
    parse_result = parse_python(
        source=content.encode("utf-8"),
        file_path=changed_file.path,
        resolver=import_path_resolver,
    )
    # TODO(commit-8): tree-sitter `has_error` detection + degraded
    # outcomes (`failed+degraded_llm`, `degraded+degraded_llm`).
    # `has_error` is a per-scope-unit dict; the value (not the key
    # presence) signals whether that unit's parse tree carries an
    # ERROR node. Until commit-8 wires the changed-region
    # intersection, treat ANY has-error scope unit as degraded
    # (over-conservative; tightens in the next commit).
    if any(parse_result.has_error.values()):
        raise NotImplementedError(
            f"analyze: degraded+degraded_llm outcome not yet implemented "
            f"(has_error scope unit ids="
            f"{sorted(uid for uid, has in parse_result.has_error.items() if has)})"
        )

    # TODO(commit-8): NO_CHANGED_SCOPE_UNITS detection. If the changed
    # regions don't intersect any scope unit, the outcome is
    # skipped+NO_CHANGED_SCOPE_UNITS (no LLM call). Commit-7 treats
    # every clean-parsed file with scope units as full_llm; if there
    # are no scope units at all, raise.
    if not parse_result.scope_units:
        raise NotImplementedError(
            f"analyze: skipped+NO_CHANGED_SCOPE_UNITS outcome not yet implemented "
            f"(parse_result.scope_units is empty for {changed_file.path!r})"
        )

    # TODO(commit-8): changed-region intersection. Commit-7 passes ALL
    # scope units as included; the prompt is over-broad but admission-
    # correct.
    included_scope_units = tuple(parse_result.scope_units)

    # TODO(commit-8): registry-query firing. Commit-7 passes an empty
    # set; OBSERVED proposals will reject at producer admission. V1
    # supported tier is JUDGED until commit-8 wires the registry.
    query_match_id_set: frozenset[str] = frozenset()

    # Step 3c: build prompt context. Commit-7 simplification: empty
    # scope-unit context block + empty query-match-id list; the patch
    # text fills the diff_hunks slot. TODO(commit-8): per-scope-unit
    # context assembly + scope-unit-clipped diff hunks.
    parts = analyze_prompt.render(
        file_path=changed_file.path,
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=changed_file.patch or "",
        pass_index=pass_index,
    )

    # Step 3d: cost gate.
    estimated_tokens = (
        _estimate_tokens(parts.system_prompt)
        + _estimate_tokens(parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
    )
    cost_exhausted = (
        estimated_tokens > per_file_cap_tokens or estimated_tokens > remaining_budget_tokens
    )

    # Step 3e: SINGLE FileExaminationEvent emission point. Outcome is
    # finalized at this step; no event fires before here, no event
    # fires after.
    if cost_exhausted:
        await file_examination_sink.emit_file_examination(
            FileExaminationEvent(
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                examination_type="analyze",
                node_id="analyze",
                parse_status="skipped",
                skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
            )
        )
        return _FileOutcome(
            parse_status="skipped",
            parser_result=None,
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            cost_usd=0.0,
            estimated_tokens=0,
        )

    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            examination_type="analyze",
            node_id="analyze",
            parse_status="clean",
            skip_reason=None,
        )
    )

    # Step 3f: LLM call + response parse.
    # Build context_summary per spec §7: one ContextManifestEntry per
    # included scope unit. Commit-7 stamps every entry with
    # `inclusion_reason="changed_scope"` because the changed-region
    # intersection isn't implemented yet (TODO commit-8). Once the
    # intersection lands, units outside the changed regions become
    # `"same_file_context"`.
    context_summary = tuple(
        ContextManifestEntry(
            file_path=changed_file.path,
            scope_unit_name=su.qualified_name or su.name,
            line_start=su.line_start,
            line_end=su.line_end,
            inclusion_reason="changed_scope",
        )
        for su in included_scope_units
    )
    request = LLMRequest(
        model=analyze_model,
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        max_tokens=analyze_prompt.MAX_TOKENS,
        temperature=analyze_prompt.TEMPERATURE,
        review_id=review_id,
        node_id="analyze",
        is_eval=is_eval,
        prompt_template_version=analyze_prompt.VERSION,
        degraded_mode=False,
        context_summary=context_summary,
    )
    # Provider failure (LLMProviderError subclasses) propagates per the
    # triage convention. No try/except — the dangling start phase event
    # is the audit signal for "this pass was interrupted."
    response: LLMResponse = await provider.complete(request)

    # Cost compute via the canonical wrapper. `total_cost_usd` on
    # AnalyzeCompletedEvent matches this float; the four-term sum is
    # the same recipe LLMCallEvent uses internally.
    cost_decimal = compute_cost_usd(
        analyze_model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
    )
    cost_usd = float(cost_decimal)

    parser_result = parse_analyze_response(
        response.text,
        review_id=review_id,
        installation_id=installation_id,
        file_path=changed_file.path,
        file_content=content,
        file_byte_length=file_byte_length,
        included_scope_units=included_scope_units,
        query_match_id_set=query_match_id_set,
        degraded_mode=False,
        active_policy_version=active_policy_version,
    )

    # Lift parser rejection payloads into audit events.
    for proposal_rej in parser_result.proposal_rejections:
        await analyze_event_sink.emit_finding_proposal_rejected(
            _lift_proposal_rejection(proposal_rej, review_id=review_id, is_eval=is_eval)
        )
    if parser_result.response_rejection is not None:
        await analyze_event_sink.emit_analyze_response_rejected(
            _lift_response_rejection(
                parser_result.response_rejection,
                review_id=review_id,
                is_eval=is_eval,
            )
        )

    # Emit one FindingEvent per admitted finding.
    for finding in parser_result.admitted_findings:
        await analyze_event_sink.emit_finding(_lift_admitted_finding(finding, is_eval=is_eval))

    return _FileOutcome(
        parse_status="clean",
        parser_result=parser_result,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cached_tokens=response.cache_read_tokens + response.cache_write_tokens,
        cost_usd=cost_usd,
        estimated_tokens=estimated_tokens,
    )


def _lift_proposal_rejection(
    rej: ProposalRejection,
    *,
    review_id: UUID,
    is_eval: bool,
) -> FindingProposalRejectedEvent:
    """Lift a parser-side `ProposalRejection` payload into a
    `FindingProposalRejectedEvent`. The parser produced the content
    fields; the node body adds the audit-context fields (`review_id`,
    `is_eval`) here. Other audit-context fields (`event_id`,
    `timestamp`, `sequence_number`, `node_id`, `event_type`) populate
    via the event's default factories / Literal defaults."""
    return FindingProposalRejectedEvent(
        review_id=review_id,
        is_eval=is_eval,
        file_path=rej.file_path,
        proposal_hash=rej.proposal_hash,
        claimed_evidence_tier=rej.claimed_evidence_tier,
        claimed_finding_type_hash=rej.claimed_finding_type_hash,
        claimed_finding_type_len=rej.claimed_finding_type_len,
        rejection_reason=rej.rejection_reason,
        rejection_detail=rej.rejection_detail,
    )


def _lift_response_rejection(
    rej: ResponseRejection,
    *,
    review_id: UUID,
    is_eval: bool,
) -> AnalyzeResponseRejectedEvent:
    """Lift a parser-side `ResponseRejection` into an
    `AnalyzeResponseRejectedEvent`. Same audit-context add as
    `_lift_proposal_rejection`."""
    return AnalyzeResponseRejectedEvent(
        review_id=review_id,
        is_eval=is_eval,
        file_path=rej.file_path,
        response_hash=rej.response_hash,
        rejection_reason=rej.rejection_reason,
        rejection_detail=rej.rejection_detail,
    )


def _lift_admitted_finding(
    finding: ReviewFinding,
    *,
    is_eval: bool,
) -> FindingEvent:
    """Lift an admitted `ReviewFinding` to a `FindingEvent` for audit.
    The finding carries every load-bearing field; the event mirrors
    them plus the audit-context default factories."""
    return FindingEvent(
        review_id=finding.review_id,
        is_eval=is_eval,
        finding_id=finding.finding_id,
        finding_type=finding.finding_type,
        severity=finding.severity,
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        dimension=finding.dimension,
        finding_content_hash=finding.content_hash,
        evidence_tier=finding.evidence_tier,
        query_match_id=finding.query_match_id,
        trace_path=finding.trace_path,
        policy_version=finding.policy_version,
    )


__all__ = [
    "DEFAULT_REVIEW_BUDGET_TOKENS",
    "PER_FILE_CAP_FRACTION",
    "analyze",
]
