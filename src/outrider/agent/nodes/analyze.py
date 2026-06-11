# See DECISIONS.md#018, #025 (proposal_hash threaded through admitted
# and rejected lift sites per #025 point 1).
# See also DECISIONS.md#024-trace-candidates-are-dotted-python-import-strings-v1
# (admission path; same-file inline handling).
"""Analyze node body — orchestration around the proof-boundary parser.

Assembles inputs, enforces triage gating, calls the provider, hands the
raw response to `analyze_parser.parse_analyze_response`, lifts parser
rejection payloads into audit events, returns state deltas. Admission
logic lives in `analyze_parser.py`; this module does NOT replicate it.

Wiring: `async def analyze(...)` with kwarg-bound deps; `build_graph`
binds them via `functools.partial` (same convention as triage).

**Provider-failure policy.** `LLMProviderError` propagates without a
try/except wrapper. On mid-loop failure, files 0..N-1's audit events
have already landed and the start `ReviewPhaseEvent` is dangling
without a matching end — that's the audit signal for "pass
interrupted." A blanket try/except would mask transport failures as
fake skip outcomes.

**Counter source-of-truth.** Local accumulators (populated from
`ParserResult.counters`) feed `AnalyzeCompletedEvent` — never re-read
from the audit stream. `_enforce_proposal_accounting` validator
backstops drift; producer-side correctness is the contract.

**File outcomes** (spec §7 step 3a):

- `clean+full_llm` — clean parse, scope units intersect changed
  regions, no `has_error` in those units, cost gate passes.
- `degraded+degraded_llm` — clean parse but either tree-sitter `has_error`
  nodes intersect a changed scope unit (`degradation_reason=
  "tree_has_error_in_changed_regions"`) OR a changed addable line
  intersects a tree error with no recovered scope
  (`degradation_reason="tree_has_error_no_scope"`, DECISIONS#033). Parser
  admits JUDGED only, gated on `span_within_file` AND
  `span_within_degraded_context` (FUP-138).
- `skipped+NO_REVIEWABLE_CONTEXT` — both `content_head` and
  `content_base` are None (V1-unreachable: `ChangedFile.enforce_status_invariants`
  guarantees every valid status has ≥1 content side) OR parse failure
  with no added text (V1-unreachable per the `failed+degraded_llm` note
  below). Branch kept as a structural slot for the future
  schema-relaxation / raw-bytes paths. No LLM call.
- `skipped+NO_CHANGED_SCOPE_UNITS` — clean parse but no scope unit
  intersects the changed regions, OR clean parse with no patch.
- `skipped+COST_BUDGET_EXHAUSTED` — cost gate fired before the LLM
  call.
- `skipped+UNSUPPORTED_LANGUAGE` — non-Python file path; the V1
  analyze adapter only handles `.py` / `.pyi`. Capability-scoped per
  `DECISIONS.md#018` Amended 2026-05-21 — the value names "today's
  analyze cannot review this," not "Outrider forever cannot."
- `skipped+ALL_SCOPES_TRIVIAL` — enforcing-mode trivial-scope filter:
  every admitted scope classified ordinary-comment-only, so the LLM
  call is skipped (the shadow default never produces this). Fires
  after the cost gate per `DECISIONS.md#018` Amended 2026-06-11.

**V1 unreachable: `failed+degraded_llm`.** Spec §7 step 3a names this
outcome; the analyze code path is wired to handle it, but in V1 the
trigger cannot fire. `parse_python` only produces `parser_outcome=
"failed"` on a UTF-8 strict-decode failure ([ast_facts/python_adapter.py]
step 2). Two upstream gates make that branch dead in V1: (a) intake's
`_classify_or_reserve_decode` rejects invalid-UTF-8 bytes with
`SkipReason.OVERSIZED` BEFORE analyze sees the file; (b) analyze
receives content as `str` from `ChangedFile` and re-encodes via
`content.encode("utf-8")` — a Python `str` round-trips to valid UTF-8
by definition. The `failed`/`parse_failed` paths remain in code as
structural slots so adding a raw-bytes intake → state path (FUP-053)
doesn't require re-introducing them.

Parser-stage skips (`OVERSIZED`, `VENDORED`, `GENERATED_FILENAME`,
`MINIFIED`, `GENERATED_BANNER`) pass through with the parser's
`skip_reason` preserved on `FileExaminationEvent`.

**Changed-region intersection.** A scope unit is "included" iff BOTH
`coordinates.scope_unit_has_added_lines` AND
`coordinates.scope_unit_diff_hunks` return non-empty. Context-only
intersections don't include the unit. Deletion-only edits inside an
otherwise-unchanged function currently route to
`NO_CHANGED_SCOPE_UNITS` — V1 limitation tracked as FUP-050.

**Registry-query firing.** For clean+full_llm, every id in
`queries.registry.REGISTERED_QUERY_IDS` is fired against the file
content; the matching subset becomes `query_match_id_set` passed to
the parser. OBSERVED claims with an id outside the set reject.

**Token estimation.** `_estimate_tokens` counts UTF-8 bytes with
ceiling division (`_BYTES_PER_TOKEN = 3`). Conservative-up for code-
heavy / multi-byte content; over-estimates the budget rather than
under-estimates. A tokenizer-grade estimate is FUP-049 scope.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from outrider.agent.nodes.analyze_parser import (
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    parse_analyze_response,
)
from outrider.agent.nodes.degradation import (
    _DegradationReason,
    _ParseStatus,
    decide_degradation,
)
from outrider.ast_facts.models import SkipReason, TrivialityReason
from outrider.ast_facts.python_adapter import parse_python
from outrider.ast_facts.triviality import (
    TRIVIAL_FILTER_VERSION,
    build_triviality_context,
    classify_scope_triviality,
)
from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingProposalRejectedEvent,
    ReviewPhaseEvent,
    ScopeExclusionEntry,
    ScopeExclusionEvent,
)
from outrider.coordinates import (
    added_line_byte_ranges,
    bound_diff_hunks_text,
    changed_line_spans,
    extract_scope_unit_body,
    lookup_patched_file,
    patched_file_has_removed_lines,
)
from outrider.llm.base import LLMRequest
from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd
from outrider.policy.canonical import compute_phase_id, compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
from outrider.prompts import safe_code_fence
from outrider.queries import registry as query_registry
from outrider.schemas import AnalysisRound
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from uuid import UUID

    from unidiff import PatchedFile

    from outrider.ast_facts.base import ImportPathResolver
    from outrider.ast_facts.models import ParseResult, ScopeUnit
    from outrider.audit.sinks import (
        AnalyzeEventSink,
        FileExaminationSink,
        PhaseEventSink,
    )
    from outrider.llm.base import LLMProvider, LLMResponse
    from outrider.schemas import (
        ReviewFinding,
        ReviewState,
        TraceCandidate,
        TraceFetchedFile,
    )
    from outrider.schemas.pr_context import ChangedFile


# One file can starve at most `1 / PER_FILE_CAP_FRACTION` others on the
# review-wide budget; richer fairness (iteration ordering, per-installation
# budgets) is FUP-044.
PER_FILE_CAP_FRACTION: Final[float] = 0.25

# Default per-review token budget; production wires a tighter value
# from settings.
DEFAULT_REVIEW_BUDGET_TOKENS: Final[int] = 200_000

# Absolute ceiling on the per-file pre-flight token estimate, applied
# alongside `PER_FILE_CAP_FRACTION * budget`. Decouples the cap from
# caller-configurable budget — a "monorepo PR" knob can't lift the
# per-file cap into call-overflow territory.
MAX_PER_FILE_TOKENS_ABSOLUTE: Final[int] = 60_000

# Bytes-per-token divisor for `_estimate_tokens`. Code-leaning (over-
# estimates vs Anthropic's prose 1:4 heuristic); the cost gate fails
# safer. Tokenizer-grade replacement is FUP-049.
_BYTES_PER_TOKEN: Final[int] = 3

# Degraded-context bounds per spec §7 step 3c: ≤100 unidiff Line objects
# AND ≤8192 chars. Either cap closes the gate.
_DEGRADED_HUNK_LINE_CAP: Final[int] = 100
_DEGRADED_HUNK_CHAR_CAP: Final[int] = 8192


def _estimate_tokens(text: str) -> int:
    """UTF-8 byte count with ceiling division by `_BYTES_PER_TOKEN`.

    Conservative-up: over-estimates rather than under-estimates so the
    cost gate fails safer. Codepoint-counting would under-count multi-
    byte sequences (a 3-byte CJK char → `1 // 3 == 0` tokens). The
    estimator's job is order-of-magnitude blowup detection, not a tight
    count; a tokenizer-grade estimate is FUP-049.
    """
    byte_len = len(text.encode("utf-8"))
    return (byte_len + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def _is_python_file(path: str) -> bool:
    """True iff `path` is a Python source file analyze can process.

    V1 ships only the Python adapter (`ast_facts/python_adapter.py`,
    `queries/python/*.scm`). `.py` and `.pyi` are the two file
    extensions tree-sitter Python parses meaningfully; everything else
    routes to a skip outcome. The check is path-based because intake
    does not populate `ChangedFile.language`. Future V1.5 multi-language
    adapters move this gate to a registry lookup; the same path check
    stays as the cheap pre-filter.
    """
    return path.endswith((".py", ".pyi"))


def _compute_per_file_cap(total_review_budget_tokens: int) -> int:
    """Min of fractional cap (`budget * PER_FILE_CAP_FRACTION`) and
    absolute cap (`MAX_PER_FILE_TOKENS_ABSOLUTE`). Budget ≤ 0 returns a
    non-positive cap, which gates every file to `COST_BUDGET_EXHAUSTED`
    — fail-closed kill switch for a misconfigured budget.

    The fractional ceiling bounds budget consumption per file
    (one file can starve at most `1/PER_FILE_CAP_FRACTION` others).
    The absolute ceiling stays independent of caller-configurable
    budget, preventing budget inflation from lifting the cap into
    call-overflow territory.
    """
    return min(
        int(total_review_budget_tokens * PER_FILE_CAP_FRACTION),
        MAX_PER_FILE_TOKENS_ABSOLUTE,
    )


def _round_ended_at(started_at: datetime, started_mono: float) -> datetime:
    """Derive an `AnalysisRound.ended_at` that is always >= `started_at` (FUP-141).

    `ended_at = started_at + (monotonic elapsed since the round started)`, NOT a
    second wall-clock read. `time.monotonic()` is non-decreasing by contract, so a
    backwards wall-clock jump (NTP step / VM resume / WSL2 skew) between round start
    and end can't make `ended_at < started_at` and trip the `AnalysisRound`
    ordering invariant mid-review. `started_at` itself stays wall-clock for the
    audit trail. `max(0.0, …)` is belt-and-suspenders against a non-monotonic clock.
    """
    return started_at + timedelta(seconds=max(0.0, time.monotonic() - started_mono))


def _model_for_tier(tier: ReviewTier, *, analyze_model: str, standard_analyze_model: str) -> str:
    """The analyze model for a pass-0 file by its triage tier: STANDARD →
    `standard_analyze_model` (the cost lever); everything else → `analyze_model`. Only
    DEEP/STANDARD reach analyze (SKIM/SKIP are filtered upstream), and trace-fetched
    files (pass 1) have no tier and call `analyze_model` directly. One place for the
    choice keeps the call site literal-free (`model-strings-from-config-not-hardcoded`).
    See `specs/2026-06-08-analyze-tiered-model-routing.md`.
    """
    return standard_analyze_model if tier is ReviewTier.STANDARD else analyze_model


async def analyze(
    state: ReviewState,
    *,
    provider: LLMProvider,
    analyze_model: str,
    standard_analyze_model: str,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    import_path_resolver: ImportPathResolver,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
    trivial_scope_filter_enabled: bool = False,
) -> dict[str, object]:
    """Run one analyze pass over the triage-classified PR.

    Returns `{"analysis_rounds": [round], "trace_candidates": [...]}`
    for LangGraph's reducer to merge into state. Per
    `reducers-dedup-not-concat`, both fields use
    `append_with_dedup_by` with content-derived stable keys.

    Step order (failure-path-significant):
      1. Emit start phase event.
      2. Triage-gate filter over `state.pr_context.changed_files`.
      3. Per kept file: parse + outcome + cost gate + trivial-scope
         classification (ScopeExclusionEvent, pass 0 only) +
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
    # `pass_index` is derived from `state.analysis_rounds`: pass 0 = no
    # rounds merged yet (the first analyze pass); pass 1 = one round
    # merged (post-trace re-entry per the M8 loop). The round_id reducer
    # dedup includes pass_index in its content-derived hash, so deriving
    # the index from state guarantees distinct round_ids across the two
    # passes (a hardcoded `pass_index = 0` would collide under the
    # reducer + silently drop the second pass). The depth-2 ceiling is
    # enforced at `agent/graph.py::_trace_router` via `MAX_ANALYSIS_ROUNDS`.
    pass_index = len(state.analysis_rounds)
    # Per `compute_phase_id`'s contract, `attempt_key` is derived from
    # `pass_index` BEFORE the round is appended — same pre-merge state on
    # replay produces the same key, so the PhaseEventSink idempotency
    # collapses re-emissions to one row.
    phase_id = compute_phase_id(
        review_id=str(state.review_id),
        node_id="analyze",
        attempt_key=f"analyze-pass-{pass_index}",
    )
    started_at = datetime.now(UTC)
    # Monotonic anchor for the round DURATION (FUP-141): `ended_at` is derived
    # from this rather than a second wall-clock read, so a backwards clock jump
    # (NTP step / VM resume / WSL2 skew) between start and end can't make
    # `ended_at < started_at` and trip the `AnalysisRound` invariant mid-review.
    # `started_at` stays wall-clock for the audit trail. Precondition: this mark
    # and the matching end read stay in one process (`monotonic()` is per-process;
    # a V1.5 parallel-analyze fan-out must re-anchor per worker, not hoist this).
    started_mono = time.monotonic()
    per_file_cap_tokens = _compute_per_file_cap(total_review_budget_tokens)

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

    # Local accumulators — single source of truth for AnalyzeCompletedEvent
    # counters. Re-reading from the audit stream would couple counter
    # correctness to emission ordering and break the proposal-accounting
    # equation under concurrent-emit refactors.
    admitted_findings: list[ReviewFinding] = []
    # Dedupe-by-(content_hash, proposal_hash) tracked alongside
    # `admitted_findings` because `AnalysisRound` enforces uniqueness
    # on both. Pass-1 fan-out (one iteration per source-finding × target
    # file) can legitimately produce identical logical findings from
    # different source-finding contexts (same file content → same
    # vulnerability under any prompt framing); without this gate, the
    # SECOND emission of the same logical finding would trip
    # `_enforce_findings_unique` at `AnalysisRound` construction. The
    # same defensive gate covers pass-0 against an LLM repeating the
    # same proposal in a single response.
    admitted_keys_seen: set[tuple[str, str]] = set()
    trace_candidates: list[TraceCandidate] = []
    files_examined: list[str] = []
    files_skipped: list[str] = []
    n_proposals_seen = 0
    n_findings_emitted = 0
    n_proposals_rejected = 0
    n_responses_rejected = 0
    n_trace_candidates_emitted = 0
    # Per-pass aggregate of malformed-trace-candidate drops. Mirrors
    # `ParserCounters.n_trace_candidates_dropped_malformed` and lands
    # on `AnalyzeCompletedEvent` for the audit row — accumulating
    # here is what makes the per-pass summary count match the per-file
    # counters.
    n_trace_candidates_dropped_malformed = 0
    n_llm_calls = 0
    # The STANDARD-tier model actually used this pass (an LLM call fired for at least
    # one STANDARD-tier file), else None — lands on `AnalyzeCompletedEvent`.
    # `analyze_model` (DEEP) is always recorded; the STANDARD model only when STANDARD
    # routing fired this pass (`specs/2026-06-08-analyze-tiered-model-routing.md`).
    standard_model_used: str | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_cache_write_tokens = 0
    total_cost_decimal = Decimal("0")
    remaining_budget_tokens = total_review_budget_tokens

    # Step 2: per-pass iteration scope.
    #
    # Pass 0 (`len(state.analysis_rounds) == 0`): iterate
    # `pr_context.changed_files` filtered by triage tier — the original
    # analyze surface. SKIM/SKIP excluded by construction; files absent
    # from the tier map are treated as SKIP (defensive against tier-map
    # gaps; per spec §7 step 2). No FileExaminationEvent fires for
    # excluded files.
    #
    # Pass 1 (`len(state.analysis_rounds) == 1`, post-trace re-entry per
    # M8 loop): iterate `state.trace_fetched_files` — files trace
    # resolved + fetched at head SHA. These are NOT PR-diff files, so
    # there's no patch, no triage classification, and no
    # changed-scope-unit intersection: analyze examines the WHOLE file
    # because trace's resolution decided the file is relevant to a
    # source finding. The parser admits INFERRED proposals only on
    # pass 1 (`pass_index > 0`) — pass 0 still rejects per the V1 stub
    # (no trace context exists yet at that point).
    triage_result = state.triage_result
    if pass_index == 0:
        for changed_file in state.pr_context.changed_files:
            tier = (
                triage_result.file_tiers.get(changed_file.path, ReviewTier.SKIP)
                if triage_result is not None
                else ReviewTier.SKIP
            )
            if tier not in (ReviewTier.DEEP, ReviewTier.STANDARD):
                continue

            # Tier → model (the cost lever, DECISIONS.md#041): STANDARD routes to
            # standard_analyze_model (Haiku by default), DEEP stays on analyze_model (Sonnet).
            model_for_file = _model_for_tier(
                tier,
                analyze_model=analyze_model,
                standard_analyze_model=standard_analyze_model,
            )

            # Step 3: per-file processing.
            file_outcome = await _process_one_file(
                changed_file=changed_file,
                review_id=state.review_id,
                installation_id=state.pr_context.installation_id,
                is_eval=state.is_eval,
                provider=provider,
                analyze_model=model_for_file,
                import_path_resolver=import_path_resolver,
                file_examination_sink=file_examination_sink,
                analyze_event_sink=analyze_event_sink,
                active_policy_version=active_policy_version,
                pass_index=pass_index,
                per_file_cap_tokens=per_file_cap_tokens,
                remaining_budget_tokens=remaining_budget_tokens,
                # Pass-0 PR-diff files ONLY — trace-fetched files (pass 1,
                # `_process_one_trace_fetched_file`) have no changed-scope
                # set, so the filter never evaluates there by design.
                trivial_scope_filter_enabled=trivial_scope_filter_enabled,
            )

            if file_outcome.parser_result is not None:
                # LLM call was made; parser ran.
                n_llm_calls += 1
                if tier is ReviewTier.STANDARD:
                    standard_model_used = standard_analyze_model
                n_proposals_seen += file_outcome.parser_result.counters.n_proposals_seen
                n_findings_emitted += file_outcome.parser_result.counters.n_findings_emitted
                n_proposals_rejected += file_outcome.parser_result.counters.n_proposals_rejected
                n_responses_rejected += file_outcome.parser_result.counters.n_responses_rejected
                n_trace_candidates_emitted += (
                    file_outcome.parser_result.counters.n_trace_candidates_emitted
                )
                n_trace_candidates_dropped_malformed += (
                    file_outcome.parser_result.counters.n_trace_candidates_dropped_malformed
                )
                for f in file_outcome.parser_result.admitted_findings:
                    key = (f.content_hash, f.proposal_hash)
                    if key in admitted_keys_seen:
                        continue
                    admitted_keys_seen.add(key)
                    admitted_findings.append(f)
                # Per DECISIONS.md#025 point 6: trace_candidates from
                # rejected-parent proposals stay in state for replay
                # ("Unjoined candidates remain forensic-only"). Trace's
                # `_bucket_candidates_by_finding` skips the unjoined
                # ones (INFO log) — that's the documented forensic
                # contract, not a bug to filter at the analyze→state
                # boundary. The audit-event counter
                # `n_trace_candidates_emitted` on
                # `AnalyzeCompletedEvent` reflects the same pre-dedup
                # set the state-side reducer ingests.
                trace_candidates.extend(file_outcome.parser_result.trace_candidates)

            total_input_tokens += file_outcome.input_tokens
            total_output_tokens += file_outcome.output_tokens
            total_cache_read_tokens += file_outcome.cache_read_tokens
            total_cache_write_tokens += file_outcome.cache_write_tokens
            total_cost_decimal += file_outcome.cost_decimal
            remaining_budget_tokens -= file_outcome.estimated_tokens

            if file_outcome.parse_status == "skipped":
                files_skipped.append(changed_file.path)
            else:
                files_examined.append(changed_file.path)
    else:
        # Pass 1+ trace-fetched-file iteration. Trace resolved these
        # files; analyze examines the whole content (no diff intersection)
        # and admits INFERRED proposals citing trace_path.
        #
        # Build `source_findings_by_id` once per pass — the post-trace
        # prompt names the originating finding's title/description/evidence
        # so the model can connect the trace-fetched file back to the
        # source finding (passing source_finding_id alone leaves the
        # model with no content to reason about). The lookup walks ALL
        # prior rounds' findings; the source finding always exists for
        # any (path, source_finding_id) pair derived from
        # `state.trace_decisions` per trace's emission contract
        # (`trace.py::_build_proposal_hash_join` admits only proposal
        # hashes from admitted findings; unjoined candidates die at
        # `_bucket_candidates_by_finding` BEFORE producing a
        # TraceDecision — DECISIONS.md#025 point 6 documents the
        # forensic-only behavior of those dropped candidates). So
        # `.get(...)` returning None is a programmer error.
        source_findings_by_id: dict[UUID, ReviewFinding] = {
            f.finding_id: f for r in state.analysis_rounds for f in r.findings
        }
        # Fan out by `(target_file, source_finding_id)` per
        # `state.trace_decisions`, NOT by `state.trace_fetched_files`
        # alone. `TraceFetchedFile.path` is dedup'd first-write-wins
        # under the reducer; iterating over fetched files would
        # process the fetched content ONCE under the first finding's
        # context only, leaving every other finding that resolved to
        # the same target with no source-specific pass-1 analysis.
        # Each `(fetched_file, source_finding)` pair runs pass-1
        # independently so every admitted source finding gets its own
        # post-trace prompt.
        fetched_files_by_path: dict[str, TraceFetchedFile] = {
            f.path: f for f in state.trace_fetched_files
        }
        # Build the (path, source_finding_id) work list from the
        # canonical state.trace_decisions stream (filter to resolved +
        # target_file in fetched_files_by_path to skip target-in-PR
        # decisions whose Phase 2 deliberately skipped per M8).
        pass_one_work: list[tuple[TraceFetchedFile, ReviewFinding]] = []
        for decision in state.trace_decisions:
            if decision.target_file is None:
                continue
            fetched = fetched_files_by_path.get(decision.target_file)
            if fetched is None:
                # Decision resolved but Phase 2 skipped (target-in-PR
                # case per M8). No content to feed pass-1.
                continue
            source_finding = source_findings_by_id.get(decision.source_finding_id)
            if source_finding is None:
                raise RuntimeError(
                    f"analyze pass {pass_index}: TraceDecision "
                    f"source_finding_id={decision.source_finding_id} "
                    f"does not appear in state.analysis_rounds. Trace's "
                    f"emission contract is broken — "
                    f"_build_proposal_hash_join only admits proposal "
                    f"hashes from admitted findings, so every "
                    f"TraceDecision.source_finding_id should resolve."
                )
            pass_one_work.append((fetched, source_finding))

        for fetched_file, source_finding in pass_one_work:
            file_outcome = await _process_one_trace_fetched_file(
                fetched_file=fetched_file,
                source_finding=source_finding,
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
                n_llm_calls += 1
                n_proposals_seen += file_outcome.parser_result.counters.n_proposals_seen
                n_findings_emitted += file_outcome.parser_result.counters.n_findings_emitted
                n_proposals_rejected += file_outcome.parser_result.counters.n_proposals_rejected
                n_responses_rejected += file_outcome.parser_result.counters.n_responses_rejected
                n_trace_candidates_emitted += (
                    file_outcome.parser_result.counters.n_trace_candidates_emitted
                )
                n_trace_candidates_dropped_malformed += (
                    file_outcome.parser_result.counters.n_trace_candidates_dropped_malformed
                )
                for f in file_outcome.parser_result.admitted_findings:
                    key = (f.content_hash, f.proposal_hash)
                    if key in admitted_keys_seen:
                        continue
                    admitted_keys_seen.add(key)
                    admitted_findings.append(f)
                # Per DECISIONS.md#025 point 6: trace_candidates from
                # rejected-parent proposals stay in state for replay
                # ("Unjoined candidates remain forensic-only"). Trace's
                # `_bucket_candidates_by_finding` skips the unjoined
                # ones (INFO log) — that's the documented forensic
                # contract, not a bug to filter at the analyze→state
                # boundary. The audit-event counter
                # `n_trace_candidates_emitted` on
                # `AnalyzeCompletedEvent` reflects the same pre-dedup
                # set the state-side reducer ingests.
                trace_candidates.extend(file_outcome.parser_result.trace_candidates)

            total_input_tokens += file_outcome.input_tokens
            total_output_tokens += file_outcome.output_tokens
            total_cache_read_tokens += file_outcome.cache_read_tokens
            total_cache_write_tokens += file_outcome.cache_write_tokens
            total_cost_decimal += file_outcome.cost_decimal
            remaining_budget_tokens -= file_outcome.estimated_tokens

            # Pass-1 fan-out can iterate the same `fetched_file.path`
            # multiple times (one per source finding targeting that
            # path). `AnalysisRound._enforce_files_examined_unique` /
            # `_enforce_files_skipped_unique` validators reject
            # duplicates, so dedup at append time.
            #
            # Examined-wins semantic: once a path lands in
            # `files_examined`, subsequent iterations are no-ops; if an
            # earlier iteration skipped the path and a later one
            # examines it, promote the path from `files_skipped` to
            # `files_examined`. In current code paths skip outcomes
            # are deterministic per file (same content → same
            # decision), so the skip→examined transition cannot occur
            # — but encoding "examined wins" defensively means a
            # future skip-reason addition that DOES depend on
            # iteration-mutable state (e.g., partial-budget) can't
            # silently leave a successfully-examined path stuck in
            # `files_skipped`.
            if fetched_file.path in files_examined:
                pass
            elif file_outcome.parse_status != "skipped":
                if fetched_file.path in files_skipped:
                    files_skipped.remove(fetched_file.path)
                files_examined.append(fetched_file.path)
            elif fetched_file.path not in files_skipped:
                files_skipped.append(fetched_file.path)

    # ended_at is monotonic-derived so it can't precede started_at (FUP-141).
    ended_at = _round_ended_at(started_at, started_mono)

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
            n_trace_candidates_dropped_malformed=n_trace_candidates_dropped_malformed,
            total_input_tokens=total_input_tokens,
            total_cache_read_tokens=total_cache_read_tokens,
            total_cache_write_tokens=total_cache_write_tokens,
            total_output_tokens=total_output_tokens,
            # Decimal-summed across files, cast to float once. Matches
            # `sum(LLMCallEvent.cost_usd)` to within one float-cast step
            # rather than per-file FP drift.
            total_cost_usd=float(total_cost_decimal),
            pricing_version=PRICING_VERSION,
            policy_version=active_policy_version,
            analyze_model=analyze_model,
            standard_analyze_model=standard_model_used,
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


@dataclass(frozen=True, slots=True)
class _FileOutcome:
    """Per-file processing result. Populated by `_process_one_file` and
    consumed by the main loop's accumulators.

    Cache reads and writes stay separate (cache_write bills at 1.25×
    base, cache_read at 0.1×); the 12.5× cost differential would be
    hidden if summed. `cost_decimal` is the per-file `Decimal` cost;
    the main loop sums Decimals and casts to float once at the
    aggregate event, eliminating per-file FP drift against the per-call
    `LLMCallEvent.cost_usd` sum.
    """

    parse_status: _ParseStatus
    parser_result: ParserResult | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_decimal: Decimal
    estimated_tokens: int


def _build_query_match_id_set(file_content_bytes: bytes) -> frozenset[str]:
    """Fire every registered query against `file_content_bytes`; return
    the set of ids that produced at least one match.

    Iterates `queries.registry.REGISTERED_QUERY_IDS` (current
    non-deprecated ids only). Per spec §7 step 3b, this set is passed
    to the parser's OBSERVED admission — a model claim whose
    `query_match_id` isn't in this set rejects with
    `query_match_id_not_in_registry`. Empty set means no registry
    query fired against this file → every OBSERVED claim rejects;
    only JUDGED proposals can land.
    """
    fired: set[str] = set()
    for query_id in query_registry.REGISTERED_QUERY_IDS:
        if query_registry.match(query_id, file_content_bytes):
            fired.add(query_id)
    return frozenset(fired)


def _filter_query_ids_to_scopes(
    query_ids: frozenset[str],
    file_content_bytes: bytes,
    scope_units: tuple[ScopeUnit, ...],
) -> frozenset[str]:
    """Keep a fired query ID iff at least one of its match envelopes
    intersects an INCLUDED scope unit's byte range.

    Used only when the trivial-scope filter excluded scopes from the
    prompt (specs/2026-06-10-trivial-scope-filter.md): IDs whose matches
    fall only in excluded scopes must not advertise — the same filtered
    set feeds both the prompt and the parser's OBSERVED admission, so a
    finding cannot cite structural proof from code the model never saw.
    Half-open intersection over `QueryMatchSpan` envelopes.
    """
    ranges = tuple((su.byte_start, su.byte_end) for su in scope_units)
    kept: set[str] = set()
    for query_id in query_ids:
        for match_span in query_registry.match(query_id, file_content_bytes):
            if any(s < match_span.byte_end and match_span.byte_start < e for s, e in ranges):
                kept.add(query_id)
                break
    return frozenset(kept)


def _classify_included_scopes(
    *,
    changed_file: ChangedFile,
    content: str,
    content_bytes: bytes,
    patched_file: PatchedFile,
    included_scope_units: tuple[ScopeUnit, ...],
) -> tuple[ScopeExclusionEntry, ...]:
    """Classify every admitted scope through the trivial-scope filter;
    return one audit entry per scope (specs/2026-06-10-trivial-scope-filter.md).

    Fail-closed pre-check: removed lines anywhere in the patch with no
    base content means base-side verification is impossible — classify
    every scope `MISSING_BASE_CONTENT` rather than reaching
    `changed_line_spans`'s misuse guard (unreachable under the intake
    contract: modified/renamed files carry `content_base`; added files
    have no removed lines).
    """
    base_text = changed_file.content_base
    if base_text is None and patched_file_has_removed_lines(patched_file):
        return tuple(
            ScopeExclusionEntry(
                scope_qualified_name=su.qualified_name or su.name,
                trivial=False,
                reason=TrivialityReason.MISSING_BASE_CONTENT,
                head_added_lines=(),
                base_removed_lines=(),
            )
            for su in included_scope_units
        )

    context = build_triviality_context(
        content_bytes, base_text.encode("utf-8") if base_text is not None else None
    )
    entries: list[ScopeExclusionEntry] = []
    for su in included_scope_units:
        changed = changed_line_spans(su, patched_file, head_source=content, base_source=base_text)
        verdict = classify_scope_triviality(changed, context)
        entries.append(
            ScopeExclusionEntry(
                scope_qualified_name=su.qualified_name or su.name,
                trivial=verdict.trivial,
                reason=verdict.reason,
                head_added_lines=tuple(e.line_no for e in changed.head_added),
                base_removed_lines=tuple(e.line_no for e in changed.base_removed),
                offending_side=verdict.offending_side,
                offending_line=verdict.offending_line,
            )
        )
    return tuple(entries)


# `coordinates.bound_diff_hunks_text` does the bounded-render math;
# this module pins the cap values per spec §7 step 3c.


def _assemble_scope_unit_context(
    *,
    included_scope_units: tuple[ScopeUnit, ...],
    file_content: str,
) -> str:
    """Render the included scope units as the prompt's `scope_unit_context` block.

    V1 shape is per-unit kind + qualified name + line range + raw body
    extract. Same-file callers/callees/imports/decorators land with the
    trace spec. Byte-slicing is delegated to
    `coordinates.extract_scope_unit_body` because the byte-range →
    text surface belongs to the coordinates module per
    `coordinates-module-is-sole-translator`.

    No internal char cap today — the cost gate at the call site is the
    fail-closed protection. Adding an assembly-time cap parallel to
    `_DEGRADED_HUNK_CHAR_CAP` for the degraded path is tracked as
    FUP-052.
    """
    source_bytes = file_content.encode("utf-8")
    blocks: list[str] = []
    for su in included_scope_units:
        body = extract_scope_unit_body(su, source_bytes)
        name = su.qualified_name or su.name
        blocks.append(
            f"### {su.kind} `{name}` (lines {su.line_start}-{su.line_end})\n"
            f"{safe_code_fence(body, lang='python')}"
        )
    return "\n\n".join(blocks)


def _assemble_query_match_id_list(query_match_id_set: frozenset[str]) -> str:
    """Render the registry-fired ids as the prompt's `query_match_id_list` block.

    Sorted for determinism (replay equivalence depends on the prompt
    bytes being identical across runs with the same inputs). Empty set
    renders an explicit "no matches" line rather than a blank
    placeholder so the model sees the structural cue and falls back to
    JUDGED rather than guessing an OBSERVED id.
    """
    if not query_match_id_set:
        return "(no registry query matches fired for these scope units; do not claim `observed`)"
    return "\n".join(f"- `{qid}`" for qid in sorted(query_match_id_set))


def _concat_clipped_hunks(clipped_per_unit: tuple[tuple[str, ...], ...]) -> str:
    """Concatenate the per-scope-unit clipped hunks into the prompt's `diff_hunks` block.

    Per-unit hunks join with single newlines; per-unit blocks separate
    with blank lines so the model can visually tell where one unit's
    diff ends and the next begins. The full PR diff is never in this
    string — only hunks already clipped to included scope unit lines.
    """
    return "\n\n".join("\n".join(hunks) for hunks in clipped_per_unit)


async def _emit_skip(
    *,
    file_examination_sink: FileExaminationSink,
    review_id: UUID,
    is_eval: bool,
    file_path: str,
    skip_reason: SkipReason,
) -> _FileOutcome:
    """Emit a single `FileExaminationEvent(parse_status="skipped", skip_reason=...)`
    and return a zero-cost `_FileOutcome`. Used by every skip path in
    `_process_one_file` to keep the emission point uniform per spec §7
    step 3e (single emission per kept file)."""
    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=file_path,
            examination_type="analyze",
            node_id="analyze",
            parse_status="skipped",
            skip_reason=skip_reason,
        )
    )
    return _FileOutcome(
        parse_status="skipped",
        parser_result=None,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_decimal=Decimal("0"),
        estimated_tokens=0,
    )


async def _process_one_file(  # noqa: PLR0913, PLR0911, PLR0912, PLR0915 — orchestration boundary; outcome branches resist further extraction without losing audit clarity
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
    trivial_scope_filter_enabled: bool = False,
) -> _FileOutcome:
    """Process one triage-kept file through parse → outcome → cost
    gate → trivial-scope classification → LLM call → parser → audit
    events.

    File outcomes — the spec §7 step 3a set, plus the parser-stage
    pass-through, plus the trivial-scope filter's skip (the
    trivial-scope-filter spec):

    - `skipped+NO_REVIEWABLE_CONTEXT` — no content at all OR (V1
      unreachable, see module docstring) parse failure with no addable
      diff text.
    - `skipped+NO_CHANGED_SCOPE_UNITS` — clean parse but no scope
      unit intersects the changed regions.
    - `skipped+COST_BUDGET_EXHAUSTED` — outcome would have made an
      LLM call but cost gate failed.
    - `skipped+ALL_SCOPES_TRIVIAL` — enforcing-mode trivial-scope
      filter: every admitted scope classified ordinary-comment-only
      (after the cost gate; the shadow default never skips).
    - `failed+degraded_llm` — V1 unreachable (intake gates invalid
      UTF-8; analyze re-encodes valid str). Kept as a structural slot
      for the future raw-bytes intake path (FUP-053). Would fire on
      parse failure with addable text; degraded LLM call
      (`degradation_reason="parse_failed"`).
    - `degraded+degraded_llm` — clean parse but `has_error` ERROR
      nodes intersect a changed scope unit
      (`degradation_reason="tree_has_error_in_changed_regions"`), OR a
      changed addable line intersects a tree error with no recovered scope
      (`degradation_reason="tree_has_error_no_scope"`, DECISIONS#033);
      degraded LLM call.
    - `clean+full_llm` — clean parse, scope units intersect changed
      regions, no `has_error` in those units.
    - Parser-stage skip — `parse_python` returned `parser_outcome=
      "skipped"` (`OVERSIZED`, `VENDORED`, etc.); the parser's
      `skip_reason` is the audit value.
    """
    # Language gate: V1 only handles Python. Triage doesn't filter by
    # language and `ChangedFile.language` is currently unpopulated, so a
    # `.js`/`.go`/`.ts`/`.rs` file classified DEEP/STANDARD would
    # otherwise reach `parse_python` (tree-sitter Python parser) and the
    # `queries/python/` registry. Routes through `SkipReason.UNSUPPORTED_LANGUAGE`
    # per `DECISIONS.md#018` Amended 2026-05-21 — capability-scoped to
    # the current analyze implementation, not a forever-claim about
    # Outrider's language support.
    if not _is_python_file(changed_file.path):
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            skip_reason=SkipReason.UNSUPPORTED_LANGUAGE,
        )

    # Explicit `is not None` (not `or`) so an empty `content_head` ("")
    # doesn't fall through to `content_base` and analyze stale content.
    if changed_file.content_head is not None:
        content = changed_file.content_head
    elif changed_file.content_base is not None:
        content = changed_file.content_base
    else:
        content = None
    if content is None:
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # `file_byte_length` computed ONCE here per spec §7 step 3a;
    # passed to parser §5 unchanged so it never recomputes per
    # proposal.
    content_bytes = content.encode("utf-8")
    file_byte_length = len(content_bytes)

    parse_result: ParseResult = parse_python(
        source=content_bytes,
        file_path=changed_file.path,
        resolver=import_path_resolver,
    )

    # Parser-stage skip (VENDORED, OVERSIZED, GENERATED_FILENAME, MINIFIED,
    # GENERATED_BANNER): pass the parser's skip_reason through. Handled BEFORE
    # `lookup_patched_file` because a skipped file may carry a malformed/duplicate
    # patch, and `lookup_patched_file` RAISES `CoordinateError` on those (it returns
    # None only for the absent cases). Skipping first keeps the clean-skip for those
    # files — `lookup_patched_file` is only safe to call for a review candidate.
    if parse_result.parser_outcome == "skipped":
        # ParseResult validator guarantees skip_reason non-None when
        # parser_outcome="skipped"; rebind narrows for mypy.
        parser_skip_reason = parse_result.skip_reason
        if parser_skip_reason is None:  # validator-impossible; kept for narrowing
            raise RuntimeError(
                "ParseResult invariant violated: parser_outcome='skipped' "
                "requires non-None skip_reason"
            )
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            skip_reason=parser_skip_reason,
        )

    # `None` covers three cases: no patch (binary / oversized response), file
    # absent from a well-formed patch, or path-validation failure (the helper
    # returns None for those). Computed after the parser-skip return because the
    # degraded prompt below also needs `patched_file`.
    patched_file = lookup_patched_file(changed_file.patch, changed_file.path)

    # Outcome determination (skip / degraded / clean) for a PARSED file lives in the
    # pure `decide_degradation` (degradation.py) — extracted so structural eval
    # scenarios can exercise it LLM-free. This node is the only place that turns the
    # decision into behavior. The `"failed"` degraded branch is V1-unreachable
    # (intake gates invalid UTF-8 with SkipReason.OVERSIZED); retained for the
    # raw-bytes intake path (FUP-053) + audit/prompt-wiring tests.
    decision = decide_degradation(parse_result, patched_file)
    if decision.mode == "skip":
        skip_reason = decision.skip_reason
        if skip_reason is None:  # DegradationDecision guard makes this impossible; narrows mypy.
            raise RuntimeError("DegradationDecision mode='skip' with skip_reason None")
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            skip_reason=skip_reason,
        )
    degradation_reason: _DegradationReason | None = decision.degradation_reason
    parse_status_for_event: _ParseStatus = decision.parse_status
    included_scope_units: tuple[ScopeUnit, ...] = decision.included_scope_units
    included_clipped_hunks: tuple[tuple[str, ...], ...] = decision.included_clipped_hunks
    degraded_mode = decision.mode == "degraded"

    # Step 3b: registry-query firing (skip for degraded mode).
    query_match_id_set: frozenset[str] = (
        frozenset() if degraded_mode else _build_query_match_id_set(content_bytes)
    )

    # Step 3c: assemble the (system, user) prompt pair.
    if degraded_mode:
        # `patched_file` and `degradation_reason` are both non-None on
        # this branch by construction. The runtime checks below narrow
        # for mypy without re-asserting upstream invariants.
        if degradation_reason is None or patched_file is None:
            raise RuntimeError(
                "analyze: degraded_mode true with degradation_reason/patched_file None "
                "— upstream outcome-determination invariant violated"
            )
        parts = analyze_prompt.render_degraded(
            file_path=changed_file.path,
            bounded_hunks=bound_diff_hunks_text(
                patched_file,
                max_lines=_DEGRADED_HUNK_LINE_CAP,
                max_chars=_DEGRADED_HUNK_CHAR_CAP,
            ),
            pass_index=pass_index,
            degradation_reason=degradation_reason,
        )
    else:
        parts = analyze_prompt.render(
            file_path=changed_file.path,
            scope_unit_context=_assemble_scope_unit_context(
                included_scope_units=included_scope_units, file_content=content
            ),
            query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
            diff_hunks=_concat_clipped_hunks(included_clipped_hunks),
            pass_index=pass_index,
        )

    # Step 3d: cost gate.
    estimated_tokens = (
        _estimate_tokens(parts.system_prompt)
        + _estimate_tokens(parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
    )
    if estimated_tokens > per_file_cap_tokens or estimated_tokens > remaining_budget_tokens:
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        )

    # Step 3d-bis: trivial-scope classification (pass-0 clean mode only;
    # degraded files are never classified). Runs AFTER the baseline cost
    # gate by pinned precedence (specs/2026-06-10-trivial-scope-filter.md):
    # COST_BUDGET_EXHAUSTED wins over ALL_SCOPES_TRIVIAL, and the gate
    # evaluated the BASELINE prompt estimate, so enabling the filter can
    # never convert a budget skip into an LLM call (the filtered prompt is
    # <= baseline). The baseline estimate also stays in
    # `_FileOutcome.estimated_tokens` for the budget deduction — a
    # deliberate conservative over-debit: later files see a tighter
    # budget under enforcement, never a looser one.
    # Shadow mode (flag off) still classifies and emits
    # `applied=False` would-exclude telemetry — the eval-backed flip's
    # production data; enforcing mode excludes trivial scopes from the
    # prompt and skips all-trivial files. Runs BEFORE step 3e so the
    # all-trivial skip routes through `_emit_skip` and the single
    # FileExaminationEvent emission point holds.
    if not degraded_mode and patched_file is not None and included_scope_units:
        entries = _classify_included_scopes(
            changed_file=changed_file,
            content=content,
            content_bytes=content_bytes,
            patched_file=patched_file,
            included_scope_units=included_scope_units,
        )
        await analyze_event_sink.emit_scope_exclusion(
            ScopeExclusionEvent(
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                applied=trivial_scope_filter_enabled,
                filter_version=TRIVIAL_FILTER_VERSION,
                entries=entries,
            )
        )
        if trivial_scope_filter_enabled and any(e.trivial for e in entries):
            kept = tuple(
                (su, hunks)
                for su, hunks, entry in zip(
                    included_scope_units, included_clipped_hunks, entries, strict=True
                )
                if not entry.trivial
            )
            if not kept:
                return await _emit_skip(
                    file_examination_sink=file_examination_sink,
                    review_id=review_id,
                    is_eval=is_eval,
                    file_path=changed_file.path,
                    skip_reason=SkipReason.ALL_SCOPES_TRIVIAL,
                )
            included_scope_units = tuple(su for su, _ in kept)
            included_clipped_hunks = tuple(hunks for _, hunks in kept)
            # Span-filter the fired query-ID set to the kept scopes. The
            # SAME filtered set feeds the prompt's query_match_id_list AND
            # the parser's OBSERVED admission below — filtering only the
            # prompt text would let a finding anchored in a shown scope
            # cite a query ID whose match lives in an excluded scope (an
            # OBSERVED proof pointing at never-shown code).
            query_match_id_set = _filter_query_ids_to_scopes(
                query_match_id_set, content_bytes, included_scope_units
            )
            # Re-render over the kept scopes; this filtered prompt is what
            # is actually sent (and what context_summary describes).
            parts = analyze_prompt.render(
                file_path=changed_file.path,
                scope_unit_context=_assemble_scope_unit_context(
                    included_scope_units=included_scope_units, file_content=content
                ),
                query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
                diff_hunks=_concat_clipped_hunks(included_clipped_hunks),
                pass_index=pass_index,
            )

    # Step 3e: SINGLE FileExaminationEvent emission point.
    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            examination_type="analyze",
            node_id="analyze",
            parse_status=parse_status_for_event,
            skip_reason=None,
        )
    )

    # Step 3f: LLM call + response parse.
    # One ContextManifestEntry per included scope unit for clean+full_llm.
    # Empty tuple for degraded — `_enforce_context_for_scope_nodes`
    # special-cases this.
    context_summary: tuple[ContextManifestEntry, ...] = (
        ()
        if degraded_mode
        else tuple(
            ContextManifestEntry(
                file_path=changed_file.path,
                scope_unit_name=su.qualified_name or su.name,
                line_start=su.line_start,
                line_end=su.line_end,
                inclusion_reason="changed_scope",
            )
            for su in included_scope_units
        )
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
        degraded_mode=degraded_mode,
        degradation_reason=degradation_reason,
        context_summary=context_summary,
    )
    # Provider failure (LLMProviderError subclasses) propagates. No
    # try/except — the dangling start phase event is the audit signal
    # for "this pass was interrupted."
    response: LLMResponse = await provider.complete(request)

    # Cost: Decimal per file, summed in Decimal arithmetic, float-cast
    # once at the aggregate event. Matches sum(LLMCallEvent.cost_usd)
    # modulo a single float-cast step rather than per-file FP drift.
    cost_decimal = compute_cost_usd(
        analyze_model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
    )

    # FUP-138: deterministic degraded context (the addable-diff byte ranges a
    # degraded JUDGED span must intersect), computed from the patch in coordinates/
    # — never recomputed from prompt text or trusted from a model span. patched_file
    # is non-None on the degraded path by construction; `()` on the clean path is
    # unused (the parser's degraded gate doesn't run).
    degraded_context_byte_ranges = (
        added_line_byte_ranges(patched_file, content)
        if degraded_mode and patched_file is not None
        else ()
    )
    parser_result = parse_analyze_response(
        response.text,
        review_id=review_id,
        installation_id=installation_id,
        file_path=changed_file.path,
        file_content=content,
        file_byte_length=file_byte_length,
        included_scope_units=included_scope_units,
        query_match_id_set=query_match_id_set,
        degraded_mode=degraded_mode,
        active_policy_version=active_policy_version,
        degraded_context_byte_ranges=degraded_context_byte_ranges,
        pass_index=pass_index,
        finish_reason=response.finish_reason,
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

    # Persist one finding (audit row + content row) per admitted finding.
    for finding in parser_result.admitted_findings:
        await analyze_event_sink.emit_finding(finding, is_eval=is_eval)

    return _FileOutcome(
        parse_status=parse_status_for_event,
        parser_result=parser_result,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cost_decimal=cost_decimal,
        estimated_tokens=estimated_tokens,
    )


async def _process_one_trace_fetched_file(  # noqa: PLR0913 — orchestration parallel to _process_one_file
    *,
    fetched_file: TraceFetchedFile,
    source_finding: ReviewFinding,
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
    """Process one trace-fetched file through parse → LLM call → parser.

    Pass-1 sibling of `_process_one_file`. Trace resolved this file via
    M8's two-phase fetch (Phase 1 probes + Phase 2 content fetch); the
    file is NOT a PR-diff file, so there's no patch, no triage
    classification, and no changed-scope-unit intersection. Analyze
    examines the WHOLE file because trace's resolution decided the file
    is relevant to a source finding.

    Outcomes (subset of `_process_one_file`'s):
      - `skipped+UNSUPPORTED_LANGUAGE` — non-Python file.
      - Parser-stage skip — `parse_python` returned skipped (vendored,
        oversized, etc.).
      - `skipped+COST_BUDGET_EXHAUSTED` — cost gate failed.
      - `clean+full_llm` — clean parse, LLM call admitted, parser ran.

    Degraded outcomes (parse_failed / tree_has_error_in_changed_regions /
    tree_has_error_no_scope) don't apply here: no changed regions, and parse failures on a
    head-SHA-fetched file are routed through the parser-stage skip path
    rather than the V1-unreachable degraded branch.

    Per spec line 25: "INFERRED findings whose source `TraceDecision.
    resolution_status` is `unresolved` or `ambiguous` downgrade to
    JUDGED." V1 enforces this by Phase 2's gate (only `resolution_status=
    "resolved"` files reach `state.trace_fetched_files`), so the
    downgrade case doesn't fire here at the parser layer — every file
    iterated in pass 1 is by construction resolved.
    """
    if not _is_python_file(fetched_file.path):
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            skip_reason=SkipReason.UNSUPPORTED_LANGUAGE,
        )

    content = fetched_file.content_head
    content_bytes = content.encode("utf-8")
    file_byte_length = len(content_bytes)

    parse_result: ParseResult = parse_python(
        source=content_bytes,
        file_path=fetched_file.path,
        resolver=import_path_resolver,
    )

    if parse_result.parser_outcome == "skipped":
        parser_skip_reason = parse_result.skip_reason
        if parser_skip_reason is None:  # validator-impossible; narrows for mypy
            raise RuntimeError(
                "ParseResult invariant violated: parser_outcome='skipped' "
                "requires non-None skip_reason"
            )
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            skip_reason=parser_skip_reason,
        )

    if parse_result.parser_outcome == "failed":
        # Parse failure on a trace-fetched file: route to
        # NO_REVIEWABLE_CONTEXT skip rather than the degraded-LLM branch.
        # The fetched-file content came from GitHub at head SHA; a parse
        # failure here is either a genuinely-unparseable Python file
        # (already unusual for production code) or an invalid-UTF-8 case
        # the encode round-trip can't produce. Trace's resolution stays
        # in the audit log; analyze pass 1 didn't admit findings, but
        # the file is reachable for forensic inspection.
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # Include all scope units WITHOUT `has_error` from the prompt's
    # deterministic-proof set. Pass-1 INFERRED admission cross-checks
    # `trace_path` elements against the included scope-unit names
    # (`evidence-tier-schema-enforced`); admitting a scope unit whose
    # tree-sitter parse contains ERROR nodes would let the model claim
    # structural proof against recovered/error-bearing AST — defeating
    # the proof boundary. The unchanged-file `_process_one_file` has
    # the same guard (it routes the whole file to degraded mode when
    # has_error fires in changed regions); for trace-fetched files we
    # do per-scope-unit filtering instead, since there are no "changed
    # regions" to scope the degradation to.
    included_scope_units = tuple(
        su for su in parse_result.scope_units if not parse_result.has_error.get(su.unit_id, False)
    )
    if not included_scope_units:
        # Pass-1 trace-fetched file has no notion of "changed" — the file
        # lives outside the PR diff. Empty included-scope-units after the
        # has_error filter means there's no reviewable structural context;
        # NO_REVIEWABLE_CONTEXT classifies this for audit telemetry.
        # Same enum the parse-failed branch above uses for the same reason.
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # Pass-1 prompt admits INFERRED proposals with non-empty `trace_path`.
    # `render_post_trace` is the pass-1 variant of `render`; same shape
    # (system + user prompt), different system-prompt instructions to
    # allow INFERRED.
    #
    # Fail-closed on partial has_error filtering: if the has_error filter
    # above dropped any scope unit, treat the whole file as degraded for
    # OBSERVED-proof purposes (force `query_match_id_set` empty). Without
    # this, the registry built from the FULL file content could authorize
    # an OBSERVED proposal against a query match that lives inside a
    # scope unit we deliberately excluded from the proof set — defeating
    # the proof boundary the has_error filter exists to defend. Mirrors
    # the pass-0 `degraded_mode → frozenset()` pattern in
    # `_process_one_file`.
    query_match_id_set: frozenset[str] = (
        frozenset()
        if len(included_scope_units) != len(parse_result.scope_units)
        else _build_query_match_id_set(content_bytes)
    )
    parts = analyze_prompt.render_post_trace(
        file_path=fetched_file.path,
        scope_unit_context=_assemble_scope_unit_context(
            included_scope_units=included_scope_units, file_content=content
        ),
        query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
        # Pass the ACTIVE source finding's id (matches the
        # title/description/evidence below), NOT
        # `fetched_file.source_finding_id` — the latter is first-write
        # provenance on `state.trace_fetched_files` (dedup'd by path
        # under the reducer); under the pass-1 fan-out (one iteration
        # per source finding targeting this fetched path), every
        # iteration after the first would attribute the SECOND/THIRD
        # finding's content to the FIRST finding's id — internally
        # inconsistent prompt.
        source_finding_id=source_finding.finding_id,
        source_finding_title=source_finding.title,
        source_finding_description=source_finding.description,
        source_finding_evidence=source_finding.evidence,
        pass_index=pass_index,
    )

    estimated_tokens = (
        _estimate_tokens(parts.system_prompt)
        + _estimate_tokens(parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
    )
    if estimated_tokens > per_file_cap_tokens or estimated_tokens > remaining_budget_tokens:
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        )

    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            examination_type="analyze",
            node_id="analyze",
            parse_status="clean",
            skip_reason=None,
        )
    )

    # `inclusion_reason="trace_expansion"` per the ContextManifestEntry
    # Literal — names the post-trace expansion-pass inclusion shape
    # (scope units from a trace-fetched file). The Literal predates the
    # trace-node arc; using it here closes the loop without a schema
    # change.
    context_summary: tuple[ContextManifestEntry, ...] = tuple(
        ContextManifestEntry(
            file_path=fetched_file.path,
            scope_unit_name=su.qualified_name or su.name,
            line_start=su.line_start,
            line_end=su.line_end,
            inclusion_reason="trace_expansion",
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
        degradation_reason=None,
        context_summary=context_summary,
    )
    response: LLMResponse = await provider.complete(request)

    cost_decimal = compute_cost_usd(
        analyze_model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
    )

    # Deterministic-proof set for INFERRED admission per the
    # `evidence-tier-schema-enforced` invariant: every scope-unit name
    # the model could legitimately have walked in this file. Uses the
    # SAME single rendered label the prompt actually shows the model
    # (`_assemble_scope_unit_context` at line ~681 renders
    # `su.qualified_name or su.name` — one label per scope unit, not
    # both). Admitting both `qualified_name` AND bare `name` would
    # weaken the proof boundary: common duplicate bare names like
    # `__init__` or `handle` across multiple classes would satisfy
    # `trace_path` membership without identifying a unique scope unit.
    # The parser's pass-1 INFERRED admission rejects any trace_path
    # element not in this set — load-bearing for
    # `evidence-tier-schema-enforced`.
    valid_trace_path_elements = frozenset(
        rendered_name
        for su in included_scope_units
        if (rendered_name := (su.qualified_name or su.name))
    )
    parser_result = parse_analyze_response(
        response.text,
        review_id=review_id,
        installation_id=installation_id,
        file_path=fetched_file.path,
        file_content=content,
        file_byte_length=file_byte_length,
        included_scope_units=included_scope_units,
        query_match_id_set=query_match_id_set,
        degraded_mode=False,
        active_policy_version=active_policy_version,
        pass_index=pass_index,
        valid_trace_path_elements=valid_trace_path_elements,
        finish_reason=response.finish_reason,
    )

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

    for finding in parser_result.admitted_findings:
        await analyze_event_sink.emit_finding(finding, is_eval=is_eval)

    return _FileOutcome(
        parse_status="clean",
        parser_result=parser_result,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cache_read_tokens=response.cache_read_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cost_decimal=cost_decimal,
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


__all__ = [
    "DEFAULT_REVIEW_BUDGET_TOKENS",
    "PER_FILE_CAP_FRACTION",
    "analyze",
]
