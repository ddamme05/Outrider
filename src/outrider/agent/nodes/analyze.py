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

**Landing scope (second landing, 2026-05-20).** Five file outcomes
land per spec §7 step 3a:

- `clean+full_llm` — file parses cleanly, has scope units in the
  changed regions, no `has_error` in those units, cost gate passes;
  full LLM call + parser invocation with the included scope units +
  pre-fired registry query-match-id set.
- `degraded+degraded_llm` — clean parse but tree-sitter `has_error`
  ERROR nodes intersect a changed scope unit; degraded LLM call with
  `degradation_reason="tree_has_error_in_changed_regions"`. Registry
  queries skipped (no structurally-trustworthy tree); parser admits
  only JUDGED via `span_within_file`.
- `failed+degraded_llm` — `parse_python` returned `parser_outcome=
  "failed"` (V1: UTF-8 decode failure) AND the patch contains added
  text; degraded LLM call with `degradation_reason="parse_failed"`.
- `skipped+NO_REVIEWABLE_CONTEXT` — no content at all (binary,
  unfetchable both sides), OR parse failure with no added text
  (pure deletion). No LLM call.
- `skipped+NO_CHANGED_SCOPE_UNITS` — clean parse but no scope unit
  intersects the changed regions (comment-only / whitespace-only /
  module-level changes), OR clean parse with no patch info. No LLM call.
- `skipped+COST_BUDGET_EXHAUSTED` — outcome would have made an LLM
  call but the cost gate's per-file ceiling OR remaining-budget
  check failed; no LLM call.

Parser-stage skips returned by `parse_python` (`OVERSIZED`,
`VENDORED`, `GENERATED_FILENAME`, `MINIFIED`, `GENERATED_BANNER`) are
passed through as `FileExaminationEvent(parse_status="skipped",
skip_reason=<parser's reason>)`. Spec §7 doesn't enumerate these —
the spec assumes upstream filtering — but `parse_python` returns
them, so analyze must route them rather than crash.

**Changed-region intersection.** Performed via
`coordinates.lookup_patched_file` (parse the patch and find this
file's `PatchedFile`) plus `coordinates.scope_unit_diff_hunks` (clip
to scope-unit line range, returning the clipped hunk text). A scope
unit is "included" iff `scope_unit_diff_hunks` returns non-empty;
both intersection and clipped-hunk text come from the same call.
Empty patch / file absent from patch → `None`, which short-circuits
to `NO_CHANGED_SCOPE_UNITS` for clean parses (no addable hunks to
analyze).

**Registry-query firing.** For clean+full_llm outcomes, every id in
`queries.registry.REGISTERED_QUERY_IDS` is fired against the file
content; the set of ids that produce at least one match becomes
`query_match_id_set` passed to the parser. The parser's OBSERVED
admission rejects any claim whose id isn't in this set.

**Token estimation.** `_BYTES_PER_TOKEN = 3` (code-leaning).
Anthropic's BPE tokenizer operates on UTF-8 byte sequences;
`_estimate_tokens` counts bytes (not Python codepoints) and rounds
up. Bytes-not-codepoints preserves the over-estimate safety
direction for multi-byte-heavy files (CJK comments, emoji,
accented identifiers); ceiling division avoids floor-rounding
small fragments down to undercount. The estimator's job is
order-of-magnitude blowup detection, not a tight count; a
`tiktoken`-grade estimate is FUP-049 scope.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, Literal
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
from outrider.coordinates import (
    bound_diff_hunks_text,
    lookup_patched_file,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
)
from outrider.llm.base import LLMRequest
from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd
from outrider.policy.canonical import compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
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

# Hard ceiling on the per-file pre-flight token estimate, applied
# alongside `PER_FILE_CAP_FRACTION * total_review_budget_tokens`. With
# `PER_FILE_CAP_FRACTION = 0.25`, a caller passing
# `total_review_budget_tokens=2_000_000` (e.g., a "monorepo PR" knob)
# would silently lift the per-file cap to 500K tokens — well past any
# reasonable Sonnet-call envelope. The absolute ceiling closes that
# gate independently of caller configuration; tuning headroom remains
# in `PER_FILE_CAP_FRACTION`.
MAX_PER_FILE_TOKENS_ABSOLUTE: Final[int] = 60_000

# Token-estimate divisor for `_estimate_tokens`. Code-leaning ratio
# (1 token ≈ 3 bytes) over-estimates token count vs Anthropic's
# published 1:4 prose heuristic, which makes the cost gate fail safer
# for code-heavy prompts. A `tiktoken`-grade estimate is FUP-049.
_BYTES_PER_TOKEN: Final[int] = 3

# Degraded-context bounds per spec §7 step 3c: ≤100 unidiff Line
# objects total AND ≤8192 chars of text content. Either cap closes
# the gate; the line count prevents many-tiny-lines fan-out, the byte
# cap prevents pathological few-very-long-lines blowup.
_DEGRADED_HUNK_LINE_CAP: Final[int] = 100
_DEGRADED_HUNK_CHAR_CAP: Final[int] = 8192


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate for the pre-flight cost gate.

    Counts UTF-8 BYTES (not Python codepoints) and rounds up. Both
    choices preserve the over-estimate safety direction the cost gate
    needs:

    - Bytes, not codepoints: Anthropic's BPE tokenizer operates on
      UTF-8 byte sequences. `len(str)` returns codepoint count, so a
      CJK character (1 codepoint, 3 bytes) under-counts as `1 // 3 == 0`
      tokens; a 4-byte emoji codepoint under-counts even worse. Switching
      to bytes restores the documented "over-estimate, fail safer" claim
      for multi-byte-heavy files (CJK comments, accented identifiers,
      emoji in tests).
    - Ceiling division `(n + d - 1) // d`: floor-division would round
      a 4-byte fragment down to 1 token (`4 // 3 == 1`); ceiling rounds
      to 2 (`-(-4 // 3) == 2`). Conservative-up is the right direction
      for a budget guard.

    The estimator's job is order-of-magnitude blowup detection, not a
    tight count; a `tiktoken`-grade estimator is FUP-049.
    """
    byte_len = len(text.encode("utf-8"))
    return (byte_len + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def _compute_per_file_cap(total_review_budget_tokens: int) -> int:
    """Per-file token cap for the cost gate.

    Combines two ceilings; the tighter one wins:
      - Fraction of total review budget (`PER_FILE_CAP_FRACTION * budget`)
        — bounds budget consumption by any single file so an oversized
        file at iteration head doesn't drain the per-review budget. With
        the V1 0.25 fraction, one file can starve at most four others.
      - Absolute ceiling (`MAX_PER_FILE_TOKENS_ABSOLUTE`) — independent of
        caller-configurable budget. Guards against budget inflation
        (e.g., a "monorepo PR" knob setting `total_review_budget_tokens`
        to 2M tokens) silently lifting the per-file cap into Sonnet-call-
        overflow territory.

    Extracted to a named helper so the policy is independently testable
    (the previous inline `min(...)` expression at the call site was
    indirectly tested via the cost-gate path; pinning the helper's return
    value directly catches drift in either ceiling without LLM-flow
    setup).
    """
    return min(
        int(total_review_budget_tokens * PER_FILE_CAP_FRACTION),
        MAX_PER_FILE_TOKENS_ABSOLUTE,
    )


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


# Spec §7 step 3f: `LLMRequest.degradation_reason` literal values per
# `_enforce_degradation_provenance` at `llm/base.py`. Aliased here so
# the outcome-determination block in `_process_one_file` carries the
# narrow type rather than `str`.
_DegradationReason = Literal["parse_failed", "tree_has_error_in_changed_regions"]

# Spec §7 step 3e: `FileExaminationEvent.parse_status` literal values.
# Aliased so the outcome-determination block carries the narrow type
# rather than `str`.
_ParseStatus = Literal["clean", "failed", "degraded", "skipped"]


def _has_addable_lines(patched_file: PatchedFile) -> bool:
    """True iff any hunk in `patched_file` carries at least one added line.

    Used by `_process_one_file` to distinguish
    `skipped+NO_REVIEWABLE_CONTEXT` (parse failure with no addable
    text — binary file or pure deletion) from `failed+degraded_llm`
    (parse failure with addable text — degraded LLM call needed). The
    discriminator is "the patch has `+` lines to analyze," not just
    "the patch exists."
    """
    return any(line.is_added for hunk in patched_file for line in hunk)


def _intersect_changed_scope_units(
    scope_units: tuple[ScopeUnit, ...],
    patched_file: PatchedFile,
) -> tuple[tuple[ScopeUnit, ...], tuple[tuple[str, ...], ...]]:
    """Return `(included_units, clipped_hunks_per_unit)` for the intersection.

    A scope unit is "included" iff
    `coordinates.scope_unit_has_added_lines` returns True AND
    `coordinates.scope_unit_diff_hunks` returns non-empty. The two
    tuples share indices: `included_units[i]` has clipped hunks
    `clipped_hunks_per_unit[i]`. Empty inputs / no intersection
    returns `((), ())`.

    Composition of two coordinates surfaces — the orchestration lives
    here (analyze decides which units feed which prompt), the
    coordinate math lives there. Backs the spec's
    `outcome="skipped+NO_CHANGED_SCOPE_UNITS"` discriminator and the
    `clean+full_llm` prompt's `diff_hunks` block.
    """
    included: list[ScopeUnit] = []
    hunks: list[tuple[str, ...]] = []
    for su in scope_units:
        if not scope_unit_has_added_lines(su, patched_file):
            continue
        clipped = scope_unit_diff_hunks(su, patched_file)
        if not clipped:
            continue
        included.append(su)
        hunks.append(clipped)
    return tuple(included), tuple(hunks)


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


# `coordinates.bound_diff_hunks_text` does the bounded-render math;
# this module pins the cap values per spec §7 step 3c.


def _assemble_scope_unit_context(
    *,
    included_scope_units: tuple[ScopeUnit, ...],
    file_content: str,
) -> str:
    """Render the included scope units as the prompt's `scope_unit_context` block.

    V1 minimum-viable shape: per-unit kind + qualified name + line
    range + raw body extract. Same-file callers/callees/imports/
    decorators (spec §5's full file-scoped context) are deferred to
    the trace-spec landing; this commit's intersection alone is the
    structural defense against the over-broad-context cost issue the
    user flagged.

    The body extract is taken via UTF-8 byte slicing (`ScopeUnit.byte_start`
    / `byte_end`) because tree-sitter byte offsets land on char
    boundaries per `parse_python`'s contract. `errors="replace"` would
    only fire under producer bug (byte offsets misaligned); we treat
    the bytes as already-validated UTF-8 from `content.encode("utf-8")`.
    """
    source_bytes = file_content.encode("utf-8")
    blocks: list[str] = []
    for su in included_scope_units:
        body = source_bytes[su.byte_start : su.byte_end].decode("utf-8", errors="replace")
        name = su.qualified_name or su.name
        blocks.append(
            f"### {su.kind} `{name}` (lines {su.line_start}-{su.line_end})\n```python\n{body}\n```"
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
        cached_tokens=0,
        cost_usd=0.0,
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
) -> _FileOutcome:
    """Process one triage-kept file through parse → outcome → cost
    gate → LLM call → parser → audit events.

    Five outcomes per spec §7 step 3a (with parser-stage skip passed
    through as a sixth):

    - `skipped+NO_REVIEWABLE_CONTEXT` — no content at all OR parse
      failure with no addable diff text.
    - `skipped+NO_CHANGED_SCOPE_UNITS` — clean parse but no scope
      unit intersects the changed regions.
    - `skipped+COST_BUDGET_EXHAUSTED` — outcome would have made an
      LLM call but cost gate failed.
    - `failed+degraded_llm` — parse failure with addable text;
      degraded LLM call (`degradation_reason="parse_failed"`).
    - `degraded+degraded_llm` — clean parse but `has_error` ERROR
      nodes intersect a changed scope unit; degraded LLM call
      (`degradation_reason="tree_has_error_in_changed_regions"`).
    - `clean+full_llm` — clean parse, scope units intersect changed
      regions, no `has_error` in those units.
    - Parser-stage skip — `parse_python` returned `parser_outcome=
      "skipped"` (`OVERSIZED`, `VENDORED`, etc.); the parser's
      `skip_reason` is the audit value.
    """
    # Step 3a: content selection + outcome determination.
    content = changed_file.content_head or changed_file.content_base
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

    # Parser-stage skip (VENDORED, OVERSIZED, GENERATED_FILENAME,
    # MINIFIED, GENERATED_BANNER). The parser already decided. Pass
    # through with its skip_reason — spec §7 doesn't enumerate this
    # path but `parse_python` returns it; routing rather than
    # crashing preserves audit visibility.
    if parse_result.parser_outcome == "skipped":
        # ParseResult validator guarantees skip_reason non-None when
        # parser_outcome="skipped"; the local rebind narrows for mypy
        # and the runtime check is documentation of an upstream
        # invariant rather than a defensive gate.
        parser_skip_reason = parse_result.skip_reason
        if parser_skip_reason is None:  # validator-impossible, kept for type narrowing
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

    # Locate the file's `PatchedFile` for the changed-region
    # intersection. None covers three cases: no patch (binary /
    # oversized GitHub response), file absent from a well-formed
    # patch, or path-validation failure on `changed_file.path` (the
    # `coordinates` helper returns None rather than raising for
    # these per its boolean-helper policy).
    patched_file = lookup_patched_file(changed_file.patch, changed_file.path)

    # Outcome branch: parser_outcome == "failed" (V1: UTF-8 decode failure).
    if parse_result.parser_outcome == "failed":
        if patched_file is None or not _has_addable_lines(patched_file):
            return await _emit_skip(
                file_examination_sink=file_examination_sink,
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
            )
        # failed+degraded_llm
        degradation_reason: _DegradationReason | None = "parse_failed"
        parse_status_for_event: _ParseStatus = "failed"
        included_scope_units: tuple[ScopeUnit, ...] = ()
        included_clipped_hunks: tuple[tuple[str, ...], ...] = ()
    else:
        # parser_outcome == "clean".
        if patched_file is None:
            return await _emit_skip(
                file_examination_sink=file_examination_sink,
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS,
            )
        included_scope_units, included_clipped_hunks = _intersect_changed_scope_units(
            tuple(parse_result.scope_units), patched_file
        )
        if not included_scope_units:
            return await _emit_skip(
                file_examination_sink=file_examination_sink,
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                skip_reason=SkipReason.NO_CHANGED_SCOPE_UNITS,
            )
        if any(parse_result.has_error.get(su.unit_id, False) for su in included_scope_units):
            degradation_reason = "tree_has_error_in_changed_regions"
            parse_status_for_event = "degraded"
        else:
            degradation_reason = None
            parse_status_for_event = "clean"

    degraded_mode = degradation_reason is not None

    # Step 3b: registry-query firing (skip for degraded mode).
    query_match_id_set: frozenset[str] = (
        frozenset() if degraded_mode else _build_query_match_id_set(content_bytes)
    )

    # Step 3c: assemble the (system, user) prompt pair.
    if degraded_mode:
        # `patched_file` is non-None on both degraded branches by
        # construction: the failed path required `_has_addable_lines`
        # (None would have AttributeError'd inside the helper), and
        # the clean path early-returned NO_CHANGED_SCOPE_UNITS when
        # patched_file was None. Same for `degradation_reason`: the
        # `degraded_mode = degradation_reason is not None` derivation
        # above pins the implication. The runtime checks below narrow
        # for mypy without re-asserting upstream invariants as
        # adversarial-input gates.
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
    # Build context_summary per spec §7: one ContextManifestEntry per
    # included scope unit for clean+full_llm. Empty tuple for degraded
    # — `_enforce_context_for_scope_nodes` special-cases this per
    # spec §7 step 3f.
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
        degraded_mode=degraded_mode,
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
        parse_status=parse_status_for_event,
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
