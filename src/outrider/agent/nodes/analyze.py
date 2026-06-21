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

import logging
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from outrider.agent.nodes.analyze_observed import (
    OBSERVED_PRODUCER_VERSION,
    compute_observed_skip_shadow,
    produce_observed_findings,
    run_observed_matches,
)
from outrider.agent.nodes.analyze_parser import (
    ANALYZE_PARSER_VERSION,
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    parse_analyze_response,
)
from outrider.agent.nodes.cache_config import CacheMode
from outrider.agent.nodes.degradation import (
    _DegradationReason,
    _ParseStatus,
    decide_degradation,
)
from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.ast_facts.analyze_bundle import extract_triviality_and_scan
from outrider.ast_facts.models import SkipReason, TrivialityReason
from outrider.ast_facts.parameterized_calls import scan_digest, scan_parameterized_calls
from outrider.ast_facts.python_adapter import parse_python
from outrider.ast_facts.triviality import (
    TRIVIAL_FILTER_VERSION,
    FileTrivialityContext,
    classify_scope_triviality,
)
from outrider.audit.events import (
    AnalyzeCompletedEvent,
    AnalyzeResponseRejectedEvent,
    CacheLookupEvent,
    CacheServeEvent,
    ContextManifestEntry,
    FileExaminationEvent,
    FindingProposalRejectedEvent,
    ReviewPhaseEvent,
    ScopeExclusionEntry,
    ScopeExclusionEvent,
    ServedTraceCandidateRef,
)
from outrider.cache import CacheStoreError, compute_analyze_cache_key
from outrider.coordinates import (
    added_line_byte_ranges,
    bound_diff_hunks_text,
    changed_line_spans,
    extract_scope_unit_body,
    lookup_patched_file,
    patched_file_has_removed_lines,
)
from outrider.llm.base import LLMRequest, _canonical_prompt_hash
from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd
from outrider.policy.canonical import (
    compute_phase_id,
    compute_round_id,
    compute_served_finding_id,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.recall import scan_added_lines_for_risk
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
from outrider.prompts import safe_code_fence
from outrider.queries import registry as query_registry
from outrider.queries.registry import QUERY_REGISTRY_DIGEST
from outrider.schemas import AnalysisRound, ReviewFinding, TraceCandidate
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_FORMAT_DIGEST,
    ANALYZE_RESPONSE_SCHEMA_JSON,
)
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from unidiff import PatchedFile

    from outrider.anomaly.sinks import AnomalySink
    from outrider.ast_facts.base import ImportPathResolver
    from outrider.ast_facts.models import ParseResult, ScopeUnit
    from outrider.audit.sinks import (
        AnalyzeEventSink,
        FileExaminationSink,
        PhaseEventSink,
    )
    from outrider.cache import AnalyzeCacheStore, CacheEntry, CacheScope
    from outrider.llm.base import LLMProvider, LLMResponse
    from outrider.schemas import (
        ReviewState,
        TraceFetchedFile,
    )
    from outrider.schemas.pr_context import ChangedFile


logger = logging.getLogger(__name__)

# One file can starve at most `1 / PER_FILE_CAP_FRACTION` others on the
# review-wide budget; richer fairness (iteration ordering, per-installation
# budgets) is FUP-044.
PER_FILE_CAP_FRACTION: Final[float] = 0.25

# Bounded high-risk reserve (specs/2026-06-17-analyze-cost-fairness.md Stage 1).
# A capped slice of the per-review budget that ONLY pass-0 files carrying a
# blatant CRITICAL-class signature (policy.recall) may draw from, so a
# late-iterated dangerous file is never starved purely by iteration position.
# The reserve is the deterministic fix for the verified PR #8 failure (a CRITICAL
# command_injection skipped COST_BUDGET_EXHAUSTED behind benign DEEP files). It is
# BOUNDED — high-risk files draw their dedicated reserve first and dip into general
# on overflow, and a file past (reserve + general) still skips; it is NOT an
# unlimited budget bypass. The fractional token reserve + PER_FILE_CAP_FRACTION are
# the only bounds (no separate slot cap — the token cap already bounds total reserve
# spend, and a single huge high-risk file still can't exceed the per-file cap). 0.25
# of a 200k budget = 50k, ~3-4 reserved files at typical DEEP per-file cost.
HIGH_RISK_RESERVE_FRACTION: Final[float] = 0.25

# Tier-descending iteration order (specs/2026-06-17-analyze-cost-fairness.md Stage 2):
# pass-0 processes DEEP-tier files before STANDARD so the per-review budget lands on
# the higher-tier files first under pressure. Lower rank = earlier. SKIM/SKIP never
# reach analyze, so they have no rank.
_PASS0_TIER_RANK: Final[Mapping[ReviewTier, int]] = MappingProxyType(
    {ReviewTier.DEEP: 0, ReviewTier.STANDARD: 1}
)

# Starvation anomaly (FUP-044 extension 3): an analyze pass that skips at least
# this many files with COST_BUDGET_EXHAUSTED emits one COST_BUDGET_STARVATION
# anomaly so operators see the structural pattern instead of counting individual
# FileExaminationEvent skips. 3 matches the FUP-044 exit-rule threshold.
COST_BUDGET_STARVATION_THRESHOLD: Final[int] = 3

# Default per-review token budget; tunable via OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS
# (AnalyzeConfig), wired through build_graph by api/lifespan.py. Before that wiring
# (Stage 0), production silently used this hardcoded default.
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
    anomaly_sink: AnomalySink,
    import_path_resolver: ImportPathResolver,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
    trivial_scope_filter_enabled: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
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

    # Analyze-cache scope, resolved once on pass 0 from the reviews row —
    # the canonical (installation_id, repo_id) tenant identity the cache
    # key requires (never PRContext's mutable owner/repo strings); the
    # cache is pass-0-only, so trace rounds skip the SELECT. Sits AFTER
    # the phase-start emit: the resolve is node work (a DB read), and
    # `phase-events-bound-work` requires node work inside the phase
    # envelope. None disables the cache for this pass: store not
    # injected (the eval driver's default), review row missing
    # (fail-open to UNCACHED, never to cross-scope), or a store DB
    # failure: the shadow cache is optional telemetry and must never
    # abort a review (`CacheStoreError` is contained, not raised). An
    # eval review does NOT disable the cache — it reads/writes scoped to
    # is_eval rows via the lookup's is_eval predicate (DECISIONS.md#046).
    cache_scope: CacheScope | None = None
    if analyze_cache_store is not None and pass_index == 0:
        try:
            cache_scope = await analyze_cache_store.resolve_scope(state.review_id)
        except CacheStoreError:
            logger.warning(
                "analyze-cache resolve_scope failed; cache disabled for this pass",
                exc_info=True,
            )
            cache_scope = None
        # No is_eval bypass (DECISIONS.md#046): eval and production reviews BOTH
        # use the cache, kept mutually invisible by the lookup's REQUIRED is_eval
        # read-isolation predicate (the scope's is_eval, passed at the lookup
        # below) plus the is_eval-stamped write.
        #
        # Defense-in-depth: the lookup partitions on cache_scope.is_eval (the
        # reviews row) while CacheLookupEvent + the serve emits are tagged
        # state.is_eval. The two are set together at review creation and cannot
        # diverge in production; but if a producer bug ever split them, reading one
        # partition while emitting the other's telemetry would be incoherent — so a
        # divergence disables the cache for this pass (fail-safe; the protection the
        # pre-#046 either-flag bypass gave, kept without the eval-wide veto).
        if cache_scope is not None and cache_scope.is_eval != state.is_eval:
            logger.warning(
                "analyze-cache is_eval divergence (scope=%s, state=%s); cache disabled",
                cache_scope.is_eval,
                state.is_eval,
            )
            cache_scope = None

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
    # FUP-044 ext 3: count COST_BUDGET_EXHAUSTED skips in this pass (pass-0
    # changed-file set) to drive the COST_BUDGET_STARVATION anomaly after the loop.
    budget_skip_count = 0
    n_proposals_seen = 0
    n_findings_emitted = 0
    n_findings_served = 0
    n_findings_observed = 0
    n_proposals_superseded_by_observed = 0
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
        # Bounded high-risk reserve (Stage 1): split the per-review budget into a
        # general pool (all files) + a capped reserve only high-risk files may
        # draw from. A high-risk file draws its reserve FIRST, dipping into general
        # only on overflow (reserved-then-general); a benign file draws general
        # only. Dedicating the reserve to high-risk files (not spending general
        # first) keeps it from being wasted when high-risk files iterate early.
        # Pass-1 (trace-fetched, no patch → never high-risk) keeps the single
        # `remaining_budget_tokens` pool below, untouched.
        remaining_reserved_tokens = int(total_review_budget_tokens * HIGH_RISK_RESERVE_FRACTION)
        remaining_general_tokens = total_review_budget_tokens - remaining_reserved_tokens

        # Tier-descending iteration order (Stage 2): build the admitted
        # (DEEP/STANDARD) worklist, then STABLE-sort DEEP-first so budget pressure
        # hits higher-tier files last. `list.sort` is stable, so `changed_files`
        # order is preserved WITHIN a tier (no path re-sort). SKIM/SKIP are excluded
        # by construction (absent from the tier map → SKIP); they never reach
        # analyze. Orthogonal to the reserve, which is signature- not tier-based.
        pass_zero_worklist: list[tuple[ChangedFile, ReviewTier]] = []
        if triage_result is not None:
            for changed_file in state.pr_context.changed_files:
                tier = triage_result.file_tiers.get(changed_file.path, ReviewTier.SKIP)
                if tier in (ReviewTier.DEEP, ReviewTier.STANDARD):
                    pass_zero_worklist.append((changed_file, tier))
            pass_zero_worklist.sort(key=lambda item: _PASS0_TIER_RANK[item[1]])

        for changed_file, tier in pass_zero_worklist:
            # Tier → model (the cost lever, DECISIONS.md#041): STANDARD routes to
            # standard_analyze_model (Haiku by default), DEEP stays on analyze_model (Sonnet).
            model_for_file = _model_for_tier(
                tier,
                analyze_model=analyze_model,
                standard_analyze_model=standard_analyze_model,
            )

            # High-risk files (a blatant CRITICAL-class signature in their ADDED
            # lines, per policy.recall) may draw the reserve on top of the general
            # pool; benign files see only the general pool. This is what keeps the
            # CRITICAL command_injection from being starved behind benign DEEP
            # files (the verified PR #8 failure).
            is_high_risk = bool(scan_added_lines_for_risk(changed_file.patch))
            available_budget_tokens = (
                remaining_general_tokens + remaining_reserved_tokens
                if is_high_risk
                else remaining_general_tokens
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
                remaining_budget_tokens=available_budget_tokens,
                # Pass-0 PR-diff files ONLY — trace-fetched files (pass 1,
                # `_process_one_trace_fetched_file`) have no changed-scope
                # set, so the filter never evaluates there by design. The
                # analyze cache is pass-0-only for the same reason.
                trivial_scope_filter_enabled=trivial_scope_filter_enabled,
                analyze_cache_store=analyze_cache_store,
                cache_scope=cache_scope,
                cache_mode=cache_mode,
            )

            if file_outcome.parser_result is not None:
                # LLM call was made; parser ran.
                n_llm_calls += 1
                if tier is ReviewTier.STANDARD:
                    standard_model_used = standard_analyze_model
                n_proposals_seen += file_outcome.parser_result.counters.n_proposals_seen
                # OBSERVED findings fire real FindingEvents, so they ride
                # n_findings_emitted; n_findings_observed tracks them for the
                # accounting equation's subtraction (they are not proposals).
                n_findings_emitted += (
                    file_outcome.parser_result.counters.n_findings_emitted
                    + file_outcome.parser_result.counters.n_findings_observed
                )
                n_findings_observed += file_outcome.parser_result.counters.n_findings_observed
                n_proposals_superseded_by_observed += (
                    file_outcome.parser_result.counters.n_proposals_superseded_by_observed
                )
                n_proposals_rejected += file_outcome.parser_result.counters.n_proposals_rejected
                n_responses_rejected += file_outcome.parser_result.counters.n_responses_rejected
                n_trace_candidates_emitted += (
                    file_outcome.parser_result.counters.n_trace_candidates_emitted
                )
                n_trace_candidates_dropped_malformed += (
                    file_outcome.parser_result.counters.n_trace_candidates_dropped_malformed
                )
                _admit_with_dedup(
                    file_outcome.parser_result.admitted_findings,
                    admitted_findings,
                    admitted_keys_seen,
                )
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
            elif file_outcome.served_result is not None:
                # Cache-served hit (Stage B): findings reconstructed from the
                # cache, NO LLM call. They ride n_findings_emitted (real
                # FindingEvents fired) AND n_findings_served (so the proposal-
                # accounting equation subtracts them — served findings have no
                # proposal lifecycle). n_llm_calls is untouched.
                served = file_outcome.served_result
                n_findings_emitted += len(served.admitted_findings)
                n_findings_served += len(served.admitted_findings)
                # n_trace_candidates_emitted stays parser/model-emitted (pre-dedup);
                # served candidates were NOT emitted this pass and fire no per-item
                # event (unlike served FINDINGS, which re-emit FindingEvents) — their
                # audit trace is CacheServeEvent.served_trace_candidates. They still
                # extend into state below for the trace loop.
                _admit_with_dedup(served.admitted_findings, admitted_findings, admitted_keys_seen)
                trace_candidates.extend(served.trace_candidates)

            total_input_tokens += file_outcome.input_tokens
            total_output_tokens += file_outcome.output_tokens
            total_cache_read_tokens += file_outcome.cache_read_tokens
            total_cache_write_tokens += file_outcome.cache_write_tokens
            total_cost_decimal += file_outcome.cost_decimal
            # Reserved-then-general: a high-risk file spends its DEDICATED reserve
            # first, dipping into general only on overflow; a benign file draws
            # general only and never touches the reserve. Dedicating the reserve to
            # high-risk files (rather than spending general first) is what keeps it
            # from being wasted: under a general-first rule, high-risk files that
            # iterate EARLY would spend general, leave the reserve unused, and let a
            # later benign file starve that the reserve could have covered — lower
            # throughput AND a weaker high-risk guarantee. Skipped files have
            # `estimated_tokens == 0`, so this deducts nothing.
            spent_tokens = file_outcome.estimated_tokens
            if is_high_risk:
                from_reserved = min(spent_tokens, remaining_reserved_tokens)
                remaining_reserved_tokens -= from_reserved
                remaining_general_tokens -= spent_tokens - from_reserved
            else:
                remaining_general_tokens -= spent_tokens

            if file_outcome.parse_status == "skipped":
                files_skipped.append(changed_file.path)
                if file_outcome.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED:
                    budget_skip_count += 1
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
                # Trace-fetched files don't run the OBSERVED producer today, so
                # n_findings_observed / n_proposals_superseded_by_observed are 0
                # here; aggregated symmetrically with the main path so accounting
                # stays correct if that ever changes.
                n_findings_emitted += (
                    file_outcome.parser_result.counters.n_findings_emitted
                    + file_outcome.parser_result.counters.n_findings_observed
                )
                n_findings_observed += file_outcome.parser_result.counters.n_findings_observed
                n_proposals_superseded_by_observed += (
                    file_outcome.parser_result.counters.n_proposals_superseded_by_observed
                )
                n_proposals_rejected += file_outcome.parser_result.counters.n_proposals_rejected
                n_responses_rejected += file_outcome.parser_result.counters.n_responses_rejected
                n_trace_candidates_emitted += (
                    file_outcome.parser_result.counters.n_trace_candidates_emitted
                )
                n_trace_candidates_dropped_malformed += (
                    file_outcome.parser_result.counters.n_trace_candidates_dropped_malformed
                )
                _admit_with_dedup(
                    file_outcome.parser_result.admitted_findings,
                    admitted_findings,
                    admitted_keys_seen,
                )
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

    # FUP-044 ext 3 starvation anomaly: if pass-0 starved >= threshold files on the
    # per-review budget, emit one COST_BUDGET_STARVATION anomaly so operators see
    # the structural pattern instead of counting individual FileExaminationEvent
    # skips. Scope: pass-0 changed-file set (the verified failure mode; pass-1
    # trace-fetched starvation is out of V1 scope). Best-effort — observability must
    # not fail the review, so an emit failure is logged, never raised (unlike
    # synthesize's correctness-critical divergence anomaly). Idempotent on
    # (review_id, rule_name) via the partial unique index.
    if budget_skip_count >= COST_BUDGET_STARVATION_THRESHOLD:
        try:
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.COST_BUDGET_STARVATION,
                severity=AnomalySeverity.MEDIUM,
                details={
                    "budget_skipped_count": budget_skip_count,
                    "total_review_budget_tokens": total_review_budget_tokens,
                    "pass_index": pass_index,
                },
                is_eval=state.is_eval,
            )
        except Exception:
            logger.exception("analyze_cost_budget_starvation_anomaly_emit_failed")

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
            n_findings_served=n_findings_served,
            n_findings_observed=n_findings_observed,
            n_proposals_superseded_by_observed=n_proposals_superseded_by_observed,
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
class _ServedResult:
    """Cache-served findings + trace candidates for one file (Stage B serve
    flip). A served hit populates this on `_FileOutcome` INSTEAD of
    `parser_result` (which stays None — no LLM call), so the main loop
    accumulates the findings WITHOUT counting an LLM call: they ride
    `n_findings_served` (subtracted from the proposal-accounting equation) and
    `n_findings_emitted` (real `FindingEvent`s fired)."""

    admitted_findings: tuple[ReviewFinding, ...]
    trace_candidates: tuple[TraceCandidate, ...]


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
    # Stage B serve flip: non-None ONLY on a cache-served hit (parser_result is
    # then None — no LLM call). The main loop's served branch consumes it.
    served_result: _ServedResult | None = None
    # Stage 2 (FUP-044 ext 3): the skip reason, mirrored from the
    # FileExaminationEvent onto the in-memory outcome so the main loop can count
    # COST_BUDGET_EXHAUSTED skips for the starvation anomaly. None on any
    # non-skipped outcome (parse_status != "skipped").
    skip_reason: SkipReason | None = None


class _ServeReconstructionError(Exception):
    """A live cache payload could not be reconstructed into served findings.

    Raised by `_serve_cache_hit` BEFORE any emit when the cached payload is
    malformed (missing key, null/non-iterable container, or a finding/candidate
    dict that fails validation). The serve short-circuit catches it and degrades
    to a real LLM call (FUP-177 edge 2) — degrade, never abort the review.
    """


async def _serve_cache_hit(
    *,
    entry: CacheEntry,
    cache_key: str,
    review_id: UUID,
    installation_id: int,
    repo_id: int,
    is_eval: bool,
    file_path: str,
    included_scope_units: tuple[ScopeUnit, ...],
    analyze_event_sink: AnalyzeEventSink,
    file_examination_sink: FileExaminationSink,
) -> _FileOutcome:
    """Serve a live analyze-cache hit (Stage B): reconstruct the cached findings
    + trace candidates onto THIS review, emit the audit trail, and return a
    served `_FileOutcome` — NO LLM call.

    Findings re-mint `finding_id` DETERMINISTICALLY (`compute_served_finding_id`)
    and re-stamp `review_id` / `installation_id` onto the new review, preserving
    all content. The rebuild routes through `ReviewFinding.model_validate` (NOT
    `model_copy`), so every validator re-runs — content_hash re-verified,
    severity re-checked against LIVE policy, proof boundary re-enforced: cache
    content is never trusted past the schema floor. Trace candidates need no
    re-mint (`candidate_id` is content-derived, review-independent).

    Served findings re-emit `FindingEvent`s (per-review self-containment for
    replay); served trace candidates emit no per-item event — their identity
    rides the `CacheServeEvent`, their full content (incl. `reason`) rides the
    returned `_ServedResult` into state and purges with the cache row. The file
    emits ONE `FileExaminationEvent(parse_status="clean")` and NO `LLMCallEvent`.
    """
    # Reconstruct from the cached payload INSIDE a degrade guard (FUP-177 edge 2):
    # a malformed-but-live payload (missing key, null/non-iterable container, or a
    # dict that fails validation) raises `_ServeReconstructionError`, which the
    # serve short-circuit catches and degrades to a real LLM call rather than
    # aborting the review. Raised BEFORE any emit, so no partial serve events
    # land. `ValueError` covers pydantic's `ValidationError` (a ValueError
    # subclass); the message carries only the exception type + path, never content.
    try:
        served_findings = tuple(
            ReviewFinding.model_validate(
                {
                    **dump,
                    "review_id": str(review_id),
                    "installation_id": installation_id,
                    "finding_id": str(
                        compute_served_finding_id(
                            review_id=review_id,
                            content_hash=dump["content_hash"],
                        )
                    ),
                    # Force analyze-time lifecycle state regardless of the cached
                    # dump: the HITL override triplet + publish_destination are set
                    # DOWNSTREAM per review (HITL re-gates, publish re-routes), never
                    # served. Today's writer caches pre-HITL findings (all None), but
                    # the serve boundary enforces it locally (defense-in-depth).
                    "publish_destination": None,
                    "original_severity": None,
                    "override_reason": None,
                    "overrider_id": None,
                }
            )
            for dump in entry.payload["findings"]
        )
        served_candidates = tuple(
            TraceCandidate.model_validate(dump) for dump in entry.payload["trace_candidates"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _ServeReconstructionError(
            f"serve reconstruction failed for {file_path}: {type(exc).__name__}"
        ) from exc

    # Pre-emit uniqueness gate (FUP-177): a malformed-but-live payload with a
    # duplicate-finding set would, under the (review_id, content_hash) re-mint,
    # produce duplicate finding_ids — appending duplicate FindingEvents / hitting
    # persister conflicts BEFORE the `AnalysisRound` validators (which run only
    # AFTER this returns) reject the round. Enforce the round's uniqueness
    # invariants HERE, before any emit; a violation raises into the degrade guard.
    # Two arms suffice: finding_id is uuid5(review_id, content_hash), so
    # finding_id-uniqueness IS content_hash-uniqueness for the served set (a
    # separate content_hash arm would be redundant) — finding_id is the
    # emit-collision key. proposal_hash is the independent second invariant
    # (`AnalysisRound._enforce_findings_proposal_hash_unique`): two findings with
    # distinct content can still share a proposal_hash.
    if len({f.finding_id for f in served_findings}) != len(served_findings) or len(
        {f.proposal_hash for f in served_findings}
    ) != len(served_findings):
        raise _ServeReconstructionError(
            f"served set for {file_path} violates per-round uniqueness "
            "(duplicate finding_id / proposal_hash)"
        )

    await analyze_event_sink.emit_cache_serve(
        CacheServeEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=file_path,
            cache_key=cache_key,
            installation_id=installation_id,
            repo_id=repo_id,
            served_finding_count=len(served_findings),
            context_summary=_build_context_manifest(
                file_path, included_scope_units, inclusion_reason="changed_scope"
            ),
            served_trace_candidates=tuple(
                ServedTraceCandidateRef(
                    candidate_id=c.candidate_id,
                    source_proposal_hash=c.source_proposal_hash,
                    import_string=c.import_string,
                )
                for c in served_candidates
            ),
            source_review_id=entry.source_review_id,
            source_cache_created_at=entry.created_at,
        )
    )

    # SINGLE FileExaminationEvent (clean — parse + prompt assembly genuinely ran;
    # only the provider call didn't). The serve short-circuit returns before the
    # normal-path step-3e emission, so it emits here itself.
    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=file_path,
    )

    # Re-emit one FindingEvent per served finding so this review's audit/findings
    # tables are self-contained for replay (the cache stores content; audit rows
    # are per-review). The deterministic finding_id keeps the persister's
    # no-resurrection content-row guard correct under checkpoint replay.
    for finding in served_findings:
        await analyze_event_sink.emit_finding(finding, is_eval=is_eval)

    return _FileOutcome(
        parse_status="clean",
        parser_result=None,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        cost_decimal=Decimal("0"),
        estimated_tokens=0,
        served_result=_ServedResult(
            admitted_findings=served_findings,
            trace_candidates=served_candidates,
        ),
    )


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
    triviality_context: FileTrivialityContext,
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

    entries: list[ScopeExclusionEntry] = []
    for su in included_scope_units:
        changed = changed_line_spans(su, patched_file, head_source=content, base_source=base_text)
        verdict = classify_scope_triviality(changed, triviality_context)
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
    source_bytes: bytes,
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
        skip_reason=skip_reason,
    )


def _build_context_manifest(
    file_path: str,
    scope_units: Iterable[ScopeUnit],
    *,
    inclusion_reason: Literal["changed_scope", "same_file_context", "trace_expansion"],
) -> tuple[ContextManifestEntry, ...]:
    """One `ContextManifestEntry` per scope unit — the `context_summary` that rides
    every analyze emit (LLMCallEvent on the clean + trace paths, CacheServeEvent on
    the serve path). Extracted (FUP-178) so the three paths cannot drift on the
    manifest shape; the clean path keeps its own `degraded_mode` empty-manifest
    guard at the call site."""
    return tuple(
        ContextManifestEntry(
            file_path=file_path,
            scope_unit_name=su.qualified_name or su.name,
            line_start=su.line_start,
            line_end=su.line_end,
            inclusion_reason=inclusion_reason,
        )
        for su in scope_units
    )


async def _emit_examination(
    *,
    file_examination_sink: FileExaminationSink,
    review_id: UUID,
    is_eval: bool,
    file_path: str,
    parse_status: _ParseStatus = "clean",
) -> None:
    """Emit the single `FileExaminationEvent` for a KEPT (non-skipped) file —
    sibling to `_emit_skip` (FUP-178). The clean, trace, and serve paths share this
    one emission shape; `parse_status` defaults to "clean" (serve + trace), and the
    clean path passes its own `parse_status_for_event`."""
    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=file_path,
            examination_type="analyze",
            node_id="analyze",
            parse_status=parse_status,
            skip_reason=None,
        )
    )


def _admit_with_dedup(
    findings: Iterable[ReviewFinding],
    admitted_findings: list[ReviewFinding],
    admitted_keys_seen: set[tuple[str, str]],
) -> None:
    """Append each finding to `admitted_findings` unless its
    `(content_hash, proposal_hash)` key was already admitted this round (FUP-178).
    The round-build merges findings from several source contexts (cold parse,
    served payload, trace-fetched files); `AnalysisRound` enforces uniqueness on
    both keys, so this dedup stops a logically-identical finding from being
    double-admitted before the round validators run."""
    for f in findings:
        key = (f.content_hash, f.proposal_hash)
        if key in admitted_keys_seen:
            continue
        admitted_keys_seen.add(key)
        admitted_findings.append(f)


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
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_scope: CacheScope | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
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
                included_scope_units=included_scope_units, source_bytes=content_bytes
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
    # FUP-170: one post-cost-gate head parse feeds BOTH the trivial-scope
    # classification (below) and the FUP-162 parameterized-call scan.
    # `parse_python` stays a separate PRE-gate parse (it fed degradation + the
    # token estimate); this runs strictly AFTER the cost gate, so a cost-skipped
    # file never reaches it (COST_BUDGET_EXHAUSTED-before-classification holds).
    # The scan rides every clean file (None in degraded mode — also exactly when
    # the file is not cacheable, so `parameterized_call_scan is not None` IS the
    # clean-mode cache gate below). The SAME scan object feeds BOTH the cache-key
    # digest AND the admission veto in parse_analyze_response, so the keyed and
    # admitted inputs can never fork (FUP-171 anti-fork). `compute_triviality`
    # mirrors the classification gate so triviality (+ its base parse) builds
    # only when there's a patch and included scopes.
    want_triviality = not degraded_mode and patched_file is not None and bool(included_scope_units)
    triviality_context, parameterized_call_scan = extract_triviality_and_scan(
        content_bytes,
        changed_file.content_base.encode("utf-8")
        if (want_triviality and changed_file.content_base is not None)
        else None,
        compute_triviality=want_triviality,
        degraded=degraded_mode,
    )

    # triviality_context is non-None iff want_triviality (which already implies a
    # patch + non-empty scopes); the patched_file re-check narrows the type.
    if triviality_context is not None and patched_file is not None:
        entries = _classify_included_scopes(
            changed_file=changed_file,
            content=content,
            triviality_context=triviality_context,
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
                    included_scope_units=included_scope_units, source_bytes=content_bytes
                ),
                query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
                diff_hunks=_concat_clipped_hunks(included_clipped_hunks),
                pass_index=pass_index,
            )

    # `parameterized_call_scan` was produced by the FUP-170 bundle above (None in
    # degraded mode); the clean-mode cache lookup below gates on it being non-None.

    # Step 3d-ter: analyze-cache shadow lookup (pass-0 clean mode only;
    # specs/2026-06-11-file-hash-analyze-cache.md). The key is computed
    # over the FINAL rendered parts — post trivial-filter re-render under
    # enforcement — so the keyed prompt is the prompt actually sent. The
    # all-trivial skip returned above without keying: skip outcomes are
    # NEVER memoized (no prompt bytes to pin them). Shadow semantics: a
    # would_hit still calls the model; nothing is served. The would-hit
    # rate accumulated here is the serve flip's evidence.
    cache_key: str | None = None
    cache_would_hit = False
    if (
        analyze_cache_store is not None
        and cache_scope is not None
        and parameterized_call_scan is not None
    ):
        cache_key = compute_analyze_cache_key(
            system_prompt=parts.system_prompt,
            user_prompt=parts.user_prompt,
            installation_id=cache_scope.installation_id,
            repo_id=cache_scope.repo_id,
            model=analyze_model,
            prompt_template_version=analyze_prompt.VERSION,
            trivial_filter_version=TRIVIAL_FILTER_VERSION,
            query_registry_digest=QUERY_REGISTRY_DIGEST,
            active_policy_version=active_policy_version,
            analyze_parser_version=ANALYZE_PARSER_VERSION,
            response_format_digest=ANALYZE_RESPONSE_FORMAT_DIGEST,
            parameterized_call_scan_digest=scan_digest(parameterized_call_scan),
            observed_producer_version=OBSERVED_PRODUCER_VERSION,
        )
        try:
            # Self-hit exclusion: a crash/retry re-execution of this node
            # must not read its own first attempt's writes as hits — that
            # would inflate the would-hit rate (the serve flip's evidence)
            # and, under serve, serve a review its own partial output.
            cache_entry = await analyze_cache_store.lookup(
                cache_key, is_eval=cache_scope.is_eval, exclude_source_review_id=review_id
            )
        except CacheStoreError:
            # Contained: a failed lookup degrades to a real LLM call (shadow OR
            # serve) — NEVER a silent skip of findings. No CacheLookupEvent (the
            # lookup didn't complete; a fabricated "miss" would be false audit
            # history). cache_key cleared so step 3g's write gate skips too.
            logger.warning(
                "analyze-cache lookup failed; cache skipped for %s",
                changed_file.path,
                exc_info=True,
            )
            cache_key = None
        else:
            if cache_mode is CacheMode.SERVE and cache_entry is not None:
                # SERVE: reconstruct the cached findings onto this review and
                # short-circuit the LLM call + parser block entirely. Every
                # deterministic downstream gate still runs on the served
                # findings (reducers → synthesize → HITL → publish); the cache
                # replaces exactly the analyze LLM call.
                try:
                    return await _serve_cache_hit(
                        entry=cache_entry,
                        cache_key=cache_key,
                        review_id=review_id,
                        installation_id=installation_id,
                        repo_id=cache_scope.repo_id,
                        is_eval=is_eval,
                        file_path=changed_file.path,
                        included_scope_units=included_scope_units,
                        analyze_event_sink=analyze_event_sink,
                        file_examination_sink=file_examination_sink,
                    )
                except _ServeReconstructionError:
                    # Malformed-but-LIVE cached payload (incl. a duplicate-finding
                    # set): degrade to a real LLM call instead of aborting the
                    # review — degrade-not-lose-findings (FUP-177 edge 2). The raise
                    # lands BEFORE any serve emit, so no partial events leaked. Do
                    # NOT fabricate a CacheLookupEvent: the lookup DID find a row,
                    # so a "miss" is false history and a "would_hit" implies serve
                    # worked. Clear cache_key — the telemetry emit AND the step-3g
                    # write both gate on it — so this degrades silently (the log is
                    # the signal), like the lookup-error path. The live poisoned row
                    # persists until expiry (write refreshes only EXPIRED rows);
                    # re-serving it just degrades again, safely. Unreachable via the
                    # V1 writer (valid payloads; the key pins policy/registry).
                    logger.warning(
                        "analyze-cache serve reconstruction failed; degrading to LLM for %s",
                        changed_file.path,
                        exc_info=True,
                    )
                    cache_key = None
            # SHADOW (any outcome) or SERVE-miss: record would-hit/miss telemetry
            # and fall through to the model call. A serve-miss is a real miss → the
            # model runs and step 3g writes. Skipped when a serve reconstruction
            # degraded above (cache_key cleared): no fabricated miss/would_hit for a
            # row that WAS found but could not be served.
            if cache_key is not None:
                cache_would_hit = cache_entry is not None
                await analyze_event_sink.emit_cache_lookup(
                    CacheLookupEvent(
                        review_id=review_id,
                        is_eval=is_eval,
                        file_path=changed_file.path,
                        outcome="would_hit" if cache_would_hit else "miss",
                        cache_key=cache_key,
                    )
                )

    # Step 3e: SINGLE FileExaminationEvent emission point.
    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=changed_file.path,
        parse_status=parse_status_for_event,
    )

    # Step 3f: LLM call + response parse.
    # One ContextManifestEntry per included scope unit for clean+full_llm.
    # Empty tuple for degraded — `_enforce_context_for_scope_nodes`
    # special-cases this.
    context_summary: tuple[ContextManifestEntry, ...] = (
        ()
        if degraded_mode
        else _build_context_manifest(
            changed_file.path, included_scope_units, inclusion_reason="changed_scope"
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
        # Constrained decoding (FUP-096): the pinned analyze response schema
        # rides every analyze call — pass-0 and trace-fetched alike — so the
        # API guarantees syntactically valid, shape-conforming JSON. The
        # parser's rejection path stays (refusal/max_tokens escapes).
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
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
        # FUP-162 veto facts, hoisted above the cache key (FUP-171): the SAME
        # scan object keyed the cache entry, so admission and key never fork.
        parameterized_call_scan=parameterized_call_scan,
    )

    # Deterministic OBSERVED-tier findings (Cost Lever 3): augment the LLM's
    # JUDGED findings with structural security-query matches. Clean-parse only
    # (the producer never runs in degraded mode); merged into
    # `parser_result.admitted_findings` BEFORE the emit/cache/return below, so
    # they ride the audit stream, the cache payload (serve reconstructs them),
    # and the round identically to LLM findings. Deduped by content_hash —
    # `AnalysisRound` requires unique content_hashes, and an OBSERVED match the
    # model ALSO flagged (same file/lines/type) is redundant. signal_only: the
    # LLM still ran; OBSERVED augments it, never skips it.
    #
    # HEAD-CONTENT ONLY (defense-in-depth): the OBSERVED producer is head-content
    # proof — its queries run on head, and `evidence` + the shadow event's
    # `side="head"` are head-derived. A normal `removed` file already skips upstream
    # at NO_CHANGED_SCOPE_UNITS (no added lines; `decide_degradation`) and never
    # reaches here. This gate makes the head-content dependency explicit AT the
    # block and guards the one residual path it would NOT catch: a `content_head is
    # None` file that still carries added lines (a ChangedFile-invariant violation)
    # would otherwise run OBSERVED on the `content_base` fallback and flag deleted
    # code with base lines treated as head.
    if not degraded_mode and changed_file.content_head is not None:
        # Single deterministic OBSERVED query pass; the findings producer and
        # (the routing increment's) skip-coverage check both read these matches.
        observed_matches = run_observed_matches(
            file_path=changed_file.path,
            head_content=content,
            included_scope_units=included_scope_units,
        )
        observed_findings = produce_observed_findings(
            observed_matches,
            file_path=changed_file.path,
            review_id=review_id,
            installation_id=installation_id,
            active_policy_version=active_policy_version,
        )
        if observed_findings:
            # prefer-OBSERVED (DECISIONS.md#054): a producer OBSERVED finding that
            # collides (same content_hash = file+line+finding_type) with an
            # admitted model JUDGED proposal EVICTS the JUDGED in place — keeping
            # the stronger, replay-verifiable query_match_id. A collision with an
            # already-OBSERVED/INFERRED admitted finding keeps the incumbent (its
            # proof is not lost) and drops the producer duplicate. Non-colliding
            # producer findings append, as before. The swap MUST happen here,
            # before round/FindingEvent construction: content_hash excludes
            # evidence_tier, so two same-hash findings would trip
            # AnalysisRound._enforce_findings_unique.
            admitted_list = list(parser_result.admitted_findings)
            index_by_hash = {f.content_hash: i for i, f in enumerate(admitted_list)}
            fresh: list[ReviewFinding] = []
            fresh_hashes: set[str] = set()
            n_superseded = 0
            for observed_finding in observed_findings:
                content_hash = observed_finding.content_hash
                collide_idx = index_by_hash.get(content_hash)
                if collide_idx is not None:
                    if admitted_list[collide_idx].evidence_tier is EvidenceTier.JUDGED:
                        admitted_list[collide_idx] = observed_finding
                        n_superseded += 1
                    # else: incumbent already carries structural proof — keep it,
                    # drop the producer duplicate.
                elif content_hash not in fresh_hashes:
                    fresh.append(observed_finding)
                    fresh_hashes.add(content_hash)
            n_observed = len(fresh) + n_superseded
            if n_observed:
                # OBSERVED findings (fresh + swapped-in) fire real FindingEvents
                # and ride the aggregate n_findings_emitted, but are NOT proposals
                # — subtracted via n_findings_observed (like n_findings_served).
                # Each swap also evicts a JUDGED proposal: drop it from the parser
                # n_findings_emitted (one fewer proposal-finding) and account it via
                # n_proposals_superseded_by_observed, which the accounting equation
                # ADDS (a proposal with no surviving finding — same side as
                # n_proposals_rejected).
                parser_result = replace(
                    parser_result,
                    admitted_findings=tuple(admitted_list) + tuple(fresh),
                    counters=replace(
                        parser_result.counters,
                        n_findings_emitted=parser_result.counters.n_findings_emitted - n_superseded,
                        n_findings_observed=n_observed,
                        n_proposals_superseded_by_observed=n_superseded,
                    ),
                )

        # Skip-routing SHADOW telemetry (Cost Lever 3, DECISIONS.md#049): record
        # the per-file skip-eligibility decision from the SAME OBSERVED matches.
        # V1 RECORDS ONLY — it never skips the LLM (which already ran above);
        # enforcement is the later evidence-gated flip. Needs the diff, so it is
        # gated on patched_file (None for binary / oversized / absent patches).
        if patched_file is not None:
            shadow_event = compute_observed_skip_shadow(
                observed_matches,
                file_path=changed_file.path,
                included_scope_units=included_scope_units,
                patched_file=patched_file,
                head_source=content,
                base_source=changed_file.content_base,
                review_id=review_id,
                is_eval=is_eval,
            )
            if shadow_event is not None:
                await analyze_event_sink.emit_observed_skip_shadow(shadow_event)

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

    # Step 3g: analyze-cache write-on-miss. Only completed clean-mode
    # calls populate the store — a response-level rejection has no
    # admitted outcome to cache (zero findings, by contrast, IS a valid
    # cacheable outcome), and a `max_tokens`-truncated response is never
    # cached even when its JSON happens to validate: the finding set may
    # be silently incomplete, and memoizing it would serve the truncated
    # outcome for the row's whole lifetime. The payload carries the
    # content tier: admitted finding content (pre-HITL, policy-stamped)
    # + FULL trace candidates including their LLM-derived `reason` —
    # content lives in the retention-bound cache row, never on the
    # audit events.
    if (
        analyze_cache_store is not None
        and cache_scope is not None
        and cache_key is not None
        and not cache_would_hit
        and parser_result.response_rejection is None
        and response.finish_reason != "max_tokens"
    ):
        try:
            await analyze_cache_store.write(
                cache_key=cache_key,
                scope=cache_scope,
                source_review_id=review_id,
                file_path=changed_file.path,
                payload={
                    "findings": [
                        f.model_dump(mode="json") for f in parser_result.admitted_findings
                    ],
                    "trace_candidates": [
                        c.model_dump(mode="json") for c in parser_result.trace_candidates
                    ],
                },
                model=analyze_model,
                prompt_template_version=analyze_prompt.VERSION,
                trivial_filter_version=TRIVIAL_FILTER_VERSION,
                query_registry_digest=QUERY_REGISTRY_DIGEST,
                active_policy_version=active_policy_version,
                analyze_parser_version=ANALYZE_PARSER_VERSION,
                prompt_hash=_canonical_prompt_hash(
                    system_prompt=parts.system_prompt, user_prompt=parts.user_prompt
                ),
            )
        except CacheStoreError:
            # Contained: the review's findings are already emitted; a
            # failed cache write loses nothing but one memoization.
            logger.warning(
                "analyze-cache write failed; outcome not cached for %s",
                changed_file.path,
                exc_info=True,
            )

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
            included_scope_units=included_scope_units, source_bytes=content_bytes
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

    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=fetched_file.path,
    )

    # `inclusion_reason="trace_expansion"` per the ContextManifestEntry
    # Literal — names the post-trace expansion-pass inclusion shape
    # (scope units from a trace-fetched file). The Literal predates the
    # trace-node arc; using it here closes the loop without a schema
    # change.
    context_summary: tuple[ContextManifestEntry, ...] = _build_context_manifest(
        fetched_file.path, included_scope_units, inclusion_reason="trace_expansion"
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
        # Constrained decoding (FUP-096): the pinned analyze response schema
        # rides every analyze call — pass-0 and trace-fetched alike — so the
        # API guarantees syntactically valid, shape-conforming JSON. The
        # parser's rejection path stays (refusal/max_tokens escapes).
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
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
        # FUP-162 veto facts — trace-fetched files run with has_error scope
        # units filtered out, not a whole-file degraded mode; the scan
        # itself returns empty for any error-bearing tree, so a partially
        # erroring file disables the veto rather than trusting recovery.
        parameterized_call_scan=scan_parameterized_calls(content_bytes),
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
