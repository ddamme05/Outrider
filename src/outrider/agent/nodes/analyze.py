# See DECISIONS.md#018, #025 (proposal_hash threaded through admitted
# and rejected lift sites per #025 point 1).
# See also DECISIONS.md#024-trace-candidates-are-dotted-python-import-strings-v1
# (admission path; same-file inline handling).
"""Review changed files and audit accepted findings.

`analyze` plans the initial pass and runs the sequential trace pass;
`analyze_file` reviews one file. `analyze_aggregate` folds worker results
in its own module. `build_graph` binds runtime dependencies, and
`analyze_parser.parse_analyze_response` owns finding admission.

File outcomes (spec §7 step 3a):

- `clean+full_llm`: changed scopes parse without errors and pass the budget gate.
- `degraded+degraded_llm`: tree errors intersect changed scopes or added lines
  without recovered scopes, or eligible module-level OBSERVED matches occur
  outside changed scopes (`module_level_observed_match`, DECISIONS.md#062).
  The module route keeps clean parse status and merges deterministic findings.
  Model proposals must be JUDGED, stay within file bounds, and intersect
  added-line context.
- `skipped+NO_REVIEWABLE_CONTEXT`: no content, or parse failure without added text.
  Current intake and `ChangedFile` validation make these cases unreachable.
- `skipped+NO_CHANGED_SCOPE_UNITS`: no changed scopes or patch, unless eligible
  module-level matches select the degraded prompt route.
- `skipped+COST_BUDGET_EXHAUSTED`: prompt estimate exceeds the available budget.
- `skipped+UNSUPPORTED_LANGUAGE`: no registered adapter for the file extension.
- `skipped+ALL_SCOPES_TRIVIAL`: enforcement excludes every admitted scope;
  the budget gate runs first. Shadow mode never takes this route.
- `skipped+PATCH_HEAD_MISALIGNED`: added patch lines disagree with head content;
  checked before routing because their coordinates cannot be trusted.

Parser-stage skips retain their reason on `FileExaminationEvent`. Scope
selection requires added lines and clipped hunks; deletion-only edits can
still miss review (FUP-050).

The `failed+degraded_llm` route remains for a future raw-byte intake path
(FUP-053). Current intake rejects invalid UTF-8 and supplies decoded strings.

Provider failures propagate. Completed-file events remain persisted, and an
interrupted phase has no end marker; failures must not become skip outcomes.
Completion counters come from parser results, never from re-reading the audit
stream, and `AnalyzeCompletedEvent` validates proposal accounting.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID  # noqa: TC003 — AnalyzeWorkerPayload resolves this at model build

from langgraph.types import Command, Send
from pydantic import BaseModel, ConfigDict, Field

from outrider.agent.nodes.analyze_budget import (
    FileBudgetRequest,
    plan_file_budgets,
    proxy_estimate_tokens,
)
from outrider.agent.nodes.analyze_observed import (
    OBSERVED_PRODUCER_VERSION,
    ObservedMatch,
    compute_observed_skip_shadow,
    import_bindings_digest,
    lexical_bindings_digest,
    module_admission_digest,
    module_admission_inputs,
    module_level_observed_matches,
    produce_observed_findings,
    run_observed_matches,
)
from outrider.agent.nodes.analyze_parser import (
    ANALYZE_PARSER_VERSION,
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    from_import_map_digest,
    parse_analyze_response,
)
from outrider.agent.nodes.analyze_worker_build import (
    worker_outcome_from_observed_coverage,
    worker_outcome_from_observed_skip,
    worker_outcome_from_parser,
    worker_outcome_from_plain_skip,
    worker_outcome_from_serve,
)
from outrider.agent.nodes.cache_config import CacheMode
from outrider.agent.nodes.degradation import (
    _ParseStatus,
    decide_degradation,
)
from outrider.agent.nodes.finding_cap import (
    _collapse_by_content_hash,
    admit_with_pair_dedup,
    cap_findings_by_severity,
)
from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.ast_facts.analyze_bundle import extract_triviality_and_scan
from outrider.ast_facts.models import SkipReason, TrivialityReason
from outrider.ast_facts.parameterized_calls import (
    ParameterizedCallScan,
    scan_digest,
    scan_parameterized_calls,
)
from outrider.ast_facts.registry import (
    JAVASCRIPT_EXTENSIONS,
    PYTHON_EXTENSIONS,
    TYPESCRIPT_DIALECT_BY_EXTENSION,
    get_adapter_factory,
    parse_source,
    supported_extensions,
)
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
    DegradationReason,
    FileExaminationEvent,
    FindingProposalRejectedEvent,
    ObservedSkipShadowEvent,
    ObservedSubsumedMatch,
    ReviewPhaseEvent,
    ScopeExclusionEntry,
    ScopeExclusionEvent,
    ServedTraceCandidateRef,
)
from outrider.cache import CacheScope, CacheStoreError, compute_analyze_cache_key
from outrider.coordinates import (
    CoordinateError,
    added_line_byte_ranges,
    bound_diff_hunks_text,
    changed_line_spans,
    extract_scope_unit_body,
    lookup_patched_file,
    patched_file_has_added_lines,
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
from outrider.policy.subsumption import SUBSUMES_DIGEST, subsumes
from outrider.prompts import analyze as analyze_prompt
from outrider.prompts import safe_code_fence
from outrider.queries import registry as query_registry
from outrider.queries.registry import OBSERVED_QUERY_IDS, QUERY_REGISTRY_DIGEST
from outrider.schemas import AnalysisRound, ReviewFinding, TraceCandidate
from outrider.schemas.analysis_round import MAX_FINDINGS_HARD_CAP, MAX_FINDINGS_PER_ROUND
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_FORMAT_DIGEST,
    ANALYZE_RESPONSE_SCHEMA_JSON,
)
from outrider.schemas.pr_context import ChangedFile  # noqa: TC001 — payload model field
from outrider.schemas.triage_result import ReviewTier

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from unidiff import PatchedFile

    from outrider.anomaly.sinks import AnomalySink
    from outrider.ast_facts.base import ImportPathResolver
    from outrider.ast_facts.models import ParseResult, ScopeUnit
    from outrider.audit.sinks import (
        AnalyzeEventSink,
        FileExaminationSink,
        PhaseEventSink,
    )
    from outrider.cache import AnalyzeCacheStore, CacheEntry
    from outrider.llm.base import LLMProvider, LLMResponse
    from outrider.schemas import (
        ReviewState,
        TraceFetchedFile,
    )
    from outrider.schemas.analyze_worker import AnalyzeWorkerOutcome
    from outrider.schemas.triage_result import TriageResult


logger = logging.getLogger(__name__)

# Limit each file's share of the review budget. Additional fairness policies
# (including per-installation budgets) remain deferred to FUP-044.
PER_FILE_CAP_FRACTION: Final[float] = 0.25

# Reserved tokens are available only to pass-0 high-risk files (policy.recall;
# specs/2026-06-17-analyze-cost-fairness.md Stage 1). They draw reserve first, then
# general tokens. This protects some funding from benign files without bypassing
# the class budget or per-file cap; exhausted high-risk files can still skip.
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

# AnalyzeConfig reads OUTRIDER_ANALYZE_REVIEW_BUDGET_TOKENS; lifespan passes the
# setting through build_graph. Direct callers may use this default.
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

# Default bound on in-flight `analyze_file` workers (the Send fan-out per
# specs/2026-07-05-parallel-analyze.md). The Send count itself is unbounded
# — every kept file gets a worker — but the gate built at build_graph
# admits at most this many worker BODIES at once PER (graph, event loop):
# on the single-loop production server that is graph-global across
# concurrent reviews; simultaneous multi-loop driving gets an independent
# cap per loop (the AnalyzeConcurrencyGate contract). Prompt-cache
# stampede note (spec): the first concurrent wave can each cache-WRITE
# the shared system prefix; the bounded 1.25× write overhead on at most
# (cap - 1) redundant writes per tier-model is accepted rather than
# serializing the first call.
ANALYZE_MAX_CONCURRENCY: Final[int] = 4

# Per-file render margin folded into the planner proxy's fixed overhead.
# The rendered user prompt carries per-file scaffolding that is NOT
# proportional to content bytes — the fired-query-id list (worst case
# ~170 tokens for the python catalog), safe_code_fence lines, scope-unit
# headers, and the file path — so `DUP_FACTOR × bytes` alone under-covers
# TINY files (the eval starvation fixture: proxy 17,419 vs real 17,449 →
# a spurious COST_BUDGET_EXHAUSTED skip on a funded-by-intent file). The
# margin over-reserves (utilization cost, measured via skip counters),
# never overspends; the proxy-covers-real calibration test is the
# regression gate for shrinking it.
_PROXY_RENDER_MARGIN_TOKENS: Final[int] = 384


def _estimate_tokens(text: str) -> int:
    """Estimate tokens from UTF-8 bytes with ceiling division by _BYTES_PER_TOKEN.

    Byte counting includes multibyte source text; codepoint counting would understate
    it. This is a budget proxy, not a tokenizer bound (FUP-049)."""
    byte_len = len(text.encode("utf-8"))
    return (byte_len + _BYTES_PER_TOKEN - 1) // _BYTES_PER_TOKEN


def _language_supported(path: str) -> bool:
    """True iff the ast_facts registry has an adapter for `path`'s
    extension — the analyze language gate (dispatch spec). Path-based
    because intake does not populate `ChangedFile.language`;
    `get_adapter_factory` normalizes case internally.
    """
    return get_adapter_factory(PurePosixPath(path).suffix) is not None


def _is_python_file(path: str) -> bool:
    """True iff `path` is Python source — the STAGE predicate, narrower
    than `_language_supported`: the parameterized-call veto scan and the
    trivial-scope classifier are Python-only surfaces and stay gated on
    this even after registry dispatch admits JS/TS files to the review
    flow. (Trace-candidate collection is per-language since the resolver
    spec — its admitted syntax comes from
    `_TRACE_CANDIDATE_FORM_BY_EXTENSION`, not this gate. The OBSERVED
    producer and the structural query-id set are per-language since the
    JS/TS OBSERVED catalog spec — their selection lives in
    `queries.registry`'s language-aware surface, not this gate.) Derived
    from the registry's extension
    group with the same case normalization as the registry gate — a
    case-sensitive copy here let `UTILS.PY` pass the gate but run with
    every Python-only stage silently disabled.
    """
    return PurePosixPath(path).suffix.lower() in PYTHON_EXTENSIONS


# Derived from the registry's extension groups so a newly registered
# language can't silently miss this table; the one genuinely local fact
# is `.jsx` → the finer jsx hint. The import-time totality assert below
# fails loud if the registry ever grows an extension this table lacks.
_FENCE_LANG_BY_EXTENSION: Final[dict[str, str]] = {
    **dict.fromkeys(PYTHON_EXTENSIONS, "python"),
    **dict.fromkeys(JAVASCRIPT_EXTENSIONS, "javascript"),
    ".jsx": "jsx",
    **TYPESCRIPT_DIALECT_BY_EXTENSION,
}

if set(_FENCE_LANG_BY_EXTENSION) != set(supported_extensions()):
    raise AssertionError(
        f"fence-lang table diverged from the registry: registry supports "
        f"{sorted(supported_extensions())} but fences cover "
        f"{sorted(_FENCE_LANG_BY_EXTENSION)}."
    )


# Per-language trace-candidate syntax (DECISIONS.md#024 Amended
# 2026-07-03), derived from the registry's extension groups exactly like
# the fence table above: a newly registered language must choose its
# candidate form here or fail loud at import time. "Not Python" does NOT
# imply "relative specifier" — a future Go/Rust adapter has neither form
# until it registers one, and defaulting it to specifier would silently
# swallow all its candidates into the malformed counter.
_TRACE_CANDIDATE_FORM_BY_EXTENSION: Final[dict[str, Literal["module", "specifier"]]] = {
    **dict.fromkeys(PYTHON_EXTENSIONS, "module"),
    **dict.fromkeys(JAVASCRIPT_EXTENSIONS, "specifier"),
    **dict.fromkeys(TYPESCRIPT_DIALECT_BY_EXTENSION, "specifier"),
}

if set(_TRACE_CANDIDATE_FORM_BY_EXTENSION) != set(supported_extensions()):
    raise AssertionError(
        f"trace-candidate-form table diverged from the registry: registry "
        f"supports {sorted(supported_extensions())} but forms cover "
        f"{sorted(_TRACE_CANDIDATE_FORM_BY_EXTENSION)}."
    )


def _trace_candidate_form_for(path: str) -> Literal["module", "specifier"]:
    """Admitted trace-candidate syntax for `path`'s language. KeyError on
    an unregistered extension is deliberate fail-loud — unreachable for
    gate-admitted files (totality assert above), and a silent default
    would be a wrong-form admission bug for a future language.
    """
    return _TRACE_CANDIDATE_FORM_BY_EXTENSION[PurePosixPath(path).suffix.lower()]


def _fence_lang_for(path: str) -> str:
    """Markdown fence hint for the scope-context blocks in the USER
    prompt. Language variation lives here and only here: the system
    prompt (`SYSTEM_PROMPT_STABLE_PREFIX`) stays byte-identical across
    languages per the cache-packing contract (DECISIONS.md#042); its
    Python exemplars are explicitly reference-only. The "text" fallback
    is unreachable for gate-admitted files (totality assert above); a
    neutral hint beats mislabeling unknown code as python.
    """
    return _FENCE_LANG_BY_EXTENSION.get(PurePosixPath(path).suffix.lower(), "text")


def _compute_per_file_cap(total_review_budget_tokens: int) -> int:
    """Limit a file to the smaller fractional or absolute token-estimate cap.

    Non-positive review budgets reject all files. The absolute ceiling prevents a
    larger review budget from permitting an oversized model call."""
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


class AnalyzeConcurrencyGate:
    """Per-event-loop semaphore factory for the `analyze_file` workers.

    A bare `asyncio.Semaphore` created at `build_graph` time lazily binds
    to the FIRST event loop it is contended on and then raises
    "bound to a different event loop" if the same compiled graph is ever
    driven on a second loop (a module-scoped graph fixture, an
    import-time build, sequential `asyncio.run` calls). This gate holds
    the permit COUNT and mints one semaphore per running loop on demand.

    CONTRACT — the bound is per-(graph, LOOP), not globally cross-loop:
    the production server drives one loop, where per-loop IS graph-global;
    two loops simultaneously driving one compiled graph would each get an
    independent cap (an accepted limitation — this gate bounds the
    fan-out's burst; cross-process/global provider throttling is the
    provider/rate-limit layer's concern, and a cross-loop limiter would
    need thread-blocking primitives inside coroutines).

    Storage is a STRONG dict pruned of closed loops on EVERY call, under
    a lock: weak keys do not work here, because a contended
    `asyncio.Semaphore` caches its bound loop (`_LoopBoundMixin`) — the
    value would strongly retain the weak key and no entry would ever
    collect — and miss-only pruning would retain a closed loop's entry
    indefinitely when no new loop ever appears. The lock serializes
    simultaneous-thread access (simultaneous loops ARE simultaneous
    threads); per-call pruning bounds the map by LIVE loops.
    """

    def __init__(self, permits: int) -> None:
        if permits < 1:
            raise ValueError(f"permits must be >= 1, got {permits}")
        self._permits = permits
        # Simultaneous loops mean simultaneous THREADS, so the map is
        # guarded by a threading.Lock: two same-instant misses (or two
        # threads pruning the same closed loop) would otherwise race the
        # iterate/del/set sequence — GIL atomicity is not a contract
        # (free-threaded builds), and a double-del raises KeyError.
        # Uncontended acquisition is nanoseconds against seconds-long
        # worker bodies.
        self._lock = threading.Lock()
        self._by_loop: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}

    def current(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        with self._lock:
            # Prune on EVERY call (the map is a handful of loops at most):
            # miss-only pruning would retain a closed secondary loop's
            # semaphore indefinitely when no new loop ever appears.
            closed_loops: list[asyncio.AbstractEventLoop] = []
            for known_loop in self._by_loop:
                if known_loop.is_closed():
                    closed_loops.append(known_loop)
            for stale in closed_loops:
                del self._by_loop[stale]
            semaphore = self._by_loop.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._permits)
                self._by_loop[loop] = semaphore
        return semaphore


class AnalyzeWorkerPayload(BaseModel):
    """Checkpointable input for one file worker, separate from closure dependencies.

    The rendered-prompt estimate must fit `allocation_tokens`; an unfunded
    file carries zero. Content and patch are copied into pending sends,
    bounded by intake's size gates, and released when the worker completes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    review_id: UUID
    installation_id: int
    is_eval: bool
    pass_index: int = Field(ge=0)
    review_tier: ReviewTier
    allocation_tokens: int = Field(ge=0)
    changed_file: ChangedFile
    cache_scope: CacheScope | None = None


async def analyze_file(
    payload: AnalyzeWorkerPayload,
    *,
    provider: LLMProvider,
    analyze_model: str,
    standard_analyze_model: str,
    import_path_resolver: ImportPathResolver,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
    trivial_scope_filter_enabled: bool = False,
    analyze_observed_skip_enforced: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
    concurrency_semaphore: asyncio.Semaphore | AnalyzeConcurrencyGate | None = None,
) -> dict[str, object]:
    """Review one file and return its outcome for the aggregate's round.

    Acquire the concurrency permit before opening the keyed phase: queueing
    is outside the work envelope (`phase-events-bound-work`). The file/pass
    key also tags per-operation audit events and the provider request.
    Worker failures propagate without worker-level retries (DECISIONS.md#064).

    `AnalyzeConcurrencyGate` supplies a semaphore per running loop; direct
    callers may pass a bare semaphore or None for unbounded execution.
    """
    phase_key = f"file:{payload.changed_file.path}#{payload.pass_index}"
    worker_phase_id = compute_phase_id(
        review_id=str(payload.review_id),
        node_id="analyze",
        attempt_key=phase_key,
    )
    gate = (
        concurrency_semaphore.current()
        if isinstance(concurrency_semaphore, AnalyzeConcurrencyGate)
        else concurrency_semaphore
    )
    async with gate if gate is not None else nullcontext():
        await phase_event_sink.emit_phase(
            ReviewPhaseEvent(
                review_id=payload.review_id,
                phase_id=worker_phase_id,
                node_id="analyze",
                marker="start",
                is_eval=payload.is_eval,
                phase_key=phase_key,
            )
        )
        model_for_file = _model_for_tier(
            payload.review_tier,
            analyze_model=analyze_model,
            standard_analyze_model=standard_analyze_model,
        )
        file_outcome = await _process_one_file(
            changed_file=payload.changed_file,
            review_id=payload.review_id,
            installation_id=payload.installation_id,
            is_eval=payload.is_eval,
            provider=provider,
            analyze_model=model_for_file,
            import_path_resolver=import_path_resolver,
            file_examination_sink=file_examination_sink,
            analyze_event_sink=analyze_event_sink,
            active_policy_version=active_policy_version,
            pass_index=payload.pass_index,
            per_file_cap_tokens=_compute_per_file_cap(total_review_budget_tokens),
            # The rendered-prompt estimate must fit this allocation. A low
            # planning estimate causes a skip rather than overspending.
            remaining_budget_tokens=payload.allocation_tokens,
            trivial_scope_filter_enabled=trivial_scope_filter_enabled,
            analyze_observed_skip_enforced=analyze_observed_skip_enforced,
            analyze_cache_store=analyze_cache_store,
            cache_scope=payload.cache_scope,
            cache_mode=cache_mode,
            profile_id=profile_id,
            reasoning_enabled=reasoning_enabled,
            profile_contract_digest=profile_contract_digest,
            phase_key=phase_key,
        )
        outcome = _worker_outcome_for(
            file_outcome,
            path=payload.changed_file.path,
            pass_index=payload.pass_index,
            review_tier=payload.review_tier,
        )
        await phase_event_sink.emit_phase(
            ReviewPhaseEvent(
                review_id=payload.review_id,
                phase_id=worker_phase_id,
                node_id="analyze",
                marker="end",
                is_eval=payload.is_eval,
                phase_key=phase_key,
            )
        )
    return {"analyze_worker_outcomes": [outcome]}


async def _plan_pass_zero(
    state: ReviewState,
    *,
    phase_event_sink: PhaseEventSink,
    total_review_budget_tokens: int,
    pass_index: int,
    phase_id: str,
    plan_phase_key: str | None,
    started_at: datetime,
    per_file_cap_tokens: int,
    cache_scope: CacheScope | None,
    triage_result: TriageResult | None,
) -> Command[Literal["analyze_file", "analyze_aggregate", "synthesize"]]:
    """Allocate budgets and dispatch one worker per kept file.

    No kept files routes directly to the aggregate for an empty round.
    `Command` requires the node's full destination union, though this planner
    only routes to `analyze_file` or `analyze_aggregate`.
    """
    pass_zero_worklist: list[tuple[ChangedFile, ReviewTier]] = []
    if triage_result is not None:
        for changed_file in state.pr_context.changed_files:
            tier = triage_result.file_tiers.get(changed_file.path, ReviewTier.SKIP)
            if tier in (ReviewTier.DEEP, ReviewTier.STANDARD):
                pass_zero_worklist.append((changed_file, tier))
        pass_zero_worklist.sort(key=lambda item: _PASS0_TIER_RANK[item[1]])

    # Include the system prompt, template scaffolding, and output reservation
    # in every estimate; omitting a fixed term would underfund every worker.
    empty_parts = analyze_prompt.render(
        file_path="",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    fixed_overhead_tokens = (
        _estimate_tokens(empty_parts.system_prompt)
        + _estimate_tokens(empty_parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
        + _PROXY_RENDER_MARGIN_TOKENS
    )

    # Allocations are final; unused tokens are not redistributed.
    # Reject duplicate paths before dispatching any worker.
    reserved_pool_tokens = int(total_review_budget_tokens * HIGH_RISK_RESERVE_FRACTION)
    general_pool_tokens = total_review_budget_tokens - reserved_pool_tokens
    plan = plan_file_budgets(
        tuple(
            FileBudgetRequest(
                path=changed_file.path,
                estimate_tokens=proxy_estimate_tokens(
                    len((changed_file.content_head or "").encode("utf-8")),
                    len((changed_file.patch or "").encode("utf-8")),
                    fixed_overhead_tokens=fixed_overhead_tokens,
                ),
                is_high_risk=bool(scan_added_lines_for_risk(changed_file.patch)),
            )
            for changed_file, _tier in pass_zero_worklist
        ),
        general_pool_tokens=general_pool_tokens,
        reserved_pool_tokens=reserved_pool_tokens,
        per_file_cap_tokens=per_file_cap_tokens,
    )
    allocation_by_path = {a.path: a for a in plan.allocations}
    sends: list[Send] = []
    for worker_file, worker_tier in pass_zero_worklist:
        sends.append(
            Send(
                "analyze_file",
                AnalyzeWorkerPayload(
                    review_id=state.review_id,
                    installation_id=state.pr_context.installation_id,
                    is_eval=state.is_eval,
                    pass_index=pass_index,
                    review_tier=worker_tier,
                    allocation_tokens=allocation_by_path[worker_file.path].allocation_tokens,
                    changed_file=worker_file,
                    cache_scope=cache_scope,
                ),
            )
        )
    # Workers and the aggregate have separate phases; planning ends here.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="end",
            is_eval=state.is_eval,
            phase_key=plan_phase_key,
        )
    )
    # Pass the wall-clock start to the aggregate across vertices. Route an
    # empty worklist there explicitly so it still produces a round.
    return Command(
        update={"analyze_pass_started_at": started_at},
        goto=sends if sends else "analyze_aggregate",
    )


async def _run_trace_pass(
    state: ReviewState,
    *,
    provider: LLMProvider,
    analyze_model: str,
    profile_id: str | None,
    reasoning_enabled: bool | None,
    profile_contract_digest: str | None,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    anomaly_sink: AnomalySink,
    import_path_resolver: ImportPathResolver,
    active_policy_version: str,
    total_review_budget_tokens: int,
    pass_index: int,
    phase_id: str,
    started_at: datetime,
    started_mono: float,
    per_file_cap_tokens: int,
) -> Command[Literal["analyze_file", "analyze_aggregate", "synthesize"]]:
    """Analyze trace-fetched files sequentially, then route to synthesis.

    Completion counters come from this pass's bookkeeping, not the audit stream.
    `Command` requires the node's full destination union; the two-round limit
    makes synthesis the only destination after this pass.
    """
    admitted_findings: list[ReviewFinding] = []
    # Different source findings can produce the same target-file proposal.
    # Deduplicate admission by both hashes; collapse content hashes later.
    admitted_keys_seen: set[tuple[str, str]] = set()
    trace_candidates: list[TraceCandidate] = []
    # Collect per-file subsumption proof for completion accounting (DECISIONS.md#055).
    pass_subsumed_matches: list[ObservedSubsumedMatch] = []
    files_examined: list[str] = []
    files_skipped: list[str] = []
    # Trace passes leave this counter at zero; starvation is measured in pass 0.
    budget_skip_count = 0
    n_proposals_seen = 0
    n_findings_emitted = 0
    n_findings_served = 0
    n_findings_observed = 0
    # Trace passes currently serve no cached findings. Retain the provenance set
    # used by the post-cap accounting below; evidence tier cannot identify a serve.
    served_content_hashes: set[str] = set()
    n_proposals_superseded_by_observed = 0
    n_proposals_rejected = 0
    n_responses_rejected = 0
    n_trace_candidates_emitted = 0
    n_trace_candidates_dropped_malformed = 0
    n_llm_calls = 0
    # Trace analysis uses the DEEP model; this field remains None.
    standard_model_used: str | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read_tokens = 0
    total_cache_write_tokens = 0
    total_cost_decimal = Decimal("0")
    remaining_budget_tokens = total_review_budget_tokens

    source_findings_by_id: dict[UUID, ReviewFinding] = {
        f.finding_id: f for r in state.analysis_rounds for f in r.findings
    }
    # Analyze each (target file, source finding) pair from the decisions.
    # Fetched files deduplicate by path, but each source needs its own prompt.
    fetched_files_by_path: dict[str, TraceFetchedFile] = {
        f.path: f for f in state.trace_fetched_files
    }
    # Only resolved decisions with fetched content enter the worklist.
    pass_one_work: list[tuple[TraceFetchedFile, ReviewFinding]] = []
    for decision in state.trace_decisions:
        if decision.target_file is None:
            continue
        fetched = fetched_files_by_path.get(decision.target_file)
        if fetched is None:
            # Targets already in the PR have no trace-fetched content.
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
            # Trace-fetched files currently have no producer OBSERVED findings.
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
            admit_with_pair_dedup(
                file_outcome.parser_result.admitted_findings,
                admitted_findings,
                admitted_keys_seen,
            )
            # Retain candidates from rejected proposals for replay (DECISIONS.md#025).
            # Trace skips candidates it cannot join to admitted findings.
            trace_candidates.extend(file_outcome.parser_result.trace_candidates)

        total_input_tokens += file_outcome.input_tokens
        total_output_tokens += file_outcome.output_tokens
        total_cache_read_tokens += file_outcome.cache_read_tokens
        total_cache_write_tokens += file_outcome.cache_write_tokens
        total_cost_decimal += file_outcome.cost_decimal
        pass_subsumed_matches.extend(file_outcome.subsumed_matches)
        remaining_budget_tokens -= file_outcome.estimated_tokens

        # One path can have several source contexts, but round paths must be
        # unique. An examined outcome takes precedence over an earlier skip.
        if fetched_file.path in files_examined:
            pass
        elif file_outcome.parse_status != "skipped":
            if fetched_file.path in files_skipped:
                files_skipped.remove(fetched_file.path)
            files_examined.append(fetched_file.path)
        elif fetched_file.path not in files_skipped:
            files_skipped.append(fetched_file.path)

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

    # Finalize and validate the kept set before emitting its FindingEvents.
    # Different proposals can share a content hash; collapse them first-wins.
    collapsed_findings = _collapse_by_content_hash(admitted_findings)

    # Gated findings survive the soft cap (hitl-gates-high-severity).
    # Exceeding the hard ceiling raises rather than dropping them.
    kept_findings, dropped_findings = cap_findings_by_severity(
        collapsed_findings, soft_cap=MAX_FINDINGS_PER_ROUND, hard_cap=MAX_FINDINGS_HARD_CAP
    )
    n_findings_dropped_over_cap = len(dropped_findings)

    # Recount kept findings after admission deduplication, collapse, and capping.
    # The sequential path identifies producer output by tier and registry membership;
    # the parallel aggregate uses recorded origin. Dropped proposals balance accounting.
    parser_proposal_emitted = n_findings_emitted - n_findings_served - n_findings_observed
    kept_served = sum(1 for f in kept_findings if f.content_hash in served_content_hashes)
    kept_observed = sum(
        1
        for f in kept_findings
        if f.content_hash not in served_content_hashes
        and f.evidence_tier is EvidenceTier.OBSERVED
        and f.query_match_id in OBSERVED_QUERY_IDS
    )
    kept_proposals = len(kept_findings) - kept_served - kept_observed
    n_proposals_dropped = parser_proposal_emitted - kept_proposals
    n_findings_emitted = len(kept_findings)
    n_findings_served = kept_served
    n_findings_observed = kept_observed

    # Unjoined candidates remain in state for replay, including candidates whose
    # findings were capped (DECISIONS.md#025). Trace will not fetch for them.
    # Validate both the round and completion accounting before emitting findings.
    # Content-derived round identity makes replay idempotent.
    round_id = compute_round_id(
        pass_index=pass_index,
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        finding_content_hashes=tuple(f.content_hash for f in kept_findings),
    )
    new_round = AnalysisRound(
        round_id=round_id,
        pass_index=pass_index,
        findings=tuple(kept_findings),
        files_examined=tuple(files_examined),
        files_skipped=tuple(files_skipped),
        started_at=started_at,
        ended_at=ended_at,
    )
    # Validate completion accounting before emitting findings.
    completed_event = AnalyzeCompletedEvent(
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
        n_proposals_dropped=n_proposals_dropped,
        n_findings_dropped_over_cap=n_findings_dropped_over_cap,
        subsumed_matches=tuple(pass_subsumed_matches),
        n_proposals_rejected=n_proposals_rejected,
        n_responses_rejected=n_responses_rejected,
        n_trace_candidates_emitted=n_trace_candidates_emitted,
        n_trace_candidates_dropped_malformed=n_trace_candidates_dropped_malformed,
        total_input_tokens=total_input_tokens,
        total_cache_read_tokens=total_cache_read_tokens,
        total_cache_write_tokens=total_cache_write_tokens,
        total_output_tokens=total_output_tokens,
        # Sum costs as Decimal; convert to float only at the event boundary.
        total_cost_usd=float(total_cost_decimal),
        pricing_version=PRICING_VERSION,
        policy_version=active_policy_version,
        analyze_model=analyze_model,
        standard_analyze_model=standard_model_used,
        # Host identity comes from build_graph (DECISIONS.md#056).
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
    )

    # Gated findings can exceed the soft cap and still reach HITL.
    # Report the overflow, but anomaly failures must not fail the review.
    if len(kept_findings) > MAX_FINDINGS_PER_ROUND:
        try:
            await anomaly_sink.emit_anomaly(
                review_id=state.review_id,
                rule_name=AnomalyRuleName.GATED_FINDINGS_OVER_CAP,
                severity=AnomalySeverity.HIGH,
                details={
                    "n_kept": len(kept_findings),
                    "soft_cap": MAX_FINDINGS_PER_ROUND,
                    "pass_index": pass_index,
                },
                is_eval=state.is_eval,
            )
        except Exception:
            logger.exception("analyze_gated_findings_over_cap_anomaly_emit_failed")

    # Emit exactly the round's kept findings.
    for finding in kept_findings:
        await analyze_event_sink.emit_finding(finding, is_eval=state.is_eval)

    await analyze_event_sink.emit_analyze_completed(completed_event)

    # Close the phase after findings and completion accounting.
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

    # Reducers require lists. The trace pass ends at synthesis.
    return Command(
        update={
            "analysis_rounds": [new_round],
            "trace_candidates": list(trace_candidates),
            # Only pass 0 produces worker outcomes.
            "analyze_worker_outcomes": [],
        },
        goto="synthesize",
    )


async def analyze(
    state: ReviewState,
    *,
    provider: LLMProvider,
    analyze_model: str,
    standard_analyze_model: str,
    # Host identity for completion accounting (DECISIONS.md#056).
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
    phase_event_sink: PhaseEventSink,
    file_examination_sink: FileExaminationSink,
    analyze_event_sink: AnalyzeEventSink,
    anomaly_sink: AnomalySink,
    import_path_resolver: ImportPathResolver,
    active_policy_version: str = ACTIVE_POLICY_VERSION,
    total_review_budget_tokens: int = DEFAULT_REVIEW_BUDGET_TOKENS,
    trivial_scope_filter_enabled: bool = False,
    analyze_observed_skip_enforced: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
) -> Command[Literal["analyze_file", "analyze_aggregate", "synthesize"]]:
    """Open the analysis phase, then plan workers or run the trace pass.

    Pass 0 resolves the cache scope and delegates to `_plan_pass_zero`;
    workers and the aggregate produce its round. Pass 1 delegates to
    `_run_trace_pass`, which reviews fetched scopes sequentially and
    routes to synthesis.
    """
    # Derive the pass from merged rounds so each pass has a distinct round_id.
    # The graph's trace router enforces MAX_ANALYSIS_ROUNDS.
    pass_index = len(state.analysis_rounds)
    # Pass 0 has a keyed planner phase; pass 1 keeps its unkeyed envelope.
    # Both identities are stable across retries from the same state.
    if pass_index == 0:
        plan_phase_key: str | None = f"plan#{pass_index}"
        phase_id = compute_phase_id(
            review_id=str(state.review_id),
            node_id="analyze",
            attempt_key=f"plan#{pass_index}",
        )
    else:
        plan_phase_key = None
        phase_id = compute_phase_id(
            review_id=str(state.review_id),
            node_id="analyze",
            attempt_key=f"analyze-pass-{pass_index}",
        )
    started_at = datetime.now(UTC)
    # The sequential trace pass derives its end time from this monotonic anchor
    # to survive backwards wall-clock jumps (FUP-141). Pass 0 ends in the aggregate.
    started_mono = time.monotonic()
    per_file_cap_tokens = _compute_per_file_cap(total_review_budget_tokens)

    # Persist the phase start before any cache read or file work.
    await phase_event_sink.emit_phase(
        ReviewPhaseEvent(
            review_id=state.review_id,
            phase_id=phase_id,
            node_id="analyze",
            marker="start",
            is_eval=state.is_eval,
            phase_key=plan_phase_key,
        )
    )

    # Resolve pass-0 cache tenancy from the reviews row, never mutable PR names.
    # Missing scope or store failure disables caching without aborting analysis.
    # Production and eval partitions remain separate (DECISIONS.md#046).
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
        # Lookups use the scope's eval flag; events use state's flag.
        # Disable caching if they disagree rather than mix partitions and telemetry.
        if cache_scope is not None and cache_scope.is_eval != state.is_eval:
            logger.warning(
                "analyze-cache is_eval divergence (scope=%s, state=%s); cache disabled",
                cache_scope.is_eval,
                state.is_eval,
            )
            cache_scope = None

    # Pass 0 plans only DEEP/STANDARD diff files; absent tiers default to SKIP.
    # Pass 1 reviews trace-fetched scope units with source-finding context.
    if pass_index == 0:
        return await _plan_pass_zero(
            state,
            phase_event_sink=phase_event_sink,
            total_review_budget_tokens=total_review_budget_tokens,
            pass_index=pass_index,
            phase_id=phase_id,
            plan_phase_key=plan_phase_key,
            started_at=started_at,
            per_file_cap_tokens=per_file_cap_tokens,
            cache_scope=cache_scope,
            triage_result=state.triage_result,
        )
    return await _run_trace_pass(
        state,
        provider=provider,
        analyze_model=analyze_model,
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        phase_event_sink=phase_event_sink,
        file_examination_sink=file_examination_sink,
        analyze_event_sink=analyze_event_sink,
        anomaly_sink=anomaly_sink,
        import_path_resolver=import_path_resolver,
        active_policy_version=active_policy_version,
        total_review_budget_tokens=total_review_budget_tokens,
        pass_index=pass_index,
        phase_id=phase_id,
        started_at=started_at,
        started_mono=started_mono,
        per_file_cap_tokens=per_file_cap_tokens,
    )


def _worker_outcome_for(
    file_outcome: _FileOutcome,
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
) -> AnalyzeWorkerOutcome:
    """Build a worker outcome from the active per-file result.

    Pass original producer objects: builders determine provenance by identity
    before cloning findings for state.
    """
    if file_outcome.parser_result is not None:
        parser_result = file_outcome.parser_result
        return worker_outcome_from_parser(
            path=path,
            pass_index=pass_index,
            review_tier=review_tier,
            parse_status=file_outcome.parse_status,  # type: ignore[arg-type]  # non-skip by branch
            admitted_findings=parser_result.admitted_findings,
            producer_findings=file_outcome.producer_findings,
            trace_candidates=parser_result.trace_candidates,
            subsumed_matches=file_outcome.subsumed_matches,
            n_proposals_seen=parser_result.counters.n_proposals_seen,
            n_proposals_rejected=parser_result.counters.n_proposals_rejected,
            n_responses_rejected=parser_result.counters.n_responses_rejected,
            n_proposals_superseded_by_observed=(
                parser_result.counters.n_proposals_superseded_by_observed
            ),
            n_trace_candidates_dropped_malformed=(
                parser_result.counters.n_trace_candidates_dropped_malformed
            ),
            input_tokens=file_outcome.input_tokens,
            output_tokens=file_outcome.output_tokens,
            cache_read_tokens=file_outcome.cache_read_tokens,
            cache_write_tokens=file_outcome.cache_write_tokens,
            cost=file_outcome.cost_decimal,
            estimated_tokens=file_outcome.estimated_tokens,
        )
    if file_outcome.served_result is not None:
        served = file_outcome.served_result
        return worker_outcome_from_serve(
            path=path,
            pass_index=pass_index,
            review_tier=review_tier,
            served_findings=served.admitted_findings,
            trace_candidates=served.trace_candidates,
            subsumed_matches=file_outcome.subsumed_matches,
            estimated_tokens=file_outcome.estimated_tokens,
        )
    if file_outcome.observed_skip_result is not None:
        skip_reason = file_outcome.skip_reason
        if skip_reason is None:
            # Deterministic coverage counts as examined, with no skip reason.
            return worker_outcome_from_observed_coverage(
                path=path,
                pass_index=pass_index,
                review_tier=review_tier,
                producer_findings=file_outcome.observed_skip_result.admitted_findings,
                estimated_tokens=file_outcome.estimated_tokens,
            )
        return worker_outcome_from_observed_skip(
            path=path,
            pass_index=pass_index,
            review_tier=review_tier,
            skip_reason=skip_reason,
            producer_findings=file_outcome.observed_skip_result.admitted_findings,
        )
    skip_reason = file_outcome.skip_reason
    if skip_reason is None:  # every remaining branch is a reasoned skip
        raise RuntimeError("plain-skip outcome without a skip_reason")
    return worker_outcome_from_plain_skip(
        path=path,
        pass_index=pass_index,
        review_tier=review_tier,
        skip_reason=skip_reason,
    )


@dataclass(frozen=True, slots=True)
class _ServedResult:
    """Cached findings and candidates restored without an LLM call.

    Worker builders classify these as served, outside proposal accounting.
    """

    admitted_findings: tuple[ReviewFinding, ...]
    trace_candidates: tuple[TraceCandidate, ...]


@dataclass(frozen=True, slots=True)
class _ObservedSkipResult:
    """Producer findings retained without an LLM call.

    Deterministic coverage counts as examined. Module-level findings also
    survive budget/trivial skips (DECISIONS.md#062). Both reach synthesis
    and HITL without entering proposal accounting.
    """

    admitted_findings: tuple[ReviewFinding, ...]


@dataclass(frozen=True, slots=True)
class _FileOutcome:
    """Per-file result for worker builders and sequential pass accounting.

    Cache reads and writes stay separate because their prices can differ.
    Costs remain Decimal until conversion at the completion-event boundary.
    """

    parse_status: _ParseStatus
    parser_result: ParserResult | None
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_decimal: Decimal
    estimated_tokens: int
    # Cache serves have no parser_result or LLM call.
    served_result: _ServedResult | None = None
    # Producer-only coverage or findings retained alongside a budget/trivial skip.
    observed_skip_result: _ObservedSkipResult | None = None
    # Mirror the examination event's reason for skip and starvation accounting.
    skip_reason: SkipReason | None = None
    # Merge or restored-cache proof for AnalyzeCompletedEvent (DECISIONS.md#055).
    subsumed_matches: tuple[ObservedSubsumedMatch, ...] = ()
    # Original producer objects let builders check survival by identity before cloning.
    producer_findings: tuple[ReviewFinding, ...] = ()


class _ServeReconstructionError(Exception):
    """A cache payload failed reconstruction before any serve event was emitted.

    The caller falls back to model analysis.
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
    phase_key: str | None,
) -> _FileOutcome:
    """Restore cached findings and candidates without an LLM call.

    Revalidate findings, reset downstream lifecycle fields, and derive IDs for
    this review. Candidate IDs remain content-derived and review-independent.
    Reconstruction and uniqueness checks precede all emits; then emit the
    cache serve and a clean examination event. FindingEvents are deferred to
    round assembly, and candidate content travels in the returned outcome.
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
        # Cross-type subsumption proof-retention records (DECISIONS.md#055): the
        # dropped OBSERVED findings are NOT in `findings`, so reconstruct their
        # telemetry from the payload so a cache HIT retains the proof too. `.get`
        # tolerates pre-#055 rows (none survive the analyze-parser-v4 bump, but the
        # default keeps reconstruction total).
        served_subsumed_matches = tuple(
            ObservedSubsumedMatch.model_validate(dump)
            for dump in entry.payload.get("subsumed_matches", [])
        )
    except (CoordinateError, KeyError, TypeError, ValueError) as exc:
        # CoordinateError is NOT a ValueError: the file_path validators on
        # ReviewFinding AND ObservedSubsumedMatch re-run validate_diff_path, which
        # raises CoordinateError on a malformed cached path. Catch it here so a
        # tampered/corrupt cache payload DEGRADES to a real LLM call rather than
        # aborting the review (the whole point of the reconstruction guard).
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
            phase_key=phase_key,
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
    # normal-path step-3e emission, so it emits here itself — with the worker's
    # key: an unkeyed event inside the keyed worker envelope is exactly what the
    # strict replay hybrid's None-branch rejects.
    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=file_path,
        phase_key=phase_key,
    )

    # Served findings re-emit one FindingEvent each so this review's audit/findings
    # tables are self-contained for replay (the cache stores content; audit rows are
    # per-review). Emission is DEFERRED to the main loop's post-cap step (FUP-180): the
    # per-round finding cap must decide the kept set before any FindingEvent fires, so
    # the returned `_ServedResult` carries the findings and the main loop emits the kept
    # ones. The deterministic finding_id keeps the persister's no-resurrection
    # content-row guard correct under checkpoint replay.

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
        subsumed_matches=served_subsumed_matches,
    )


def _build_query_match_id_set(file_content_bytes: bytes, *, file_path: str) -> frozenset[str]:
    """Return IDs of non-deprecated structural queries that fired for this file.

    The registry selects the language catalog and grammar, including JS/TS dialects.
    The parser requires OBSERVED IDs to belong to this supplied set (spec §7 step 3b);
    an empty set rejects every model OBSERVED claim."""
    language = query_registry.query_language_for_path(file_path)
    grammar = query_registry.grammar_for_path(file_path)
    if language is None or grammar is None:
        return frozenset()
    fired: set[str] = set()
    for query_id in query_registry.structural_query_ids_for(language):
        if query_registry.match(query_id, file_content_bytes, grammar=grammar):
            fired.add(query_id)
    return frozenset(fired)


def _filter_query_ids_to_scopes(
    query_ids: frozenset[str],
    file_content_bytes: bytes,
    scope_units: tuple[ScopeUnit, ...],
    *,
    file_path: str,
) -> frozenset[str]:
    """Keep a fired query ID iff at least one of its match envelopes
    intersects an INCLUDED scope unit's byte range. `file_path` selects the
    grammar the re-match runs under — the same selection that fired the ids
    in `_build_query_match_id_set` (unreachable in practice for a file with
    no catalog: its fired set is already empty).

    Used only when the trivial-scope filter excluded scopes from the
    prompt (specs/2026-06-10-trivial-scope-filter.md): IDs whose matches
    fall only in excluded scopes must not advertise — the same filtered
    set feeds both the prompt and the parser's OBSERVED admission, so a
    finding cannot cite structural proof from code the model never saw.
    Half-open intersection over `QueryMatchSpan` envelopes.
    """
    grammar = query_registry.grammar_for_path(file_path)
    if grammar is None:
        return frozenset()
    ranges = tuple((su.byte_start, su.byte_end) for su in scope_units)
    kept: set[str] = set()
    for query_id in query_ids:
        for match_span in query_registry.match(query_id, file_content_bytes, grammar=grammar):
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
    fence_lang: str = "python",
) -> str:
    """Render scope kind, name, source-line range and fenced body for each unit.

    coordinates.extract_scope_unit_body owns byte slicing
    (`coordinates-module-is-sole-translator`). This block adds no caller/import
    graph context. The call-site budget gate limits size; an assembly-time cap
    remains deferred (FUP-052)."""
    blocks: list[str] = []
    for scope_unit in included_scope_units:
        body = extract_scope_unit_body(scope_unit, source_bytes)
        name = scope_unit.qualified_name or scope_unit.name
        blocks.append(
            f"### {scope_unit.kind} `{name}` "
            f"(lines {scope_unit.line_start}-{scope_unit.line_end})\n"
            f"{safe_code_fence(body, lang=fence_lang)}"
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
    phase_key: str | None,
) -> _FileOutcome:
    """Emit a single `FileExaminationEvent(parse_status="skipped", skip_reason=...)`
    and return a zero-cost `_FileOutcome`. Used by every skip path in
    `_process_one_file` to keep the emission point uniform per spec §7
    step 3e (single emission per kept file)."""
    # One stdout line per skip so a live run shows WHY a file produced no findings
    # (UNSUPPORTED_LANGUAGE / VENDORED / OVERSIZED / NO_CHANGED_SCOPE_UNITS / ...) — otherwise
    # "fewer findings than files" reads as a silent failure. The event below is the durable
    # record; this is the live-watch signal (INFO, so it stays out of the WARNING+ error tee).
    logger.info("analyze: skipped %s (%s)", file_path, skip_reason.value)
    await file_examination_sink.emit_file_examination(
        FileExaminationEvent(
            review_id=review_id,
            is_eval=is_eval,
            file_path=file_path,
            examination_type="analyze",
            node_id="analyze",
            parse_status="skipped",
            skip_reason=skip_reason,
            phase_key=phase_key,
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


async def _emit_skip_with_module_findings(
    *,
    file_examination_sink: FileExaminationSink,
    review_id: UUID,
    is_eval: bool,
    file_path: str,
    skip_reason: SkipReason,
    module_matches: tuple[ObservedMatch, ...],
    installation_id: int,
    active_policy_version: str,
    phase_key: str | None,
) -> _FileOutcome:
    """A skip that still emits the module arm's OBSERVED findings
    (DECISIONS.md#062): the deterministic finding costs ZERO LLM tokens, so an
    early skip — cost budget, all-trivial — must not drop it (the finding is
    exactly what the module route exists to emit). The skip
    FileExaminationEvent and its accounting (skip_reason on the outcome, e.g.
    the COST_BUDGET_EXHAUSTED starvation counter) stand unchanged; the
    findings ride out on `_ObservedSkipResult`, the enforced-skip carrier the
    main loop admits with zero LLM accounting (they reach synthesize + HITL).
    Same-content_hash producer duplicates collapse first-wins, mirroring the
    enforced-skip branch — content_hash excludes evidence_tier, so two
    same-span OBSERVED findings would otherwise trip
    `AnalysisRound._enforce_findings_unique`."""
    outcome = await _emit_skip(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=file_path,
        skip_reason=skip_reason,
        phase_key=phase_key,
    )
    produced = produce_observed_findings(
        module_matches,
        file_path=file_path,
        review_id=review_id,
        installation_id=installation_id,
        active_policy_version=active_policy_version,
    )
    deduped = _collapse_by_content_hash(produced)
    return replace(outcome, observed_skip_result=_ObservedSkipResult(tuple(deduped)))


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
    phase_key: str | None,
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
            phase_key=phase_key,
        )
    )


def _merge_producer_findings(
    parser_result: ParserResult, observed_findings: tuple[ReviewFinding, ...]
) -> tuple[ParserResult, list[ObservedSubsumedMatch]]:
    """Merge producer findings and retain proof for subsumed matches."""
    subsumed_matches: list[ObservedSubsumedMatch] = []
    # Same-hash producer findings replace JUDGED proposals to retain query proof
    # (DECISIONS.md#054). Keep incumbents carrying proof; append fresh findings
    # in producer order.
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
        elif content_hash not in fresh_hashes:
            fresh.append(observed_finding)
            fresh_hashes.add(content_hash)
    n_observed = len(fresh) + n_superseded
    if n_observed:
        # Replacements remove proposal findings and record supersessions;
        # fresh and replacement producer findings count as observed.
        # Reuse the original tuple when no incumbent was replaced.
        base_admitted = (
            parser_result.admitted_findings if n_superseded == 0 else tuple(admitted_list)
        )
        parser_result = replace(
            parser_result,
            admitted_findings=base_admitted + tuple(fresh),
            counters=replace(
                parser_result.counters,
                n_findings_emitted=parser_result.counters.n_findings_emitted - n_superseded,
                n_findings_observed=n_observed,
                n_proposals_superseded_by_observed=n_superseded,
            ),
        )

    # Subsumption follows replacement so triple collisions use the surviving
    # producer finding (DECISIONS.md#055). Only same-span JUDGED findings can
    # subsume producer OBSERVED findings; exact spans avoid absorbing unrelated
    # findings. Retain the dropped query proof in subsumed_matches.
    admitted = parser_result.admitted_findings
    judged_by_span: dict[tuple[int, int], list[ReviewFinding]] = {}
    for jf in admitted:
        if jf.evidence_tier is EvidenceTier.JUDGED:
            judged_by_span.setdefault((jf.line_start, jf.line_end), []).append(jf)
    if judged_by_span:
        surviving: list[ReviewFinding] = []
        for af in admitted:
            subsumer: ReviewFinding | None = None
            # Producer registry IDs exclude model-cited structural OBSERVED
            # findings, whose removal would decrement the wrong counter.
            if (
                af.evidence_tier is EvidenceTier.OBSERVED
                and af.query_match_id is not None
                and af.query_match_id in OBSERVED_QUERY_IDS
            ):
                for yf in judged_by_span.get((af.line_start, af.line_end), ()):
                    if subsumes(yf.finding_type, af.finding_type):
                        subsumer = yf
                        break
            if subsumer is None:
                surviving.append(af)
            else:
                subsumed_matches.append(
                    ObservedSubsumedMatch(
                        file_path=af.file_path,
                        query_match_id=af.query_match_id,
                        finding_type=af.finding_type,
                        subsumed_by_finding_type=subsumer.finding_type,
                        line_start=af.line_start,
                        line_end=af.line_end,
                        dropped_content_hash=af.content_hash,
                        subsumer_content_hash=subsumer.content_hash,
                    )
                )
        n_subsumed = len(admitted) - len(surviving)
        if n_subsumed:
            # Decrease observed only: aggregate emitted decreases with it,
            # so proposal accounting still balances.
            parser_result = replace(
                parser_result,
                admitted_findings=tuple(surviving),
                counters=replace(
                    parser_result.counters,
                    n_findings_observed=parser_result.counters.n_findings_observed - n_subsumed,
                ),
            )
    return parser_result, subsumed_matches


def _price_model_response(response: LLMResponse) -> Decimal:
    """Price the returned token counts and actual billing tier, keeping Decimal precision."""
    return compute_cost_usd(
        response.profile_id,
        response.model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
        billed_prompt_tokens=response.billed_prompt_tokens,
        service_tier=response.service_tier_actual,
    )


async def _emit_parser_rejections(
    *,
    parser_result: ParserResult,
    analyze_event_sink: AnalyzeEventSink,
    review_id: UUID,
    is_eval: bool,
    phase_key: str | None,
) -> None:
    """Audit rejected proposals in order, then any whole-response rejection."""
    for rejection in parser_result.proposal_rejections:
        await analyze_event_sink.emit_finding_proposal_rejected(
            _lift_proposal_rejection(
                rejection, review_id=review_id, is_eval=is_eval, phase_key=phase_key
            )
        )
    if parser_result.response_rejection is not None:
        await analyze_event_sink.emit_analyze_response_rejected(
            _lift_response_rejection(
                parser_result.response_rejection,
                review_id=review_id,
                is_eval=is_eval,
                phase_key=phase_key,
            )
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
    analyze_observed_skip_enforced: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_scope: CacheScope | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    profile_id: str | None = None,
    reasoning_enabled: bool | None = None,
    profile_contract_digest: str | None = None,
    phase_key: str | None,
) -> _FileOutcome:
    """Review a triage-kept file through eligibility, context, and budget gates.

    Filter scopes, try the clean cache, then check deterministic coverage.
    If a model call remains necessary, price and parse its response, merge
    producer findings, audit rejections, and cache the completed result.
    Finding emission waits for the caller's round cap.
    """
    # Triage can keep files without a registered language adapter; skip them
    # before parsing (DECISIONS.md#018).
    if not _language_supported(changed_file.path):
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            phase_key=phase_key,
            skip_reason=SkipReason.UNSUPPORTED_LANGUAGE,
        )

    # An empty head is current content; only None permits the base fallback.
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
            phase_key=phase_key,
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # Reuse the byte length for every proposal's span check.
    content_bytes = content.encode("utf-8")
    file_byte_length = len(content_bytes)

    parse_result: ParseResult = parse_source(
        content_bytes,
        changed_file.path,
        import_path_resolver,
    )

    # Handle parser exclusions before patch lookup, which can raise on malformed
    # or duplicate patches. Pass the skip reason through without a model call.
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
            phase_key=phase_key,
            skip_reason=parser_skip_reason,
        )

    # Missing or path-rejected patch context returns None. A malformed patch can
    # raise here, after parser exclusions have already returned.
    patched_file = lookup_patched_file(changed_file.patch, changed_file.path)

    # Validate patch/head alignment before routing: every later span uses this
    # coordinate frame. The coordinates layer raises on mismatched added lines;
    # only that mismatch becomes a skip (FUP-217).
    if patched_file is not None:
        if changed_file.content_head is None:
            misaligned = patched_file_has_added_lines(patched_file)
        else:
            try:
                added_line_byte_ranges(patched_file, changed_file.content_head)
                misaligned = False
            except CoordinateError:
                misaligned = True
        if misaligned:
            logger.warning(
                "patch/head-content misalignment for %s — file skipped "
                "(PATCH_HEAD_MISALIGNED; review continues)",
                changed_file.path,
            )
            return await _emit_skip(
                file_examination_sink=file_examination_sink,
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                phase_key=phase_key,
                skip_reason=SkipReason.PATCH_HEAD_MISALIGNED,
            )

    # Derive module inputs once for routing, producer admission, and cache identity
    # (DECISIONS.md#062). The catalog sweep remains lazy.
    module_all_scope_units, module_added_ranges = module_admission_inputs(
        parse_result, patched_file, changed_file.content_head
    )

    # The pure degradation decision selects skip, bounded hunks, or scoped review.
    # Its parse-failure route remains for future raw-byte intake (FUP-053).
    decision = decide_degradation(parse_result, patched_file)

    # Sweep only when module matches could rescue NO_CHANGED_SCOPE_UNITS.
    # Reuse those matches for production; ineligible languages short-circuit
    # (DECISIONS.md#062).
    module_matches: tuple[ObservedMatch, ...] = ()
    if (
        decision.mode == "skip"
        and decision.skip_reason is SkipReason.NO_CHANGED_SCOPE_UNITS
        and module_added_ranges
    ):
        module_matches = module_level_observed_matches(
            file_path=changed_file.path,
            head_content=content,
            all_scope_units=module_all_scope_units,
            added_line_ranges=module_added_ranges,
            import_refs=parse_result.imports,
            lexical_bindings=parse_result.lexical_bindings,
        )
        if module_matches:
            decision = decide_degradation(
                parse_result, patched_file, module_level_observed_candidate=True
            )
    if decision.mode == "skip":
        skip_reason = decision.skip_reason
        if skip_reason is None:  # DegradationDecision guard makes this impossible; narrows mypy.
            raise RuntimeError("DegradationDecision mode='skip' with skip_reason None")
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            phase_key=phase_key,
            skip_reason=skip_reason,
        )
    degradation_reason: DegradationReason | None = decision.degradation_reason
    parse_status_for_event: _ParseStatus = decision.parse_status
    included_scope_units: tuple[ScopeUnit, ...] = decision.included_scope_units
    included_clipped_hunks: tuple[tuple[str, ...], ...] = decision.included_clipped_hunks
    degraded_mode = decision.mode == "degraded"

    # Run structural queries under the registered language grammar. Degraded
    # prompts cannot cite structural proof; an empty fired set rejects OBSERVED
    # model proposals.
    is_python = _is_python_file(changed_file.path)
    query_match_id_set: frozenset[str] = (
        frozenset()
        if degraded_mode
        else _build_query_match_id_set(content_bytes, file_path=changed_file.path)
    )

    # Use the same imports for cache identity and parser candidate correction.
    # Correction applies only to module-form candidates.
    trace_import_refs = parse_result.imports

    # Build scoped context for clean files, bounded diff context for degraded ones.
    if degraded_mode:
        # The degradation decision guarantees both values; this check narrows them.
        if degradation_reason is None or patched_file is None:
            raise RuntimeError(
                "analyze: degraded_mode true with degradation_reason/patched_file None "
                "— upstream outcome-determination invariant violated"
            )
        prompt_parts = analyze_prompt.render_degraded(
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
        prompt_parts = analyze_prompt.render(
            file_path=changed_file.path,
            scope_unit_context=_assemble_scope_unit_context(
                included_scope_units=included_scope_units,
                source_bytes=content_bytes,
                fence_lang=_fence_lang_for(changed_file.path),
            ),
            query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
            diff_hunks=_concat_clipped_hunks(included_clipped_hunks),
            pass_index=pass_index,
        )

    # Gate the full prompt before scope filtering can reduce it.
    estimated_tokens = (
        _estimate_tokens(prompt_parts.system_prompt)
        + _estimate_tokens(prompt_parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
    )
    if estimated_tokens > per_file_cap_tokens or estimated_tokens > remaining_budget_tokens:
        # A budget skip must retain zero-token module findings (DECISIONS.md#062).
        # Reuse routing matches, or sweep lazily if the clean route has not needed them.
        budget_module_matches = module_matches or (
            module_level_observed_matches(
                file_path=changed_file.path,
                head_content=content,
                all_scope_units=module_all_scope_units,
                added_line_ranges=module_added_ranges,
                import_refs=parse_result.imports,
                lexical_bindings=parse_result.lexical_bindings,
            )
            if module_added_ranges
            else ()
        )
        if budget_module_matches:
            # Keep the budget skip and retain the zero-token deterministic findings.
            return await _emit_skip_with_module_findings(
                file_examination_sink=file_examination_sink,
                review_id=review_id,
                is_eval=is_eval,
                file_path=changed_file.path,
                skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
                module_matches=budget_module_matches,
                installation_id=installation_id,
                active_policy_version=active_policy_version,
                phase_key=phase_key,
            )
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            phase_key=phase_key,
            skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        )

    # Classify scopes only after the baseline budget gate: filtering must not turn
    # a budget skip into a model call. Keep the baseline estimate for conservative
    # budget accounting. Shadow mode records exclusions without applying them.
    #
    # One post-gate parse supplies triviality and parameterized-call facts. The same
    # scan feeds cache identity and parser admission; build base-side triviality
    # only when a patch and included scopes require it.
    want_triviality = (
        is_python and not degraded_mode and patched_file is not None and bool(included_scope_units)
    )
    if is_python:
        triviality_context, parameterized_call_scan = extract_triviality_and_scan(
            content_bytes,
            changed_file.content_base.encode("utf-8")
            if (want_triviality and changed_file.content_base is not None)
            else None,
            compute_triviality=want_triviality,
            degraded=degraded_mode,
        )
    else:
        # These scans are Python-only. Clean non-Python files get an empty scan so they
        # remain cacheable; None denotes degraded mode. The SQL veto stays inert.
        triviality_context = None
        parameterized_call_scan = None if degraded_mode else ParameterizedCallScan()

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
                phase_key=phase_key,
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
                # Preserve changed module findings when every function scope is trivial
                # (DECISIONS.md#062). This guard is currently unreachable: triviality is
                # Python-only and the eligible module query is JavaScript-only.
                trivial_module_matches = module_level_observed_matches(
                    file_path=changed_file.path,
                    head_content=content,
                    all_scope_units=module_all_scope_units,
                    added_line_ranges=module_added_ranges,
                    import_refs=parse_result.imports,
                    lexical_bindings=parse_result.lexical_bindings,
                )
                if trivial_module_matches:
                    return await _emit_skip_with_module_findings(
                        file_examination_sink=file_examination_sink,
                        review_id=review_id,
                        is_eval=is_eval,
                        file_path=changed_file.path,
                        skip_reason=SkipReason.ALL_SCOPES_TRIVIAL,
                        module_matches=trivial_module_matches,
                        installation_id=installation_id,
                        active_policy_version=active_policy_version,
                        phase_key=phase_key,
                    )
                return await _emit_skip(
                    file_examination_sink=file_examination_sink,
                    review_id=review_id,
                    is_eval=is_eval,
                    file_path=changed_file.path,
                    phase_key=phase_key,
                    skip_reason=SkipReason.ALL_SCOPES_TRIVIAL,
                )
            included_scope_units = tuple(su for su, _ in kept)
            included_clipped_hunks = tuple(hunks for _, hunks in kept)
            # Keep proof IDs within the retained scopes. Both prompt and parser use this
            # filtered set, so a finding cannot cite proof from excluded code.
            query_match_id_set = _filter_query_ids_to_scopes(
                query_match_id_set, content_bytes, included_scope_units, file_path=changed_file.path
            )
            # Re-render over the kept scopes; this filtered prompt is what
            # is actually sent (and what context_summary describes).
            prompt_parts = analyze_prompt.render(
                file_path=changed_file.path,
                scope_unit_context=_assemble_scope_unit_context(
                    included_scope_units=included_scope_units,
                    source_bytes=content_bytes,
                    fence_lang=_fence_lang_for(changed_file.path),
                ),
                query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
                diff_hunks=_concat_clipped_hunks(included_clipped_hunks),
                pass_index=pass_index,
            )

    # Look up clean-pass cache entries after scope filtering. The key describes
    # the actual prompt and admission inputs; degraded files are not cacheable.
    cache_key: str | None = None
    cache_would_hit = False
    if (
        analyze_cache_store is not None
        and cache_scope is not None
        and parameterized_call_scan is not None
    ):
        cache_key = compute_analyze_cache_key(
            system_prompt=prompt_parts.system_prompt,
            user_prompt=prompt_parts.user_prompt,
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
            subsumes_digest=SUBSUMES_DIGEST,
            # Candidate correction also depends on imports outside the prompt (#024).
            from_import_map_digest=from_import_map_digest(trace_import_refs),
            # Key imported API bindings used by producer admission, including imports
            # outside the rendered scopes. Lexical shadowing is keyed separately below.
            import_bindings_digest=import_bindings_digest(trace_import_refs),
            # Shadowing may come from enclosing scopes outside the prompt.
            lexical_bindings_digest=lexical_bindings_digest(parse_result.lexical_bindings),
            # Module admission uses added bytes and all scope spans, beyond prompt context.
            module_admission_digest=module_admission_digest(
                module_added_ranges, module_all_scope_units, content_bytes
            ),
            profile_id=profile_id,
            reasoning_enabled=reasoning_enabled,
            profile_contract_digest=profile_contract_digest,
        )
        try:
            # A retry must not count or serve this review's own previous write.
            cache_entry = await analyze_cache_store.lookup(
                cache_key, is_eval=cache_scope.is_eval, exclude_source_review_id=review_id
            )
        except CacheStoreError:
            # A failed lookup falls back to the model without a false miss event.
            # Clear the key to suppress writes for this failed lookup as well.
            logger.warning(
                "analyze-cache lookup failed; cache skipped for %s",
                changed_file.path,
                exc_info=True,
            )
            cache_key = None
        else:
            if cache_mode is CacheMode.SERVE and cache_entry is not None:
                # Restore cached findings for this review and skip the model call. Record
                # served findings separately from current-pass model proposals.
                try:
                    return await _serve_cache_hit(
                        phase_key=phase_key,
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
                    # Fall through to the model before any serve event. Clear the key
                    # to suppress fabricated lookup telemetry and cache writes (FUP-177).
                    logger.warning(
                        "analyze-cache serve reconstruction failed; degrading to LLM for %s",
                        changed_file.path,
                        exc_info=True,
                    )
                    cache_key = None
            # Record completed shadow lookups and serve misses before continuing analysis.
            # Failed lookups take the exception path without fabricating a miss event.
            if cache_key is not None:
                cache_would_hit = cache_entry is not None
                await analyze_event_sink.emit_cache_lookup(
                    CacheLookupEvent(
                        review_id=review_id,
                        phase_key=phase_key,
                        is_eval=is_eval,
                        file_path=changed_file.path,
                        outcome="would_hit" if cache_would_hit else "miss",
                        cache_key=cache_key,
                    )
                )

    # Count the file as examined before deterministic coverage or a model call.
    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=changed_file.path,
        parse_status=parse_status_for_event,
        phase_key=phase_key,
    )

    # Compute deterministic coverage before the model call so enforcement can
    # skip it. Shadow telemetry remains deferred until a successful call.
    observed_matches: tuple[ObservedMatch, ...] = ()
    observed_skip_event: ObservedSkipShadowEvent | None = None
    # The producer selects its registered grammar and eligible security queries.
    # Degraded parse-error routes exclude structural production; the clean-parse
    # module route reuses its routing matches.
    module_level_route = decision.is_module_level_route
    if changed_file.content_head is not None and decision.runs_observed_producer:
        if module_level_route:
            # One sweep, zero drift: the routing sweep's admitted matches
            # ARE the module route's final set — non-empty by construction
            # (an empty sweep never re-decides onto this route).
            observed_matches = module_matches
        else:
            observed_matches = run_observed_matches(
                file_path=changed_file.path,
                head_content=content,
                included_scope_units=included_scope_units,
                # Pass imports and lexical facts unchanged: receiver binding checks use both,
                # and the cache key digests these same inputs.
                import_refs=parse_result.imports,
                lexical_bindings=parse_result.lexical_bindings,
                # On the clean route, combine changed-scope and module matches using the
                # same module admission inputs already used for routing and cache identity.
                all_scope_units=module_all_scope_units,
                added_line_ranges=module_added_ranges,
            )
        if patched_file is not None and decision.emits_skip_shadow:
            observed_skip_event = compute_observed_skip_shadow(
                observed_matches,
                file_path=changed_file.path,
                included_scope_units=included_scope_units,
                patched_file=patched_file,
                head_source=content,
                base_source=changed_file.content_base,
                review_id=review_id,
                is_eval=is_eval,
            )
        if (
            analyze_observed_skip_enforced
            and observed_skip_event is not None
            and observed_skip_event.outcome == "would_skip"
        ):
            # Full deterministic coverage can skip the model call. The file still counts
            # as examined, and its findings pass through ordinary synthesis and approval.
            produced = produce_observed_findings(
                observed_matches,
                file_path=changed_file.path,
                review_id=review_id,
                installation_id=installation_id,
                active_policy_version=active_policy_version,
            )
            # Collapse duplicate producer claims before admission; rounds require unique
            # content hashes even when distinct queries match the same span.
            skip_findings = _collapse_by_content_hash(produced)
            # Revalidate the skip_enforced => would_skip event contract when stamping it.
            await analyze_event_sink.emit_observed_skip_shadow(
                ObservedSkipShadowEvent.model_validate(
                    {
                        **observed_skip_event.model_dump(),
                        "skip_enforced": True,
                        "phase_key": phase_key,
                    }
                )
            )
            # Return findings for the caller's round cap before any FindingEvent (FUP-180).
            return _FileOutcome(
                parse_status=parse_status_for_event,
                parser_result=None,
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                cost_decimal=Decimal("0"),
                estimated_tokens=0,
                observed_skip_result=_ObservedSkipResult(tuple(skip_findings)),
            )

    # The model receives a manifest of clean scopes. Degraded context uses no
    # structural manifest, as allowed by the request validator.
    context_summary: tuple[ContextManifestEntry, ...] = (
        ()
        if degraded_mode
        else _build_context_manifest(
            changed_file.path, included_scope_units, inclusion_reason="changed_scope"
        )
    )
    request = LLMRequest(
        model=analyze_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=analyze_prompt.MAX_TOKENS,
        temperature=analyze_prompt.TEMPERATURE,
        review_id=review_id,
        node_id="analyze",
        is_eval=is_eval,
        prompt_template_version=analyze_prompt.VERSION,
        degraded_mode=degraded_mode,
        degradation_reason=degradation_reason,
        # Providers mirror the worker's phase key onto LLMCallEvent.
        phase_key=phase_key,
        context_summary=context_summary,
        # Both passes use the pinned schema; refusal and truncation still need parser checks.
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
    )
    # Provider failures propagate; the open phase records the interruption.
    response: LLMResponse = await provider.complete(request)

    cost_decimal = _price_model_response(response)

    # Derive allowed degraded spans from the patch through coordinates, not from
    # model or prompt text. Clean-mode admission does not use these ranges.
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
        # Admission uses the same parameterized-call facts included in the cache key.
        parameterized_call_scan=parameterized_call_scan,
        # Use this file's imports to add corrected module-form siblings alongside
        # the original candidates. This is the same input digested by the cache key.
        import_refs=trace_import_refs,
        # The registry owns candidate form: dotted modules for Python, relative
        # specifiers for JS/TS. The parser enforces form and repository containment
        # (DECISIONS.md#024).
        trace_candidate_form=_trace_candidate_form_for(changed_file.path),
    )
    if parser_result.counters.n_trace_candidates_module_corrected:
        logger.info(
            "analyze: corrected %d trace candidate module prefix(es) against "
            "the from-imports of %s",
            parser_result.counters.n_trace_candidates_module_corrected,
            changed_file.path,
        )

    # Merge deterministic findings before emission and caching (DECISIONS.md#054,
    # #055). Only head content proves a current finding; the base fallback must
    # not produce OBSERVED findings for deleted code. Parse-error routes exclude
    # the producer; the clean-parse module route permits it.
    subsumed_matches: list[ObservedSubsumedMatch] = []
    producer_originals: tuple[ReviewFinding, ...] = ()
    if changed_file.content_head is not None and decision.runs_observed_producer:
        # Reuse pre-call matches so coverage decisions and emitted findings agree.
        observed_findings = produce_observed_findings(
            observed_matches,
            file_path=changed_file.path,
            review_id=review_id,
            installation_id=installation_id,
            active_policy_version=active_policy_version,
        )
        producer_originals = tuple(observed_findings)
        if observed_findings:
            parser_result, new_subsumed = _merge_producer_findings(parser_result, observed_findings)
            subsumed_matches.extend(new_subsumed)

        # Emit shadow coverage only after a successful model call, preserving failure
        # semantics and audit order. The enforced branch already emitted and returned
        # (DECISIONS.md#049).
        if observed_skip_event is not None:
            await analyze_event_sink.emit_observed_skip_shadow(
                # Revalidate when adding the worker's phase key to the coverage event.
                ObservedSkipShadowEvent.model_validate(
                    {**observed_skip_event.model_dump(), "phase_key": phase_key}
                )
            )

    await _emit_parser_rejections(
        parser_result=parser_result,
        analyze_event_sink=analyze_event_sink,
        review_id=review_id,
        is_eval=is_eval,
        phase_key=phase_key,
    )

    # Return admitted findings for the caller's round cap before any FindingEvent (FUP-180).

    # Cache only completed clean passes, using the earlier lookup key. Persist
    # admitted findings and subsumption proof together; rejected responses are
    # not reusable review results. Even valid JSON with max_tokens is incomplete
    # and must not be cached.
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
                    # Retain query proof for subsumed findings missing from `findings` (#055).
                    "subsumed_matches": [m.model_dump(mode="json") for m in subsumed_matches],
                },
                model=analyze_model,
                prompt_template_version=analyze_prompt.VERSION,
                trivial_filter_version=TRIVIAL_FILTER_VERSION,
                query_registry_digest=QUERY_REGISTRY_DIGEST,
                active_policy_version=active_policy_version,
                analyze_parser_version=ANALYZE_PARSER_VERSION,
                prompt_hash=_canonical_prompt_hash(
                    system_prompt=prompt_parts.system_prompt, user_prompt=prompt_parts.user_prompt
                ),
                # Telemetry uses the same host fields as the cache key.
                profile_id=profile_id,
                reasoning_enabled=reasoning_enabled,
            )
        except CacheStoreError:
            # Admission has completed; a failed write only loses memoization.
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
        subsumed_matches=tuple(subsumed_matches),
        producer_findings=producer_originals,
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
    """Review error-free scopes for a resolved trace target without a PR patch.

    Check eligibility and budget, then call the model with the active source
    finding's context. Price and parse the response, audit rejections, and
    return findings for the caller's round cap. Excluding an error-bearing
    scope also disables OBSERVED query proofs for the file.
    """
    if not _language_supported(fetched_file.path):
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
            skip_reason=SkipReason.UNSUPPORTED_LANGUAGE,
        )

    content = fetched_file.content_head
    content_bytes = content.encode("utf-8")
    file_byte_length = len(content_bytes)

    parse_result: ParseResult = parse_source(
        content_bytes,
        fetched_file.path,
        import_path_resolver,
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
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
            skip_reason=parser_skip_reason,
        )

    if parse_result.parser_outcome == "failed":
        # No patch context exists for a degraded prompt; retain trace resolution
        # in the audit trail and record an analysis skip.
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # INFERRED admission checks trace paths against included scope names.
    # Exclude error-bearing scopes so they cannot authorize structural proof.
    included_scope_units = tuple(
        su for su in parse_result.scope_units if not parse_result.has_error.get(su.unit_id, False)
    )
    if not included_scope_units:
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # Queries scan full file content. If a scope was excluded, clear query IDs
    # so a match in that scope cannot authorize OBSERVED proof.
    is_python = _is_python_file(fetched_file.path)
    query_match_id_set: frozenset[str] = (
        frozenset()
        if len(included_scope_units) != len(parse_result.scope_units)
        else _build_query_match_id_set(content_bytes, file_path=fetched_file.path)
    )
    prompt_parts = analyze_prompt.render_post_trace(
        file_path=fetched_file.path,
        scope_unit_context=_assemble_scope_unit_context(
            included_scope_units=included_scope_units,
            source_bytes=content_bytes,
            fence_lang=_fence_lang_for(fetched_file.path),
        ),
        query_match_id_list=_assemble_query_match_id_list(query_match_id_set),
        # Each source finding gets its own pass over the target. Use its active ID
        # with its content; fetched_file retains only the first source's provenance.
        source_finding_id=source_finding.finding_id,
        source_finding_title=source_finding.title,
        source_finding_description=source_finding.description,
        source_finding_evidence=source_finding.evidence,
        pass_index=pass_index,
    )

    estimated_tokens = (
        _estimate_tokens(prompt_parts.system_prompt)
        + _estimate_tokens(prompt_parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
    )
    if estimated_tokens > per_file_cap_tokens or estimated_tokens > remaining_budget_tokens:
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=fetched_file.path,
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
            skip_reason=SkipReason.COST_BUDGET_EXHAUSTED,
        )

    await _emit_examination(
        file_examination_sink=file_examination_sink,
        review_id=review_id,
        is_eval=is_eval,
        file_path=fetched_file.path,
        phase_key=None,  # pass 1 is the legacy un-keyed stream by design
    )

    # Mark these scopes as trace expansion rather than PR changes.
    context_summary: tuple[ContextManifestEntry, ...] = _build_context_manifest(
        fetched_file.path, included_scope_units, inclusion_reason="trace_expansion"
    )
    request = LLMRequest(
        model=analyze_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=analyze_prompt.MAX_TOKENS,
        temperature=analyze_prompt.TEMPERATURE,
        review_id=review_id,
        node_id="analyze",
        is_eval=is_eval,
        prompt_template_version=analyze_prompt.VERSION,
        degraded_mode=False,
        degradation_reason=None,
        context_summary=context_summary,
        # Both passes use the pinned schema; refusal and truncation still need parser checks.
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
    )
    response: LLMResponse = await provider.complete(request)

    cost_decimal = _price_model_response(response)

    # INFERRED trace paths may cite only the single rendered name of each scope
    # (evidence-tier-schema-enforced). Adding bare-name aliases could let duplicate
    # methods in different classes satisfy the same proof.
    valid_trace_path_elements = frozenset(
        rendered_name
        for scope_unit in included_scope_units
        if (rendered_name := (scope_unit.qualified_name or scope_unit.name))
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
        # This Python-only scan returns empty for any error-bearing tree, even
        # after scope filtering. Other languages receive an inert empty scan.
        parameterized_call_scan=scan_parameterized_calls(content_bytes)
        if is_python
        else ParameterizedCallScan(),
        # Correction is a no-op here (candidate collection is pass-0-only)
        # but threading the imports keeps both call sites uniform.
        import_refs=parse_result.imports,
    )

    await _emit_parser_rejections(
        parser_result=parser_result,
        analyze_event_sink=analyze_event_sink,
        review_id=review_id,
        is_eval=is_eval,
        phase_key=None,  # pass 1: legacy un-keyed stream
    )

    # Return admitted findings for the caller's round cap before any FindingEvent (FUP-180).

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
    phase_key: str | None,
) -> FindingProposalRejectedEvent:
    """Lift a parser-side `ProposalRejection` payload into a
    `FindingProposalRejectedEvent`. The parser produced the content
    fields; the node body adds the audit-context fields (`review_id`,
    `is_eval`) here. Other audit-context fields (`event_id`,
    `timestamp`, `sequence_number`, `node_id`, `event_type`) populate
    via the event's default factories / Literal defaults."""
    return FindingProposalRejectedEvent(
        phase_key=phase_key,
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
    phase_key: str | None,
) -> AnalyzeResponseRejectedEvent:
    """Lift a parser-side `ResponseRejection` into an
    `AnalyzeResponseRejectedEvent`. Same audit-context add as
    `_lift_proposal_rejection`."""
    return AnalyzeResponseRejectedEvent(
        phase_key=phase_key,
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
