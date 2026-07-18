# See DECISIONS.md#018, #025 (proposal_hash threaded through admitted
# and rejected lift sites per #025 point 1).
# See also DECISIONS.md#024-trace-candidates-are-dotted-python-import-strings-v1
# (admission path; same-file inline handling).
"""Analyze node body — orchestration around the proof-boundary parser.

Assembles inputs, enforces triage gating, calls the provider, hands the
raw response to `analyze_parser.parse_analyze_response`, lifts parser
rejection payloads into audit events, returns state deltas. Admission
logic lives in `analyze_parser.py`; this module does NOT replicate it.

Wiring: TWO graph vertices live here since the fan-out cutover
(specs/2026-07-05-parallel-analyze.md) — `analyze` (the pass-0 PLANNER
+ the sequential pass-1 body, Command-routing) and `analyze_file` (the
per-file Send worker wrapping `_process_one_file`); the third vertex,
`analyze_aggregate`, lives in `analyze_aggregate.py` with the fold it
consumes. All kwarg-bound deps; `build_graph` binds them via
`functools.partial` (same convention as triage).

**Provider-failure policy.** `LLMProviderError` propagates without a
try/except wrapper — no worker-level retry machinery. On a mid-pass
failure, the finished files' audit events have already landed and the
start `ReviewPhaseEvent` is dangling without a matching end — that's
the audit signal for "pass interrupted." A blanket try/except would
mask transport failures as fake skip outcomes.

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
  (`degradation_reason="tree_has_error_no_scope"`, DECISIONS#033) OR — the
  one non-parse-defect cause — a diff whose ADDED lines all sit outside
  any scope unit carries an eligible module-level OBSERVED match (`degradation_reason=
  "module_level_observed_match"`, the module-scope admission arm: the
  producer RUNS on this route and its OBSERVED findings merge with the
  degraded pass; `parse_status` stays "clean"). Parser
  admits JUDGED only, gated on `span_within_file` AND
  `span_within_degraded_context` (FUP-138).
- `skipped+NO_REVIEWABLE_CONTEXT` — both `content_head` and
  `content_base` are None (V1-unreachable: `ChangedFile.enforce_status_invariants`
  guarantees every valid status has ≥1 content side) OR parse failure
  with no added text (V1-unreachable per the `failed+degraded_llm` note
  below). Branch kept as a structural slot for the future
  schema-relaxation / raw-bytes paths. No LLM call.
- `skipped+NO_CHANGED_SCOPE_UNITS` — clean parse but no scope unit
  intersects the changed regions, OR clean parse with no patch — UNLESS an
  eligible module-level OBSERVED match sits on the added lines, which
  routes to `degraded+degraded_llm` (`module_level_observed_match`) above
  instead of skipping.
- `skipped+COST_BUDGET_EXHAUSTED` — cost gate fired before the LLM
  call.
- `skipped+UNSUPPORTED_LANGUAGE` — no registered ast_facts adapter
  for the file's extension (`.go`, `.rs`, `.vue`, …; the registry
  covers Python + JS/TS/TSX). Capability-scoped per `DECISIONS.md#018`
  Amended 2026-05-21 — the value names "today's analyze cannot review
  this," not "Outrider forever cannot."
- `skipped+ALL_SCOPES_TRIVIAL` — enforcing-mode trivial-scope filter:
  every admitted scope classified ordinary-comment-only, so the LLM
  call is skipped (the shadow default never produces this). Fires
  after the cost gate per `DECISIONS.md#018` Amended 2026-06-11.

**V1 unreachable: `failed+degraded_llm`.** Spec §7 step 3a names this
outcome; the analyze code path is wired to handle it, but in V1 the
trigger cannot fire. The `parse_*` entry points only produce
`parser_outcome="failed"` on a UTF-8 strict-decode failure ([ast_facts/python_adapter.py]
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

**Registry-query firing.** For clean+full_llm, every structural query
id registered for the file's catalog language
(`queries.registry.structural_query_ids_for(language)`) is fired
against the file content; the matching subset becomes
`query_match_id_set` passed to the parser. OBSERVED claims with an id
outside the set reject. A language registering no structural queries
(JS/TS today) gets the empty set — every model OBSERVED claim rejects.

**Token estimation.** `_estimate_tokens` counts UTF-8 bytes with
ceiling division (`_BYTES_PER_TOKEN = 3`). Conservative-up for code-
heavy / multi-byte content; over-estimates the budget rather than
under-estimates. A tokenizer-grade estimate is FUP-049 scope.
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
from outrider.agent.nodes.finding_cap import admit_with_pair_dedup, cap_findings_by_severity
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
    """UTF-8 byte count with ceiling division by `_BYTES_PER_TOKEN`.

    Conservative-up: over-estimates rather than under-estimates so the
    cost gate fails safer. Codepoint-counting would under-count multi-
    byte sequences (a 3-byte CJK char → `1 // 3 == 0` tokens). The
    estimator's job is order-of-magnitude blowup detection, not a tight
    count; a tokenizer-grade estimate is FUP-049.
    """
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
            for stale in [known for known in self._by_loop if known.is_closed()]:
                del self._by_loop[stale]
            semaphore = self._by_loop.get(loop)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._permits)
                self._by_loop[loop] = semaphore
        return semaphore


class AnalyzeWorkerPayload(BaseModel):
    """The self-contained Send payload for one `analyze_file` worker.

    Pure data (specs/2026-07-05-parallel-analyze.md): a Send worker
    receives ONLY this payload as its input — never the parent
    `ReviewState` — so it carries everything `_process_one_file` needs
    that isn't a closure dep. `allocation_tokens` is the planner's
    pre-flight budget verdict (`plan_file_budgets`): the worker's real
    rendered-prompt estimate is gated against it, so a worker can never
    spend past its allocation and N concurrent workers can never
    overshoot the pools. An unfunded file carries 0 and skips
    COST_BUDGET_EXHAUSTED at the gate, exactly like the sequential
    drawdown's starved files.

    Checkpoint weight: the `ChangedFile` (content + patch) duplicates
    into the superstep's pending sends; bounded by intake's size gates
    and dropped once the worker completes.
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
    """The per-file Send worker: one `(file, pass)` slot per invocation.

    Wraps `_process_one_file` (all per-file audit events fire inside it,
    under the logical `node_id="analyze"` per `DECISIONS.md#064`) and
    returns the slot's `AnalyzeWorkerOutcome` for the slot-guard reducer;
    the aggregate step folds outcomes into the pass's round. Emits no
    round and no pass-level event; it DOES own its keyed phase pair —
    `phase_key = file:<path>#<pass>` (injective even for paths containing
    `#`: the pass index is an integer final segment) — and stamps the
    same key on every per-operation event `_process_one_file` emits,
    including `LLMRequest.phase_key` for the provider to mirror onto
    `LLMCallEvent`. The pair opens AFTER the semaphore admits the body:
    the envelope bounds work, and queueing is not work
    (`phase-events-bound-work`).

    Failure policy (spec): NO worker-level retry machinery — provider
    retries are unchanged, and a worker exception fails the pass, parity
    with the sequential loop's abort behavior.

    `concurrency_semaphore` bounds in-flight worker bodies
    (`ANALYZE_MAX_CONCURRENCY`): build_graph closes in an
    `AnalyzeConcurrencyGate` (one semaphore per running loop — a bare
    Semaphore would bind to the first loop and crash on a second); tests
    may pass a bare `asyncio.Semaphore`. None (direct invocation) runs
    unbounded.
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
            # WORKER-SIDE ALLOCATION ENFORCEMENT: the existing cost gate
            # compares the REAL rendered-prompt estimate against this value,
            # so the pre-flight allocation is the worker's whole budget —
            # a proxy under-estimate costs coverage (a skip), never budget.
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


async def analyze(
    state: ReviewState,
    *,
    provider: LLMProvider,
    analyze_model: str,
    standard_analyze_model: str,
    # Host-identity triad (DECISIONS.md#056) closed in at build_graph for the AnalyzeCompletedEvent
    # (default None = unqualified until lifespan wiring; the event validator enforces coherence).
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
    """The analyze entry vertex: pass-0 PLANNER, pass-1 sequential body.

    Pass 0 (the fan-out cutover, specs/2026-07-05-parallel-analyze.md):
    emit the pass's phase start, resolve the cache scope, build the
    kept-file worklist, allocate per-file budgets pre-flight
    (`plan_file_budgets` over the bytes-based proxy estimate), and
    return `Command(goto=[Send("analyze_file", payload), ...])` — one
    self-contained payload per kept file. Zero kept files routes to
    "analyze_aggregate" directly (the empty pass still folds one empty
    round + completed event). No per-file work happens here; the
    planner emits no per-file events.

    Pass 1 (trace re-entry) stays SEQUENTIAL and byte-unchanged: the
    trace-fetched loop, round assembly, event emissions, and phase end
    all run in this body, and the Command routes to "synthesize" (the
    depth-2 ceiling means a post-pass-1 round count can never route
    back to trace).

    Counter source-of-truth (pass 1): per-file local bookkeeping
    accumulators. NEVER re-read from the audit stream.
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
    # Phase identity per pass shape (increment 4): pass 0 emits the keyed
    # PLANNER pair (`plan#<pass>`; attempt_key = phase_key VERBATIM, so
    # phase_id inherits the key's retry stability and collision-freedom);
    # the sequential pass-1 body keeps the legacy un-keyed
    # `analyze-pass-<n>` envelope. Both derive from pre-merge state, so
    # replay re-emission collapses on the PhaseEventSink idempotency.
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
    # Monotonic anchor for the round DURATION (FUP-141): `ended_at` is derived
    # from this rather than a second wall-clock read, so a backwards clock jump
    # (NTP step / VM resume / WSL2 skew) between start and end can't make
    # `ended_at < started_at` and trip the `AnalysisRound` invariant mid-review.
    # `started_at` stays wall-clock for the audit trail. Precondition: this mark
    # and the matching end read stay in one process (`monotonic()` is per-process;
    # a V1.5 parallel-analyze fan-out must re-anchor per worker, not hoist this).
    started_mono = time.monotonic()
    per_file_cap_tokens = _compute_per_file_cap(total_review_budget_tokens)

    # Step 1: start phase event (the plan# pair on pass 0; the legacy
    # envelope on pass 1). If this raises (audit infra outage), the node
    # fails before any work — no dangling start.
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
    # Cross-type subsumption proof-retention records (DECISIONS.md#055), aggregated
    # across files for the per-pass AnalyzeCompletedEvent. Set on every _FileOutcome
    # (normal merge + cache serve), so it accumulates regardless of cache hit/miss.
    pass_subsumed_matches: list[ObservedSubsumedMatch] = []
    files_examined: list[str] = []
    files_skipped: list[str] = []
    # FUP-044 ext 3: count COST_BUDGET_EXHAUSTED skips in this pass (pass-0
    # changed-file set) to drive the COST_BUDGET_STARVATION anomaly after the loop.
    budget_skip_count = 0
    n_proposals_seen = 0
    n_findings_emitted = 0
    n_findings_served = 0
    n_findings_observed = 0
    # FUP-180: content_hashes of cache-SERVED findings, so the post-loop cap can
    # classify a dropped served finding back to its origin (served findings can be
    # any evidence_tier, so origin isn't derivable from the finding alone — only the
    # serve branch knows). Producer-OBSERVED vs proposal IS derivable (OBSERVED_QUERY_IDS).
    served_content_hashes: set[str] = set()
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
        # ---- THE PLANNER STEP (fan-out cutover) ----
        # Kept-file worklist exactly as the sequential loop built it:
        # DEEP/STANDARD only, tier-descending stable order (Send dispatch
        # order follows it, so a bounded semaphore admits higher-tier
        # files first under pressure — the Stage 2 ordering survives).
        pass_zero_worklist: list[tuple[ChangedFile, ReviewTier]] = []
        if triage_result is not None:
            for changed_file in state.pr_context.changed_files:
                tier = triage_result.file_tiers.get(changed_file.path, ReviewTier.SKIP)
                if tier in (ReviewTier.DEEP, ReviewTier.STANDARD):
                    pass_zero_worklist.append((changed_file, tier))
            pass_zero_worklist.sort(key=lambda item: _PASS0_TIER_RANK[item[1]])

        # Fixed proxy overhead, computed ONCE per review from the real
        # prompt constants: system prefix + empty user-template scaffolding
        # + the output reservation — the same three fixed terms the
        # worker's real estimate carries (omitting any would under-fund
        # every file by that constant and convert the gap into spurious
        # COST_BUDGET_EXHAUSTED skips).
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

        # Pre-flight allocation replaces the sequential drawdown: the
        # reserved/general split is unchanged (Stage 1), but allocations
        # are final — unspent tokens are NOT redistributed (the
        # later-files-benefit dynamic is deliberately given up per the
        # spec's Non-goals; utilization is measured via skip counters).
        # plan_file_budgets fails loud on a duplicate path BEFORE any
        # Send (vendor-data corruption, never silently deduped).
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
        sends = [
            Send(
                "analyze_file",
                AnalyzeWorkerPayload(
                    review_id=state.review_id,
                    installation_id=state.pr_context.installation_id,
                    is_eval=state.is_eval,
                    pass_index=pass_index,
                    review_tier=tier,
                    allocation_tokens=allocation_by_path[changed_file.path].allocation_tokens,
                    changed_file=changed_file,
                    cache_scope=cache_scope,
                ),
            )
            for changed_file, tier in pass_zero_worklist
        ]
        # Close the planner's own phase pair: planning work is bounded
        # here; the workers and the aggregate own their own envelopes.
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
        # `analyze_pass_started_at` rides state to the aggregate (the pass
        # spans vertices; a monotonic anchor cannot). Zero-worker route:
        # no Sends would strand the aggregate — route to it by name so the
        # empty pass still folds one empty round + completed event.
        return Command(
            update={"analyze_pass_started_at": started_at},
            goto=sends if sends else "analyze_aggregate",
        )
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
                admit_with_pair_dedup(
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
            # Cross-type subsumption records (DECISIONS.md#055) from this file's
            # outcome — empty for trace-fetched files (no OBSERVED producer runs
            # there) and for cache misses with nothing subsumed.
            pass_subsumed_matches.extend(file_outcome.subsumed_matches)
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

    # Step 3h (FUP-180): finalize the round's findings BEFORE any FindingEvent fires, so
    # the emitted audit stream == the round by construction (no mid-assembly strand).
    #
    # 1. Collapse content_hash duplicates (first-wins). The cold-parse proposal path can
    #    admit two findings sharing a content_hash but differing in proposal_hash (same
    #    file/line/finding_type, different prose); `_admit_with_dedup` keys on the PAIR so
    #    both survive, and `AnalysisRound._enforce_findings_unique` (content_hash) would
    #    then raise — the same collapse the augment/skip paths apply via fresh_hashes /
    #    skip_hashes (FUP-180 review finding A).
    collapsed_findings: list[ReviewFinding] = []
    seen_content_hashes: set[str] = set()
    for finding in admitted_findings:
        if finding.content_hash not in seen_content_hashes:
            collapsed_findings.append(finding)
            seen_content_hashes.add(finding.content_hash)

    # 2. Gated-aware severity cap. NON-gated findings drop to the soft cap; gated
    #    (CRITICAL/HIGH) are NEVER dropped to fit it (hitl-gates-high-severity), only the
    #    hard runaway ceiling bounds them (FUP-180 review design call).
    kept_findings, dropped_findings = cap_findings_by_severity(
        collapsed_findings, soft_cap=MAX_FINDINGS_PER_ROUND, hard_cap=MAX_FINDINGS_HARD_CAP
    )
    n_findings_dropped_over_cap = len(dropped_findings)

    # 3. Reconcile the emitted-set counters to the KEPT set (post collapse + cap). The
    #    pre-dedup loop accumulators counted what the parser admitted; the actual
    #    FindingEvents fire only over kept_findings, so recompute from kept, classified by
    #    ORIGIN (not evidence_tier): served (content_hash tracked above), producer-OBSERVED
    #    (tier OBSERVED + query_match_id in the producer registry), else a model proposal.
    #    `n_proposals_dropped` = parser proposal-emitted minus surviving proposals — covers
    #    ALL pre-emission drops (cross-source dedup, content-hash collapse, AND the cap),
    #    keeping `_enforce_proposal_accounting` balanced (FUP-180 review finding B).
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

    # NOTE: trace_candidates is intentionally NOT filtered to surviving findings. A
    # candidate whose source finding did not survive (cap-dropped, or a rejected-parent
    # proposal) stays in state.trace_candidates as a forensic-only record per
    # DECISIONS.md#025 point 6 — trace's proposal-hash join already INFO-skips any
    # unjoinable candidate (no GitHub fetch), so a cap-orphaned candidate is handled
    # identically and safely. (An earlier filter here over-removed the rejected-parent
    # forensic class; reverted.)

    # Step 4: build AnalysisRound AND construct AnalyzeCompletedEvent — BOTH BEFORE any
    # side effect. Their validators (round uniqueness + round_id; the proposal-accounting
    # equation) run here, so a validator raise crashes cleanly with NOTHING emitted,
    # closing the strand class fully (FUP-180 review finding A + the residual: the
    # completed-event accounting validators must run before the FindingEvent emit too).
    # `round_id` is content-derived so re-emission is idempotent on replay.
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
    # Counters from local accumulators — the producer-side source of truth (spec §7 step
    # 5). Constructed (accounting-validated) HERE, before any emit.
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
        # Decimal-summed across files, cast to float once. Matches
        # `sum(LLMCallEvent.cost_usd)` to within one float-cast step.
        total_cost_usd=float(total_cost_decimal),
        pricing_version=PRICING_VERSION,
        policy_version=active_policy_version,
        analyze_model=analyze_model,
        standard_analyze_model=standard_model_used,
        # Host-identity triad (DECISIONS.md#056), closed in at build_graph (None until lifespan
        # wiring; coherence enforced by the event validator).
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
    )

    # All validation done. Side effects now, in order:
    # Gated-overflow anomaly (best-effort). `len(kept) > soft cap` only when gated findings
    # ALONE exceed it — all kept (gated are never dropped; >hard_cap fails loud earlier),
    # still reaching HITL — a loud capacity signal. Observability must never fail the review.
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

    # One FindingEvent per kept finding — the emitted set equals the round by construction.
    for finding in kept_findings:
        await analyze_event_sink.emit_finding(finding, is_eval=state.is_eval)

    await analyze_event_sink.emit_analyze_completed(completed_event)

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

    # Step 7: state delta + routing. list shape (not tuple) per canonical
    # docs/spec.md §7.1 — the `append_with_dedup_by` reducer expects
    # list-of-T. Only pass 1 reaches this tail (pass 0 returned at the
    # planner), and the post-pass-1 round count can never satisfy the
    # trace router's `rounds == 1` predicate — synthesize, always.
    return Command(
        update={
            "analysis_rounds": [new_round],
            "trace_candidates": list(trace_candidates),
            # Always empty here: only pass 1 reaches this tail, and the
            # trace-fetched pass stays sequential and out of fan-out scope
            # (workers own the pass-0 writes).
            "analyze_worker_outcomes": [],
        },
        goto="synthesize",
    )


def _worker_outcome_for(
    file_outcome: _FileOutcome,
    *,
    path: str,
    pass_index: int,
    review_tier: ReviewTier,
) -> AnalyzeWorkerOutcome:
    """Map one `_FileOutcome` to its `AnalyzeWorkerOutcome`: the per-file
    branch union discriminates the source, and the builders receive the
    ORIGINAL objects (producer originals pre-clone — origin truth intersects
    by object identity; cloning happens inside the builders). Called by the
    `analyze_file` worker; the aggregate folds the outcomes into the pass's
    round (the sequential-parity contract lives in the wiring tests)."""
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
            # The #049 ENFORCED coverage skip: clean FileExaminationEvent,
            # no SkipReason — the file is EXAMINED, the LLM just never ran.
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
    """Cache-served findings + trace candidates for one file (Stage B serve
    flip). A served hit populates this on `_FileOutcome` INSTEAD of
    `parser_result` (which stays None — no LLM call), so the main loop
    accumulates the findings WITHOUT counting an LLM call: they ride
    `n_findings_served` (subtracted from the proposal-accounting equation) and
    `n_findings_emitted` (real `FindingEvent`s fired)."""

    admitted_findings: tuple[ReviewFinding, ...]
    trace_candidates: tuple[TraceCandidate, ...]


@dataclass(frozen=True, slots=True)
class _ObservedSkipResult:
    """Deterministic OBSERVED findings for a file whose LLM pass did NOT run.
    Three producers: the ENFORCED coverage skip (Step 3b-mechanism — every
    changed scope covered by `skip_safe` matches), and the two module-arm
    early-skip ride-outs (DECISIONS.md#062 as amended: COST_BUDGET_EXHAUSTED
    and ALL_SCOPES_TRIVIAL skips carry the zero-token module-level findings
    instead of dropping them). Populated on `_FileOutcome` INSTEAD of
    `parser_result` (which stays None). The main loop admits these like the
    augment path's OBSERVED findings — `n_findings_emitted` (real
    FindingEvents fired) AND `n_findings_observed` (the proposal-accounting
    subtraction channel; OBSERVED findings have no proposal lifecycle) — with
    NO LLM call counted. They still flow to synthesize + HITL."""

    admitted_findings: tuple[ReviewFinding, ...]


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
    # Non-None ONLY when the file skipped the LLM but produced OBSERVED findings:
    # the enforced skip_safe-coverage skip, or a module-finding ride-out on a
    # budget/trivial skip (parser_result is then None — no LLM call). The main
    # loop's observed-skip branch admits these findings so they reach
    # synthesize + HITL.
    observed_skip_result: _ObservedSkipResult | None = None
    # Stage 2 (FUP-044 ext 3): the skip reason, mirrored from the
    # FileExaminationEvent onto the in-memory outcome so the main loop can count
    # COST_BUDGET_EXHAUSTED skips for the starvation anomaly. None on any
    # non-skipped outcome (parse_status != "skipped").
    skip_reason: SkipReason | None = None
    # Cross-type subsumption proof-retention records (DECISIONS.md#055), set on
    # BOTH paths — the normal merge assembles them, the cache serve reconstructs
    # them from the payload — so the main loop accumulates them onto the per-pass
    # AnalyzeCompletedEvent uniformly regardless of cache hit/miss.
    subsumed_matches: tuple[ObservedSubsumedMatch, ...] = ()
    # RAW producer output (original objects, pre-clone) from the #054 merge
    # site — the parallel-analyze origin-truth source: the worker-outcome
    # builder intersects these BY OBJECT IDENTITY with the admitted set, so
    # they must be the same objects the merge placed (never clones). Empty
    # on every non-parser path.
    producer_findings: tuple[ReviewFinding, ...] = ()


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
    phase_key: str | None,
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
    """Fire every structural query registered for `file_path`'s catalog
    language against `file_content_bytes`; return the set of ids that
    produced at least one match.

    Iterates `queries.registry.structural_query_ids_for(language)` (current
    non-deprecated ids of the file's language only — the registry-language-
    aware selector per the JS/TS OBSERVED catalog spec). Per spec §7 step
    3b, this set is passed to the parser's OBSERVED admission — a model
    claim whose `query_match_id` isn't in this set rejects with
    `query_match_id_not_in_registry`. Empty set means no structural query
    is registered for (or fired against) this file → every model OBSERVED
    claim rejects; only JUDGED proposals can land. JS/TS files register no
    structural queries in V1, so their set is empty by registration — the
    dispatch-era rejection behavior, now data-driven.
    """
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
    deduped: list[ReviewFinding] = []
    seen_hashes: set[str] = set()
    for finding in produced:
        if finding.content_hash not in seen_hashes:
            deduped.append(finding)
            seen_hashes.add(finding.content_hash)
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
      unit intersects the changed regions, UNLESS an eligible
      module-level OBSERVED match sits on the added lines (the
      module-scope arm routes that to `degraded+degraded_llm` below).
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
      (`degradation_reason="tree_has_error_no_scope"`, DECISIONS#033), OR
      a diff whose added lines all sit outside any scope unit carries an
      eligible module-level OBSERVED match
      (`degradation_reason="module_level_observed_match"` — clean parse,
      `parse_status="clean"`; the producer runs and its OBSERVED findings
      merge with the degraded pass);
      degraded LLM call.
    - `clean+full_llm` — clean parse, scope units intersect changed
      regions, no `has_error` in those units.
    - Parser-stage skip — `parse_source` returned `parser_outcome=
      "skipped"` (`OVERSIZED`, `VENDORED`, etc.); the parser's
      `skip_reason` is the audit value.
    """
    # Language gate: registry dispatch (dispatch spec). Triage doesn't
    # filter by language and `ChangedFile.language` is unpopulated, so a
    # file with no registered adapter (`.go`, `.rs`, …) classified
    # DEEP/STANDARD would otherwise reach a parser that can't handle it.
    # Routes through `SkipReason.UNSUPPORTED_LANGUAGE` per
    # `DECISIONS.md#018` Amended 2026-05-21 — capability-scoped to the
    # current registry, not a forever-claim about language support.
    if not _language_supported(changed_file.path):
        return await _emit_skip(
            file_examination_sink=file_examination_sink,
            review_id=review_id,
            is_eval=is_eval,
            file_path=changed_file.path,
            phase_key=phase_key,
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
            phase_key=phase_key,
            skip_reason=SkipReason.NO_REVIEWABLE_CONTEXT,
        )

    # `file_byte_length` computed ONCE here per spec §7 step 3a;
    # passed to parser §5 unchanged so it never recomputes per
    # proposal.
    content_bytes = content.encode("utf-8")
    file_byte_length = len(content_bytes)

    parse_result: ParseResult = parse_source(
        content_bytes,
        changed_file.path,
        import_path_resolver,
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
            phase_key=phase_key,
            skip_reason=parser_skip_reason,
        )

    # `None` covers three cases: no patch (binary / oversized response), file
    # absent from a well-formed patch, or path-validation failure (the helper
    # returns None for those). Computed after the parser-skip return because the
    # degraded prompt below also needs `patched_file`.
    patched_file = lookup_patched_file(changed_file.patch, changed_file.path)

    # FUP-217: patch/head misalignment probe. `added_line_byte_ranges` raises
    # CoordinateError when a patch target line lies beyond the fetched source
    # (force-push racing intake's files-list vs content fetches). Probed ONCE
    # here, before any route, because every downstream coordinate anchor for
    # the file is unsound against mismatched content: the degraded span veto
    # recomputes these ranges AFTER the LLM call (an uncontained raise there
    # aborted the whole review), the module-scope arm denies on them, and a
    # "clean" review would hand publish wrong-line comment locations. Added
    # lines are HEAD-side coordinates, so the probe anchors on head content
    # ONLY: when head is absent, an added-line patch has nothing to anchor
    # against and skips as the same misalignment class — probing the base
    # fallback would validate head coordinates against the wrong text and
    # pass wrongly. Deletion-only patches over base content (removed files)
    # carry no added lines and proceed unprobed, unchanged.
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

    # Module-scope arm inputs (DECISIONS.md#062), derived through the ONE
    # gated helper (fully error-free parse; patch/head-misalignment
    # contained to an inert arm — production-unreachable now that the
    # probe above skips misaligned files first; kept as defense) — needed
    # on the clean route too (the with-scopes arm + its cache-key
    # digest), so derived up front. The ROUTING SWEEP below is lazy; only
    # this derivation is unconditional.
    module_all_scope_units, module_added_ranges = module_admission_inputs(
        parse_result, patched_file, changed_file.content_head
    )

    # Outcome determination (skip / degraded / clean) for a PARSED file lives in the
    # pure `decide_degradation` (degradation.py) — extracted so structural eval
    # scenarios can exercise it LLM-free. This node is the only place that turns the
    # decision into behavior. The `"failed"` degraded branch is V1-unreachable
    # (intake gates invalid UTF-8 with SkipReason.OVERSIZED); retained for the
    # raw-bytes intake path (FUP-053) + audit/prompt-wiring tests.
    decision = decide_degradation(parse_result, patched_file)

    # Module-scope routing (DECISIONS.md#062), evaluated LAZILY: the catalog
    # sweep runs only when the file would otherwise skip at
    # NO_CHANGED_SCOPE_UNITS — the ONE branch that consults the candidate —
    # and `module_level_observed_matches` itself short-circuits languages
    # with no eligible query, so the common route (files with changed
    # scopes) and every Python file pay nothing. `decide_degradation` is
    # pure, so re-deciding on a non-empty sweep is free; the admitted
    # matches are REUSED as the module route's final set below (one sweep,
    # zero drift between routing and production).
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

    # Step 3b: registry-query firing (skip for degraded mode). The builder
    # is registry-language-aware (JS/TS OBSERVED catalog spec): it selects
    # the structural queries registered for THIS file's language and runs
    # them under the file's grammar, so a query never fires over bytes of
    # another language (no error-recovery garbage matches). A language with
    # no structural queries (JS/TS in V1) or no catalog at all yields the
    # empty set — the prompt stays honest ("do not claim observed") and any
    # model observed claim on such a file rejects at admission, exactly the
    # dispatch-era behavior, now by registration instead of a Python gate.
    is_python = _is_python_file(changed_file.path)
    query_match_id_set: frozenset[str] = (
        frozenset()
        if degraded_mode
        else _build_query_match_id_set(content_bytes, file_path=changed_file.path)
    )

    # The from-import refs the trace-candidate machinery may consume.
    # Feeds BOTH the cache key digest and the parser call below from one
    # binding (FUP-171 anti-fork). Unconditional since the resolver spec:
    # every registry language collects candidates. NOTE the correction
    # map is NOT necessarily empty for JS/TS (a bare specifier like
    # 'express' is a valid identifier and enters the map); what keeps
    # corrections off JS/TS is the parser's module-form-only correction
    # gate — the map's only effect for specifier-form files is keying
    # the cache digest.
    trace_import_refs = parse_result.imports

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
                included_scope_units=included_scope_units,
                source_bytes=content_bytes,
                fence_lang=_fence_lang_for(changed_file.path),
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
        # Budget exhaustion must not drop the module arm's zero-token
        # OBSERVED findings (DECISIONS.md#062 as amended: the early-skip
        # ride-out). Two shapes reach here: the module ROUTE already carries
        # its routing-sweep matches; a WITH-SCOPES clean-route file (changed
        # function + changed module-level line) never ran the sweep — run it
        # now, lazily, only because the gate actually fired (the sweep
        # short-circuits ineligible languages and empty ranges).
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
            # The LLM pass is skipped — and counted, the
            # COST_BUDGET_EXHAUSTED accounting stands — but the
            # deterministic findings cost zero tokens and ride out.
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
    # The pre-gate `parse_source` parse stays separate (it fed degradation + the
    # token estimate); this runs strictly AFTER the cost gate, so a cost-skipped
    # file never reaches it (COST_BUDGET_EXHAUSTED-before-classification holds).
    # The scan rides every clean file (None in degraded mode — also exactly when
    # the file is not cacheable, so `parameterized_call_scan is not None` IS the
    # clean-mode cache gate below). The SAME scan object feeds BOTH the cache-key
    # digest AND the admission veto in parse_analyze_response, so the keyed and
    # admitted inputs can never fork (FUP-171 anti-fork). `compute_triviality`
    # mirrors the classification gate so triviality (+ its base parse) builds
    # only when there's a patch and included scopes.
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
        # Non-Python: the trivial-scope classifier and the FUP-162 scan
        # are Python-grammar surfaces (dispatch spec). Clean files carry
        # an EMPTY scan, NOT None — None means degraded/not-cacheable
        # (the clean-mode cache gate below keys on `is not None`), while
        # an empty scan keeps the sql_injection veto naturally inert
        # (no execute-like calls to match) and `scan_digest(empty)` keys
        # the cache stably.
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
                # Before dropping to the trivial skip: a changed
                # module-level match is NOT trivial-scope coverage and must
                # not be silently dropped with it (DECISIONS.md#062). The
                # clean route never ran the routing sweep, so run it here —
                # lazily, only when the skip actually fires; the sweep
                # short-circuits ineligible languages. Structurally
                # unreachable today (triviality is Python-gated; the sole
                # eligible query is JavaScript) — this guards the first
                # Python eligible query / JS trivial classifier.
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
            # Span-filter the fired query-ID set to the kept scopes. The
            # SAME filtered set feeds the prompt's query_match_id_list AND
            # the parser's OBSERVED admission below — filtering only the
            # prompt text would let a finding anchored in a shown scope
            # cite a query ID whose match lives in an excluded scope (an
            # OBSERVED proof pointing at never-shown code).
            query_match_id_set = _filter_query_ids_to_scopes(
                query_match_id_set, content_bytes, included_scope_units, file_path=changed_file.path
            )
            # Re-render over the kept scopes; this filtered prompt is what
            # is actually sent (and what context_summary describes).
            parts = analyze_prompt.render(
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
            subsumes_digest=SUBSUMES_DIGEST,
            # Candidate correction's per-file input (#024 from-import
            # amendment): corrected siblings depend on imports the
            # rendered prompt doesn't carry.
            from_import_map_digest=from_import_map_digest(trace_import_refs),
            # Import-binding admission's per-file input: `_binding_admits`
            # joins OBSERVED matches against ALL imports (module + value
            # marker + names), most of which the from-import map excludes —
            # same refs the producer consumes
            # (`trace_import_refs IS parse_result.imports`).
            import_bindings_digest=import_bindings_digest(trace_import_refs),
            # The shadowing guard's per-file input — bindings can live in
            # enclosing-but-not-included scopes the prompt never shows.
            lexical_bindings_digest=lexical_bindings_digest(parse_result.lexical_bindings),
            # The module-scope arm's per-file input: added-line ranges +
            # the module-level bytes they cover + every parsed scope span
            # (the disjointness input) — all outside prompt bytes.
            module_admission_digest=module_admission_digest(
                module_added_ranges, module_all_scope_units, content_bytes
            ),
            profile_id=profile_id,
            reasoning_enabled=reasoning_enabled,
            profile_contract_digest=profile_contract_digest,
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
                        phase_key=phase_key,
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
        phase_key=phase_key,
    )

    # Step 3e.5 (Step 3b-mechanism): compute the OBSERVED coverage decision BEFORE the
    # LLM call so an enforced `would_skip` short-circuits provider.complete. The matches
    # are reused by the post-LLM OBSERVED block; the shadow event is EMITTED at its
    # existing post-LLM point for the non-enforced path (byte-identical audit ordering —
    # a provider failure still leaves no shadow event), and only HERE, pre-LLM-return, on
    # an enforced skip. Same clean-parse + head-content gate as the post-LLM block.
    observed_matches: tuple[ObservedMatch, ...] = ()
    observed_skip_event: ObservedSkipShadowEvent | None = None
    # Language selection lives INSIDE `run_observed_matches` (JS/TS OBSERVED
    # catalog spec): the producer runs the queries registered for this
    # file's language under this file's grammar — Python files run the
    # Python catalog, JS/TS files the javascript catalog, and a language
    # with no catalog selects zero queries (inert producer, the
    # dispatch-era safety preserved by registration). No queries ever
    # execute over bytes of another language's grammar.
    # The module-scope degraded route (`module_level_observed_match`) is the
    # ONE degraded reason under which the producer runs — its module-level
    # arm only (no included scopes exist on that route, so the containment
    # arm is naturally inert). Every other degraded reason keeps the producer
    # OFF (an error-recovered/failed tree is not trusted for structural
    # proof). Module-level matches are excluded from #049 skip coverage
    # (DECISIONS.md#062) at three layers: the shadow call below is gated
    # off the module ROUTE, `compute_observed_skip_shadow` filters
    # `module_level` matches out of coverage in clean mode, and the schema
    # floor rejects eligible+SKIP_SAFE queries outright.
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
                # Import-binding admission: the producer proves a
                # name-anchored match binds to its dangerous API via the
                # file's extracted imports — and rejects it when a local
                # binding shadows the anchor (or a guarded global) at the
                # match site.
                import_refs=parse_result.imports,
                lexical_bindings=parse_result.lexical_bindings,
                # Module-scope arm inputs — the with-scopes case (a changed
                # function AND a changed module-level line in one file)
                # admits through the same arm; `module_added_ranges` is
                # empty for error-bearing parses and patch-less files,
                # keeping the arm inert exactly there.
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
            # ENFORCED SKIP: every changed scope is skip_safe-covered, so the LLM is
            # NOT called. Emit the OBSERVED findings (they reach synthesize + HITL via
            # the round, preserving hitl-gates-high-severity) + the shadow event with
            # skip_enforced=True, then return. The FileExaminationEvent (clean) was
            # already emitted above. No #054/#055 merge (no LLM parser_result to merge
            # against); no cache write (skip outcomes are never memoized).
            produced = produce_observed_findings(
                observed_matches,
                file_path=changed_file.path,
                review_id=review_id,
                installation_id=installation_id,
                active_policy_version=active_policy_version,
            )
            # Collapse same-content_hash producer duplicates BEFORE emit/admit — the
            # same first-wins reason as the augment path's #054 `fresh_hashes`:
            # content_hash excludes evidence_tier, so two OBSERVED findings of the same
            # finding_type matching the same span (two queries) would otherwise fire
            # duplicate FindingEvents AND trip AnalysisRound._enforce_findings_unique.
            # No JUDGED to supersede here (the LLM never ran).
            skip_findings: list[ReviewFinding] = []
            skip_hashes: set[str] = set()
            for produced_finding in produced:
                if produced_finding.content_hash not in skip_hashes:
                    skip_findings.append(produced_finding)
                    skip_hashes.add(produced_finding.content_hash)
            # model_validate (not model_copy) so the skip_enforced=True event is
            # RE-VALIDATED (skip_enforced => would_skip); the branch already guarantees
            # would_skip, but model_copy bypasses validators — re-validation keeps the
            # audit-event contract enforced even if the gate above ever changes.
            await analyze_event_sink.emit_observed_skip_shadow(
                ObservedSkipShadowEvent.model_validate(
                    {
                        **observed_skip_event.model_dump(),
                        "skip_enforced": True,
                        "phase_key": phase_key,
                    }
                )
            )
            # FindingEvent emission for these OBSERVED findings is DEFERRED to the
            # main loop's post-cap step (FUP-180); they ride out on the returned
            # `_ObservedSkipResult`.
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
        # Worker phase attribution (parallel-analyze increment 4): providers
        # mirror this verbatim onto LLMCallEvent.phase_key.
        phase_key=phase_key,
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
    # modulo a single float-cast step rather than per-file FP drift —
    # priced on (response.profile_id, response.model), the SAME host+model the
    # provider/persister bill against, so SDK model substitution can't drift the
    # aggregate from the per-call event costs.
    cost_decimal = compute_cost_usd(
        response.profile_id,
        response.model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
        # Response-derived pricing context (openai-native-host spec): tier-echo
        # expectation derives INSIDE pricing.py from profile_id, so an
        # echo-expecting host priced without these would classify absent_tier
        # and raise AFTER the billed call. A deviant response never reaches
        # this site — the provider raises the terminal contract error first —
        # so the default-tier context here always prices flat/long.
        billed_prompt_tokens=response.billed_prompt_tokens,
        service_tier=response.service_tier_actual,
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
        # From-import candidate correction: the file's own imports are the
        # deterministic ground truth for a symbol's module; a candidate
        # whose module prefix they contradict gains a corrected
        # module-form sibling at admission (emitted alongside the
        # original, never instead). `trace_import_refs` is the SAME
        # object the cache key digested (FUP-171 anti-fork: keyed input
        # and consumed input cannot diverge).
        import_refs=trace_import_refs,
        # Per-language candidate form (DECISIONS.md#024 Amended
        # 2026-07-03, ANALYZE_PARSER_VERSION v7): Python admits dotted
        # module strings, JS/TS admit leading-dot relative specifiers —
        # the parser drops the wrong form and enforces repo-escape
        # containment at admission for specifiers. Registry-derived
        # (totality-asserted), not an is_python ternary — the form is a
        # per-language registration, not a Python/other binary.
        trace_candidate_form=_trace_candidate_form_for(changed_file.path),
    )
    if parser_result.counters.n_trace_candidates_module_corrected:
        logger.info(
            "analyze: corrected %d trace candidate module prefix(es) against "
            "the from-imports of %s",
            parser_result.counters.n_trace_candidates_module_corrected,
            changed_file.path,
        )

    # Deterministic OBSERVED-tier findings (Cost Lever 3): augment the LLM's
    # JUDGED findings with structural security-query matches. Runs in clean
    # mode and on the ONE clean-parse degraded route
    # (`module_level_observed_match` — the module-scope arm's whole point);
    # the error-caused degraded routes keep the producer off. Merged into
    # `parser_result.admitted_findings` BEFORE the emit/cache/return below, so
    # they ride the audit stream, the cache payload (serve reconstructs them),
    # and the round identically to LLM findings. Collapsed by content_hash —
    # `AnalysisRound` requires unique content_hashes. On a same-(file,lines,type)
    # collision with a model JUDGED finding, prefer-OBSERVED (DECISIONS.md#054)
    # EVICTS the JUDGED and keeps the OBSERVED (its query_match_id is the stronger
    # proof); a collision with an already-OBSERVED incumbent keeps the incumbent.
    # signal_only: the LLM still ran; OBSERVED augments it, never skips it.
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
    # Cross-type subsumption records (DECISIONS.md#055), assembled in the merge
    # below and threaded out for the per-pass `AnalyzeCompletedEvent`; stays empty
    # in degraded mode and when nothing is subsumed.
    subsumed_matches: list[ObservedSubsumedMatch] = []
    producer_originals: tuple[ReviewFinding, ...] = ()
    if changed_file.content_head is not None and decision.runs_observed_producer:
        # `observed_matches` was computed PRE-LLM (Step 3b-mechanism) so an enforced
        # `would_skip` could short-circuit the LLM; reuse the same matches here for the
        # findings producer + the #054 merge (a single deterministic OBSERVED query pass).
        # On the module-scope degraded route the same merge applies: the producer's
        # module-level OBSERVED findings join (and #054-evict colliding JUDGED from)
        # the degraded pass's admitted findings.
        observed_findings = produce_observed_findings(
            observed_matches,
            file_path=changed_file.path,
            review_id=review_id,
            installation_id=installation_id,
            active_policy_version=active_policy_version,
        )
        producer_originals = tuple(observed_findings)
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
                    # else: incumbent already carries its own proof artifact —
                    # keep it, drop the producer duplicate. The reachable
                    # incumbent here is a MODEL-CITED structural OBSERVED:
                    # content_hash excludes query_match_id and the model
                    # chooses finding_type freely, so a structural citation
                    # with a security type at the producer's line collides.
                    # INFERRED cannot reach this merge — pass 0 rejects every
                    # INFERRED proposal (no trace context), and pass 1 never
                    # runs the OBSERVED producer.
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
                # No substitution (the common case — the producer flags lines the
                # model missed) means admitted_list is an untouched copy, so reuse
                # the original tuple instead of rebuilding an identical one.
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

            # Cross-type subsumption (DECISIONS.md#055): over the POST-#054
            # admitted set, drop an OBSERVED finding X when a same-span JUDGED
            # finding Y of a more-specific finding_type subsumes it — the INVERSE
            # tiebreaker from #054 (accuracy over proof when the claims DIFFER, not
            # agree). The two-sided tier gate is load-bearing: Y JUDGED keeps the
            # survivor honestly JUDGED; X OBSERVED keeps the dropped side
            # structural, so the accounting nets out through n_findings_observed
            # alone and the both-JUDGED case is a non-goal by construction.
            # Running over the post-#054 set (not the pending producer list)
            # handles the triple-collision: a redundant model JUDGED weak_crypto
            # has already been swapped to OBSERVED by #054, so this pass finds and
            # drops it. The dropped query_match_id is retained in subsumed_matches
            # (the signal_only match is absent from the skip-shadow telemetry).
            # Exact-span match only: a multi-line envelope mismatch is an accepted
            # recall gap, never loose overlap (which could absorb an unrelated
            # weak_crypto sharing lines with a password-hash subsumer).
            admitted = parser_result.admitted_findings
            judged_by_span: dict[tuple[int, int], list[ReviewFinding]] = {}
            for jf in admitted:
                if jf.evidence_tier is EvidenceTier.JUDGED:
                    judged_by_span.setdefault((jf.line_start, jf.line_end), []).append(jf)
            if judged_by_span:
                surviving: list[ReviewFinding] = []
                for af in admitted:
                    subsumer: ReviewFinding | None = None
                    # Only a PRODUCER-origin OBSERVED finding is subsumable: its
                    # query_match_id is in OBSERVED_QUERY_IDS (the security producer
                    # registry), and ONLY producer findings are counted in
                    # n_findings_observed — the term the drop decrements. A
                    # MODEL-cited OBSERVED finding (the model claimed `observed`
                    # citing a fired STRUCTURAL query id, which the parser admits
                    # without binding finding_type to the query) rides the parser's
                    # n_findings_emitted instead; dropping it would underflow
                    # n_findings_observed and crash the pass. The two id sets are
                    # disjoint, so this gate excludes the model-cited case cleanly.
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
                    # The dropped finding is always OBSERVED, so it rode
                    # n_findings_observed (and the aggregate n_findings_emitted via
                    # the parser_emitted + parser_observed sum). Decrement ONLY
                    # n_findings_observed; the aggregate emitted drops with it and
                    # the two cancel in _enforce_proposal_accounting — no new term.
                    parser_result = replace(
                        parser_result,
                        admitted_findings=tuple(surviving),
                        counters=replace(
                            parser_result.counters,
                            n_findings_observed=parser_result.counters.n_findings_observed
                            - n_subsumed,
                        ),
                    )

        # Skip-routing telemetry (Cost Lever 3, DECISIONS.md#049): the skip-eligibility
        # decision (`observed_skip_event`) was computed PRE-LLM (Step 3b-mechanism);
        # EMIT it here for the non-enforced path so audit ordering + failure semantics
        # stay byte-identical (a `provider.complete` failure above leaves no shadow
        # event — this point is never reached). The enforced `would_skip` branch already
        # emitted it (skip_enforced=True) and returned; `skip_enforced` stays False here
        # (the LLM ran, so no skip was enforced).
        if observed_skip_event is not None:
            await analyze_event_sink.emit_observed_skip_shadow(
                # Re-validated (not model_copy) to stamp the worker's phase
                # key — compute_observed_skip_shadow is phase-blind by design.
                ObservedSkipShadowEvent.model_validate(
                    {**observed_skip_event.model_dump(), "phase_key": phase_key}
                )
            )

    # Lift parser rejection payloads into audit events.
    for proposal_rej in parser_result.proposal_rejections:
        await analyze_event_sink.emit_finding_proposal_rejected(
            _lift_proposal_rejection(
                proposal_rej, review_id=review_id, is_eval=is_eval, phase_key=phase_key
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

    # FindingEvent emission (audit row + content row per admitted finding) is
    # DEFERRED to the main loop's post-cap step (FUP-180): the per-round finding cap
    # must decide the kept set before any FindingEvent fires. The admitted findings
    # ride out on the returned `parser_result.admitted_findings`.

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
                    # Cross-type subsumption proof-retention records (DECISIONS.md#055):
                    # the dropped OBSERVED findings are absent from `findings`, so without
                    # this the cache-hit path would lose their query_match_id telemetry.
                    "subsumed_matches": [m.model_dump(mode="json") for m in subsumed_matches],
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
                # Host-triad telemetry columns (FUP-194): the SAME values folded into
                # cache_key above, so the denormalized columns match the key.
                profile_id=profile_id,
                reasoning_enabled=reasoning_enabled,
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
    """Process one trace-fetched file through parse → LLM call → parser.

    Pass-1 sibling of `_process_one_file`. Trace resolved this file via
    M8's two-phase fetch (Phase 1 probes + Phase 2 content fetch); the
    file is NOT a PR-diff file, so there's no patch, no triage
    classification, and no changed-scope-unit intersection. Analyze
    examines the WHOLE file because trace's resolution decided the file
    is relevant to a source finding.

    Outcomes (subset of `_process_one_file`'s):
      - `skipped+UNSUPPORTED_LANGUAGE` — no registered adapter for
        the extension.
      - Parser-stage skip — `parse_source` returned skipped (vendored,
        oversized, etc.).
      - `skipped+COST_BUDGET_EXHAUSTED` — cost gate failed.
      - `clean+full_llm` — clean parse, LLM call admitted, parser ran.

    Degraded outcomes (parse_failed / tree_has_error_in_changed_regions /
    tree_has_error_no_scope / module_level_observed_match) don't apply
    here: no changed regions — the module-scope arm is diff-anchored, so a
    whole-file pass has no added-line ranges to anchor on — and parse
    failures on a
    head-SHA-fetched file are routed through the parser-stage skip path
    rather than the V1-unreachable degraded branch.

    Per spec line 25: "INFERRED findings whose source `TraceDecision.
    resolution_status` is `unresolved` or `ambiguous` downgrade to
    JUDGED." V1 enforces this by Phase 2's gate (only `resolution_status=
    "resolved"` files reach `state.trace_fetched_files`), so the
    downgrade case doesn't fire here at the parser layer — every file
    iterated in pass 1 is by construction resolved.
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
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
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
            phase_key=None,  # pass 1 is the legacy un-keyed stream by design
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
    # Language-awareness mirrors pass-0: the builder selects this file's
    # own structural queries under its own grammar (empty for a language
    # with none registered — JS/TS in V1 — so no id can authorize an
    # OBSERVED claim there; the veto scan further down stays Python-gated
    # for its own reason).
    is_python = _is_python_file(fetched_file.path)
    query_match_id_set: frozenset[str] = (
        frozenset()
        if len(included_scope_units) != len(parse_result.scope_units)
        else _build_query_match_id_set(content_bytes, file_path=fetched_file.path)
    )
    parts = analyze_prompt.render_post_trace(
        file_path=fetched_file.path,
        scope_unit_context=_assemble_scope_unit_context(
            included_scope_units=included_scope_units,
            source_bytes=content_bytes,
            fence_lang=_fence_lang_for(fetched_file.path),
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
        response.profile_id,
        response.model,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
        # Response-derived pricing context (openai-native-host spec): tier-echo
        # expectation derives INSIDE pricing.py from profile_id, so an
        # echo-expecting host priced without these would classify absent_tier
        # and raise AFTER the billed call. A deviant response never reaches
        # this site — the provider raises the terminal contract error first —
        # so the default-tier context here always prices flat/long.
        billed_prompt_tokens=response.billed_prompt_tokens,
        service_tier=response.service_tier_actual,
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
        # Python-grammar surface: non-Python fetched files — reached on
        # every resolved JS/TS trace since the resolver spec — get the
        # inert empty scan; the gate keeps Python-grammar scanning off
        # non-Python bytes.
        parameterized_call_scan=scan_parameterized_calls(content_bytes)
        if is_python
        else ParameterizedCallScan(),
        # Correction is a no-op here (candidate collection is pass-0-only)
        # but threading the imports keeps both call sites uniform.
        import_refs=parse_result.imports,
    )

    for proposal_rej in parser_result.proposal_rejections:
        await analyze_event_sink.emit_finding_proposal_rejected(
            _lift_proposal_rejection(
                proposal_rej, review_id=review_id, is_eval=is_eval, phase_key=None
            )
        )
    if parser_result.response_rejection is not None:
        await analyze_event_sink.emit_analyze_response_rejected(
            _lift_response_rejection(
                parser_result.response_rejection,
                review_id=review_id,
                is_eval=is_eval,
                phase_key=None,  # pass 1: legacy un-keyed stream
            )
        )

    # FindingEvent emission DEFERRED to the main loop's post-cap step (FUP-180), same
    # as the pass-0 path — the trace-fetched (pass-1) round is capped identically. The
    # admitted findings ride out on `parser_result.admitted_findings`.

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
