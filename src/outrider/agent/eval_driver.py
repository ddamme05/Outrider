# Fixture-driven eval graph driver: run_review + the three src-local replay adapters.
# See specs/2026-06-01-eval-graph-driver.md (resolution A). LLM-provider boundary #8 +
# input boundary #5: the adapters implement the LLMProvider / GitHub / GitHubPublisher
# Protocols WITHOUT importing anthropic/githubkit — they replay fixture data only.
"""Drive the real review graph from JSON fixtures with persisted audit data.

run_review owns an ephemeral test database and stops at completion or a HITL
interrupt. run_review_persisting retains a caller-owned database;
run_review_with_resume rebuilds the graph/checkpointer and supplies an explicit
approval. Eval entry points require OUTRIDER_IS_EVAL=1 and isolated test URLs.

Fixture adapters replay model and GitHub data without vendor SDKs. An optional
provider_factory supplies a real/captured provider whose cleanup the driver
owns; GitHub reads and publication still stay fixture-local. Scripted responses
prove pipeline handling, not a real model's output quality.

This code lives in src because the eval entry points cannot import tests;
see specs/2026-06-01-eval-graph-driver.md (resolution A)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, Self
from uuid import UUID, uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from outrider.agent.checkpoint_serde import build_checkpoint_serde
from outrider.agent.graph import build_graph
from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS
from outrider.agent.nodes.cache_config import CacheMode
from outrider.agent.nodes.hitl_config import HITLConfig
from outrider.agent.nodes.patch_config import PatchConfig
from outrider.anomaly.persister import AnomalyPersister
from outrider.audit.config import RetentionSettings
from outrider.audit.persister import AuditPersister
from outrider.coordinates import validate_diff_path
from outrider.coordinates.errors import CoordinateError
from outrider.db.review_status_persister import ReviewStatusPersister
from outrider.eval_support import (
    EVAL_DB_NAME_PREFIX,
    EXPECTED_TEST_PORT,
    assert_no_is_eval_violations,
    ephemeral_database,
    redact_url_password,
    require_eval_mode,
    run_alembic_upgrade_head,
)
from outrider.github.authz import LiveAuthOutcome, LiveAuthResult
from outrider.llm.config import ModelConfig
from outrider.schemas.hitl import (
    HITLDecision,
    HITLRequest,
    PerFindingDecision,
    PerFindingOutcome,
)
from outrider.schemas.pr_context import PRContext
from outrider.schemas.publish import GitHubReviewCreated
from outrider.schemas.review_state import ReviewState

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from outrider.ast_facts.base import ImportPathResolver
    from outrider.cache import AnalyzeCacheStore
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMExchangePersister, LLMProvider, LLMRequest, LLMResponse
    from outrider.schemas.analysis_round import AnalysisRound
    from outrider.schemas.publish import InlineComment
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_report import ReviewMetrics
    from outrider.schemas.trace_decision import TraceDecision

# reviews.repo_id is an opaque int here; eval scenarios never assert on it.
_EVAL_REPO_ID = 100
# Env var naming the base postgres-test URL (port 5433); the ephemeral DB is
# carved off it per run.
_TEST_DB_URL_ENV_VAR = "TEST_DATABASE_URL"


class EvalDriverError(RuntimeError):
    """A fixture or scripted-response problem made the run unrunnable.

    Distinct from production `LLMProviderError` / `BuildGraphError`: this is a
    test-harness misconfiguration (missing scripted response, unset
    TEST_DATABASE_URL, malformed fixture), surfaced loudly so a scenario fails
    with a clear cause rather than a confusing downstream error.
    """


class _FixtureContentNotFoundError(Exception):
    """Mimics githubkit's 404 `RequestFailed` for a path the fixture's repo
    does not contain.

    Carries `.response.status_code == 404` so consumers that special-case 404
    treat an absent path as a real GitHub "file not found" — matching production
    wire behavior. This is required for the trace node's two-phase probe, which
    fetches BOTH candidate paths for a dotted import (e.g. `app/models.py` AND
    `app/models/__init__.py`) and relies on a 404 to learn which one exists
    (see `trace.py` — it reads `exc.response.status_code`). No `githubkit`
    import — only the `.response.status_code` shape those consumers read.
    """

    def __init__(self, path: str, ref: str) -> None:
        self.response = SimpleNamespace(status_code=404)
        super().__init__(f"fixture repo has no file at path={path!r} ref={ref!r} (GitHub 404)")


# ---------------------------------------------------------------------------
# Fixture schema
# ---------------------------------------------------------------------------


class EvalFixtureFile(BaseModel):
    """One changed file in the PR fixture. `content_*` follow GitHub status
    semantics: `added` → head only, `removed` → base only, `modified` → both,
    `renamed` → both + `previous_path`."""

    model_config = ConfigDict(extra="forbid")

    path: str
    status: Literal["added", "removed", "modified", "renamed"]
    additions: int
    deletions: int
    patch: str | None = None
    previous_path: str | None = None
    content_base: str | None = None
    content_head: str | None = None


class EvalFixture(BaseModel):
    """A complete PR fixture: identity + changed files + scripted LLM responses.

    `llm_responses` maps `node_id` → the ordered list of raw response strings
    that node's calls receive (index 0 = first call). The strings are the
    exact text the real node parser consumes (triage tiers JSON, the analyze
    findings JSON, synthesize summary prose) — the shape proven by
    `tests/integration/test_e2e_smoke.py`.
    """

    model_config = ConfigDict(extra="forbid")

    installation_id: int
    owner: str
    repo: str
    pr_number: int
    base_sha: str
    head_sha: str
    pr_title: str
    pr_body: str | None = None
    author: str
    total_additions: int
    total_deletions: int
    files: tuple[EvalFixtureFile, ...]
    # Repository content OUTSIDE the PR diff, served by the fake GitHub client's
    # `async_get_content` at `head_sha` (the ref trace probes — see trace.py).
    # Keyed by repo-relative path → file content. Empty for non-trace fixtures.
    # The trace node fetches beyond-diff files (e.g. a model imported by a changed
    # handler) through this; it goes through the SAME content path as PR files, so
    # the base64 wire-shape normalization is exercised identically.
    repository_contents_head: dict[str, str] = Field(default_factory=dict)
    llm_responses: dict[str, list[str]]
    # Opt-in CONCURRENCY-SAFE analyze scripting (parallel-analyze increment 6):
    # pass-0 analyze responses keyed BY FILE PATH instead of call order. Under
    # `analyze_max_concurrency > 1` worker completion order is nondeterministic,
    # so the index-keyed `llm_responses["analyze"]` list cannot attribute
    # responses to files — the provider serves from this map via the worker's
    # `LLMRequest.phase_key` (`file:<path>#<pass>`) instead. Keyless analyze
    # calls (the sequential pass-1 trace-fetched pass) still draw from
    # `llm_responses["analyze"]` in call order. None (the default) keeps the
    # index-keyed behavior, which requires analyze_max_concurrency == 1.
    analyze_responses_by_path: dict[str, str] | None = None
    # Opt-in per-review analyze token budget (analyze-cost-fairness Stage 1c seam).
    # None → use build_graph's DEFAULT_REVIEW_BUDGET_TOKENS (200k). A starvation /
    # budget-pressure scenario sets a tight value here so the analyze cost gate
    # fires deterministically. Same opt-in shape as `trivial_scope_filter_enabled`
    # / `cache_mode` — read in `_build_eval_graph`, threaded to `build_graph`.
    total_review_budget_tokens: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _repo_content_is_beyond_the_diff(self) -> Self:
        """`repository_contents_head` must NOT overlap any path in the diff.

        It is written into the same `(path, head_sha)` content map as the PR
        files' `content_head` (see `_github_factory_for`), so an overlapping key
        would silently override what intake reads — and a trace scenario would
        stop proving the beyond-diff fetch path without failing. Paths are
        compared AFTER `validate_diff_path` canonicalization — the SAME normalizer
        intake + trace apply before fetch — so a non-canonical spelling
        (`./app/x.py` vs `app/x.py`) can't bypass the check. Invalid paths
        (traversal, absolute, …) are rejected here too.
        """

        def _canon(p: str) -> str:
            try:
                return validate_diff_path(p)
            except CoordinateError as exc:
                raise ValueError(f"invalid fixture path {p!r}: {exc}") from exc

        diff_paths = {_canon(f.path) for f in self.files}
        diff_paths |= {_canon(f.previous_path) for f in self.files if f.previous_path}

        # Detect repository_contents_head keys that canonicalize to the SAME path:
        # two raw spellings collapse to one `(path, head_sha)` map entry in
        # `_github_factory_for`, so one would silently overwrite the other.
        repo_canon_to_raw: dict[str, list[str]] = {}
        for k in self.repository_contents_head:
            repo_canon_to_raw.setdefault(_canon(k), []).append(k)
        collisions = {c: sorted(raws) for c, raws in repo_canon_to_raw.items() if len(raws) > 1}
        if collisions:
            raise ValueError(
                f"repository_contents_head has keys that canonicalize to the same path "
                f"(one would silently overwrite the other): {collisions}"
            )

        overlap = diff_paths & set(repo_canon_to_raw)
        if overlap:
            raise ValueError(
                f"repository_contents_head must be beyond the diff, but {sorted(overlap)} "
                f"also appear(s) in `files` (changed-file path or rename source, after path "
                f"canonicalization); a changed file's content comes from `files`."
            )
        return self


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalRunResult:
    """The terminal product of a driven review.

    Iterable + len-able over `.findings` so finding-scenarios can write
    `findings = run_review(...)` / `[f for f in findings ...]` / `len(...)`
    verbatim; also exposes `.findings` / `.trace_decisions` / `.review_id` /
    `.published_comments` so trace-accuracy + FUP-137 scenarios can reach
    graph-state and the captured publish (the contract is a result object,
    not a bare tuple — see the spec's return-type fork).

    **HITL gating is a legitimate terminal state, not a failure.** `run_review`
    is a single-pass driver: when a CRITICAL/HIGH finding trips the HITL gate,
    the graph `interrupt()`s and this driver STOPS there — it does NOT
    auto-approve (that would auto-publish a gated finding, defeating trust
    boundary #6; resume semantics are the separate `run_review_with_resume`
    driver's job, which approves explicitly and returns a `ResumedRunResult`).
    On a gated run:
      - `.findings` is still populated (synthesize ran before hitl);
      - `.published_comments` is `()` (publish never ran — the gate held);
      - `.hitl_gated` is `True` and `.hitl_request` carries the gate payload.
    So `published_comments == ()` with `hitl_gated=True` means "gated", which a
    scenario distinguishes from "no findings" via the flag.
    """

    review_id: UUID
    findings: tuple[ReviewFinding, ...]
    trace_decisions: tuple[TraceDecision, ...]
    published_comments: tuple[InlineComment, ...]
    hitl_gated: bool
    hitl_request: HITLRequest | None = None
    # The synthesize node's populated `ReviewMetrics` (FUP-093: LLM aggregates
    # are summed from the review's `LLMCallEvent` rows). None only if synthesize
    # never ran (e.g. an early-failed run). Lets scenarios pin metric population.
    review_metrics: ReviewMetrics | None = None

    def __iter__(self) -> Any:
        return iter(self.findings)

    def __len__(self) -> int:
        return len(self.findings)


@dataclass(frozen=True)
class ResumedRunResult:
    """Result collected after explicitly approving a saved HITL request.

    The resume driver rebuilds graph and checkpointer on the same database/thread.
    analysis_rounds records the rehydrated state; hitl_decision records the scripted
    approval. Publication may have no inline comments, so review_status is separate."""

    review_id: UUID
    analysis_rounds: tuple[AnalysisRound, ...]
    published_comments: tuple[InlineComment, ...]
    hitl_gated: bool
    hitl_decision: HITLDecision | None
    review_status: str


# ---------------------------------------------------------------------------
# Adapter 1 — scripted LLM provider (LLMProvider Protocol; no `anthropic`)
# ---------------------------------------------------------------------------


@dataclass
class CostProbe:
    """Opt-in cost-measurement hook for `run_review` (zero-spend).

    When attached, `_FixtureScriptedProvider` counts REAL prompt tokens via
    `token_estimator(system_prompt + user_prompt)` instead of the fixed sentinel
    (100/50), so a driven review's cost flows through the production pricing path
    (`compute_cost_outcome` + `LLMCallEvent` + the aggregate SUM) on real prompt sizes.
    `output_tokens` models the response size (NOT measurable without a real
    completion); `None` estimates it from the scripted response text. Each call's
    metrics land in `.calls` for a per-node breakdown. Default OFF — eval
    correctness runs keep the fixed sentinels (their assertions are over findings,
    not cost).

    `model_cache=False` (default): cache tokens stay 0 — the measurement ignores
    prompt caching entirely. `model_cache=True`: the provider double MODELS
    Anthropic's cache deterministically, mirroring the documented contract the
    same way input counts are real-prompt-derived: per `(model, system_prompt_hash)`
    entry tracking (first occurrence above the model's `min_cacheable_tokens`
    floor → cache WRITE of the estimated system tokens; repeats → cache READ;
    below-floor or `cache_control=False` → uncached, both 0 — the silent no-op
    the real API exhibits). The model ignores TTL (driven reviews run far inside
    the 5-minute window) but is IN-FLIGHT-AWARE for concurrent first calls: a
    written entry becomes warm only when the writing call COMPLETES, so under
    `analyze_max_concurrency > 1` the overlapping first wave each records a
    WRITE — the real API's stampede (the parallel-analyze spec's accepted
    1.25x note) — and marking warm at decision time would underprice
    concurrent reviews. Cache-modeled estimates feed the same pricing
    multipliers as production
    (1.25x write / 0.1x read), so before/after packing comparisons price the
    cache, not just raw token movement.
    """

    token_estimator: Callable[[str], int]
    output_tokens: int | None = None
    model_cache: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    # Modeled-cache state: (model, system_prompt_hash) entries already "written".
    # Per-probe (= per driven review), matching one review's cache locality.
    _cache_entries: set[tuple[str, str]] = field(default_factory=set)


class _FixtureScriptedProvider:
    """Returns canned responses keyed by `(request.node_id, call-index)` —
    or, for pass-0 analyze calls when the fixture supplies
    `analyze_responses_by_path`, BY FILE PATH via the worker's
    `LLMRequest.phase_key` (`file:<path>#<pass>`): attribution is then
    correct under ANY worker completion order, which call-index keying
    cannot give at `analyze_max_concurrency > 1` (the driver entry points
    refuse that unsafe combination). Keyless analyze calls (the sequential
    pass-1 trace-fetched pass) and every other node stay call-index keyed.

    Implements the `LLMProvider` Protocol; imports no vendor SDK. Token counts
    are fixed sentinels; cost is derived from them via the real pricing table.
    Like the real provider, `complete()` emits a faithful `LLMCallEvent` (+
    `llm_call_content`) via the injected persister BEFORE returning — so
    driver-backed eval reviews carry real `llm_call` rows for synthesize's
    FUP-093 aggregate SUM (a double that emitted nothing would report
    false-zero metrics). Raises `EvalDriverError` loudly when a node makes more
    calls than the fixture scripts, so a scenario fails with a clear cause.
    """

    def __init__(
        self,
        responses: dict[str, list[str]],
        *,
        persister: LLMExchangePersister,
        probe: CostProbe | None = None,
        host: str | None = None,
        analyze_responses_by_path: dict[str, str] | None = None,
    ) -> None:
        self._responses = responses
        self._counts: dict[str, int] = {}
        self._persister = persister
        self._probe = probe
        # Concurrency-safe pass-0 analyze scripting (increment 6): responses
        # keyed by file path, served via the worker's LLMRequest.phase_key —
        # completion order can't misattribute them. None = index-keyed legacy.
        self._analyze_responses_by_path = analyze_responses_by_path
        # Host-identity triad stamped on every response/event (DECISIONS.md#056).
        # Default None -> anthropic: resolve_host_identity("anthropic", False) returns
        # the SAME (ANTHROPIC_PROFILE_ID, False, ANTHROPIC_CONTRACT_DIGEST) the
        # AnthropicProvider stamps, so the default path is byte-identical. host="baseten"
        # stamps the baseten triad so a full-graph eval can run a non-anthropic host.
        # host_profiles is SDK-free, so this import stays off the SDK module-load path.
        from outrider.llm.host_profiles import ANTHROPIC_PROFILE_ID, resolve_host_identity

        self._profile_id, self._reasoning_enabled, self._profile_contract_digest = (
            resolve_host_identity(host or ANTHROPIC_PROFILE_ID, reasoning=False)
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        from outrider.audit.events import LLMCallEvent

        # Stamp the host-identity triad resolved in __init__ (DECISIONS.md#056) on the
        # response + event so the persister host-qualifies cost + cross-checks
        # event-vs-response, and pricing keys on `(profile_id, model)`. Default host is
        # anthropic (the fixture mirrors AnthropicProvider's claude-family ModelConfig);
        # a baseten-host run stamps the baseten triad + keys pricing on `("baseten", model)`.
        from outrider.llm.base import (
            LLMResponse,
            _canonical_prompt_hash,
            _canonical_system_prompt_hash,
        )
        from outrider.llm.host_profiles import ANTHROPIC_PROFILE_ID as _ANTHROPIC_ID
        from outrider.llm.pricing import (
            PRICING_VERSION,
            TIER_ECHO_EXPECTED_PROFILE_IDS,
            Priced,
            compute_cost_outcome,
            min_cacheable_tokens,
        )

        node = request.node_id
        by_path = self._analyze_responses_by_path
        request_key = request.phase_key
        if node == "analyze" and by_path is not None and request_key is not None:
            # Pass-0 worker call: serve BY FILE PATH (phase_key is
            # `file:<path>#<pass>`; the integer final segment disambiguates
            # paths containing `#`). Deterministic under any completion order.
            path = request_key.removeprefix("file:").rsplit("#", 1)[0]
            try:
                text_out = by_path[path]
            except KeyError as exc:
                raise EvalDriverError(
                    f"no scripted analyze response for file {path!r} in the "
                    f"fixture's analyze_responses_by_path (has: "
                    f"{sorted(by_path)}). Every LLM-reaching pass-0 file needs "
                    f"an entry."
                ) from exc
        else:
            idx = self._counts.get(node, 0)
            try:
                text_out = self._responses[node][idx]
            except (KeyError, IndexError) as exc:
                raise EvalDriverError(
                    f"no scripted LLM response for node_id={node!r} call-index={idx} "
                    f"(fixture scripts {len(self._responses.get(node, []))} call(s) for "
                    f"this node). Add the response to the fixture's llm_responses."
                ) from exc
            self._counts[node] = idx + 1
        # Token counts: fixed sentinels for correctness runs; REAL prompt-derived
        # counts when a CostProbe is attached (zero-spend cost measurement). The
        # graph has ALREADY rendered the real system+user prompts by this point, so
        # the input count is grounded; output is modeled (no real completion).
        cache_read_tokens, cache_write_tokens = 0, 0
        # In-flight-aware cache model: a written prefix becomes WARM only when
        # the writing call COMPLETES (set just before `return response`), so a
        # concurrent sibling that starts while the write is in flight records
        # its own write — the real API's first-wave stampede (the
        # parallel-analyze spec's accepted 1.25× note). Marking warm at
        # decision time would let overlapping workers record reads and
        # underprice concurrent reviews.
        pending_cache_entry: tuple[str, str] | None = None
        probe = self._probe
        if probe is not None:
            output_tokens = (
                probe.output_tokens
                if probe.output_tokens is not None
                else probe.token_estimator(text_out)
            )

            def _combined_estimate() -> int:
                # Shared uncached-input recipe for the cache-off and
                # below-floor branches; lazy so the cache-modeled path
                # never estimates a concatenation it doesn't use.
                return probe.token_estimator(f"{request.system_prompt}\n{request.user_prompt}")

            if probe.model_cache and request.cache_control:
                # Deterministic cache model (see CostProbe docstring): system
                # prompt is the single V1 cacheable block; the accounting
                # identity `total_input = cache_read + cache_creation +
                # input_tokens` holds by construction. That identity is
                # ANTHROPIC accounting (write outside input) — on a
                # writes-reported host the cold-call system prefix stays INSIDE
                # billed input, so modeling it here would underprice every cold
                # call. Fail closed rather than silently mis-account
                # (openai-native-host spec, Codex round-4 finding).
                if self._profile_id != _ANTHROPIC_ID:
                    raise EvalDriverError(
                        f"CostProbe(model_cache=True) models ANTHROPIC cache accounting "
                        f"only; host {self._profile_id!r} has host-specific cache "
                        f"accounting (auto-prefix / writes-reported) this double does "
                        f"not model. Run the probe cache-off for this host, or extend "
                        f"the cache model per-host first."
                    )
                system_tokens = probe.token_estimator(request.system_prompt)
                try:
                    floor = min_cacheable_tokens(self._profile_id, request.model)
                except KeyError as exc:
                    # Same loud-failure contract as the pricing KeyError
                    # below — a bare KeyError mid-complete() hides the cause.
                    raise EvalDriverError(
                        f"eval fixture model {request.model!r} "
                        f"(node_id={request.node_id!r}) has no "
                        f"MIN_CACHEABLE_TOKENS floor — the cache model needs "
                        f"a priced+floored model. Use a priced model in the "
                        f"eval's ModelConfig."
                    ) from exc
                if floor is None:
                    # `None` is the #056 unknown-floor sentinel (a priced host with
                    # no documented threshold). The deterministic cache model needs a
                    # concrete floor; anthropic models always have one, so this only
                    # fires under an env override to a floorless host.
                    raise EvalDriverError(
                        f"eval fixture model {request.model!r} "
                        f"(node_id={request.node_id!r}) has a None (unknown) "
                        f"cacheable floor — the cache model needs a concretely "
                        f"floored model."
                    )
                if system_tokens >= floor:
                    entry = (
                        request.model,
                        _canonical_system_prompt_hash(request.system_prompt),
                    )
                    if entry in probe._cache_entries:
                        cache_read_tokens = system_tokens
                    else:
                        cache_write_tokens = system_tokens
                        pending_cache_entry = entry  # warm at COMPLETION, below
                    input_tokens = probe.token_estimator(request.user_prompt)
                else:
                    # Below the model's floor: processed without caching, no
                    # error returned — the real API's silent no-op.
                    input_tokens = _combined_estimate()
            else:
                input_tokens = _combined_estimate()
        else:
            input_tokens, output_tokens = 100, 50
        # Pricing-context shape derives from the modeled provider FAMILY (matching
        # each real provider's response shape): the anthropic native path stamps
        # neither field; EVERY compat-modeled host records the billed prompt count
        # (the real OpenAICompatibleProvider stamps usage.prompt_tokens for all its
        # hosts — GLM included), computed as input + cache_read per the
        # includes-cached §8a arms (exact here: the Anthropic-only cache model above
        # never runs for compat hosts, so no writes-subtraction ambiguity exists).
        # The tier echo is "default" only for echo-expecting hosts (the versioned
        # set); GLM hosts realistically echo nothing.
        is_compat_host = self._profile_id != _ANTHROPIC_ID
        expects_tier_echo = self._profile_id in TIER_ECHO_EXPECTED_PROFILE_IDS
        response = LLMResponse(
            text=text_out,
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            finish_reason="end_turn",
            latency_ms=1,
            profile_id=self._profile_id,
            reasoning_enabled=self._reasoning_enabled,
            profile_contract_digest=self._profile_contract_digest,
            billed_prompt_tokens=(input_tokens + cache_read_tokens) if is_compat_host else None,
            service_tier_actual="default" if expects_tier_echo else None,
        )
        # Emit LLMCallEvent + llm_call_content BEFORE returning, exactly as the real
        # LLMProvider contract requires (mirrors AnthropicProvider Step 9). FUP-093:
        # synthesize SUMs these rows for its ReviewMetrics aggregates, so a double
        # that emitted nothing would report false-zero metrics. `persist()` cross-
        # checks the prompt/system hashes, the request/response fields, and the
        # recomputed cost (in-transaction) — so any drift from the real provider's
        # shape fails loudly here rather than landing a silently-wrong audit row.
        try:
            outcome = compute_cost_outcome(
                response.profile_id,
                response.model,
                input_tokens=response.input_tokens,
                cache_write_tokens=response.cache_write_tokens,
                cache_read_tokens=response.cache_read_tokens,
                output_tokens=response.output_tokens,
                billed_prompt_tokens=response.billed_prompt_tokens,
                service_tier=response.service_tier_actual,
            )
        except KeyError as exc:
            # Mirror the real provider's named pricing error (it raises
            # LLMPricingMissingError) so a misconfigured eval ModelConfig surfaces a
            # clear cause instead of a bare KeyError mid-complete(). ModelConfig
            # defaults (claude-haiku-4-5 / claude-sonnet-5) are priced; this only
            # fires under an env override to a valid-but-unpriced model.
            raise EvalDriverError(
                f"eval fixture model {response.model!r} (node_id={request.node_id!r}) is "
                f"not in the pricing RATE_TABLE — the fixture provider mirrors the real "
                f"provider's cost path. Use a priced model in the eval's ModelConfig."
            ) from exc
        # Mirror the real provider's Priced/Unpriced split (OpenAICompatibleProvider
        # Step 8): a billing-unadjudicated host (openai — DECISIONS.md#056 / FUP-247)
        # yields Unpriced, so cost is honestly withheld as cost_usd=None + the typed
        # reason. Priced hosts (anthropic, glm) are unaffected.
        if isinstance(outcome, Priced):
            cost: float | None = float(outcome.cost_usd)
            unpriced_reason = None
        else:
            cost = None
            unpriced_reason = outcome.reason
        if self._probe is not None:
            self._probe.calls.append(
                {
                    "node_id": request.node_id,
                    "model": response.model,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cache_read_tokens": response.cache_read_tokens,
                    "cache_write_tokens": response.cache_write_tokens,
                    "cost_usd": cost,
                }
            )
        event = LLMCallEvent(
            review_id=request.review_id,
            timestamp=datetime.now(UTC),
            is_eval=request.is_eval,
            model=response.model,
            # DECISIONS.md#016 Amended 2026-06-30: mirror finish_reason; persister cross-checks it.
            finish_reason=response.finish_reason,
            node_id=request.node_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_tokens=response.cache_read_tokens,
            cost_usd=cost,
            cost_unpriced_reason=unpriced_reason,
            pricing_version=PRICING_VERSION,
            latency_ms=response.latency_ms,
            prompt_hash=_canonical_prompt_hash(
                system_prompt=request.system_prompt, user_prompt=request.user_prompt
            ),
            cache_hit=(response.cache_read_tokens > 0),
            context_summary=request.context_summary,
            prompt_template_version=request.prompt_template_version,
            system_prompt_hash=_canonical_system_prompt_hash(request.system_prompt),
            degraded_mode=request.degraded_mode,
            degradation_reason=request.degradation_reason,
            phase_key=request.phase_key,
            # FUP-096: mirror the real provider's provenance pass-through, or
            # the persister's event-request cross-check rejects every scripted
            # analyze call (the request carries the pinned schema).
            response_format_digest=request.response_format_digest,
            # Host-identity triad (DECISIONS.md#056) mirrored from response.* —
            # the single-source pattern the real providers use, so the persister's
            # event-vs-response cross-check compares two copies of one source.
            profile_id=response.profile_id,
            reasoning_enabled=response.reasoning_enabled,
            profile_contract_digest=response.profile_contract_digest,
            # Pricing context mirrored from response.* (openai-native-host spec),
            # matching each modeled provider's REAL event shape: the anthropic
            # native path stamps none of these (its provider is unchanged), the
            # compat path mirrors them — the persister's fresh guard verifies the
            # mirror on echo-expecting hosts and mirror-if-present elsewhere.
            service_tier=response.service_tier_actual,
            billed_prompt_tokens=response.billed_prompt_tokens,
            cache_write_tokens=(
                None if self._profile_id == _ANTHROPIC_ID else response.cache_write_tokens
            ),
        )
        await self._persister.persist(event, request, response)
        if pending_cache_entry is not None and self._probe is not None:
            # The write has now completed: later calls read; overlapped ones wrote.
            self._probe._cache_entries.add(pending_cache_entry)
        return response

    async def aclose(self) -> None:
        # Owns no transport resources (no SDK client); satisfies the
        # `LLMProvider.aclose` member formalized in DECISIONS.md#035 (retained #058) so this
        # double passes build_graph's runtime isinstance(provider, LLMProvider).
        return None


# ---------------------------------------------------------------------------
# Adapter 2 — fake GitHub read client (intake's two-phase fetch; no `githubkit`)
# ---------------------------------------------------------------------------


@dataclass
class _FixtureFileMeta:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    previous_filename: str | None


@dataclass
class _FixtureContentFile:
    encoding: str
    content: str


@dataclass
class _FixtureResponse:
    parsed_data: Any


class _FixtureReposAPI:
    """Serves `async_get_content` from fixture content, keyed by (path, ref).

    Content is base64 with newline wrapping (`base64.encodebytes`) to match
    GitHub's real contents-API wire shape — intake strips `\\n`/`\\r` then
    decodes with `validate=True` (see `github/fetch.py`), so an unwrapped stub
    would not exercise the strip path real GitHub triggers
    (`feedback_test_stubs_match_wire_format`).
    """

    def __init__(self, content_by_path_ref: dict[tuple[str, str], str]) -> None:
        self._content = content_by_path_ref

    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _FixtureResponse:
        try:
            raw = self._content[(path, ref)]
        except KeyError as exc:
            # Absent path → mimic GitHub's 404 (real wire behavior), NOT a hard
            # error: trace's candidate probe fetches paths that legitimately
            # don't exist and reads the 404 to resolve. Intake only requests
            # paths the fixture supplied, so a 404 there would itself be a
            # genuine fetch failure (surfaced loudly upstream).
            raise _FixtureContentNotFoundError(path, ref) from exc
        wrapped_b64 = base64.encodebytes(raw.encode()).decode("ascii")
        return _FixtureResponse(
            parsed_data=_FixtureContentFile(encoding="base64", content=wrapped_b64)
        )


class _FixturePullsAPI:
    def __init__(self, metas: list[_FixtureFileMeta]) -> None:
        self._metas = metas

    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _FixtureResponse:
        return _FixtureResponse(parsed_data=list(self._metas))


class _FixtureRestAPI:
    def __init__(self, repos: _FixtureReposAPI, pulls: _FixturePullsAPI) -> None:
        self.repos = repos
        self.pulls = pulls


class _FixtureGitHubClient:
    def __init__(self, rest: _FixtureRestAPI) -> None:
        self.rest = rest


# ---------------------------------------------------------------------------
# Adapter 3 — capturing publisher (GitHubPublisher Protocol; no POST)
# ---------------------------------------------------------------------------


@dataclass
class _CapturingPublisher:
    """Records `create_review` instead of POSTing; reports no prior review."""

    create_review_calls: list[dict[str, Any]] = field(default_factory=list)

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        body: str | None = None,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        self.create_review_calls.append(
            {"review_status": review_status, "comments": tuple(comments), "body": body}
        )
        return GitHubReviewCreated(github_review_id=999, comments_posted=len(comments))

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        return None


# ---------------------------------------------------------------------------
# Adapter 4 — default resolver for analyze's local import context
# ---------------------------------------------------------------------------


class _NoOpImportPathResolver:
    """Default resolver that supplies no local-filesystem candidate paths.

    Trace resolves through GitHub probes independently. Callers may inject a real
    resolver for analyze's scope/import context."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        return []


# ---------------------------------------------------------------------------
# Wiring helpers
# ---------------------------------------------------------------------------


async def _eval_always_authorized(installation_id: int, repo_id: int) -> LiveAuthResult:
    """Eval double for build_graph's `installation_authorizer` (#065). Eval fixtures are
    trusted and there is no live GitHub App to probe, so the live-auth gate always
    authorizes; the reviews row `_seed_review` writes carries the matching identity that
    intake's `_load_review_gate_state` verifies before this runs."""
    return LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "eval fixture (trusted)")


def _github_factory_for(fixture: EvalFixture) -> Any:
    """Build a `github_factory(installation_id)` serving the fixture's PR."""
    metas: list[_FixtureFileMeta] = []
    for metadata_file in fixture.files:
        metas.append(
            _FixtureFileMeta(
                filename=metadata_file.path,
                status=metadata_file.status,
                additions=metadata_file.additions,
                deletions=metadata_file.deletions,
                patch=metadata_file.patch,
                previous_filename=metadata_file.previous_path,
            )
        )
    content: dict[tuple[str, str], str] = {}
    for fixture_file in fixture.files:
        # Key by the CANONICAL path intake/trace fetch (`validate_diff_path`), so a
        # non-canonical fixture spelling still resolves. Paths are already
        # validated at `EvalFixture` construction, so this won't raise.
        # `previous_path` is the base-side path for renames; base content is read
        # there, head content at the new path (per intake's rename handling).
        head_path = validate_diff_path(fixture_file.path)
        base_path = validate_diff_path(
            fixture_file.previous_path
            if (fixture_file.status == "renamed" and fixture_file.previous_path)
            else fixture_file.path
        )
        if fixture_file.content_base is not None:
            content[(base_path, fixture.base_sha)] = fixture_file.content_base
        if fixture_file.content_head is not None:
            content[(head_path, fixture.head_sha)] = fixture_file.content_head
    # Beyond-diff repository content (trace's two-phase probe), served at head_sha
    # through the same content map as PR files, keyed by the canonical path.
    for path, repo_content in fixture.repository_contents_head.items():
        content[(validate_diff_path(path), fixture.head_sha)] = repo_content
    client = _FixtureGitHubClient(
        _FixtureRestAPI(_FixtureReposAPI(content), _FixturePullsAPI(metas))
    )

    async def factory(installation_id: int) -> Any:
        if installation_id != fixture.installation_id:
            raise EvalDriverError(
                f"unexpected installation_id {installation_id} "
                f"(fixture is {fixture.installation_id})"
            )
        return client

    return factory


def _seed_state(fixture: EvalFixture, review_id: UUID) -> ReviewState:
    """Seed state with EMPTY changed_files — intake enriches via the fixture
    github client, exercising the real two-phase fetch (per DECISIONS.md#020)."""
    return ReviewState(
        review_id=review_id,
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=fixture.installation_id,
            owner=fixture.owner,
            repo=fixture.repo,
            pr_number=fixture.pr_number,
            base_sha=fixture.base_sha,
            head_sha=fixture.head_sha,
            pr_title=fixture.pr_title,
            pr_body=fixture.pr_body,
            author=fixture.author,
            total_additions=fixture.total_additions,
            total_deletions=fixture.total_deletions,
            changed_files=(),
        ),
        is_eval=True,
    )


async def _seed_installation(engine: AsyncEngine, fixture: EvalFixture) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:id, 'eval-app', 1, :login, 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": fixture.installation_id, "login": fixture.owner},
        )


async def _seed_review(engine: AsyncEngine, review_id: UUID, fixture: EvalFixture) -> None:
    """Seed the review row with `is_eval=True` (the integrity gate enforces it)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, is_eval, retention_expires_at) VALUES "
                "(:id, :iid, :repo, :pr, :sha, 'running', TRUE, NOW() + INTERVAL '180 days')"
            ),
            {
                "id": review_id,
                "iid": fixture.installation_id,
                "repo": _EVAL_REPO_ID,
                "pr": fixture.pr_number,
                "sha": fixture.head_sha,
            },
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def _run_is_eval_gate(engine: AsyncEngine) -> None:
    """Run the is_eval integrity gate against a live connection on `engine`."""
    async with engine.connect() as conn:
        await assert_no_is_eval_violations(conn)


def _build_eval_graph(
    *,
    fixture: EvalFixture,
    session_factory: async_sessionmaker[AsyncSession],
    persister: AuditPersister,
    provider: LLMProvider,
    publisher: _CapturingPublisher,
    checkpointer: Any,
    trivial_scope_filter_enabled: bool = False,
    analyze_observed_skip_enforced: bool = False,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    model_config: ModelConfig | None = None,
    host: str | None = None,
    analyze_max_concurrency: int = 1,
    import_path_resolver: ImportPathResolver | None = None,
) -> Any:
    """Build the seven-logical-node graph with eval sinks and the supplied provider.

    The single build-graph-deps bundle shared by the single-pass `_drive` and the
    resume driver, so their default sinks/deps cannot drift. Their default
    `checkpointer` differs (`InMemorySaver` for single-pass;
    `AsyncPostgresSaver` for resume — durability is what makes resume possible).

    `model_config` lets a caller (the eval scorecard runner) vary per-node models
    in-process without mutating `OUTRIDER_MODEL_*` env between runs; default `None`
    reads the env exactly as production does — `ModelConfig.for_host(host)` (host
    defaulting to anthropic, which is byte-identical to the old `ModelConfig()`).
    """
    # Budget seam (Stage 1c): the fixture's budget, or build_graph's default 200k
    # when unset. A starvation scenario sets a tight value so the analyze cost gate
    # fires deterministically.
    review_budget = (
        fixture.total_review_budget_tokens
        if fixture.total_review_budget_tokens is not None
        else DEFAULT_REVIEW_BUDGET_TOKENS
    )
    # Qualify the eval completion events like production (#056 step 4d): close the
    # host-identity triad into build_graph — else the persister's fresh-write guard
    # rejects the unqualified AnalyzeCompletedEvent / SynthesizeCompletedEvent (and the
    # eval cache key would stay host-unqualified). Default host is anthropic and stays
    # byte-identical (resolve_host_identity + ModelConfig.for_host on "anthropic"
    # reproduce the prior constants + ModelConfig()); host="baseten" builds the graph with
    # the baseten identity (its analyze/synthesize completion events + provider calls carry
    # the triad; publish/hitl emit none) — the e2e coverage the GLM scorecard (analyze-only)
    # lacks.
    from outrider.llm.host_profiles import ANTHROPIC_PROFILE_ID, resolve_host_identity

    host_id = host or ANTHROPIC_PROFILE_ID
    profile_id, reasoning_enabled, profile_contract_digest = resolve_host_identity(
        host_id, reasoning=False
    )
    return build_graph(
        db_factory=session_factory,
        github_factory=_github_factory_for(fixture),
        installation_authorizer=_eval_always_authorized,
        provider=provider,
        model_config=model_config or ModelConfig.for_host(host_id),
        profile_id=profile_id,
        reasoning_enabled=reasoning_enabled,
        profile_contract_digest=profile_contract_digest,
        phase_event_sink=persister,
        file_examination_sink=persister,
        analyze_event_sink=persister,
        publish_event_sink=persister,
        trace_sink=persister,
        hitl_event_sink=persister,
        synthesize_event_sink=persister,
        review_status_sink=ReviewStatusPersister(session_factory=session_factory),
        anomaly_sink=AnomalyPersister(session_factory=session_factory),
        hitl_config=HITLConfig(),
        # Suggested-patch pass OFF by default in eval: existing scenarios script LLM
        # responses by node_id+call-index and do NOT script the extra synthesize patch
        # call, so enabling it would raise EvalDriverError. A patch-specific scenario
        # opts in by scripting the second synthesize response (DECISIONS.md#040).
        patch_config=PatchConfig(patches_enabled=False),
        checkpointer=checkpointer,
        publisher=publisher,
        import_path_resolver=(
            import_path_resolver if import_path_resolver is not None else _NoOpImportPathResolver()
        ),
        # Default 1: index-keyed `llm_responses["analyze"]` scripting depends
        # on deterministic call order. A concurrency scenario opts in with
        # `analyze_max_concurrency > 1` + the fixture's
        # `analyze_responses_by_path` map (served via LLMRequest.phase_key,
        # correct under ANY completion order); the entry points fail loud on
        # the unsafe combination.
        analyze_max_concurrency=analyze_max_concurrency,
        # Shadow mode by default, matching production: the classifier runs
        # and audits would-exclude verdicts on every eval scenario. An
        # enforce-mode scenario opts in through this seam
        # (specs/2026-06-10-trivial-scope-filter.md).
        trivial_scope_filter_enabled=trivial_scope_filter_enabled,
        # Step 3b-mechanism: enforced OBSERVED skip seam — same opt-in shape as
        # trivial_scope_filter_enabled. Default False; a skip-enforcement scenario
        # opts in (paired with a test-local skip_safe promotion).
        analyze_observed_skip_enforced=analyze_observed_skip_enforced,
        # Eval reviews use the cache like production, kept isolated by the lookup's
        # required is_eval read-isolation predicate (DECISIONS.md#046) — no bypass.
        # Default None still disables it; a cache eval scenario injects its own
        # store fixture through this seam (the same opt-in shape as
        # `trivial_scope_filter_enabled` above).
        analyze_cache_store=analyze_cache_store,
        # Default shadow; the serve eval scenario injects CacheMode.SERVE with a
        # pre-seeded store through this seam.
        cache_mode=cache_mode,
        total_review_budget_tokens=review_budget,
    )


async def _drive(
    fixture: EvalFixture,
    db_url: str,
    *,
    probe: CostProbe | None = None,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    analyze_observed_skip_enforced: bool = False,
    model_config: ModelConfig | None = None,
    host: str | None = None,
    analyze_max_concurrency: int = 1,
    provider_factory: Callable[[LLMExchangePersister], LLMProvider] | None = None,
    import_path_resolver: ImportPathResolver | None = None,
) -> EvalRunResult:
    """Run the graph once against `db_url` (already migrated) and collect results.

    The is_eval integrity gate runs on BOTH the success and failure paths before
    `_drive` disposes the engine (and, for the `_arun_review` caller, before its
    `ephemeral_database` drops the DB — the `run_review_persisting` caller owns the
    DB and keeps it): on success a violation fails the run; on failure it is
    best-effort (suppressed) so it never masks the root-cause exception.
    """
    engine = create_async_engine(db_url, poolclass=NullPool)
    owned_provider: LLMProvider | None = None
    run_failed = True
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        await _seed_installation(engine, fixture)
        review_id = uuid4()
        await _seed_review(engine, review_id, fixture)

        persister = AuditPersister(
            session_factory=session_factory, retention_settings=RetentionSettings()
        )
        publisher = _CapturingPublisher()
        provider: LLMProvider
        if provider_factory is None:
            provider = _FixtureScriptedProvider(
                fixture.llm_responses,
                persister=persister,
                probe=probe,
                host=host,
                analyze_responses_by_path=fixture.analyze_responses_by_path,
            )
        else:
            owned_provider = provider_factory(persister)
            provider = owned_provider
        graph = _build_eval_graph(
            fixture=fixture,
            host=host,
            session_factory=session_factory,
            persister=persister,
            provider=provider,
            publisher=publisher,
            checkpointer=InMemorySaver(serde=build_checkpoint_serde()),
            analyze_max_concurrency=analyze_max_concurrency,
            analyze_cache_store=analyze_cache_store,
            cache_mode=cache_mode,
            analyze_observed_skip_enforced=analyze_observed_skip_enforced,
            model_config=model_config,
            import_path_resolver=import_path_resolver,
        )

        result = await graph.ainvoke(
            _seed_state(fixture, review_id),
            config={"configurable": {"thread_id": str(review_id)}},
        )

        # A CRITICAL/HIGH finding trips the HITL gate -> the graph interrupt()s
        # and LangGraph returns the accumulated channel state PLUS `__interrupt__`
        # (a list of Interrupt objects; `.value` is hitl's request payload). We
        # record the gate but do NOT resume — single-pass driver (see EvalRunResult).
        interrupts = result.get("__interrupt__")
        hitl_gated = bool(interrupts)
        hitl_request = HITLRequest.model_validate(interrupts[0].value) if interrupts else None

        report = result.get("review_report")
        findings: tuple[ReviewFinding, ...] = tuple(report.findings) if report else ()
        trace_decisions: tuple[TraceDecision, ...] = tuple(result.get("trace_decisions", ()))
        published: tuple[InlineComment, ...] = tuple(
            c for call in publisher.create_review_calls for c in call["comments"]
        )
        eval_result = EvalRunResult(
            review_id=review_id,
            findings=findings,
            trace_decisions=trace_decisions,
            published_comments=published,
            hitl_gated=hitl_gated,
            hitl_request=hitl_request,
            review_metrics=report.metrics if report else None,
        )

        # is_eval integrity gate (success path): a violation fails the run.
        await _run_is_eval_gate(engine)
        run_failed = False
        return eval_result
    except Exception:
        # Failed run: still verify is_eval discipline before the DB is dropped,
        # but never let a gate violation mask the root-cause exception.
        with contextlib.suppress(Exception):
            await _run_is_eval_gate(engine)
        raise
    finally:
        try:
            if owned_provider is not None:
                if run_failed:
                    # Preserve the review failure if provider cleanup also fails.
                    with contextlib.suppress(Exception):
                        await owned_provider.aclose()
                else:
                    await owned_provider.aclose()
        finally:
            await engine.dispose()


async def _arun_review(
    fixture: EvalFixture,
    base_url: str,
    *,
    probe: CostProbe | None = None,
    model_config: ModelConfig | None = None,
    host: str | None = None,
    analyze_max_concurrency: int = 1,
    provider_factory: Callable[[LLMExchangePersister], LLMProvider] | None = None,
    import_path_resolver: ImportPathResolver | None = None,
) -> EvalRunResult:
    async with ephemeral_database(base_url=base_url) as db_url:
        await run_alembic_upgrade_head(db_url)
        return await _drive(
            fixture,
            db_url,
            probe=probe,
            model_config=model_config,
            host=host,
            analyze_max_concurrency=analyze_max_concurrency,
            provider_factory=provider_factory,
            import_path_resolver=import_path_resolver,
        )


def _require_concurrency_safe_scripting(fixture: EvalFixture, analyze_max_concurrency: int) -> None:
    """Fail loud on the unsafe combination: concurrent workers + index-keyed
    analyze scripting. With ANY scripted analyze response and no by-path
    map, completion order decides which file gets which response (how many
    files reach the LLM is runtime behavior, so one response is no safer
    than five) — a silently misattributed scenario is worse than a refused
    one."""
    if (
        analyze_max_concurrency > 1
        and fixture.analyze_responses_by_path is None
        and len(fixture.llm_responses.get("analyze", [])) >= 1
    ):
        # >= 1, not > 1: how many files actually reach the LLM is runtime
        # behavior (budget, cache, coverage skips), so even a single indexed
        # response can misattribute when two files call — and the first
        # worker would PERSIST the misattributed exchange before the second
        # aborts on a misleading call-index error. Response-list length is
        # the only static signal; refuse whenever any indexed analyze
        # response exists.
        raise EvalDriverError(
            "analyze_max_concurrency > 1 with index-keyed analyze scripting: "
            "the fixture scripts llm_responses['analyze'] entries but no "
            "analyze_responses_by_path map, so worker completion order would "
            "decide response attribution. Add analyze_responses_by_path to the "
            "fixture (or run with analyze_max_concurrency=1)."
        )


def run_review(
    fixture_path: str | os.PathLike[str],
    *,
    probe: CostProbe | None = None,
    model_config: ModelConfig | None = None,
    host: str | None = None,
    analyze_max_concurrency: int = 1,
    provider_factory: Callable[[LLMExchangePersister], LLMProvider] | None = None,
    import_path_resolver: ImportPathResolver | None = None,
) -> EvalRunResult:
    """Drive the real 7-logical-node graph against a JSON PR fixture; return the result.

    Synchronous wrapper (the eval scenarios call it without `await`); the async
    graph runs inside `asyncio.run`. Fail-closed: requires `OUTRIDER_IS_EVAL=1`
    and a port-5433/"test" `TEST_DATABASE_URL` before any database work.

    Pass a `CostProbe` to measure grounded cost-per-review on real prompt sizes
    (zero Anthropic spend); default `None` keeps the fixed-sentinel token counts.

    Pass a `model_config` to run under a specific per-node model matrix (the eval
    scorecard runner's cost pass); default `None` reads `OUTRIDER_MODEL_*` from env
    exactly as production does.

    Pass `host` to run the graph under a non-anthropic provider host (e.g. "baseten"):
    the scripted provider stamps that host's identity triad on every LLM call, and
    analyze/synthesize stamp it on their completion events; `ModelConfig.for_host(host)`
    selects its models, and pricing keys on `(host, model)`. (Single pass — a critical
    finding still gates at hitl; see `run_review_with_resume` to reach publish.) Default
    `None` -> anthropic (byte-identical to before).

    `provider_factory` opts into a real or captured provider bound to this run's
    persister. The driver closes it before dropping the database. With no factory,
    responses remain scripted. `import_path_resolver` overrides the no-op resolver.
    """
    require_eval_mode()
    if provider_factory is not None and probe is not None:
        raise EvalDriverError("CostProbe requires scripted responses; it cannot price a real run")
    try:
        base_url = os.environ[_TEST_DB_URL_ENV_VAR]
    except KeyError as exc:
        raise EvalDriverError(
            f"{_TEST_DB_URL_ENV_VAR} is not set; the eval driver needs the "
            "postgres-test URL. Run under pytest with --is-eval after "
            "`source .env`, or export it explicitly."
        ) from exc

    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))

    if provider_factory is None:
        _require_concurrency_safe_scripting(fixture, analyze_max_concurrency)
    if probe is not None and probe.model_cache and analyze_max_concurrency > 1:
        # Narrow to the CACHE-MODELED probe only: the in-flight-aware model
        # prices a concurrent first wave CORRECTLY, but HOW MANY workers land
        # in that wave depends on real persist() I/O timing — run-to-run the
        # modeled cost varies 1..cap writes, defeating the deterministic
        # before/after purpose (COST_ANALYSIS workflow). A cache-OFF probe is
        # timing-independent (per-call token counts + per-call pricing;
        # only the .calls list ORDER varies) and stays allowed.
        raise EvalDriverError(
            "CostProbe(model_cache=True) with analyze_max_concurrency > 1: "
            "the modeled cache-write count depends on worker timing, so probe "
            "runs would not be reproducible. Measure cache-modeled cost at "
            "analyze_max_concurrency=1, or drop model_cache for a concurrent "
            "run (per-call token counts are concurrency-independent)."
        )
    return asyncio.run(
        _arun_review(
            fixture,
            base_url,
            probe=probe,
            model_config=model_config,
            host=host,
            analyze_max_concurrency=analyze_max_concurrency,
            provider_factory=provider_factory,
            import_path_resolver=import_path_resolver,
        )
    )


# ---------------------------------------------------------------------------
# Resume driver — interrupt + checkpoint + process-restart + resume + publish
# ---------------------------------------------------------------------------


def _build_approve_all_decision(hitl_request: HITLRequest) -> HITLDecision:
    """Scripted approve-everything decision: one `APPROVE` per gated finding.

    The eval stand-in for a human reviewer's input through the production
    `Command(resume=...)` path — an EXPLICIT approval, never a model-set field or a
    default. `APPROVE` carries no override fields and allows an empty reason (per
    `PerFindingDecision.enforce_override_fields`). The HITL node re-validates this
    decision's finding-set against the gate request as defense-in-depth.
    """
    return HITLDecision(
        reviewer_id="eval",
        decisions=tuple(
            PerFindingDecision(
                finding_id=finding_id,
                outcome=PerFindingOutcome.APPROVE,
                reason="",
            )
            for finding_id in hitl_request.findings_requiring_approval
        ),
        annotation=None,
        decided_at=datetime.now(UTC),
    )


async def _assert_checkpoint_persisted(engine: AsyncEngine, review_id: UUID) -> None:
    """Guard the resume-identity invariant: the interrupt MUST have written a
    checkpoint row for this thread, else a fresh saver resumes nothing (it would
    start a new run instead of continuing the gated one)."""
    async with engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT 1 FROM checkpoints WHERE thread_id = :tid LIMIT 1"),
                {"tid": str(review_id)},
            )
        ).first()
    if row is None:
        raise EvalDriverError(
            f"no checkpoint persisted for thread_id={review_id} after the HITL "
            "interrupt; cannot resume (the gated run never reached Postgres, so a "
            "fresh checkpointer would resume nothing)."
        )


async def _read_review_status(engine: AsyncEngine, review_id: UUID) -> str:
    async with engine.connect() as conn:
        status = (
            await conn.execute(
                text("SELECT status FROM reviews WHERE id = :id"),
                {"id": review_id},
            )
        ).scalar_one()
    return str(status)


def _validate_eval_db_url(db_url: str) -> None:
    """Require the test port and per-run EVAL_DB_NAME_PREFIX before database work.

    The prefix excludes the shared base database. These are structural URL checks,
    not proof of a database's ownership; callers supply the already-migrated DB."""
    try:
        parsed = make_url(db_url)
    except (ArgumentError, ValueError) as exc:
        raise EvalDriverError(
            f"db_url is not a parseable database URL: {redact_url_password(db_url)!r}."
        ) from exc
    db_name = parsed.database or ""
    if parsed.port != EXPECTED_TEST_PORT or not db_name.startswith(EVAL_DB_NAME_PREFIX):
        raise EvalDriverError(
            f"db_url must be a per-test eval database (port {EXPECTED_TEST_PORT}, name "
            f"{EVAL_DB_NAME_PREFIX!r}*); got {redact_url_password(db_url)!r}. A "
            "caller-supplied-db_url eval driver (resume or serve) runs against a "
            "fixture-owned ephemeral DB, not the shared base."
        )


async def run_review_with_resume(
    fixture_path: str | os.PathLike[str],
    *,
    db_url: str,
    host: str | None = None,
    analyze_max_concurrency: int = 1,
) -> ResumedRunResult:
    """Drive to HITL, rebuild the graph/checkpointer, then explicitly approve and resume.

    The caller owns the already-migrated database so audit data survives for replay.
    Two sequential Postgres savers use the same review/thread ID; the second graph
    restores the checkpoint. This simulates graph/checkpointer reconstruction in one
    process, while fixture provider/publisher objects stay shared between both legs.
    Approval enters through Command(resume=...) (boundary #6).

    host selects the graph/provider identity (DECISIONS.md#056); publication itself
    makes no model call. Eval mode and the per-run database URL gate run before
    database or checkpointer work."""
    require_eval_mode()
    _validate_eval_db_url(db_url)

    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))
    _require_concurrency_safe_scripting(fixture, analyze_max_concurrency)

    # psycopg wants a bare URL; strip SQLAlchemy's driver suffix (mirrors lifespan).
    checkpoint_url = db_url.replace("postgresql+psycopg://", "postgresql://", 1)

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver  # noqa: PLC0415

    engine = create_async_engine(db_url, poolclass=NullPool)
    try:
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        # ONE id: seeds the review AND is the thread_id for BOTH legs. A mismatch
        # would make saver B resume a fresh run, not the gated one.
        review_id = uuid4()
        thread_config = {"configurable": {"thread_id": str(review_id)}}
        await _seed_installation(engine, fixture)
        await _seed_review(engine, review_id, fixture)

        persister = AuditPersister(
            session_factory=session_factory, retention_settings=RetentionSettings()
        )
        publisher = _CapturingPublisher()
        provider = _FixtureScriptedProvider(
            fixture.llm_responses,
            persister=persister,
            host=host,
            analyze_responses_by_path=fixture.analyze_responses_by_path,
        )

        # First graph/checkpointer: drive to the interrupt, then close the saver.
        async with AsyncPostgresSaver.from_conn_string(
            checkpoint_url, serde=build_checkpoint_serde()
        ) as saver_a:
            await saver_a.setup()
            graph_a = _build_eval_graph(
                fixture=fixture,
                session_factory=session_factory,
                persister=persister,
                provider=provider,
                publisher=publisher,
                checkpointer=saver_a,
                host=host,
                analyze_max_concurrency=analyze_max_concurrency,
            )
            result_a = await graph_a.ainvoke(_seed_state(fixture, review_id), config=thread_config)

        interrupts = result_a.get("__interrupt__")
        if not interrupts:
            raise EvalDriverError(
                "run_review_with_resume fixture did not trip the HITL gate; the "
                "resume scenario needs a CRITICAL/HIGH finding (no interrupt means "
                "there is nothing to resume)."
            )
        hitl_request = HITLRequest.model_validate(interrupts[0].value)
        await _assert_checkpoint_persisted(engine, review_id)
        decision = _build_approve_all_decision(hitl_request)

        # Rebuild graph/checkpointer on the same DB/thread to restore saved state.
        # The scripted provider and publisher remain shared between both legs.
        async with AsyncPostgresSaver.from_conn_string(
            checkpoint_url, serde=build_checkpoint_serde()
        ) as saver_b:
            graph_b = _build_eval_graph(
                fixture=fixture,
                session_factory=session_factory,
                persister=persister,
                provider=provider,
                publisher=publisher,
                checkpointer=saver_b,
                host=host,
                analyze_max_concurrency=analyze_max_concurrency,
            )
            result_b = await graph_b.ainvoke(
                Command(resume=decision.model_dump(mode="json")), config=thread_config
            )

        # LangGraph's serde rehydrates state-channel Pydantic models as instances
        # (not dicts) on the Postgres round-trip, so these are AnalysisRound objects.
        analysis_rounds: tuple[AnalysisRound, ...] = tuple(result_b.get("analysis_rounds", ()))
        published: tuple[InlineComment, ...] = tuple(
            c for call in publisher.create_review_calls for c in call["comments"]
        )
        review_status = await _read_review_status(engine, review_id)
        resumed = ResumedRunResult(
            review_id=review_id,
            analysis_rounds=analysis_rounds,
            published_comments=published,
            hitl_gated=True,
            hitl_decision=decision,
            review_status=review_status,
        )
        await _run_is_eval_gate(engine)
        return resumed
    except Exception:
        # Failure path: still verify is_eval discipline before the engine closes,
        # best-effort so a gate violation never masks the root-cause exception.
        # Mirrors `_drive` — the resume driver doesn't assume its caller gates
        # (it may run under `ephemeral_database`, not only the eval_db fixture).
        with contextlib.suppress(Exception):
            await _run_is_eval_gate(engine)
        raise
    finally:
        await engine.dispose()


async def run_review_persisting(
    fixture_path: str | os.PathLike[str],
    *,
    db_url: str,
    analyze_cache_store: AnalyzeCacheStore | None = None,
    cache_mode: CacheMode = CacheMode.SHADOW,
    analyze_observed_skip_enforced: bool = False,
    analyze_max_concurrency: int = 1,
) -> EvalRunResult:
    """Drive the graph ONCE against a caller-supplied, already-migrated `db_url`.

    Unlike the self-contained `run_review` (which owns + drops an ephemeral DB),
    this runs against the caller's `eval_db` so the audit stream AND the
    `analyze_file_cache` rows SURVIVE the call — the serve cache scenario needs
    both: an `AuditReplayer` reads the stream AFTER this returns, and a SECOND
    `run_review_persisting` against the same DB re-reviews the SAME FILE in a
    DISTINCT PR (the `reviews` natural key forbids reviewing one PR head twice), so
    the first drive's cache write becomes the second drive's serve hit (each call
    mints a fresh `review_id`, so the lookup's self-hit exclusion never suppresses
    the cross-review hit).

    `analyze_cache_store` must be bound to the SAME `db_url` (the scenario builds
    it from `eval_db`). Eval reviews use the cache like production, isolated by the
    lookup's required is_eval read-isolation predicate (DECISIONS.md#046) — so the
    serve scenario needs no special bypass override, just a store + CacheMode.SERVE.
    Fail-closed: `require_eval_mode()` + `_validate_eval_db_url` (a per-test eval
    DB, not the shared base) run before any DB work.
    """
    require_eval_mode()
    _validate_eval_db_url(db_url)

    with open(fixture_path, encoding="utf-8") as fh:
        fixture = EvalFixture.model_validate(json.load(fh))

    _require_concurrency_safe_scripting(fixture, analyze_max_concurrency)
    return await _drive(
        fixture,
        db_url,
        analyze_cache_store=analyze_cache_store,
        cache_mode=cache_mode,
        analyze_observed_skip_enforced=analyze_observed_skip_enforced,
        analyze_max_concurrency=analyze_max_concurrency,
    )
