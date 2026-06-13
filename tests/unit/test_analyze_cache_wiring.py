# Per specs/2026-06-11-file-hash-analyze-cache.md — analyze-node shadow wiring.
"""Analyze-cache shadow wiring through the analyze node.

Pins the Stage-B contracts: store-or-None is the enable switch (None =
zero cache behavior); a miss emits `CacheLookupEvent(outcome="miss")`,
calls the model, and writes the store with the composed key + content
payload + the full version-component set; a would-hit emits
`outcome="would_hit"`, STILL calls the model (shadow — nothing served),
and writes nothing; an eval review never touches a wired store (by
EITHER the resolved scope's flag or state.is_eval); the lookup excludes
the review's own prior writes (crash-resume self-hits); a
response-level rejection and a `max_tokens`-truncated response cache
nothing; and a `CacheStoreError` from any store call is contained — the
shadow cache must never abort a review.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from outrider.agent.nodes.analyze import DEFAULT_REVIEW_BUDGET_TOKENS, analyze
from outrider.agent.nodes.analyze_parser import ANALYZE_PARSER_VERSION
from outrider.agent.nodes.cache_config import CacheMode
from outrider.ast_facts.parameterized_calls import scan_digest, scan_parameterized_calls
from outrider.ast_facts.triviality import TRIVIAL_FILTER_VERSION
from outrider.audit.events import compute_finding_content_hash
from outrider.cache import CacheEntry, CacheScope, CacheStoreError, compute_analyze_cache_key
from outrider.llm.base import LLMRequest, LLMResponse, _canonical_prompt_hash
from outrider.policy import EvidenceTier, FindingType
from outrider.policy.canonical import compute_served_finding_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION, SEVERITY_POLICY
from outrider.prompts import analyze as analyze_prompt
from outrider.queries.registry import QUERY_REGISTRY_DIGEST
from outrider.schemas import ChangedFile, PRContext, ReviewFinding, ReviewState
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_FORMAT_DIGEST,
    ANALYZE_RESPONSE_SCHEMA_JSON,
)
from outrider.schemas.triage_result import (
    ReviewDimension,
    ReviewTier,
    RiskLevel,
    TriageResult,
)

_REVIEW_ID = UUID("11112222-3333-4444-5555-666677778888")


class _FakeCacheStore:
    """Records calls; lookup behavior is scripted per test.

    `raise_on` scripts a `CacheStoreError` from one named method —
    the containment tests' lever.
    """

    def __init__(
        self,
        *,
        scope: CacheScope | None,
        entry: CacheEntry | None = None,
        raise_on: str | None = None,
    ) -> None:
        self._scope = scope
        self._entry = entry
        self._raise_on = raise_on
        self.resolve_calls: list[UUID] = []
        self.lookup_calls: list[tuple[str, UUID | None]] = []
        self.write_calls: list[dict[str, Any]] = []

    async def resolve_scope(self, review_id: UUID) -> CacheScope | None:
        self.resolve_calls.append(review_id)
        if self._raise_on == "resolve_scope":
            raise CacheStoreError("scripted resolve failure")
        return self._scope

    async def lookup(
        self, cache_key: str, *, exclude_source_review_id: UUID | None = None
    ) -> CacheEntry | None:
        self.lookup_calls.append((cache_key, exclude_source_review_id))
        if self._raise_on == "lookup":
            raise CacheStoreError("scripted lookup failure")
        return self._entry

    async def write(self, **kwargs: Any) -> None:
        self.write_calls.append(kwargs)
        if self._raise_on == "write":
            raise CacheStoreError("scripted write failure")


class _StubLLMProvider:
    def __init__(self, response_text: str | None = None, finish_reason: str = "end_turn") -> None:
        self.calls: list[LLMRequest] = []
        self._text = response_text if response_text is not None else json.dumps({"findings": []})
        self._finish_reason = finish_reason

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return LLMResponse(
            text=self._text,
            model=request.model,
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason=self._finish_reason,
            latency_ms=10,
        )


class _NoOpPhaseSink:
    async def emit_phase(self, event: Any) -> None:  # noqa: ARG002
        return None


class _RecordingFileExaminationSink:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def emit_file_examination(self, event: Any) -> None:
        self.events.append(event)


class _RecordingAnalyzeEventSink:
    def __init__(self) -> None:
        self.findings: list[Any] = []
        self.proposal_rejections: list[Any] = []
        self.response_rejections: list[Any] = []
        self.completed: list[Any] = []
        self.scope_exclusions: list[Any] = []
        self.cache_lookups: list[Any] = []
        self.cache_serves: list[Any] = []

    async def emit_finding(self, finding: Any, *, is_eval: bool) -> None:
        self.findings.append((finding, is_eval))

    async def emit_finding_proposal_rejected(self, event: Any) -> None:
        self.proposal_rejections.append(event)

    async def emit_analyze_response_rejected(self, event: Any) -> None:
        self.response_rejections.append(event)

    async def emit_analyze_completed(self, event: Any) -> None:
        self.completed.append(event)

    async def emit_scope_exclusion(self, event: Any) -> None:
        self.scope_exclusions.append(event)

    async def emit_cache_lookup(self, event: Any) -> None:
        self.cache_lookups.append(event)

    async def emit_cache_serve(self, event: Any) -> None:
        self.cache_serves.append(event)


_HEAD = """\
import os


def alpha():
    y = len(os.sep)
    return y
"""

_BASE = """\
import os


def alpha():
    return os.sep
"""

_PATCH = (
    "--- a/src/cached.py\n+++ b/src/cached.py\n"
    "@@ -4,2 +4,3 @@\n"
    " def alpha():\n"
    "+    y = len(os.sep)\n"
    "     return os.sep\n"
)

_SCOPE = CacheScope(
    installation_id=42,
    repo_id=7,
    is_eval=False,
    retention_expires_at=datetime(2027, 1, 1, tzinfo=UTC),
)


def _state(*, is_eval: bool = False) -> ReviewState:
    changed = ChangedFile(
        path="src/cached.py",
        status="modified",
        additions=1,
        deletions=0,
        patch=_PATCH,
        content_base=_BASE,
        content_head=_HEAD,
        previous_path=None,
        language="python",
    )
    pr_context = PRContext(
        installation_id=42,
        owner="acme",
        repo="widget",
        pr_number=9,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title="t",
        pr_body=None,
        author="someone",
        total_additions=1,
        total_deletions=0,
        changed_files=(changed,),
    )
    triage = TriageResult(
        file_tiers={"src/cached.py": ReviewTier.DEEP},
        overall_risk=RiskLevel.MEDIUM,
        relevant_dimensions=(ReviewDimension.SECURITY,),
        reasoning="test",
    )
    # Default is the production-real combination: state.is_eval matches
    # the reviews row both False. The eval-veto tests override one side.
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context,
        triage_result=triage,
        is_eval=is_eval,
    )


async def _run(
    store: _FakeCacheStore | None,
    *,
    response_text: str | None = None,
    finish_reason: str = "end_turn",
    state_is_eval: bool = False,
    cache_mode: CacheMode = CacheMode.SHADOW,
) -> tuple[_StubLLMProvider, _RecordingAnalyzeEventSink]:
    provider = _StubLLMProvider(response_text, finish_reason=finish_reason)
    sink = _RecordingAnalyzeEventSink()
    await analyze(
        _state(is_eval=state_is_eval),
        provider=provider,  # type: ignore[arg-type]
        analyze_model="claude-sonnet-4-6",
        standard_analyze_model="claude-sonnet-4-6",
        phase_event_sink=_NoOpPhaseSink(),
        file_examination_sink=_RecordingFileExaminationSink(),
        analyze_event_sink=sink,
        import_path_resolver=MagicMock(),
        active_policy_version=ACTIVE_POLICY_VERSION,
        total_review_budget_tokens=DEFAULT_REVIEW_BUDGET_TOKENS,
        analyze_cache_store=store,  # type: ignore[arg-type]
        cache_mode=cache_mode,
    )
    return provider, sink


@pytest.mark.asyncio
async def test_no_store_means_zero_cache_behavior() -> None:
    provider, sink = await _run(None)
    assert len(provider.calls) == 1
    assert sink.cache_lookups == []


@pytest.mark.asyncio
async def test_miss_emits_event_calls_model_and_writes() -> None:
    store = _FakeCacheStore(scope=_SCOPE, entry=None)
    provider, sink = await _run(store)

    assert store.resolve_calls == [_REVIEW_ID]
    # Self-hit exclusion: the lookup names the current review so a
    # crash-resume re-run can't count its own writes as hits.
    [(looked_up_key, excluded)] = store.lookup_calls
    assert excluded == _REVIEW_ID
    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert event.is_eval is False  # threaded from state, not hardcoded
    assert len(provider.calls) == 1  # shadow: model always called
    [write] = store.write_calls
    # The written key is exactly the recomputed full key (prompt digest
    # + ten explicit components) over the request actually sent.
    [request] = provider.calls
    # FUP-096: the request that produced the cached payload rode with the
    # pinned schema — the key's response_format_digest describes it truly.
    assert request.response_schema_json == ANALYZE_RESPONSE_SCHEMA_JSON
    expected_key = compute_analyze_cache_key(
        system_prompt=request.system_prompt,
        user_prompt=request.user_prompt,
        installation_id=_SCOPE.installation_id,
        repo_id=_SCOPE.repo_id,
        model="claude-sonnet-4-6",
        prompt_template_version=analyze_prompt.VERSION,
        trivial_filter_version=TRIVIAL_FILTER_VERSION,
        query_registry_digest=QUERY_REGISTRY_DIGEST,
        active_policy_version=ACTIVE_POLICY_VERSION,
        analyze_parser_version=ANALYZE_PARSER_VERSION,
        response_format_digest=ANALYZE_RESPONSE_FORMAT_DIGEST,
        parameterized_call_scan_digest=scan_digest(scan_parameterized_calls(_HEAD.encode("utf-8"))),
    )
    assert write["cache_key"] == expected_key == event.cache_key == looked_up_key
    assert write["source_review_id"] == _REVIEW_ID
    assert write["payload"]["findings"] == []  # zero findings IS cacheable
    assert write["payload"]["trace_candidates"] == []
    assert write["prompt_hash"] == _canonical_prompt_hash(
        system_prompt=request.system_prompt, user_prompt=request.user_prompt
    )
    # The denormalized component columns must carry the SAME values the
    # key was composed from — a write-side value drifting from the key
    # side (e.g. module constant instead of the threaded policy version)
    # would silently misdescribe rows in the Stage-B telemetry.
    assert write["model"] == "claude-sonnet-4-6"
    assert write["prompt_template_version"] == analyze_prompt.VERSION
    assert write["trivial_filter_version"] == TRIVIAL_FILTER_VERSION
    assert write["query_registry_digest"] == QUERY_REGISTRY_DIGEST
    assert write["active_policy_version"] == ACTIVE_POLICY_VERSION
    assert write["analyze_parser_version"] == ANALYZE_PARSER_VERSION


@pytest.mark.asyncio
async def test_would_hit_emits_event_still_calls_model_writes_nothing() -> None:
    entry = CacheEntry(
        cache_key="0" * 64,
        payload={"findings": [], "trace_candidates": []},
        source_review_id=uuid4(),
        file_path="src/cached.py",
        created_at=datetime(2026, 6, 1, tzinfo=UTC),
    )
    store = _FakeCacheStore(scope=_SCOPE, entry=entry)
    provider, sink = await _run(store)

    [event] = sink.cache_lookups
    assert event.outcome == "would_hit"
    assert len(provider.calls) == 1  # SHADOW: nothing is served
    assert store.write_calls == []


@pytest.mark.asyncio
async def test_is_eval_review_never_touches_a_wired_store() -> None:
    """Belt-and-suspenders for the eval-bypass rule: a wired store with
    an is_eval scope is disabled for the whole pass."""
    eval_scope = CacheScope(
        installation_id=42,
        repo_id=7,
        is_eval=True,
        retention_expires_at=datetime(2027, 1, 1, tzinfo=UTC),
    )
    store = _FakeCacheStore(scope=eval_scope)
    provider, sink = await _run(store)

    assert store.resolve_calls == [_REVIEW_ID]  # scope was resolved...
    assert store.lookup_calls == []  # ...then the cache disabled
    assert store.write_calls == []
    assert sink.cache_lookups == []
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_state_is_eval_also_disables_a_wired_store() -> None:
    """The veto consults EITHER flag: state.is_eval=True disables the
    cache even when the resolved scope says is_eval=False — a divergence
    between the two sources can never write production cache rows."""
    store = _FakeCacheStore(scope=_SCOPE)  # scope says is_eval=False
    provider, sink = await _run(store, state_is_eval=True)

    assert store.resolve_calls == [_REVIEW_ID]
    assert store.lookup_calls == []
    assert store.write_calls == []
    assert sink.cache_lookups == []
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_resolve_scope_failure_is_contained() -> None:
    """A CacheStoreError from resolve_scope disables the cache for the
    pass — the shadow cache must never abort a review."""
    store = _FakeCacheStore(scope=_SCOPE, raise_on="resolve_scope")
    provider, sink = await _run(store)

    assert store.resolve_calls == [_REVIEW_ID]
    assert store.lookup_calls == []
    assert store.write_calls == []
    assert sink.cache_lookups == []
    assert len(provider.calls) == 1  # the review proceeded uncached


@pytest.mark.asyncio
async def test_lookup_failure_is_contained_no_event_no_write() -> None:
    """A CacheStoreError from lookup skips the cache for the file: the
    model is still called, NO CacheLookupEvent is emitted (the lookup
    never completed — a fabricated 'miss' would be false audit
    history), and the write gate skips."""
    store = _FakeCacheStore(scope=_SCOPE, raise_on="lookup")
    provider, sink = await _run(store)

    assert len(store.lookup_calls) == 1  # the lookup was attempted
    assert sink.cache_lookups == []  # ...but no event for a failed lookup
    assert store.write_calls == []
    assert len(provider.calls) == 1  # the review proceeded uncached


@pytest.mark.asyncio
async def test_write_failure_is_contained() -> None:
    """A CacheStoreError from the write loses one memoization, nothing
    else — findings are already emitted and the review completes."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None, raise_on="write")
    provider, sink = await _run(store)

    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert len(store.write_calls) == 1  # attempted, raised, contained
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_truncated_response_is_not_cached() -> None:
    """finish_reason='max_tokens' with JSON that still validates is NOT
    memoized: the finding set may be silently incomplete, and a cached
    truncation would be served for the row's whole lifetime."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None)
    provider, sink = await _run(store, finish_reason="max_tokens")

    [event] = sink.cache_lookups
    assert event.outcome == "miss"  # the lookup itself was fine
    assert len(provider.calls) == 1
    assert store.write_calls == []  # ...but the truncated outcome never lands


@pytest.mark.asyncio
async def test_response_rejection_caches_nothing() -> None:
    """A response-level parse failure has no admitted outcome to cache —
    the lookup event still fires (the lookup happened), the write must not."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None)
    provider, sink = await _run(store, response_text="NOT JSON {{{")

    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert len(provider.calls) == 1
    assert store.write_calls == []


# ---------------------------------------------------------------------------
# Stage B serve flip (cache_mode=serve).
# ---------------------------------------------------------------------------


def _build_cached_finding() -> ReviewFinding:
    """A valid JUDGED-tier finding for a cache payload. Severity is the live
    policy baseline so the served reconstruction's `_enforce_severity_matches_policy`
    passes; review_id/installation_id are the SOURCE review's (serve re-stamps)."""
    return ReviewFinding(
        review_id=uuid4(),
        installation_id=999,
        policy_version=ACTIVE_POLICY_VERSION,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=SEVERITY_POLICY[FindingType.SQL_INJECTION],
        evidence_tier=EvidenceTier.JUDGED,
        file_path="src/cached.py",
        line_start=4,
        line_end=6,
        title="SQL injection",
        description="User input concatenated into the SQL string.",
        evidence="cursor.execute('SELECT ... ' + user_id)",
        query_match_id=None,
        trace_path=None,
        proposal_hash="a" * 64,
        content_hash=compute_finding_content_hash(
            file_path="src/cached.py",
            line_start=4,
            line_end=6,
            finding_type=FindingType.SQL_INJECTION,
        ),
    )


def test_served_finding_id_is_deterministic() -> None:
    """The re-mint keystone: finding_id is a pure function of (new review,
    content_hash, proposal_hash) so a checkpoint replay reproduces it, keeping
    the persister's no-resurrection content-row guard correct."""
    a = compute_served_finding_id(
        review_id=_REVIEW_ID, content_hash="x" * 64, proposal_hash="y" * 64
    )
    b = compute_served_finding_id(
        review_id=_REVIEW_ID, content_hash="x" * 64, proposal_hash="y" * 64
    )
    assert a == b
    # A different review, content, or proposal yields a different id.
    assert a != compute_served_finding_id(
        review_id=uuid4(), content_hash="x" * 64, proposal_hash="y" * 64
    )
    assert a != compute_served_finding_id(
        review_id=_REVIEW_ID, content_hash="z" * 64, proposal_hash="y" * 64
    )
    assert a != compute_served_finding_id(
        review_id=_REVIEW_ID, content_hash="x" * 64, proposal_hash="z" * 64
    )


@pytest.mark.asyncio
async def test_serve_hit_short_circuits_and_reemits_finding() -> None:
    """A live hit under cache_mode=serve: NO LLM call; a CacheServeEvent (not a
    CacheLookupEvent); the cached finding re-emitted on THIS review with a
    deterministic re-mint; no write; accounting rides n_findings_served."""
    source_finding = _build_cached_finding()
    entry = CacheEntry(
        cache_key="c" * 64,
        payload={"findings": [source_finding.model_dump(mode="json")], "trace_candidates": []},
        source_review_id=uuid4(),
        file_path="src/cached.py",
        created_at=datetime(2026, 6, 13, 12, 0, 0, tzinfo=UTC),
    )
    store = _FakeCacheStore(scope=_SCOPE, entry=entry)
    provider, sink = await _run(store, cache_mode=CacheMode.SERVE)

    assert provider.calls == []  # served — the model was never called
    assert sink.cache_lookups == []
    [serve] = sink.cache_serves
    assert serve.served_finding_count == 1
    # The serve event records the key the node COMPOSED + looked up (the fake
    # store returns the entry regardless of key), not the seeded entry's field.
    [(looked_up_key, _excluded)] = store.lookup_calls
    assert serve.cache_key == looked_up_key

    [(served_finding, _is_eval)] = sink.findings
    assert served_finding.review_id == _REVIEW_ID  # re-stamped onto this review
    assert served_finding.finding_id == compute_served_finding_id(
        review_id=_REVIEW_ID,
        content_hash=source_finding.content_hash,
        proposal_hash=source_finding.proposal_hash,
    )
    assert served_finding.content_hash == source_finding.content_hash  # content preserved
    assert store.write_calls == []  # a serve hit writes nothing

    [completed] = sink.completed
    assert completed.n_llm_calls == 0
    assert completed.n_findings_emitted == 1
    assert completed.n_findings_served == 1


@pytest.mark.asyncio
async def test_serve_miss_calls_model_and_writes() -> None:
    """A serve-miss is a real miss: the model runs, miss telemetry fires, and
    step 3g writes the new outcome (same as shadow-miss)."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None)
    provider, sink = await _run(store, cache_mode=CacheMode.SERVE)
    assert len(provider.calls) == 1
    assert sink.cache_serves == []
    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert len(store.write_calls) == 1


@pytest.mark.asyncio
async def test_serve_lookup_error_degrades_to_model() -> None:
    """A contained CacheStoreError on a SERVE lookup degrades to a real LLM
    call — NEVER a silent skip of findings. No serve, no lookup event."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None, raise_on="lookup")
    provider, sink = await _run(store, cache_mode=CacheMode.SERVE)
    assert len(provider.calls) == 1
    assert sink.cache_serves == []
    assert sink.cache_lookups == []
