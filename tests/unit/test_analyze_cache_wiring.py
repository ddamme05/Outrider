# Per specs/2026-06-11-file-hash-analyze-cache.md — analyze-node shadow wiring.
"""Analyze-cache shadow wiring through the analyze node.

Pins the Stage-B contracts: store-or-None is the enable switch (None =
zero cache behavior); a miss emits `CacheLookupEvent(outcome="miss")`,
calls the model, and writes the store with the composed key + content
payload; a would-hit emits `outcome="would_hit"`, STILL calls the model
(shadow — nothing served), and writes nothing; an is_eval review never
touches a wired store; a response-level rejection caches nothing.
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
from outrider.ast_facts.triviality import TRIVIAL_FILTER_VERSION
from outrider.cache import CacheEntry, CacheScope, compute_analyze_cache_key
from outrider.llm.base import LLMRequest, LLMResponse, _canonical_prompt_hash
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
from outrider.queries.registry import QUERY_REGISTRY_DIGEST
from outrider.schemas import ChangedFile, PRContext, ReviewState
from outrider.schemas.triage_result import (
    ReviewDimension,
    ReviewTier,
    RiskLevel,
    TriageResult,
)

_REVIEW_ID = UUID("11112222-3333-4444-5555-666677778888")


class _FakeCacheStore:
    """Records calls; lookup behavior is scripted per test."""

    def __init__(self, *, scope: CacheScope | None, entry: CacheEntry | None = None) -> None:
        self._scope = scope
        self._entry = entry
        self.resolve_calls: list[UUID] = []
        self.lookup_calls: list[str] = []
        self.write_calls: list[dict[str, Any]] = []

    async def resolve_scope(self, review_id: UUID) -> CacheScope | None:
        self.resolve_calls.append(review_id)
        return self._scope

    async def lookup(self, cache_key: str) -> CacheEntry | None:
        self.lookup_calls.append(cache_key)
        return self._entry

    async def write(self, **kwargs: Any) -> None:
        self.write_calls.append(kwargs)


class _StubLLMProvider:
    def __init__(self, response_text: str | None = None) -> None:
        self.calls: list[LLMRequest] = []
        self._text = response_text if response_text is not None else json.dumps({"findings": []})

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
            finish_reason="end_turn",
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


def _state() -> ReviewState:
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
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime(2026, 6, 11, 12, 0, 0, tzinfo=UTC),
        pr_context=pr_context,
        triage_result=triage,
        is_eval=True,
    )


async def _run(
    store: _FakeCacheStore | None, *, response_text: str | None = None
) -> tuple[_StubLLMProvider, _RecordingAnalyzeEventSink]:
    provider = _StubLLMProvider(response_text)
    sink = _RecordingAnalyzeEventSink()
    await analyze(
        _state(),
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
    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert len(provider.calls) == 1  # shadow: model always called
    [write] = store.write_calls
    # The written key is exactly the recomputed full key (prompt digest
    # + eight explicit components) over the request actually sent.
    [request] = provider.calls
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
    )
    assert write["cache_key"] == expected_key == event.cache_key
    assert write["source_review_id"] == _REVIEW_ID
    assert write["payload"]["findings"] == []  # zero findings IS cacheable
    assert write["payload"]["trace_candidates"] == []
    assert write["prompt_hash"] == _canonical_prompt_hash(
        system_prompt=request.system_prompt, user_prompt=request.user_prompt
    )


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
async def test_response_rejection_caches_nothing() -> None:
    """A response-level parse failure has no admitted outcome to cache —
    the lookup event still fires (the lookup happened), the write must not."""
    store = _FakeCacheStore(scope=_SCOPE, entry=None)
    provider, sink = await _run(store, response_text="NOT JSON {{{")

    [event] = sink.cache_lookups
    assert event.outcome == "miss"
    assert len(provider.calls) == 1
    assert store.write_calls == []
