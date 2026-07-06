# Parallel fan-out concurrency pins per specs/2026-07-05-parallel-analyze.md (increment 6).
"""Genuinely CONCURRENT analyze workers through the real compiled graph.

The eval scenario (tests/eval/scenarios/parallel/) covers the real-Postgres
end of increment 6 (persisted keyed streams, resume, cache writes); this
file covers what needs a controllable provider: DETERMINISTIC overlap (a
rendezvous barrier, not sleeps — the test deadlocks-and-fails rather than
flakes if concurrency is broken), per-file attribution under that overlap,
strict-hybrid verification over the full recorded stream, the
stamp-omission failure on an otherwise-real stream, and the no-retry
worker-failure policy.
"""

from __future__ import annotations

import asyncio

# Reuse the compiled-graph harness: stub factories, sinks, response builders.
import base64
import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from test_analyze_graph_wiring import (
    _SEED_INSTALLATION_ID,
    _SEED_OWNER,
    _SEED_PULL_NUMBER,
    _SEED_REPO,
    _build_kwargs,
    _lift_finding_event,
    _StubContentFile,
    _StubFileMeta,
    _StubResponse,
)

from outrider.agent.graph import build_graph
from outrider.audit.events import FileExaminationEvent, LLMCallEvent
from outrider.audit.replay import (
    ReplayEquivalenceError,
    _group_phases,
    _verify_phase_wellformed,
)
from outrider.llm.anthropic_provider import _ANTHROPIC_CONTRACT_DIGEST, _ANTHROPIC_PROFILE_ID
from outrider.llm.base import LLMRequest, LLMResponse
from outrider.schemas import ReviewState
from outrider.schemas.pr_context import ChangedFile, PRContext

_PATHS = ("src/f1.py", "src/f2.py", "src/f3.py")
_HEAD = "def work(x):\n    return x + 1\n"
_PATCH_TEMPLATE = "--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,2 @@\n+def work(x):\n+    return x + 1\n"


def _triage_all_deep() -> str:
    return json.dumps(
        {
            "file_tiers": dict.fromkeys(_PATHS, "deep"),
            "overall_risk": "medium",
            "relevant_dimensions": ["code_quality"],
            "reasoning": "test",
        }
    )


def _analyze_response_for(path: str) -> str:
    """One LOW (non-gated) finding naming its own file in the title, so
    cross-attribution is observable."""
    return json.dumps(
        {
            "findings": [
                {
                    "finding_type": "missing_error_handling",
                    "evidence_tier": "judged",
                    "query_match_id": None,
                    "trace_path": None,
                    "title": f"finding for {path}",
                    "description": "d",
                    "evidence": "return x + 1",
                    "line_start": 1,
                    "line_end": 2,
                    "trace_candidates": [],
                }
            ]
        }
    )


class _RendezvousProvider:
    """Serves triage + per-file analyze responses (keyed by the worker's
    `LLMRequest.phase_key`) and BLOCKS each analyze call until
    `rendezvous_n` calls are concurrently in flight — deterministic proof
    of overlap: if the semaphore (or the fan-out itself) serialized the
    workers, the barrier would never fill and the test fails on the
    wait timeout instead of flaking."""

    def __init__(self, *, rendezvous_n: int, fail_path: str | None = None) -> None:
        self.rendezvous_n = rendezvous_n
        self.fail_path = fail_path
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[LLMRequest] = []
        self._barrier_filled = asyncio.Event()
        # Set by the runner: production providers emit the LLMCallEvent
        # (mirroring request.phase_key verbatim — the pass-through
        # contract); this stub does the same into the combined stream.
        self.llm_event_stream: list[Any] | None = None

    async def aclose(self) -> None:
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        if request.node_id == "triage":
            return self._response(request, _triage_all_deep())
        if request.node_id == "synthesize":
            return self._response(request, "summary")
        assert request.node_id == "analyze"
        assert request.phase_key is not None, "pass-0 worker calls must carry their key"
        path = request.phase_key.removeprefix("file:").rsplit("#", 1)[0]
        if path == self.fail_path:
            raise RuntimeError(f"scripted worker failure for {path}")
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= self.rendezvous_n:
            self._barrier_filled.set()
        try:
            await asyncio.wait_for(self._barrier_filled.wait(), timeout=5.0)
        finally:
            self.in_flight -= 1
        self._emit_llm_call(request)
        return self._response(request, _analyze_response_for(path))

    def _emit_llm_call(self, request: LLMRequest) -> None:
        if self.llm_event_stream is None:
            return
        import hashlib

        self.llm_event_stream.append(
            LLMCallEvent(
                review_id=request.review_id,
                model=request.model,
                node_id="analyze",
                input_tokens=100,
                output_tokens=50,
                cached_tokens=0,
                cost_usd=0.01,
                pricing_version="v2",
                latency_ms=5,
                prompt_hash=hashlib.sha256(b"p").hexdigest(),
                cache_hit=False,
                context_summary=(),
                prompt_template_version="analyze.v1",
                system_prompt_hash=hashlib.sha256(b"s").hexdigest(),
                degraded_mode=False,
                is_eval=request.is_eval,
                # The provider pass-through contract: mirrored VERBATIM.
                phase_key=request.phase_key,
            )
        )

    def _response(self, request: LLMRequest, text: str) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=5,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )


class _MultiFileGitHub:
    """Three-file stub GitHub (the wiring harness's stub serves ONE file;
    intake re-enriches changed_files from this listing, so the fan-out
    needs its own)."""

    class _Repos:
        async def async_get_content(
            self, owner: str, repo: str, path: str, *, ref: str
        ) -> _StubResponse:
            content = _HEAD if ref == "b" * 40 else ""
            return _StubResponse(
                parsed_data=_StubContentFile(
                    encoding="base64",
                    content=base64.b64encode(content.encode()).decode("ascii"),
                )
            )

    class _Pulls:
        async def async_list_files(
            self, owner: str, repo: str, pull_number: int, **kwargs: Any
        ) -> _StubResponse:
            return _StubResponse(
                parsed_data=[
                    _StubFileMeta(
                        filename=path,
                        status="added",
                        additions=2,
                        deletions=0,
                        patch=_PATCH_TEMPLATE.format(path=path),
                    )
                    for path in _PATHS
                ]
            )

    def __init__(self) -> None:
        from types import SimpleNamespace

        self.rest = SimpleNamespace(repos=self._Repos(), pulls=self._Pulls())


def _multi_file_github_factory(installation_id: int) -> Any:
    assert installation_id == _SEED_INSTALLATION_ID
    return _MultiFileGitHub()


class _PermissivePublisher:
    """Publish genuinely runs here (the LOW findings are eligible without
    HITL) — record the create instead of raising like the wiring harness's
    unreachable stub."""

    def __init__(self) -> None:
        self.create_review_calls: list[dict[str, Any]] = []

    async def create_review(self, **kwargs: Any) -> Any:
        from outrider.schemas.publish import GitHubReviewCreated

        self.create_review_calls.append(kwargs)
        return GitHubReviewCreated(
            github_review_id=999, comments_posted=len(kwargs.get("comments", ()))
        )

    async def find_existing_review_on_head_sha(self, **kwargs: Any) -> int | None:  # noqa: ARG002
        return None


def _multi_file_state() -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=_SEED_INSTALLATION_ID,
            owner=_SEED_OWNER,
            repo=_SEED_REPO,
            pr_number=_SEED_PULL_NUMBER,
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="parallel",
            pr_body=None,
            author="someone",
            total_additions=6,
            total_deletions=0,
            changed_files=tuple(
                ChangedFile(
                    path=path,
                    status="added",
                    additions=2,
                    deletions=0,
                    patch=_PATCH_TEMPLATE.format(path=path),
                    content_base=None,
                    content_head=_HEAD,
                    previous_path=None,
                    language="python",
                )
                for path in _PATHS
            ),
        ),
        is_eval=True,
    )


class _CombinedRecorder:
    """One ORDERED stream across the three analyze-facing sinks — the shape
    `_verify_phase_wellformed` / `_group_phases` consume. Mirrors the
    persister's FindingEvent lift so the recorded stream matches what the
    audit table would carry."""

    def __init__(self) -> None:
        self.stream: list[Any] = []

    # PhaseEventSink
    async def emit_phase(self, event: Any) -> None:
        self.stream.append(event)

    # FileExaminationSink
    async def emit_file_examination(self, event: Any) -> None:
        self.stream.append(event)

    # AnalyzeEventSink
    async def emit_finding(
        self, finding: Any, *, is_eval: bool, phase_key: str | None = None
    ) -> None:
        self.stream.append(_lift_finding_event(finding, is_eval=is_eval, phase_key=phase_key))

    async def emit_finding_proposal_rejected(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_analyze_response_rejected(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_analyze_completed(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_scope_exclusion(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_cache_lookup(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_cache_serve(self, event: Any) -> None:
        self.stream.append(event)

    async def emit_observed_skip_shadow(self, event: Any) -> None:
        self.stream.append(event)


async def _run_parallel_graph(
    provider: _RendezvousProvider,
) -> tuple[dict[str, Any], _CombinedRecorder]:
    recorder = _CombinedRecorder()
    state = _multi_file_state()
    kwargs = _build_kwargs(
        provider=provider,  # type: ignore[arg-type]
        phase_event_sink=recorder,  # type: ignore[arg-type]
        file_examination_sink=recorder,  # type: ignore[arg-type]
        analyze_event_sink=recorder,  # type: ignore[arg-type]
    )
    kwargs["analyze_max_concurrency"] = 4
    kwargs["github_factory"] = _multi_file_github_factory
    kwargs["publisher"] = _PermissivePublisher()
    provider.llm_event_stream = recorder.stream
    graph = build_graph(**kwargs)
    result = await graph.ainvoke(
        state, config={"configurable": {"thread_id": str(state.review_id)}}
    )
    return result, recorder


@pytest.mark.asyncio
async def test_concurrent_workers_overlap_and_attribute_correctly() -> None:
    """DETERMINISTIC overlap (all three workers must be in flight at once
    for the rendezvous to release) + per-file attribution + one folded
    round + complete slot outcomes — the fan-out actually ran in parallel
    and nothing crossed between workers."""
    provider = _RendezvousProvider(rendezvous_n=3)
    result, _recorder = await _run_parallel_graph(provider)

    assert provider.max_in_flight == 3  # genuine concurrency, proven not sampled
    (round_,) = result["analysis_rounds"]
    assert set(round_.files_examined) == set(_PATHS)
    titles = {f.file_path: f.title for f in round_.findings}
    assert titles == {path: f"finding for {path}" for path in _PATHS}  # no cross-attribution
    outcomes = result["analyze_worker_outcomes"]
    assert sorted(o.path for o in outcomes) == sorted(_PATHS)  # every slot filled once


@pytest.mark.asyncio
async def test_concurrent_stream_verifies_and_groups_by_identity() -> None:
    """The full recorded stream (interleaved keyed worker events, legacy
    un-keyed phases for the other nodes) passes the strict hybrid verifier,
    and grouping places each worker's LLM call + examination event in its
    OWN envelope with the findings under the aggregate's."""
    provider = _RendezvousProvider(rendezvous_n=3)
    _result, recorder = await _run_parallel_graph(provider)
    stream = tuple(recorder.stream)

    _verify_phase_wellformed(stream)  # strict hybrid, whole-graph stream

    phases = _group_phases(stream)
    by_key = {p.phase_key: p for p in phases if p.phase_key is not None}
    for path in _PATHS:
        worker_phase = by_key[f"file:{path}#0"]
        llm_events = [e for e in worker_phase.events if isinstance(e, LLMCallEvent)]
        fe_events = [e for e in worker_phase.events if isinstance(e, FileExaminationEvent)]
        assert len(llm_events) == 1 and llm_events[0].phase_key == f"file:{path}#0"  # noqa: PT018
        assert len(fe_events) == 1 and fe_events[0].file_path == path  # noqa: PT018
    aggregate_phase = by_key["aggregate#0"]
    finding_paths = {
        e.file_path for e in aggregate_phase.events if type(e).__name__ == "FindingEvent"
    }
    assert finding_paths == set(_PATHS)


@pytest.mark.asyncio
async def test_stamp_omission_on_a_real_stream_fails_loud() -> None:
    """Surgical mutation of an otherwise-REAL stream: strip the key from one
    worker's FileExaminationEvent and the strict None-branch must reject the
    stream — a stamping regression cannot pass as legacy data."""
    provider = _RendezvousProvider(rendezvous_n=3)
    _result, recorder = await _run_parallel_graph(provider)

    def strip_one(event: Any) -> Any:
        if isinstance(event, FileExaminationEvent) and event.phase_key == f"file:{_PATHS[0]}#0":
            return FileExaminationEvent.model_validate({**event.model_dump(), "phase_key": None})
        return event

    mutated = tuple(strip_one(e) for e in recorder.stream)
    assert mutated != tuple(recorder.stream)  # the mutation actually landed
    with pytest.raises(ReplayEquivalenceError, match="stamp-omission"):
        _verify_phase_wellformed(mutated)


@pytest.mark.asyncio
async def test_worker_failure_fails_the_pass_with_no_retry() -> None:
    """The pinned failure policy: NO worker-level retry machinery — one
    worker's provider exception fails the whole pass, parity with the
    sequential loop's abort. The failed file's LLM call happened once."""
    provider = _RendezvousProvider(rendezvous_n=2, fail_path=_PATHS[1])
    with pytest.raises(RuntimeError, match="scripted worker failure"):
        await _run_parallel_graph(provider)
    failing_calls = [
        r for r in provider.calls if r.node_id == "analyze" and r.phase_key == f"file:{_PATHS[1]}#0"
    ]
    assert len(failing_calls) == 1  # once — never retried
