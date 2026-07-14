"""Unit tests for the eval driver's fixture-replay doubles (no DB, no graph).

Covers the four `src`-local adapters in `outrider.agent.eval_driver` in
isolation: the scripted LLM provider's `(node, call-index)` keying, the fake
GitHub client's wire-shape fidelity (newline-wrapped base64 that survives
intake's strip-then-`validate=True` decode), the capturing publisher, and the
`EvalFixture` schema's `extra="forbid"` discipline. The full DB-backed run is
covered by `tests/eval/test_run_review_driver.py`.
"""

from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.eval_driver import (
    CostProbe,
    EvalDriverError,
    EvalFixture,
    EvalRunResult,
    _build_approve_all_decision,
    _CapturingPublisher,
    _FixtureContentNotFoundError,
    _FixtureScriptedProvider,
    _github_factory_for,
    _validate_eval_db_url,
)
from outrider.llm.base import LLMRequest
from outrider.schemas.hitl import HITLRequest, PerFindingOutcome

# asyncio_mode = "auto" (pyproject) auto-detects async tests; no marker needed.

_INSTALLATION_ID = 4242
_BASE_SHA = "a" * 40
_HEAD_SHA = "b" * 40
# >60 chars so base64.encodebytes wraps with newlines (the real GitHub shape).
_HEAD_CONTENT = (
    "def vulnerable(user_input):\n    return run('SELECT * FROM t WHERE x=' + user_input)\n"
)
_BASE_CONTENT = "def vulnerable(user_input):\n    return 1\n"


def _fixture(**overrides: object) -> EvalFixture:
    base: dict[str, object] = {
        "installation_id": _INSTALLATION_ID,
        "owner": "acme",
        "repo": "widget",
        "pr_number": 7,
        "base_sha": _BASE_SHA,
        "head_sha": _HEAD_SHA,
        "pr_title": "t",
        "author": "someone",
        "total_additions": 1,
        "total_deletions": 0,
        "files": [
            {
                "path": "app/views.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": (
                    "@@ -1,2 +1,2 @@\n def vulnerable(user_input):\n"
                    "-    return 1\n+    return run('x')\n"
                ),
                "content_base": _BASE_CONTENT,
                "content_head": _HEAD_CONTENT,
            }
        ],
        "llm_responses": {"triage": ["{}"], "analyze": ["{}"]},
    }
    base.update(overrides)
    return EvalFixture.model_validate(base)


def _triage_request() -> LLMRequest:
    # triage does not require context_summary (only analyze does).
    return LLMRequest(
        system_prompt="s",
        user_prompt="u",
        model="claude-haiku-4-5",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",
        prompt_template_version="v1",
        degraded_mode=False,
    )


# ---------------------------------------------------------------------------
# _FixtureScriptedProvider
# ---------------------------------------------------------------------------


class _RecordingPersist:
    """Minimal `LLMExchangePersister` double: records persisted `LLMCallEvent`s,
    no DB. Lets the unit tests exercise the fixture's now-persisting `complete()`
    (FUP-093) without Postgres; the eval suite covers the real `persist()`
    cross-checks end-to-end."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def persist(self, event: Any, request: Any, response: Any) -> None:  # noqa: ARG002
        self.events.append(event)


async def test_scripted_provider_returns_response_by_node_and_call_index() -> None:
    persist = _RecordingPersist()
    provider = _FixtureScriptedProvider({"triage": ["first", "second"]}, persister=persist)
    req = _triage_request()
    r1 = await provider.complete(req)
    r2 = await provider.complete(req)
    assert r1.text == "first"
    assert r2.text == "second"
    # echoes the request model + emits a valid LLMResponse shape.
    assert r1.model == "claude-haiku-4-5"
    assert r1.finish_reason == "end_turn"
    # FUP-093: each complete() emits a faithful LLMCallEvent (cost computed from the
    # real pricing table) via the persister before returning.
    assert len(persist.events) == 2
    assert persist.events[0].node_id == "triage"
    assert persist.events[0].input_tokens == 100
    assert persist.events[0].cost_usd > 0


async def test_scripted_provider_raises_loud_when_exhausted() -> None:
    provider = _FixtureScriptedProvider({"triage": ["only-one"]}, persister=_RecordingPersist())
    req = _triage_request()
    await provider.complete(req)
    with pytest.raises(EvalDriverError, match="no scripted LLM response"):
        await provider.complete(req)


async def test_scripted_provider_raises_for_unscripted_node() -> None:
    provider = _FixtureScriptedProvider({"triage": ["x"]}, persister=_RecordingPersist())
    # node_id="synthesize" is unscripted in this {"triage": ...} provider.
    req = LLMRequest(
        system_prompt="s",
        user_prompt="u",
        model="m",
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="synthesize",
        prompt_template_version="v1",
        degraded_mode=False,
    )
    with pytest.raises(EvalDriverError, match="no scripted LLM response"):
        await provider.complete(req)


# ---------------------------------------------------------------------------
# CostProbe cache modeling (model_cache=True)
# ---------------------------------------------------------------------------

# token_estimator=len → 1 char = 1 token, so floor crossings are exact:
# claude-haiku-4-5 floor is 4096 (MIN_CACHEABLE_TOKENS).
_ABOVE_FLOOR_SYSTEM = "S" * 5000
_USER = "diff under review"


def _cache_request(
    system: str,
    *,
    cache_control: bool = True,
    review_id: Any = None,
) -> LLMRequest:
    return LLMRequest(
        system_prompt=system,
        user_prompt=_USER,
        model="claude-haiku-4-5",
        max_tokens=100,
        temperature=0.0,
        cache_control=cache_control,
        review_id=review_id or uuid4(),
        node_id="triage",
        prompt_template_version="v1",
        degraded_mode=False,
    )


async def test_probe_cache_model_first_call_writes_then_reads() -> None:
    """Above-floor stable system prompt: first call models a cache WRITE,
    repeats model READs; the accounting identity holds per call; the event
    mirrors (cached_tokens, cache_hit); the read-priced call costs less."""
    persist = _RecordingPersist()
    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    provider = _FixtureScriptedProvider({"triage": ["one", "two"]}, persister=persist, probe=probe)
    req = _cache_request(_ABOVE_FLOOR_SYSTEM)
    r1 = await provider.complete(req)
    r2 = await provider.complete(req)
    assert (r1.cache_write_tokens, r1.cache_read_tokens) == (5000, 0)
    assert (r2.cache_write_tokens, r2.cache_read_tokens) == (0, 5000)
    for r in (r1, r2):  # total_input = cache_read + cache_creation + input_tokens
        assert r.cache_read_tokens + r.cache_write_tokens + r.input_tokens == 5000 + len(_USER)
    assert persist.events[0].cache_hit is False
    assert persist.events[1].cache_hit is True
    assert persist.events[1].cached_tokens == 5000
    assert probe.calls[0]["cache_write_tokens"] == 5000
    assert probe.calls[1]["cache_read_tokens"] == 5000
    # 0.1x read rate < 1.25x write rate → warm call is cheaper.
    assert probe.calls[1]["cost_usd"] < probe.calls[0]["cost_usd"]


async def test_probe_cache_model_below_floor_is_silent_noop() -> None:
    """System prompt under the model's floor: processed without caching —
    both cache fields 0, input falls back to the combined estimate."""
    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    provider = _FixtureScriptedProvider(
        {"triage": ["one", "two"]}, persister=_RecordingPersist(), probe=probe
    )
    req = _cache_request("S" * 100)  # < 4096 haiku floor
    r1 = await provider.complete(req)
    r2 = await provider.complete(req)
    for r in (r1, r2):
        assert (r.cache_write_tokens, r.cache_read_tokens) == (0, 0)
        assert r.input_tokens == len(f"{'S' * 100}\n{_USER}")


async def test_probe_cache_model_unfloored_model_raises_named_error() -> None:
    """A model with no MIN_CACHEABLE_TOKENS floor fails the cache-model
    path with the module's named EvalDriverError (same loud-failure
    contract as the pricing KeyError), not a bare KeyError."""
    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    provider = _FixtureScriptedProvider(
        {"triage": ["one"]}, persister=_RecordingPersist(), probe=probe
    )
    req = LLMRequest(
        system_prompt=_ABOVE_FLOOR_SYSTEM,
        user_prompt=_USER,
        model="claude-opus-9-9",  # valid shape, not in MIN_CACHEABLE_TOKENS
        max_tokens=100,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",
        prompt_template_version="v1",
        degraded_mode=False,
    )
    with pytest.raises(EvalDriverError, match="MIN_CACHEABLE_TOKENS"):
        await provider.complete(req)


async def test_probe_cache_model_respects_cache_control_off() -> None:
    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    provider = _FixtureScriptedProvider(
        {"triage": ["one"]}, persister=_RecordingPersist(), probe=probe
    )
    r = await provider.complete(_cache_request(_ABOVE_FLOOR_SYSTEM, cache_control=False))
    assert (r.cache_write_tokens, r.cache_read_tokens) == (0, 0)


async def test_probe_default_model_cache_off_keeps_cache_zero() -> None:
    """model_cache defaults False — existing cost-measurement baselines keep
    their uncached semantics unless a test opts in."""
    probe = CostProbe(token_estimator=len, output_tokens=10)
    provider = _FixtureScriptedProvider(
        {"triage": ["one"]}, persister=_RecordingPersist(), probe=probe
    )
    r = await provider.complete(_cache_request(_ABOVE_FLOOR_SYSTEM))
    assert (r.cache_write_tokens, r.cache_read_tokens) == (0, 0)
    assert r.input_tokens == len(f"{_ABOVE_FLOOR_SYSTEM}\n{_USER}")


async def test_probe_cache_model_distinct_system_prompts_never_hit() -> None:
    """Per-file system prompts (today's packing) each write their own entry —
    no cross-prompt reads. This is the BEFORE shape the repartition fixes."""
    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    provider = _FixtureScriptedProvider(
        {"triage": ["one", "two"]}, persister=_RecordingPersist(), probe=probe
    )
    r1 = await provider.complete(_cache_request("A" * 5000))
    r2 = await provider.complete(_cache_request("B" * 5000))
    assert (r1.cache_write_tokens, r1.cache_read_tokens) == (5000, 0)
    assert (r2.cache_write_tokens, r2.cache_read_tokens) == (5000, 0)


class _BlockingPersist(_RecordingPersist):
    """Holds every persist() until released — forces two complete() calls to
    OVERLAP deterministically (the concurrent-worker shape)."""

    def __init__(self) -> None:
        super().__init__()
        import asyncio

        self.release = asyncio.Event()

    async def persist(self, event: Any, request: Any, response: Any) -> None:
        await self.release.wait()
        await super().persist(event, request, response)


async def test_probe_cache_model_concurrent_first_wave_all_write() -> None:
    """IN-FLIGHT-AWARE cache model: a written prefix is warm only when the
    writing call COMPLETES, so two calls that overlap (the parallel-analyze
    first wave) BOTH record cache writes — the real API's documented
    stampede. Marking warm at decision time would let the sibling record a
    read and underprice concurrent reviews (1.25× write vs 0.1× read). A
    call arriving AFTER completion still reads (the sequential contract,
    unchanged)."""
    import asyncio

    probe = CostProbe(token_estimator=len, output_tokens=10, model_cache=True)
    persist = _BlockingPersist()
    provider = _FixtureScriptedProvider(
        {"triage": ["one", "two", "three"]}, persister=persist, probe=probe
    )
    system = "A" * 5000
    task1 = asyncio.ensure_future(provider.complete(_cache_request(system)))
    task2 = asyncio.ensure_future(provider.complete(_cache_request(system)))
    await asyncio.sleep(0)  # both calls reach the blocked persist — overlapped
    persist.release.set()
    r1, r2 = await asyncio.gather(task1, task2)
    assert (r1.cache_write_tokens, r1.cache_read_tokens) == (5000, 0)
    assert (r2.cache_write_tokens, r2.cache_read_tokens) == (5000, 0)  # the stampede
    # Post-completion arrival: warm — the sequential read contract holds.
    r3 = await provider.complete(_cache_request(system))
    assert (r3.cache_write_tokens, r3.cache_read_tokens) == (0, 5000)


# ---------------------------------------------------------------------------
# _FixtureGitHubClient (via _github_factory_for) — wire-shape fidelity
# ---------------------------------------------------------------------------


async def test_github_factory_rejects_wrong_installation_id() -> None:
    factory = _github_factory_for(_fixture())
    with pytest.raises(EvalDriverError, match="unexpected installation_id"):
        await factory(_INSTALLATION_ID + 1)


async def test_github_double_serves_file_list() -> None:
    client = await _github_factory_for(_fixture())(_INSTALLATION_ID)
    resp = await client.rest.pulls.async_list_files("acme", "widget", 7)
    metas = resp.parsed_data
    assert len(metas) == 1
    assert metas[0].filename == "app/views.py"
    assert metas[0].status == "modified"
    assert metas[0].previous_filename is None


async def test_github_content_base64_wrapped_survives_intake_decode() -> None:
    repos = (await _github_factory_for(_fixture())(_INSTALLATION_ID)).rest.repos
    resp = await repos.async_get_content("acme", "widget", "app/views.py", ref=_HEAD_SHA)
    data = resp.parsed_data
    assert data.encoding == "base64"
    # Real GitHub wraps base64 with newlines; assert ours does too (content >60 chars).
    assert "\n" in data.content, "fixture base64 must be newline-wrapped to match GitHub wire shape"
    # Mirror intake's decode (github/fetch.py): strip \n/\r then b64decode(validate=True).
    normalized = data.content.replace("\n", "").replace("\r", "")
    decoded = base64.b64decode(normalized, validate=True).decode()
    assert decoded == _HEAD_CONTENT


async def test_github_double_serves_base_vs_head_by_ref() -> None:
    repos = (await _github_factory_for(_fixture())(_INSTALLATION_ID)).rest.repos
    base = await repos.async_get_content("acme", "widget", "app/views.py", ref=_BASE_SHA)
    head = await repos.async_get_content("acme", "widget", "app/views.py", ref=_HEAD_SHA)
    assert base64.b64decode(base.parsed_data.content.replace("\n", "")).decode() == _BASE_CONTENT
    assert base64.b64decode(head.parsed_data.content.replace("\n", "")).decode() == _HEAD_CONTENT


async def test_github_double_mimics_404_on_missing_content() -> None:
    # An absent path mimics GitHub's 404 (not a hard error) so trace's candidate
    # probe can read `.response.status_code == 404` to resolve which path exists.
    repos = (await _github_factory_for(_fixture())(_INSTALLATION_ID)).rest.repos
    with pytest.raises(_FixtureContentNotFoundError) as exc_info:
        await repos.async_get_content("acme", "widget", "app/views.py", ref="c" * 40)
    assert exc_info.value.response.status_code == 404


async def test_github_double_renamed_file_keys_base_content_at_previous_path() -> None:
    fixture = _fixture(
        files=[
            {
                "path": "app/new_name.py",
                "status": "renamed",
                "additions": 1,
                "deletions": 1,
                "previous_path": "app/old_name.py",
                "content_base": _BASE_CONTENT,
                "content_head": _HEAD_CONTENT,
            }
        ]
    )
    repos = (await _github_factory_for(fixture)(_INSTALLATION_ID)).rest.repos
    # base content is read at the OLD path; head at the NEW path (intake's rename shape).
    base = await repos.async_get_content("acme", "widget", "app/old_name.py", ref=_BASE_SHA)
    head = await repos.async_get_content("acme", "widget", "app/new_name.py", ref=_HEAD_SHA)
    assert base64.b64decode(base.parsed_data.content.replace("\n", "")).decode() == _BASE_CONTENT
    assert base64.b64decode(head.parsed_data.content.replace("\n", "")).decode() == _HEAD_CONTENT


# ---------------------------------------------------------------------------
# _CapturingPublisher
# ---------------------------------------------------------------------------


async def test_capturing_publisher_records_create_review_and_reports_no_prior() -> None:
    publisher = _CapturingPublisher()
    assert (
        await publisher.find_existing_review_on_head_sha(
            gh=None,  # type: ignore[arg-type]
            owner="acme",
            repo="widget",
            pull_number=7,
            head_sha=_HEAD_SHA,
            body_marker="m",
        )
        is None
    )
    created = await publisher.create_review(
        gh=None,  # type: ignore[arg-type]
        owner="acme",
        repo="widget",
        pull_number=7,
        head_sha=_HEAD_SHA,
        review_status="COMMENT",
        body_marker="m",
        comments=(),
    )
    assert created.comments_posted == 0
    assert len(publisher.create_review_calls) == 1
    assert publisher.create_review_calls[0]["review_status"] == "COMMENT"


# ---------------------------------------------------------------------------
# EvalFixture schema + EvalRunResult contract
# ---------------------------------------------------------------------------


def test_eval_fixture_forbids_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        _fixture(language="python")  # not a fixture field — intake derives it


def test_eval_fixture_rejects_repo_content_overlapping_a_changed_file() -> None:
    # repository_contents_head must be BEYOND the diff; overlapping a changed file
    # path would silently override intake's content map. "app/views.py" is the
    # changed file in `_fixture`.
    with pytest.raises(ValidationError, match="beyond the diff"):
        _fixture(repository_contents_head={"app/views.py": "class X: ..."})


def test_eval_fixture_rejects_repo_content_overlapping_a_rename_source() -> None:
    with pytest.raises(ValidationError, match="beyond the diff"):
        _fixture(
            files=[
                {
                    "path": "app/new_name.py",
                    "status": "renamed",
                    "additions": 1,
                    "deletions": 1,
                    "previous_path": "app/old_name.py",
                    "content_base": _BASE_CONTENT,
                    "content_head": _HEAD_CONTENT,
                }
            ],
            repository_contents_head={"app/old_name.py": "class X: ..."},  # rename source
        )


def test_eval_fixture_overlap_guard_normalizes_non_canonical_paths() -> None:
    # A non-canonical spelling must NOT bypass the overlap guard: "./app/views.py"
    # and "app/views.py" both canonicalize (validate_diff_path) to "app/views.py",
    # which is what intake/trace actually fetch.
    with pytest.raises(ValidationError, match="beyond the diff"):
        _fixture(
            files=[
                {
                    "path": "./app/views.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 0,
                    "patch": "@@ -1 +1 @@\n-a\n+b\n",
                    "content_base": _BASE_CONTENT,
                    "content_head": _HEAD_CONTENT,
                }
            ],
            repository_contents_head={"app/views.py": "class X: ..."},
        )


def test_eval_fixture_rejects_invalid_repo_content_key() -> None:
    with pytest.raises(ValidationError, match="invalid fixture path"):
        _fixture(repository_contents_head={"../../etc/passwd": "x"})


def test_eval_fixture_rejects_canonically_duplicate_repo_content_keys() -> None:
    # Two raw keys canonicalizing to the same path would collapse to one
    # (path, head_sha) map entry -> silent last-wins; reject the ambiguity.
    with pytest.raises(ValidationError, match="canonicalize to the same path"):
        _fixture(repository_contents_head={"./app/models.py": "a", "app/models.py": "b"})


def test_eval_run_result_iterates_and_lens_over_findings_and_exposes_state() -> None:
    r = EvalRunResult(
        review_id=uuid4(),
        findings=(),
        trace_decisions=(),
        published_comments=(),
        hitl_gated=True,
    )
    assert list(r) == []
    assert len(r) == 0
    assert r.hitl_gated is True
    assert r.hitl_request is None


# ---------------------------------------------------------------------------
# _build_approve_all_decision (resume driver's scripted reviewer input)
# ---------------------------------------------------------------------------


def test_build_approve_all_decision_covers_exactly_the_gated_set() -> None:
    # The resume driver's stand-in for human input: one APPROVE per gated finding,
    # covering exactly the request's `findings_requiring_approval` (the HITL node
    # re-validates the decision set against the request, so a miss would raise).
    fid1, fid2 = uuid4(), uuid4()
    request = HITLRequest(
        findings_requiring_approval=(fid1, fid2),
        auto_post_findings=(),
        created_at=datetime(2026, 6, 2, tzinfo=UTC),
        expires_at=datetime(2026, 6, 3, tzinfo=UTC),
    )
    decision = _build_approve_all_decision(request)

    assert decision.reviewer_id == "eval"
    # Exactly the gated set, one decision each (compare as sets — the request
    # canonicalizes/sorts the finding tuple).
    assert {d.finding_id for d in decision.decisions} == {fid1, fid2}
    assert all(d.outcome == PerFindingOutcome.APPROVE for d in decision.decisions)
    # APPROVE carries no override fields and an empty reason is valid for it.
    assert all(d.reason == "" for d in decision.decisions)
    assert all(
        d.override_severity is None and d.original_severity is None for d in decision.decisions
    )


def test_build_approve_all_decision_empty_gate_yields_no_decisions() -> None:
    # Defensive: an empty gated set yields an empty decisions tuple (the driver
    # only calls this when an interrupt fired, but the builder must not invent
    # decisions for a finding that wasn't gated).
    request = HITLRequest(
        findings_requiring_approval=(),
        auto_post_findings=(uuid4(),),
        created_at=datetime(2026, 6, 2, tzinfo=UTC),
        expires_at=datetime(2026, 6, 3, tzinfo=UTC),
    )
    decision = _build_approve_all_decision(request)
    assert decision.decisions == ()


# ---------------------------------------------------------------------------
# _validate_eval_db_url (caller-supplied-db_url driver's fail-closed DB-isolation guard)
# ---------------------------------------------------------------------------


def test_validate_eval_db_url_accepts_per_test_eval_db() -> None:
    # A per-test ephemeral eval DB (port 5433, outrider_eval_* name) passes.
    _validate_eval_db_url("postgresql+psycopg://u:p@localhost:5433/outrider_eval_abc12345")


def test_validate_eval_db_url_rejects_shared_base_db() -> None:
    # The base `outrider_test` is ALSO on 5433 but is NOT a per-test DB — a
    # caller-supplied-db_url driver would create tables + seed rows into the shared
    # base. The prefix check (not port alone) is what catches this.
    with pytest.raises(EvalDriverError, match="per-test eval database"):
        _validate_eval_db_url("postgresql+psycopg://u:p@localhost:5433/outrider_test")


def test_validate_eval_db_url_rejects_non_test_port() -> None:
    # Port 5432 is the dev/prod container — refuse it even with the eval name.
    with pytest.raises(EvalDriverError, match="per-test eval database"):
        _validate_eval_db_url("postgresql+psycopg://u:p@localhost:5432/outrider_eval_abc12345")


def test_validate_eval_db_url_rejects_unparseable_url() -> None:
    # A non-numeric port makes make_url raise — surfaced as a typed EvalDriverError,
    # not a raw SQLAlchemy error (the file's fail-loud-with-clear-cause discipline).
    with pytest.raises(EvalDriverError, match="not a parseable database URL"):
        _validate_eval_db_url("postgresql+psycopg://u:p@localhost:NOTAPORT/outrider_eval_x")


def test_concurrency_scripting_guard_refuses_any_indexed_analyze_response() -> None:
    """>= 1, not > 1: how many files reach the LLM is runtime behavior, so
    even a single index-keyed analyze response can misattribute under
    concurrency — and the first worker PERSISTS the misattributed exchange
    before the second aborts. By-path (or no analyze scripting at all)
    passes; any indexed response refuses."""
    from outrider.agent.eval_driver import _require_concurrency_safe_scripting

    base = _fixture().model_dump()
    base["llm_responses"] = {"triage": ["t"], "analyze": ["one"], "synthesize": ["s"]}
    indexed = EvalFixture.model_validate(base)
    with pytest.raises(EvalDriverError, match="analyze_responses_by_path"):
        _require_concurrency_safe_scripting(indexed, 2)
    _require_concurrency_safe_scripting(indexed, 1)  # sequential: fine

    base["analyze_responses_by_path"] = {"app/views.py": "one"}
    by_path = EvalFixture.model_validate(base)
    _require_concurrency_safe_scripting(by_path, 4)  # keyed: fine

    base["analyze_responses_by_path"] = None
    base["llm_responses"] = {"triage": ["t"], "synthesize": ["s"]}
    no_analyze = EvalFixture.model_validate(base)
    _require_concurrency_safe_scripting(no_analyze, 4)  # nothing to misattribute
