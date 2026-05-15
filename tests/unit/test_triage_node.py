"""Triage node body unit tests per the triage-node spec.

Tests cover:
- Happy path with phase-event bracket
- Schema-violation propagation (malformed JSON, wrong enum casing, >500 char reasoning)
- Provider-failure propagation (LLMProviderError subclasses)
- Policy-violation: SKIP, unknown path, missing path — each pinned with
  dangling-start phase-event pattern
- Helper isolation tests (_enforce_triage_policy)
- Input-boundary regression (PR title with format metacharacters)
- Model-config wiring (request.model is the closure-captured value)

Tests use a `MockLLMProvider` constructed inline (not a fixture) for
clarity — each test sets up the canned response it wants.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.triage import (
    TriagePolicyViolationError,
    _enforce_triage_policy,
    triage,
)
from outrider.llm.base import LLMRequest, LLMResponse, LLMTimeoutError

if TYPE_CHECKING:
    from outrider.audit.events import ReviewPhaseEvent
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_finding import ReviewDimension
from outrider.schemas.review_state import ReviewState
from outrider.schemas.triage_result import ReviewTier, RiskLevel, TriageResult


class _RecordingPhaseEventSinkLike(Protocol):
    """Type stub for the `recording_phase_event_sink` fixture defined in
    tests/conftest.py. We can't `from tests.conftest import ...` because
    pyproject.toml's `pythonpath = ["src"]` doesn't include `tests/`.
    Pytest auto-injects the fixture by parameter name; this Protocol gives
    mypy the .events attribute and the async emit_phase method shape."""

    events: list[ReviewPhaseEvent]

    async def emit_phase(self, event: ReviewPhaseEvent) -> None: ...


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_changed_file(
    *,
    path: str = "src/example.py",
    status: Literal["added", "modified", "removed", "renamed"] = "modified",
) -> ChangedFile:
    return ChangedFile(
        path=path,
        status=status,
        additions=5,
        deletions=2,
        patch="@@ -1 +1 @@\n-old\n+new\n",
        content_base="old\n",
        content_head="new\n",
        previous_path=None,
        language=None,
    )


def _build_state(
    *,
    files: tuple[ChangedFile, ...] | None = None,
    is_eval: bool = False,
) -> ReviewState:
    if files is None:
        files = (_build_changed_file(),)
    return ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        is_eval=is_eval,
        pr_context=PRContext(
            installation_id=12345,
            owner="acme",
            repo="widget",
            pr_number=42,
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="Test PR",
            pr_body="Body",
            author="someone",
            total_additions=sum(cf.additions for cf in files),
            total_deletions=sum(cf.deletions for cf in files),
            changed_files=files,
        ),
    )


def _build_triage_json(
    *,
    file_tiers: dict[str, str] | None = None,
    overall_risk: str = "low",
    relevant_dimensions: list[str] | None = None,
    reasoning: str = "Standard refactor.",
) -> str:
    """Build a JSON-string TriageResult that the mock provider returns."""
    if file_tiers is None:
        file_tiers = {"src/example.py": "standard"}
    if relevant_dimensions is None:
        relevant_dimensions = ["code_quality"]
    return json.dumps(
        {
            "file_tiers": file_tiers,
            "overall_risk": overall_risk,
            "relevant_dimensions": relevant_dimensions,
            "reasoning": reasoning,
        }
    )


@dataclasses.dataclass
class _Plan:
    """What the MockLLMProvider does on .complete(request)."""

    response_text: str | None = None
    raise_exc: BaseException | None = None


class MockLLMProvider:
    """Test-only LLMProvider satisfying the runtime-checkable Protocol.

    Records every received request; returns the canned response or raises
    the canned exception.
    """

    def __init__(self, plan: _Plan) -> None:
        self.plan = plan
        self.received_requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.received_requests.append(request)
        if self.plan.raise_exc is not None:
            raise self.plan.raise_exc
        assert self.plan.response_text is not None, (
            "MockLLMProvider configured with neither response nor exception"
        )
        return LLMResponse(
            text=self.plan.response_text,
            model=request.model,
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=0,
            cache_write_tokens=0,
            finish_reason="end_turn",
            latency_ms=42,
        )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_happy_path_returns_validated_triage_result(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Provider returns valid JSON → node returns {'triage_result': <validated>}."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    result = await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    assert "triage_result" in result
    triage_result = result["triage_result"]
    assert isinstance(triage_result, TriageResult)
    assert triage_result.overall_risk == RiskLevel.LOW
    assert triage_result.file_tiers["src/example.py"] == ReviewTier.STANDARD
    assert triage_result.relevant_dimensions == (ReviewDimension.CODE_QUALITY,)


@pytest.mark.asyncio
async def test_triage_emits_start_and_end_phase_events_bracketing(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Happy path: exactly TWO ReviewPhaseEvent entries — start then end —
    bracketing the provider call. Both carry the same review_id and same
    phase_id. Pins phase-events-bound-work."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    events = recording_phase_event_sink.events
    assert len(events) == 2, f"expected 2 phase events, got {len(events)}"

    start, end = events
    assert start.marker == "start"
    assert end.marker == "end"
    assert start.node_id == "triage"
    assert end.node_id == "triage"
    assert start.review_id == state.review_id
    assert end.review_id == state.review_id
    # The same phase_id pairs them for replay
    assert start.phase_id == end.phase_id
    # phase_key is None for V1 (single-instance triage per review)
    assert start.phase_key is None
    assert end.phase_key is None


# ---------------------------------------------------------------------------
# Provider-failure propagation (dangling start)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_propagates_llm_provider_error_with_dangling_start(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Provider raises LLMProviderError subclass → triage propagates it.
    The start event was already emitted; the end event was NOT.
    Pins phase-events-bound-work's failure-path dangling-start pattern."""
    state = _build_state()
    # Simulate a transport failure via a concrete subclass (LLMProviderError
    # is abstract-by-construction; LLMTimeoutError is one of the concrete
    # subclasses with retry_at_layer set).
    plan = _Plan(raise_exc=LLMTimeoutError("simulated timeout"))
    provider = MockLLMProvider(plan)

    with pytest.raises(LLMTimeoutError, match="simulated timeout"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=recording_phase_event_sink,
        )

    # Start was emitted; end was NOT
    events = recording_phase_event_sink.events
    assert len(events) == 1
    assert events[0].marker == "start"


# ---------------------------------------------------------------------------
# Schema-violation propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_json_response,desc",
    [
        ("not-json-at-all", "non-JSON"),
        (json.dumps({"file_tiers": {}, "overall_risk": "LOW"}), "uppercase risk"),
        (
            json.dumps(
                {
                    "file_tiers": {"src/example.py": "DEEP"},  # uppercase
                    "overall_risk": "low",
                    "relevant_dimensions": [],
                    "reasoning": "x",
                }
            ),
            "uppercase tier",
        ),
        (json.dumps({"file_tiers": {}, "extra_unknown_field": "x"}), "extra field"),
        # 600-char reasoning exceeds Field(max_length=500)
        (
            json.dumps(
                {
                    "file_tiers": {"src/example.py": "standard"},
                    "overall_risk": "low",
                    "relevant_dimensions": [],
                    "reasoning": "x" * 600,
                }
            ),
            "reasoning too long",
        ),
        # unknown dimension value (not in ReviewDimension enum)
        (
            json.dumps(
                {
                    "file_tiers": {"src/example.py": "standard"},
                    "overall_risk": "low",
                    "relevant_dimensions": ["unknown_dimension"],
                    "reasoning": "x",
                }
            ),
            "unknown dimension",
        ),
    ],
)
async def test_triage_propagates_schema_violation_with_dangling_start(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
    bad_json_response: str,
    desc: str,
) -> None:
    """Provider returns schema-invalid JSON → triage propagates ValidationError.
    No partial state; start emitted, end NOT emitted."""
    state = _build_state()
    plan = _Plan(response_text=bad_json_response)
    provider = MockLLMProvider(plan)

    with pytest.raises(ValidationError):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=recording_phase_event_sink,
        )

    events = recording_phase_event_sink.events
    assert len(events) == 1, f"({desc}) expected only start event"
    assert events[0].marker == "start"


# ---------------------------------------------------------------------------
# Policy-violation: SKIP (with dangling-start assertion per Round 6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_rejects_skip_tier_with_dangling_start(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LLM produces SKIP → TriagePolicyViolationError. Pins non-goal #1
    AND the failure-path dangling-start pattern for the policy-validation
    failure mode (per Round 6 cleanup)."""
    state = _build_state()
    plan = _Plan(
        response_text=_build_triage_json(file_tiers={"src/example.py": "skip"}),
    )
    provider = MockLLMProvider(plan)

    with pytest.raises(TriagePolicyViolationError, match="policy-gate scope"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=recording_phase_event_sink,
        )

    events = recording_phase_event_sink.events
    assert len(events) == 1
    assert events[0].marker == "start"


# ---------------------------------------------------------------------------
# Policy-violation: unknown path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_rejects_unknown_path(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LLM returns a path not in changed_files → TriagePolicyViolationError
    listing the offending path. Also pins the dangling-start phase-event
    pattern: start emitted, end NOT emitted on policy failure."""
    state = _build_state(files=(_build_changed_file(path="src/foo.py"),))
    plan = _Plan(
        response_text=_build_triage_json(
            file_tiers={"unrelated/bar.py": "standard", "src/foo.py": "deep"}
        ),
    )
    provider = MockLLMProvider(plan)

    with pytest.raises(TriagePolicyViolationError, match="unrelated/bar.py"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=recording_phase_event_sink,
        )

    events = recording_phase_event_sink.events
    assert len(events) == 1
    assert events[0].marker == "start"


# ---------------------------------------------------------------------------
# Policy-violation: missing path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_rejects_missing_path(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LLM omits a changed-file path from file_tiers → TriagePolicyViolationError
    listing the missing path. Also pins the dangling-start phase-event
    pattern: start emitted, end NOT emitted on policy failure."""
    state = _build_state(
        files=(
            _build_changed_file(path="src/a.py"),
            _build_changed_file(path="src/b.py"),
        ),
    )
    # Only classify a.py; b.py missing
    plan = _Plan(response_text=_build_triage_json(file_tiers={"src/a.py": "deep"}))
    provider = MockLLMProvider(plan)

    with pytest.raises(TriagePolicyViolationError, match="src/b.py"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=recording_phase_event_sink,
        )

    events = recording_phase_event_sink.events
    assert len(events) == 1
    assert events[0].marker == "start"


# ---------------------------------------------------------------------------
# _enforce_triage_policy isolation tests (pure function)
# ---------------------------------------------------------------------------


def _build_triage_result(
    *,
    file_tiers: dict[str, ReviewTier] | None = None,
    overall_risk: RiskLevel = RiskLevel.LOW,
    relevant_dimensions: tuple[ReviewDimension, ...] = (ReviewDimension.CODE_QUALITY,),
    reasoning: str = "ok",
) -> TriageResult:
    if file_tiers is None:
        file_tiers = {"src/a.py": ReviewTier.STANDARD}
    return TriageResult(
        file_tiers=file_tiers,
        overall_risk=overall_risk,
        relevant_dimensions=relevant_dimensions,
        reasoning=reasoning,
    )


def test_enforce_policy_happy_path_does_not_raise() -> None:
    """Pure-function isolation test: matching paths, no SKIP → does not raise."""
    result = _build_triage_result(
        file_tiers={"src/a.py": ReviewTier.DEEP, "src/b.py": ReviewTier.STANDARD}
    )
    # Function returns None implicitly; the absence of a raise IS the contract.
    _enforce_triage_policy(result, expected_paths={"src/a.py", "src/b.py"})


def test_enforce_policy_accepts_frozenset() -> None:
    """expected_paths typed as Set (abstract) — accepts frozenset too."""
    result = _build_triage_result()
    # No raise
    _enforce_triage_policy(result, expected_paths=frozenset({"src/a.py"}))


def test_enforce_policy_accepts_dict_keys_view() -> None:
    """Set-like dict_keys is the third common variant. Pins the abstract-Set
    contract is real, not just frozenset-flavored."""
    result = _build_triage_result()
    expected = {"src/a.py": object()}.keys()
    _enforce_triage_policy(result, expected_paths=expected)


def test_enforce_policy_rejects_skip_value() -> None:
    """Rule (a) — any SKIP value → TriagePolicyViolationError."""
    result = _build_triage_result(file_tiers={"src/a.py": ReviewTier.SKIP})
    with pytest.raises(TriagePolicyViolationError, match="SKIP"):
        _enforce_triage_policy(result, expected_paths={"src/a.py"})


def test_enforce_policy_rejects_unknown_path() -> None:
    """Rule (b) — file_tiers key not in expected_paths."""
    result = _build_triage_result(file_tiers={"src/extra.py": ReviewTier.DEEP})
    with pytest.raises(TriagePolicyViolationError, match="unknown paths"):
        _enforce_triage_policy(result, expected_paths={"src/a.py"})


def test_enforce_policy_rejects_missing_path() -> None:
    """Rule (c) — expected_paths key missing from file_tiers."""
    result = _build_triage_result(file_tiers={"src/a.py": ReviewTier.DEEP})
    with pytest.raises(TriagePolicyViolationError, match="missing"):
        _enforce_triage_policy(result, expected_paths={"src/a.py", "src/b.py"})


def test_enforce_policy_empty_expected_with_empty_actual() -> None:
    """Degenerate: 0 expected files, 0 file_tiers → no violation. Documents
    what 'no changed files' means semantically (vacuously satisfies all
    three rules)."""
    result = _build_triage_result(file_tiers={})
    _enforce_triage_policy(result, expected_paths=frozenset())


# ---------------------------------------------------------------------------
# Input-boundary regression (webhook-strings-are-data-not-format-strings)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pr_title_with_format_metacharacters_does_not_escape(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """PR title containing literal `{system_prompt}` characters must
    survive into user_prompt as DATA (not template-substituted). Pins the
    `webhook-strings-are-data-not-format-strings` invariant at the node
    level — the prompts/triage.py render() test pins it at the template
    level; this one pins that the node's call-site doesn't inadvertently
    apply str.format somewhere it shouldn't."""
    # Build a state with a hostile title; the node calls prompts.triage.render()
    # which str.format()s the template — values pass through opaquely
    hostile_state = ReviewState(
        review_id=uuid4(),
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=12345,
            owner="acme",
            repo="widget",
            pr_number=42,
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="Refactor {system_prompt} via {user_prompt}",
            pr_body=None,
            author="someone",
            total_additions=5,
            total_deletions=2,
            changed_files=(_build_changed_file(path="src/example.py"),),
        ),
    )
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        hostile_state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    # The hostile substring appears in user_prompt as DATA
    assert len(provider.received_requests) == 1
    request = provider.received_requests[0]
    assert "{system_prompt}" in request.user_prompt
    assert "{user_prompt}" in request.user_prompt


# ---------------------------------------------------------------------------
# Model-config wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_uses_injected_triage_model_not_hardcoded(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """LLMRequest.model must equal the closure-captured triage_model
    argument — never a hardcoded string in the node body. Pins
    `model-strings-from-config-not-hardcoded`."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    custom_model_id = "claude-test-model-id-not-real"
    await triage(
        state,
        provider=provider,
        triage_model=custom_model_id,
        phase_event_sink=recording_phase_event_sink,
    )

    assert provider.received_requests[0].model == custom_model_id


@pytest.mark.asyncio
async def test_triage_threads_is_eval_false_from_state_to_request_and_phase_events(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Production state (is_eval=False) → LLMRequest.is_eval=False AND
    both ReviewPhaseEvent emissions carry is_eval=False. The downstream
    audit rows (LLMCallEvent + 2x ReviewPhaseEvent) will all be tagged
    production. Pins the canonical default path."""
    state = _build_state(is_eval=False)
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    assert provider.received_requests[0].is_eval is False
    for event in recording_phase_event_sink.events:
        assert event.is_eval is False, (
            f"phase event {event.marker} should carry is_eval=False; got {event.is_eval}"
        )


@pytest.mark.asyncio
async def test_triage_threads_is_eval_true_from_state_to_request_and_phase_events(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Eval state (is_eval=True from eval-harness factory) → LLMRequest.
    is_eval=True AND BOTH phase events carry is_eval=True. Without this,
    eval runs of triage would pollute the production audit stream
    through either the LLMCallEvent OR the ReviewPhaseEvent rows —
    exactly the bug docs/testing.md "Eval isolation end-to-end" is
    designed to prevent. Pin both threadings."""
    state = _build_state(is_eval=True)
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    assert provider.received_requests[0].is_eval is True
    for event in recording_phase_event_sink.events:
        assert event.is_eval is True, (
            f"phase event {event.marker} should carry is_eval=True; got {event.is_eval}"
        )


@pytest.mark.asyncio
async def test_triage_request_carries_canonical_constants(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """The request sent to the provider must use the canonical constants
    from prompts.triage (VERSION, MAX_TOKENS, TEMPERATURE, node_id).
    Pins the wiring from prompts/triage.py through the node."""
    from outrider.prompts.triage import MAX_TOKENS, TEMPERATURE, VERSION

    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    request = provider.received_requests[0]
    assert request.prompt_template_version == VERSION
    assert request.max_tokens == MAX_TOKENS
    assert request.temperature == TEMPERATURE
    assert request.node_id == "triage"
    assert request.degraded_mode is False
    # review_id flows from state
    assert request.review_id == state.review_id


# ---------------------------------------------------------------------------
# State mutability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_triage_does_not_mutate_input_state(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Defensive: the node should not mutate the input ReviewState. Future
    LangGraph upgrades that change state-passing semantics could expose
    mutation as a bug; pin pre-image now."""
    state = _build_state()
    snapshot = state.model_dump(mode="json")
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    assert state.model_dump(mode="json") == snapshot


# ---------------------------------------------------------------------------
# review_id type pin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase_event_review_id_matches_state_review_id_type(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """ReviewState.review_id is UUID; ReviewPhaseEvent.review_id is UUID
    too. The closure must pass them through unchanged."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    for event in recording_phase_event_sink.events:
        assert isinstance(event.review_id, UUID)
        assert event.review_id == state.review_id


class _RaiseOnEndSink:
    """Test sink that records start but raises on end. Pins failure-source #5
    in the documented post-start failure-mode matrix: the end-phase emission
    itself raising (per PhaseEventSink Protocol's "Implementations MUST
    persist before returning, OR raise" rule)."""

    def __init__(self) -> None:
        self.events: list[ReviewPhaseEvent] = []

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        self.events.append(event)
        if event.marker == "end":
            raise RuntimeError("simulated durable-sink failure on end emission")


@pytest.mark.asyncio
async def test_triage_propagates_end_phase_sink_failure_after_successful_work() -> None:
    """Failure source #5 in the post-start failure matrix: the end-phase
    emission itself raises. This is the case where everything went right
    EXCEPT the durable audit write — LLM call succeeded, schema validated,
    policy passed, but the sink couldn't persist the end-event row.

    Expected: the sink's exception propagates from triage; the start event
    landed; the end event was attempted (and may have partially landed
    depending on sink transactionality); `{"triage_result": ...}` is never
    returned, so the partial state cannot reach downstream nodes.

    Pins that triage doesn't silently swallow audit-infra failures."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)
    raise_on_end = _RaiseOnEndSink()

    with pytest.raises(RuntimeError, match="simulated durable-sink failure"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=raise_on_end,
        )

    # Both start AND end events appended to the sink's list — the start
    # was emitted normally; the end was emitted and recorded BEFORE the
    # sink raised. (A production sink that raises mid-transaction wouldn't
    # have a recorded row, but the test sink records-then-raises so we can
    # assert the order without DB-machinery.) The propagated exception is
    # the load-bearing pin: triage must not silently swallow audit failures.
    assert len(raise_on_end.events) == 2
    assert raise_on_end.events[0].marker == "start"
    assert raise_on_end.events[1].marker == "end"


@pytest.mark.asyncio
async def test_triage_propagates_start_phase_sink_failure_before_any_work() -> None:
    """Symmetric failure-mode pin: if start-emit raises (audit infra
    outage at node entry), the node fails BEFORE any LLM work. No
    provider call, no partial state, no dangling-start in the sink.
    The exception propagates as-is."""

    class _RaiseOnStartSink:
        def __init__(self) -> None:
            self.events: list[ReviewPhaseEvent] = []

        async def emit_phase(self, event: ReviewPhaseEvent) -> None:
            if event.marker == "start":
                raise RuntimeError("simulated audit infra outage at start")
            self.events.append(event)

    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)
    raise_on_start = _RaiseOnStartSink()

    with pytest.raises(RuntimeError, match="simulated audit infra outage"):
        await triage(
            state,
            provider=provider,
            triage_model="claude-haiku-4-5",
            phase_event_sink=raise_on_start,
        )

    # Provider call never happened — start raised first
    assert len(provider.received_requests) == 0
    # No events recorded; start raised before append
    assert raise_on_start.events == []


@pytest.mark.asyncio
async def test_triage_handles_empty_changed_files_end_to_end(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """Edge case: a zero-file PR (constructible if intake's GitHub fetch
    returns an empty list — rare but documented behavior). The node must
    not raise on the policy gate with `expected_paths=frozenset()` and
    `file_tiers={}`; both happy and end-state must complete.

    The helper isolation test `test_enforce_policy_empty_expected_with_empty_actual`
    covers the pure function; this test covers the full node flow."""
    state = _build_state(files=())
    plan = _Plan(response_text=_build_triage_json(file_tiers={}))
    provider = MockLLMProvider(plan)

    result = await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    triage_result = result["triage_result"]
    assert isinstance(triage_result, TriageResult)
    assert dict(triage_result.file_tiers) == {}
    # Both phase events still bracketed even on zero-file degenerate case
    assert len(recording_phase_event_sink.events) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,desc",
    [
        ("src/with spaces.py", "ascii spaces"),
        ("src/日本語.py", "unicode (CJK)"),
        ("src/sub/deep/nested/path.py", "deep nesting"),
        ("src/__init__.py", "dunder"),
    ],
)
async def test_triage_handles_special_path_characters(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
    path: str,
    desc: str,
) -> None:
    """Policy gate uses set arithmetic on path strings. Spaces, unicode,
    and deep paths must survive the round-trip through render → JSON →
    `_enforce_triage_policy` set arithmetic without normalization
    artifacts. Sibling to the input-boundary regression test, but at the
    path-content layer rather than the format-string layer."""
    state = _build_state(files=(_build_changed_file(path=path),))
    plan = _Plan(
        response_text=_build_triage_json(file_tiers={path: "standard"}),
    )
    provider = MockLLMProvider(plan)

    result = await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    triage_result = result["triage_result"]
    assert path in dict(triage_result.file_tiers), f"({desc}) path missing"


@pytest.mark.asyncio
async def test_phase_id_is_a_valid_uuid_string(
    recording_phase_event_sink: _RecordingPhaseEventSinkLike,
) -> None:
    """phase_id is str per ReviewPhaseEvent schema; the node uses
    str(uuid4()) which is a valid UUID hex string."""
    state = _build_state()
    plan = _Plan(response_text=_build_triage_json())
    provider = MockLLMProvider(plan)

    await triage(
        state,
        provider=provider,
        triage_model="claude-haiku-4-5",
        phase_event_sink=recording_phase_event_sink,
    )

    events = recording_phase_event_sink.events
    assert len(events) >= 1
    # str(uuid4()) is valid input to UUID(...) — round-trip test
    UUID(events[0].phase_id)
