"""build_graph guard tests per the triage-node + intake-and-webhook specs.

Narrowly scoped to dependency / None guards on `agent/graph.py::build_graph`.
Functional behavior (the compiled graph actually runs) is covered by the
integration tests in `tests/integration/test_review_state_langgraph_merge.py`.

Rejection contracts pinned here (six gates after intake-and-webhook landed):
  1. provider=None → BuildGraphError
  2. model_config=None → BuildGraphError
  3. phase_event_sink=None → BuildGraphError
  4. file_examination_sink=None → BuildGraphError
  5. db_factory=None → BuildGraphError
  6. github_factory=None → BuildGraphError
  7. provider lacking `complete` member → BuildGraphError (isinstance gate)
  8. phase_event_sink lacking `emit_phase` member → BuildGraphError (isinstance gate)
  9. file_examination_sink lacking `emit_file_examination` member → BuildGraphError

PEP 544 caveat: the isinstance gates check MEMBER PRESENCE only — wrong
signature or async-shape falls through to fail at the first call. Tests
deliberately use plain `object()` rather than wrong-signature classes to
match the gate's actual semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from outrider.agent.graph import BuildGraphError, build_graph
from outrider.llm.config import ModelConfig

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from outrider.audit.events import FileExaminationEvent, ReviewPhaseEvent
    from outrider.github import InstallationGitHubClient
    from outrider.llm.base import LLMRequest, LLMResponse


# ---------------------------------------------------------------------------
# Minimal protocol-satisfying stubs
# ---------------------------------------------------------------------------


class _StubProvider:
    """Satisfies LLMProvider Protocol structurally (has `complete`)."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("test stub")


class _StubPhaseSink:
    """Satisfies PhaseEventSink Protocol structurally (has `emit_phase`)."""

    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        return None


class _StubFileExaminationSink:
    """Satisfies FileExaminationSink Protocol structurally."""

    async def emit_file_examination(self, event: FileExaminationEvent) -> None:
        return None


def _stub_db_factory() -> async_sessionmaker[AsyncSession]:
    """A bare object passes the None-check + duck-typed runtime use; tests
    that exercise actual DB calls live elsewhere (integration tests)."""
    return object()  # type: ignore[return-value]


def _stub_github_factory(installation_id: int) -> InstallationGitHubClient:
    """A callable that satisfies the type at the call site; never invoked
    in these unit tests."""
    raise NotImplementedError("test stub")


def _valid_args() -> dict[str, Any]:
    """Build a complete, valid set of kwargs. Tests perturb one at a time."""
    return {
        "provider": _StubProvider(),
        "model_config": ModelConfig(),
        "phase_event_sink": _StubPhaseSink(),
        "file_examination_sink": _StubFileExaminationSink(),
        "db_factory": _stub_db_factory(),
        "github_factory": _stub_github_factory,
    }


# ---------------------------------------------------------------------------
# Happy path: all three deps valid → returns compiled graph
# ---------------------------------------------------------------------------


def test_build_graph_happy_path_returns_compiled_graph() -> None:
    """Sanity test: with all three valid Protocol-satisfying deps, the
    factory returns a compiled graph object. Validates the gate doesn't
    fire on correct inputs."""
    graph = build_graph(**_valid_args())
    # CompiledStateGraph has an `ainvoke` method per LangGraph 1.1.6
    assert callable(graph.ainvoke)


# ---------------------------------------------------------------------------
# None rejections (3 sibling tests)
# ---------------------------------------------------------------------------


def test_build_graph_rejects_provider_none() -> None:
    args = _valid_args()
    args["provider"] = None
    with pytest.raises(BuildGraphError, match="provider must not be None"):
        build_graph(**args)


def test_build_graph_rejects_model_config_none() -> None:
    args = _valid_args()
    args["model_config"] = None
    with pytest.raises(BuildGraphError, match="model_config must not be None"):
        build_graph(**args)


def test_build_graph_rejects_phase_event_sink_none() -> None:
    args = _valid_args()
    args["phase_event_sink"] = None
    with pytest.raises(BuildGraphError, match="phase_event_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_file_examination_sink_none() -> None:
    args = _valid_args()
    args["file_examination_sink"] = None
    with pytest.raises(BuildGraphError, match="file_examination_sink must not be None"):
        build_graph(**args)


def test_build_graph_rejects_db_factory_none() -> None:
    args = _valid_args()
    args["db_factory"] = None
    with pytest.raises(BuildGraphError, match="db_factory must not be None"):
        build_graph(**args)


def test_build_graph_rejects_github_factory_none() -> None:
    args = _valid_args()
    args["github_factory"] = None
    with pytest.raises(BuildGraphError, match="github_factory must not be None"):
        build_graph(**args)


# ---------------------------------------------------------------------------
# Structural Protocol rejection
# ---------------------------------------------------------------------------


def test_build_graph_rejects_provider_missing_complete_member() -> None:
    """`isinstance(provider, LLMProvider)` fails on objects without
    `complete`. Pin the gate; PEP 544 member-presence-only semantics —
    wrong-signature `complete` would NOT be caught here."""
    args = _valid_args()
    args["provider"] = object()  # no `complete` attribute
    with pytest.raises(BuildGraphError, match="provider does not satisfy LLMProvider"):
        build_graph(**args)


def test_build_graph_rejects_phase_event_sink_missing_emit_phase_member() -> None:
    """`isinstance(sink, PhaseEventSink)` fails on objects without
    `emit_phase`. Pins the no-silent-phase-drop guarantee at the
    construction-time gate."""
    args = _valid_args()
    args["phase_event_sink"] = object()  # no `emit_phase` attribute
    with pytest.raises(BuildGraphError, match="phase_event_sink does not satisfy PhaseEventSink"):
        build_graph(**args)


def test_build_graph_rejects_file_examination_sink_missing_member() -> None:
    """`isinstance(sink, FileExaminationSink)` fails on objects without
    `emit_file_examination`. Same shape as the phase-event sink gate."""
    args = _valid_args()
    args["file_examination_sink"] = object()
    with pytest.raises(
        BuildGraphError,
        match="file_examination_sink does not satisfy FileExaminationSink",
    ):
        build_graph(**args)


# ---------------------------------------------------------------------------
# Other shapes: a class that has BOTH missing methods is rejected by the
# FIRST gate (provider). Documents the gate order.
# ---------------------------------------------------------------------------


def test_build_graph_provider_gate_fires_before_sink_gate() -> None:
    """If multiple deps fail simultaneously, the provider gate fires
    first. Useful so future hardening (e.g., aggregating errors) is an
    intentional decision, not an accidental change of behavior."""
    args = _valid_args()
    args["provider"] = object()
    args["phase_event_sink"] = object()
    with pytest.raises(BuildGraphError, match="provider"):
        build_graph(**args)


# ---------------------------------------------------------------------------
# build_graph is keyword-only — positional invocation fails
# ---------------------------------------------------------------------------


def test_build_graph_signature_is_keyword_only() -> None:
    """Defensive: build_graph uses keyword-only args. A positional call
    would be a TypeError — pinning prevents accidental signature
    relaxation."""
    with pytest.raises(TypeError):
        # Positional invocation should fail
        build_graph(
            _StubProvider(),  # type: ignore[misc]
            ModelConfig(),
            _StubPhaseSink(),
        )
