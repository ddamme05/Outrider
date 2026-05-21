"""Graph-wiring introspection — Command is the sole routing mechanism from intake.

Closes FUP-030. Spec line 96 (`specs/2026-05-17-intake-and-webhook.md`):

> Also assert by introspecting `build_graph(...)` output that no static
> `add_edge('intake', 'triage')` and no `add_conditional_edges('intake', ...)`
> are present — `Command` is the sole routing mechanism.

The invariant matters because LangGraph 1.1.6 semantics let a static
edge fire ALONGSIDE a `Command(goto=...)` returned by the node — both
destinations get traversed. A regression here (someone re-adding
`add_edge('intake', 'triage')` because "the test still passes when I
forget the Command branch") would silently double-route on success and
not break any current test. Without this introspection guard, the
contract is unguarded.

The test reads two attributes of the compiled graph's builder:
  - `builder.edges` — set of static `(src, dst)` tuples
  - `builder.branches` — dict[node_id, dict[branch_key, Branch]] for
    conditional edges

Both must be free of intake-as-source entries other than the canonical
START → intake admission edge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langgraph.graph import START

from outrider.agent.graph import build_graph

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.audit.events import (
        AnalyzeCompletedEvent,
        AnalyzeResponseRejectedEvent,
        FileExaminationEvent,
        FindingEvent,
        FindingProposalRejectedEvent,
        ReviewPhaseEvent,
    )
    from outrider.llm.base import LLMRequest, LLMResponse


# ---------------------------------------------------------------------------
# Minimal stubs — the introspection test doesn't invoke any nodes; it
# only constructs the graph to read its wiring. So the deps just need
# to satisfy the build_graph protocol gates, not behave correctly.
# ---------------------------------------------------------------------------


class _StubProvider:
    async def complete(self, request: LLMRequest) -> LLMResponse:  # noqa: ARG002
        msg = "introspection test never invokes provider"
        raise NotImplementedError(msg)


class _StubPhaseSink:
    async def emit_phase(self, event: ReviewPhaseEvent) -> None:  # noqa: ARG002
        return None


class _StubFileSink:
    async def emit_file_examination(self, event: FileExaminationEvent) -> None:  # noqa: ARG002
        return None


class _StubAnalyzeEventSink:
    async def emit_finding(self, event: FindingEvent) -> None:  # noqa: ARG002
        return None

    async def emit_finding_proposal_rejected(self, event: FindingProposalRejectedEvent) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_response_rejected(self, event: AnalyzeResponseRejectedEvent) -> None:  # noqa: ARG002
        return None

    async def emit_analyze_completed(self, event: AnalyzeCompletedEvent) -> None:  # noqa: ARG002
        return None


class _StubImportPathResolver:
    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:  # noqa: ARG002
        return []


class _StubModelConfig:
    triage_model = "stub-model"
    analyze_model = "stub-analyze-model"


def _stub_db_factory() -> Any:
    """Returns an async-session-shaped sentinel; introspection won't touch it."""
    msg = "introspection test never opens a session"
    raise NotImplementedError(msg)


def _stub_github_factory(installation_id: int) -> Any:  # noqa: ARG001
    msg = "introspection test never builds a GitHub client"
    raise NotImplementedError(msg)


@pytest.fixture
def compiled_graph() -> Any:
    """Build a compiled graph with stub deps; only the wiring is read."""
    # Stubs intentionally violate the precise types — `build_graph`'s
    # gates are member-presence-only at runtime, so a duck-typed stub
    # is sufficient for introspection-only tests. mypy needs the
    # ignores because static-shape compatibility is stricter.
    return build_graph(
        db_factory=_stub_db_factory,  # type: ignore[arg-type]
        github_factory=_stub_github_factory,
        provider=_StubProvider(),
        model_config=_StubModelConfig(),  # type: ignore[arg-type]
        phase_event_sink=_StubPhaseSink(),
        file_examination_sink=_StubFileSink(),
        analyze_event_sink=_StubAnalyzeEventSink(),
        import_path_resolver=_StubImportPathResolver(),
    )


def test_no_static_edge_from_intake(compiled_graph: Any) -> None:
    """No `(src='intake', dst=*)` entry in `builder.edges`.

    A static edge from intake would fire alongside intake's
    `Command(goto=...)` return — both destinations get traversed.
    Asserts the spec's "Command is sole routing" invariant holds.
    """
    static_edges_from_intake = [
        (src, dst) for src, dst in compiled_graph.builder.edges if src == "intake"
    ]
    assert static_edges_from_intake == [], (
        f"Unexpected static edge(s) from intake: {static_edges_from_intake}. "
        f"Intake routes via Command(goto=...) only — a static edge would "
        f"fire alongside the Command's dynamic edge and double-route. "
        f"Remove `builder.add_edge('intake', ...)` from build_graph."
    )


def test_no_conditional_edge_from_intake(compiled_graph: Any) -> None:
    """No entry under `builder.branches['intake']`.

    A conditional edge from intake would require a new state slot to
    drive the branch function, conflicting with the canonical
    ReviewState ownership rule (per `DECISIONS.md#020`, intake enriches
    `pr_context.changed_files` only — no new top-level slots from intake).
    """
    intake_branches = compiled_graph.builder.branches.get("intake", {})
    assert intake_branches == {}, (
        f"Unexpected conditional edge(s) from intake: "
        f"{dict(intake_branches)}. Intake routes via Command(goto=...) "
        f"only — a conditional edge would require a new state slot, "
        f"conflicting with the canonical ReviewState ownership rule "
        f"(DECISIONS.md#020). Remove `builder.add_conditional_edges("
        f"'intake', ...)` from build_graph."
    )


def test_intake_is_admitted_only_via_start_edge(compiled_graph: Any) -> None:
    """The only edge with `intake` as a destination is START → intake.

    Catches the inverse regression where someone wires a node BACK to
    intake (e.g., a `triage → intake` re-do edge), which would loop the
    graph on the same review_id and emit duplicate phase events.
    """
    edges_into_intake = [(src, dst) for src, dst in compiled_graph.builder.edges if dst == "intake"]
    assert edges_into_intake == [(START, "intake")], (
        f"Unexpected admission edges into intake: {edges_into_intake}. "
        f"The only valid admission edge is START → intake."
    )


def test_intake_runs_via_command_goto_triage_on_happy_path(
    compiled_graph: Any,
) -> None:
    """Sanity: with no edges from intake, the graph CAN still reach triage
    via intake's `Command(goto='triage')` return. This test doesn't run
    the real intake (deps are stubs that raise) — instead it confirms
    the compiled graph has both nodes registered, so the routing target
    exists.

    Pins that the introspection-asserted absence of static edges hasn't
    accidentally removed the destination node too.
    """
    nodes = set(compiled_graph.builder.nodes.keys())
    assert "intake" in nodes
    assert "triage" in nodes


# ---------------------------------------------------------------------------
# Behavioral routing — Command(goto="triage") AND Command(goto=END) actually
# route correctly. The introspection tests above pin the ABSENCE of static
# /conditional edges; these tests pin that the replacement (Command-based
# routing) actually delivers triage / END destinations. Without these, a
# regression that removed the intake node's Command return (e.g., refactor
# to plain `return state_update`) would silently halt the graph at intake
# but the introspection tests would
# still pass.
# ---------------------------------------------------------------------------


def _make_compiled_graph_with_routing_intake(
    intake_command: Any,
    triage_calls: list[int],
) -> Any:
    """Build a compiled graph with a stub intake that returns the given
    `Command`, and a stub triage that records its invocation count.

    Bypasses `build_graph` (which composes real intake + triage) and
    constructs a minimal StateGraph directly so the routing assertion
    is isolated from the production node bodies. The wiring (no
    add_edge/add_conditional_edges from intake) mirrors `build_graph`.

    Uses `dict` for the state schema since these tests assert ONLY
    routing behavior (which node ran), not state propagation. Channel
    plumbing for state updates is out of scope; the introspection
    tests above already pin "intake has no static/conditional edge to
    triage" which is the structural contract.
    """
    from langgraph.graph import END, START, StateGraph

    async def stub_intake(state: dict[str, Any]) -> Any:  # noqa: ARG001
        return intake_command

    async def stub_triage(state: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
        triage_calls.append(1)
        return {}

    builder = StateGraph(dict)
    builder.add_node("intake", stub_intake)
    builder.add_node("triage", stub_triage)
    builder.add_edge(START, "intake")
    builder.add_edge("triage", END)
    # Deliberately NO add_edge("intake", "triage") / add_conditional_edges.
    return builder.compile()


@pytest.mark.asyncio
async def test_command_goto_triage_actually_invokes_triage() -> None:
    """`Command(goto="triage")` on a graph with the same wiring shape as
    `build_graph` (no static edge from intake; `triage → END`) routes
    correctly: the destination node runs.

    Smoke test for LangGraph 1.1.6's Command-routing primitive against
    the wiring shape Outrider uses, NOT a test of the production
    `intake()` function returning the right Command. The latter is
    pinned by `tests/unit/test_intake_node.py::test_happy_path_returns_command_to_triage`
    which exercises real intake + asserts `result.goto == "triage"` on
    the returned Command. Splitting "framework primitive works" from
    "our node returns the right Command" lets each test fail loudly on
    one regression class.
    """
    from langgraph.types import Command

    triage_calls: list[int] = []
    graph = _make_compiled_graph_with_routing_intake(
        intake_command=Command(goto="triage"),
        triage_calls=triage_calls,
    )

    await graph.ainvoke({})

    # Triage ran exactly once via the Command-driven routing.
    assert triage_calls == [1]


@pytest.mark.asyncio
async def test_command_goto_end_actually_skips_triage() -> None:
    """`Command(goto=END)` from intake → triage is NOT invoked.

    Pins the size-gate skip path: intake's `Command(goto=END)` short-
    circuits the graph without running triage. A regression that
    treated END as just-another-node-name would silently invoke triage
    when the size gate fired.
    """
    from langgraph.graph import END
    from langgraph.types import Command

    triage_calls: list[int] = []
    graph = _make_compiled_graph_with_routing_intake(
        intake_command=Command(goto=END),
        triage_calls=triage_calls,
    )

    await graph.ainvoke({})

    # Triage never ran — END short-circuited.
    assert triage_calls == []


# ---------------------------------------------------------------------------
# Graph-wiring introspection is sync — exercise the compiled graph's builder
# fields directly to confirm no event loop is required for the contract under
# test. Pins the contract independent of pytest-asyncio's `asyncio_mode` /
# loop policy (the previous `asyncio.get_running_loop()` check was
# runner-policy-dependent, not contract-dependent).
# ---------------------------------------------------------------------------


def test_graph_wiring_inspectable_without_event_loop(compiled_graph: Any) -> None:
    """Builder fields (`nodes`, `edges`, `branches`) are inspectable
    synchronously — no event-loop required. The compiled-graph fixture
    is constructed in sync context and the introspection above touches
    only these fields.
    """
    assert "intake" in compiled_graph.builder.nodes
    assert "triage" in compiled_graph.builder.nodes
    # `edges` is a set of (src, dst) tuples; `branches` is a dict.
    # Touching them must not require asyncio state.
    assert isinstance(compiled_graph.builder.edges, set)
    assert isinstance(compiled_graph.builder.branches, dict)
