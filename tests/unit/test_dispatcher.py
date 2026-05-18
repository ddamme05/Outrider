"""Tests for the dispatcher Protocol and V1 BackgroundTasksDispatcher.

Covers:
  - `ReviewDispatcher` Protocol membership semantics (mirrors the
    PhaseEventSink / FileExaminationSink test shape).
  - `BackgroundTasksDispatcher.dispatch(state)` enqueues the run_graph
    callable and survives the JSON-roundtrip fail-loud gate.
  - The gate actually fails on a non-JSON-serializable state — defends
    `state-is-pure-data` at the V1 dispatch site.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from fastapi import BackgroundTasks

from outrider.agent.state import ReviewState
from outrider.dispatcher import BackgroundTasksDispatcher, ReviewDispatcher
from outrider.schemas.pr_context import PRContext

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


def _build_seed_state() -> ReviewState:
    """A minimally-valid seed ReviewState matching the webhook-receipt shape."""
    pr_context = PRContext(
        installation_id=12345,
        owner="acme",
        repo="widgets",
        pr_number=42,
        pr_title="Test PR",
        pr_body=None,
        base_sha="b" * 40,
        head_sha="h" * 40,
        author="alice",
        changed_files=(),
        total_additions=0,
        total_deletions=0,
    )
    return ReviewState(
        review_id=uuid4(),
        pr_context=pr_context,
        received_at=datetime.now(UTC),
    )


class _MissingDispatch:
    """No `dispatch` method — should fail Protocol membership."""

    async def some_other_method(self, state: ReviewState) -> None:
        return None


def test_dispatcher_protocol_member_presence() -> None:
    """`BackgroundTasksDispatcher` satisfies the `ReviewDispatcher`
    Protocol via the `dispatch` member; a class without `dispatch` fails."""
    # We don't construct a real dispatcher here — the Protocol check is
    # structural and class-level via hasattr.
    assert hasattr(BackgroundTasksDispatcher, "dispatch")
    assert not isinstance(_MissingDispatch(), ReviewDispatcher)


def test_dispatch_enqueues_run_graph() -> None:
    """`dispatch(state)` calls `background_tasks.add_task(run_graph, state)`."""
    bg_tasks = BackgroundTasks()
    captured: list[ReviewState] = []

    async def run_graph(s: ReviewState) -> None:
        captured.append(s)

    dispatcher = BackgroundTasksDispatcher(
        background_tasks=bg_tasks,
        run_graph=run_graph,
    )
    state = _build_seed_state()

    asyncio.run(dispatcher.dispatch(state))

    # BackgroundTasks stores tasks for FastAPI to run after the response;
    # we don't run them in the unit test (that would require a TestClient).
    # We assert the task got added.
    assert len(bg_tasks.tasks) == 1
    assert bg_tasks.tasks[0].func is run_graph
    # No state was passed to run_graph yet (background tasks haven't run).
    assert captured == []


def test_dispatch_passes_state_to_run_graph_when_tasks_execute() -> None:
    """When `BackgroundTasks` runs its queued tasks (via its `__call__`
    interface), the dispatched state reaches `run_graph` unchanged.

    Bypasses FastAPI's full request flow by invoking the bg_tasks
    callable directly — this is the same hook FastAPI uses internally."""
    bg_tasks = BackgroundTasks()
    captured: list[ReviewState] = []

    async def run_graph(s: ReviewState) -> None:
        captured.append(s)

    dispatcher = BackgroundTasksDispatcher(
        background_tasks=bg_tasks,
        run_graph=run_graph,
    )
    state = _build_seed_state()

    asyncio.run(dispatcher.dispatch(state))
    # Run the queued tasks (FastAPI invokes BackgroundTasks like a callable).
    asyncio.run(bg_tasks())

    assert len(captured) == 1
    assert captured[0].review_id == state.review_id
    assert captured[0].pr_context.pr_number == 42


def test_dispatch_json_roundtrip_gate_succeeds_for_pure_state() -> None:
    """The fail-loud `state.model_dump_json()` gate inside `dispatch`
    succeeds for a regular `ReviewState`. This is the positive case —
    every production review must pass through it cleanly."""
    bg_tasks = BackgroundTasks()

    async def run_graph(s: ReviewState) -> None:
        return None

    dispatcher = BackgroundTasksDispatcher(
        background_tasks=bg_tasks,
        run_graph=run_graph,
    )
    state = _build_seed_state()
    # Must not raise.
    asyncio.run(dispatcher.dispatch(state))


def test_dispatch_propagates_json_roundtrip_failure_without_queueing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `state.model_dump_json()` raises (non-JSON-serializable field
    sneaks in), the exception propagates from `dispatch` AND no
    background task is queued. Pins the fail-loud V1↔V2 parity gate:
    if a future refactor removed the round-trip entirely, the positive
    test would still pass, but this test fails because the gate isn't
    firing on the bad state.
    """
    bg_tasks = BackgroundTasks()

    async def run_graph(s: ReviewState) -> None:
        return None

    dispatcher = BackgroundTasksDispatcher(
        background_tasks=bg_tasks,
        run_graph=run_graph,
    )
    state = _build_seed_state()

    class _SimulatedDumpError(RuntimeError):
        pass

    def _raising_dump_json(self: object) -> str:  # noqa: ARG001
        msg = "simulated non-serializable field in ReviewState"
        raise _SimulatedDumpError(msg)

    monkeypatch.setattr(ReviewState, "model_dump_json", _raising_dump_json)

    with pytest.raises(_SimulatedDumpError, match="simulated non-serializable"):
        asyncio.run(dispatcher.dispatch(state))

    # No background task was queued — `add_task` runs AFTER the
    # dump+validate round-trip, so a dump failure must short-circuit
    # before any task is enqueued.
    assert bg_tasks.tasks == []


def test_dispatcher_satisfies_runtime_checkable_protocol() -> None:
    """A real `BackgroundTasksDispatcher` instance passes the
    `isinstance(d, ReviewDispatcher)` check — pins the runtime-checkable
    Protocol gate for callers that introspect."""
    bg_tasks = BackgroundTasks()

    async def run_graph(s: ReviewState) -> None:
        return None

    dispatcher = BackgroundTasksDispatcher(
        background_tasks=bg_tasks,
        run_graph=run_graph,
    )
    assert isinstance(dispatcher, ReviewDispatcher)


@pytest.fixture
def run_graph_noop() -> Callable[[ReviewState], Awaitable[None]]:
    async def _run(state: ReviewState) -> None:
        return None

    return _run
