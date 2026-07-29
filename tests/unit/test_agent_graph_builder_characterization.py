"""Characterize `build_graph` dependency validation and error precedence.

Each test perturbs a valid argument set as narrowly as possible so the asserted
message identifies the guard that won.
"""

from __future__ import annotations

from typing import Any

import pytest
from test_agent_graph_builder import _valid_args

from outrider.agent.graph import BuildGraphError, build_graph


def _args(**overrides: Any) -> dict[str, Any]:
    """A valid kwarg set with specific arguments perturbed.

    Each test changes as few arguments as it can. A test that perturbs two
    arguments without meaning to would pass on the wrong guard.
    """
    arguments = _valid_args()
    arguments.update(overrides)
    return arguments


class _NotAProtocol:
    """Satisfies nothing. Every runtime-checkable Protocol check rejects it."""


# Required dependency None checks


def test_hitl_event_sink_none_is_rejected() -> None:
    with pytest.raises(BuildGraphError, match="hitl_event_sink must not be None"):
        build_graph(**_args(hitl_event_sink=None))


def test_review_status_sink_none_is_rejected() -> None:
    with pytest.raises(BuildGraphError, match="review_status_sink must not be None"):
        build_graph(**_args(review_status_sink=None))


def test_hitl_config_none_is_rejected() -> None:
    with pytest.raises(BuildGraphError, match="hitl_config must not be None"):
        build_graph(**_args(hitl_config=None))


def test_patch_config_none_is_rejected() -> None:
    with pytest.raises(BuildGraphError, match="patch_config must not be None"):
        build_graph(**_args(patch_config=None))


def test_checkpointer_none_is_rejected_with_the_rehydration_rationale() -> None:
    """The message carries the WHY -- cross-process state rehydration for HITL
    resume -- and that rationale is the part a rewrite would drop first.
    """
    with pytest.raises(BuildGraphError, match="checkpointer must not be None"):
        build_graph(**_args(checkpointer=None))

    with pytest.raises(BuildGraphError, match="durable checkpointer"):
        build_graph(**_args(checkpointer=None))


# Value and callability checks


@pytest.mark.parametrize("value", [0, -1, True, "3", 1.0])
def test_analyze_max_concurrency_rejects_non_positive_ints_and_bools(value: object) -> None:
    """`True` is in this list on purpose: `isinstance(True, int)` is true in Python,
    so a guard written only as an int check would accept it. The guard rejects
    bool explicitly, and that exclusion is behaviour worth pinning.
    """
    with pytest.raises(BuildGraphError, match="analyze_max_concurrency must be an int >= 1"):
        build_graph(**_args(analyze_max_concurrency=value))


def test_analyze_max_concurrency_accepts_one() -> None:
    """The boundary the guard admits. Without this, tightening `>= 1` to `> 1`
    would break nothing in the suite."""
    graph = build_graph(**_args(analyze_max_concurrency=1))
    assert callable(graph.ainvoke)


def test_resolve_slack_target_rejects_a_non_callable() -> None:
    with pytest.raises(BuildGraphError, match="resolve_slack_target must be an async callable"):
        build_graph(**_args(resolve_slack_target=_NotAProtocol()))


def test_resolve_slack_target_accepts_none() -> None:
    """The guard is conditional: None is the documented 'not configured' value,
    and only a non-None non-callable is an error."""
    graph = build_graph(**_args(resolve_slack_target=None))
    assert callable(graph.ainvoke)


# Protocol and instance checks


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("hitl_event_sink", "hitl_event_sink does not satisfy HITLEventSink Protocol"),
        (
            "synthesize_event_sink",
            "synthesize_event_sink does not satisfy SynthesizeEventSink Protocol",
        ),
        ("anomaly_sink", "anomaly_sink does not satisfy AnomalySink Protocol"),
        ("review_status_sink", "review_status_sink does not satisfy ReviewStatusSink Protocol"),
    ],
)
def test_protocol_checks_reject_an_object_satisfying_nothing(argument: str, expected: str) -> None:
    with pytest.raises(BuildGraphError, match=expected):
        build_graph(**_args(**{argument: _NotAProtocol()}))


def test_hitl_config_rejects_a_non_hitl_config() -> None:
    """An instance check, not a Protocol check -- the message says
    'must be a HITLConfig instance', and that distinction is deliberate."""
    with pytest.raises(BuildGraphError, match="hitl_config must be a HITLConfig instance"):
        build_graph(**_args(hitl_config=_NotAProtocol()))


def test_patch_config_rejects_a_non_patch_config() -> None:
    with pytest.raises(BuildGraphError, match="patch_config must be a PatchConfig"):
        build_graph(**_args(patch_config=_NotAProtocol()))


# Error precedence when multiple arguments are invalid


def test_provider_none_precedes_checkpointer_none() -> None:
    """The first guard wins over one that had no assertion before this file."""
    with pytest.raises(BuildGraphError, match="provider must not be None"):
        build_graph(**_args(provider=None, checkpointer=None))


def test_hitl_config_none_precedes_checkpointer_none() -> None:
    """`checkpointer` reads like the more fundamental dependency but is the later
    guard, so the expected winner here is not the intuitive one."""
    with pytest.raises(BuildGraphError, match="hitl_config must not be None"):
        build_graph(**_args(hitl_config=None, checkpointer=None))


def test_hitl_config_none_precedes_patch_config_none() -> None:
    """Adjacent guards, so swapping two neighbours is visible."""
    with pytest.raises(BuildGraphError, match="hitl_config must not be None"):
        build_graph(**_args(hitl_config=None, patch_config=None))


def test_review_status_sink_none_precedes_analyze_max_concurrency_check() -> None:
    """Spans the None-check block into the value-check block: hoisting the value
    checks changes this message while every other test still passes."""
    with pytest.raises(BuildGraphError, match="review_status_sink must not be None"):
        build_graph(**_args(review_status_sink=None, analyze_max_concurrency=0))
