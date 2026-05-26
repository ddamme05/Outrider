"""Unit tests for the AnomalySink Protocol + the AnomalyPersister
config-error path.

Durable persister behavior (on_conflict_do_nothing semantics, partial-
index match) is covered in integration tests against postgres-test;
this file pins:
  - Protocol surface (runtime_checkable, single emit method)
  - AnomalyPersister config-error path (None session_factory raises)
  - AnomalyRuleName enum value pin
"""

from __future__ import annotations

from typing import Any
from uuid import UUID  # noqa: TC003  (runtime: Protocol method signature)

import pytest

from outrider.anomaly import (
    AnomalyPersister,
    AnomalyPersisterConfigError,
    AnomalyRuleName,
    AnomalySink,
)


def test_anomaly_rule_name_hitl_timeout_value_pinned() -> None:
    """Canonical anomaly rule name string per docs/spec.md §16."""
    assert AnomalyRuleName.HITL_TIMEOUT.value == "hitl_timeout"


def test_anomaly_rule_name_only_one_v1_value() -> None:
    """V1 ships one rule. New rules extend the enum + need matching
    partial unique indexes on the anomalies table."""
    assert {m.value for m in AnomalyRuleName} == {"hitl_timeout"}


def test_persister_rejects_none_session_factory() -> None:
    """Fail-loud at construction time per the sibling persister
    precedents (ReviewStatusPersister, AuditPersister)."""
    with pytest.raises(AnomalyPersisterConfigError):
        AnomalyPersister(session_factory=None)  # type: ignore[arg-type]


def test_recording_sink_satisfies_protocol_structurally() -> None:
    """A duck-typed recording double satisfies AnomalySink."""

    class _Recording:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def emit_anomaly(
            self,
            *,
            review_id: UUID,
            rule_name: AnomalyRuleName,
            severity: str,
            details: dict[str, Any],
        ) -> None:
            self.calls.append(
                {
                    "review_id": review_id,
                    "rule_name": rule_name,
                    "severity": severity,
                    "details": details,
                }
            )

    sink = _Recording()
    assert isinstance(sink, AnomalySink)


def test_partial_sink_rejected_by_runtime_check() -> None:
    """A class missing emit_anomaly fails isinstance(AnomalySink)."""

    class _Partial:
        pass

    assert not isinstance(_Partial(), AnomalySink)


def test_anomaly_sink_declares_one_method() -> None:
    """Protocol surface check."""
    assert hasattr(AnomalySink, "emit_anomaly")
