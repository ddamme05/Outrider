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
    AnomalySeverity,
    AnomalySink,
)


def test_anomaly_rule_name_hitl_timeout_value_pinned() -> None:
    """Canonical anomaly rule name string per docs/spec.md §16."""
    assert AnomalyRuleName.HITL_TIMEOUT.value == "hitl_timeout"


def test_anomaly_rule_name_v1_value_set() -> None:
    """V1 ships two rules. New rules extend the enum + need matching
    partial unique indexes on the anomalies table.

    HITL_TIMEOUT — sweep-emitted from `sweep/hitl_expiry.py` per
    `docs/spec.md` §16.
    CROSS_ROUND_SEVERITY_DIVERGENCE — graph-emitted from
    `agent/nodes/synthesize.py` per the synthesize-node spec; surfaces
    cross-round `(content_hash, severity)` divergence as corruption
    (severity-set-by-policy invariant + per-element validator chain
    guarantee same content_hash + same finding_type => same severity).
    """
    assert {m.value for m in AnomalyRuleName} == {
        "hitl_timeout",
        "cross_round_severity_divergence",
    }


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
            severity: AnomalySeverity,
            details: dict[str, Any],
            is_eval: bool,
        ) -> None:
            self.calls.append(
                {
                    "review_id": review_id,
                    "rule_name": rule_name,
                    "severity": severity,
                    "details": details,
                    "is_eval": is_eval,
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
    """Protocol surface check — exact membership, not just presence.

    Pins the public API to `{"emit_anomaly"}`. A new public method
    added to `AnomalySink` (e.g., a V1.5 `emit_recovery` for the
    `hitl_recovery` anomaly the dashboard might consume) fails this
    test loudly — same shape as the
    `tests/unit/test_github_publisher.py::test_publish_event_sink_method_set`
    pin per the `Class-10` doctrine (centrally-pinned contracts
    require call-side registration).
    """
    actual_public_methods = {name for name in AnomalySink.__dict__ if not name.startswith("_")}
    assert actual_public_methods == {"emit_anomaly"}, (
        f"AnomalySink public surface drift: missing={ {'emit_anomaly'} - actual_public_methods }, "
        f"extra={actual_public_methods - {'emit_anomaly'}}. Update this test AND verify "
        f"every AnomalySink consumer + test fixture carries the new method."
    )
