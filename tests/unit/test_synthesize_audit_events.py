# See specs/2026-05-28-synthesize-node.md §Audit append-only.
"""SynthesizeCompletedEvent shape + validator + discriminator tests.

Pins:
- Discriminator routing through `AuditEventAdapter` (`event_type` ==
  `"synthesize_completed"`) — sibling to the analyze-completed routing
  test at `tests/unit/test_analyze_audit_events.py:310`.
- `Optional[int]` / `Optional[float]` accept None on LLM-aggregate
  fields (kept nullable for append-only historical-row read-compat per
  #030; populated from the audit stream going forward per FUP-093).
- `summary_content_hash` matches SHA-256 pattern at the schema layer.
- `overall_risk` accepts the canonical RiskLevel ladder.
- frozen=True + extra="forbid" (the AuditEventBase contract).

Companion to `test_analyze_audit_events.py` — same shape/style, mirror
for the synthesize event so a future schema-shape regression on the
new event surfaces in the unit tier rather than at replay/integration
time.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventAdapter, SynthesizeCompletedEvent
from outrider.schemas import RiskLevel


def _completed_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "summary_content_hash": "a" * 64,  # valid SHA-256 hex (64 chars)
        "overall_risk": RiskLevel.MEDIUM,
        "n_findings": 0,
        "files_examined": 0,
        "files_traced_beyond_diff": 0,
        # LLM-aggregate Optional fields default to None (the schema default,
        # kept nullable for historical-row read-compat per #030). Tests for
        # the int-accepting case provide explicit overrides.
        "wall_clock_seconds": 0.0,
        "pricing_version": "v1",
        "policy_version": "1.0.0",
        "synthesize_model": "claude-sonnet-4-6",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Shape + validator tests
# ---------------------------------------------------------------------------


def test_synthesize_completed_constructs_with_minimal_kwargs() -> None:
    """Required-only construction succeeds; LLM aggregates default to None."""
    event = SynthesizeCompletedEvent(**_completed_kwargs())
    assert event.event_type == "synthesize_completed"
    assert event.node_id == "synthesize"
    # Schema default: None on the LLM-aggregate fields (nullable per #030).
    assert event.llm_calls_made is None
    assert event.total_input_tokens is None
    assert event.total_output_tokens is None
    assert event.total_cost_usd is None


def test_synthesize_completed_accepts_explicit_llm_aggregates() -> None:
    """Optional[int]/[float] fields accept concrete values.

    Synthesize populates these from the audit-stream SUM over `LLMCallEvent`
    rows (FUP-093); the schema must admit non-None values. None remains valid
    for historical rows (kept nullable per #030).
    """
    event = SynthesizeCompletedEvent(
        **_completed_kwargs(
            llm_calls_made=4,
            total_input_tokens=12000,
            total_output_tokens=900,
            total_cost_usd=0.12,
        )
    )
    assert event.llm_calls_made == 4
    assert event.total_input_tokens == 12000
    assert event.total_output_tokens == 900
    assert event.total_cost_usd == pytest.approx(0.12)


def test_synthesize_completed_rejects_negative_llm_aggregates() -> None:
    """`ge=0` floor fires when an explicit negative is supplied
    (None is admitted by the Optional union; -1 is not).
    """
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        SynthesizeCompletedEvent(**_completed_kwargs(llm_calls_made=-1))


def test_synthesize_completed_rejects_oversize_total_cost_usd() -> None:
    """`le=100.0` cap on total_cost_usd defends against `float('inf')`
    propagating into JSONB. A runaway $200 cost is rejected at construction.
    """
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        SynthesizeCompletedEvent(**_completed_kwargs(total_cost_usd=200.0))


def test_synthesize_completed_rejects_oversize_wall_clock_seconds() -> None:
    """`le=86400` (24h) cap on wall_clock_seconds. A multi-day review
    is a bug, not a workload.
    """
    with pytest.raises(ValidationError, match="less than or equal to 86400"):
        SynthesizeCompletedEvent(**_completed_kwargs(wall_clock_seconds=90000.0))


def test_synthesize_completed_rejects_bad_summary_content_hash() -> None:
    """`summary_content_hash` must match SHA-256 hex pattern (64
    lowercase hex chars). Off-pattern values raise at construction.
    """
    with pytest.raises(ValidationError, match="String should match pattern"):
        SynthesizeCompletedEvent(**_completed_kwargs(summary_content_hash="too-short"))


def test_synthesize_completed_admits_all_risk_levels() -> None:
    """`overall_risk` accepts the full RiskLevel ladder per the
    canonical TriageResult.overall_risk ladder."""
    for risk in (RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
        event = SynthesizeCompletedEvent(**_completed_kwargs(overall_risk=risk))
        assert event.overall_risk is risk


def test_synthesize_completed_rejects_extra_fields() -> None:
    """extra="forbid" per AuditEventBase contract — silent-extras
    are how audit-event schemas drift; loud rejection prevents that."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SynthesizeCompletedEvent(**_completed_kwargs(unexpected_field="value"))


def test_synthesize_completed_frozen_rejects_assignment() -> None:
    """frozen=True per AuditEventBase. Post-construction mutation
    of audit-event rows would break the append-only contract; the
    schema-layer frozen flag is defense in depth at the in-memory layer."""
    event = SynthesizeCompletedEvent(**_completed_kwargs())
    with pytest.raises(ValidationError, match="Instance is frozen"):
        event.n_findings = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Discriminator routing — the AuditEvent tagged union picks
# SynthesizeCompletedEvent correctly via TypeAdapter.
# ---------------------------------------------------------------------------


def test_audit_event_adapter_routes_synthesize_completed() -> None:
    """`AuditEventAdapter` (the discriminated-union TypeAdapter at
    events.py) resolves `event_type="synthesize_completed"` payloads
    to `SynthesizeCompletedEvent` — pins the replay-path subtype
    selection that the persister + replay both depend on.

    Closes the audit gap that existing discriminator-routing tests
    cover analyze + analyze_response_rejected + finding_proposal_rejected
    but not synthesize_completed.
    """
    kwargs = _completed_kwargs()
    payload: dict[str, Any] = {**kwargs, "event_type": "synthesize_completed"}
    payload["review_id"] = str(payload["review_id"])
    payload["overall_risk"] = payload["overall_risk"].value
    event = AuditEventAdapter.validate_python(payload)
    assert isinstance(event, SynthesizeCompletedEvent)
    assert event.event_type == "synthesize_completed"


def test_audit_event_adapter_round_trips_synthesize_completed() -> None:
    """model_dump → validate_python round-trips cleanly via the
    discriminated union — the audit-event persistence/replay path
    relies on this for any event that goes through the union.
    """
    original = SynthesizeCompletedEvent(
        **_completed_kwargs(
            llm_calls_made=2,
            total_input_tokens=5000,
            total_output_tokens=400,
            total_cost_usd=0.05,
        )
    )
    payload = original.model_dump(mode="json")
    reconstructed = AuditEventAdapter.validate_python(payload)
    assert isinstance(reconstructed, SynthesizeCompletedEvent)
    assert reconstructed.summary_content_hash == original.summary_content_hash
    assert reconstructed.overall_risk is original.overall_risk
    assert reconstructed.n_findings == original.n_findings
    assert reconstructed.llm_calls_made == 2
    assert reconstructed.total_cost_usd == pytest.approx(0.05)


def test_historical_null_aggregates_round_trip_through_adapter() -> None:
    """FUP-093 / #030 read-compat guard: a persisted `SynthesizeCompletedEvent`
    with `null` LLM aggregates (the pre-FUP-093 shape) still deserializes through
    `AuditEventAdapter` — the replay read path. MUST stay green; a future revert of
    the four fields to required `int`/`float` would fail this, which is exactly why
    #030 keeps them nullable (append-only historical rows serialize `null`)."""
    original = SynthesizeCompletedEvent(**_completed_kwargs())  # aggregates default None
    payload = original.model_dump(mode="json")
    assert payload["llm_calls_made"] is None  # the historical persisted shape
    reconstructed = AuditEventAdapter.validate_python(payload)
    assert isinstance(reconstructed, SynthesizeCompletedEvent)
    assert reconstructed.llm_calls_made is None
    assert reconstructed.total_cost_usd is None
