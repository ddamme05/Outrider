"""Eval-harness factory schema compliance.

Each factory's `.create()` produces a schema-valid instance of its target
canonical type. Catches drift between factory-generated shapes and the
underlying canonical types when those types evolve.

Backs the harness's "factories own setting `is_eval=True`" discipline:
audit-event factories construct frozen+extra=forbid Pydantic models with
`is_eval=True` set on every instance.

The `PRContext`-shape factory + the webhook-input-schema validation
arrives in a later spec (when `api/webhooks/schemas.py` exists), per the
Input boundary held item in the eval-harness spec.
"""

from uuid import UUID

from outrider.audit.events import (
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    TraceDecisionEvent,
)
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.schemas import PerFindingDecision, ReviewFinding

from .fixtures import (
    FindingEventFactory,
    FindingFactory,
    HITLDecisionEventFactory,
    HITLRequestEventFactory,
    ReviewFactory,
    TraceDecisionEventFactory,
)


def test_review_factory_produces_dict_with_is_eval_true() -> None:
    """ReviewFactory returns a dict shaped for `Review(**dict)` insertion."""
    row = ReviewFactory.create()
    assert isinstance(row, dict)
    assert row["is_eval"] is True
    assert isinstance(row["id"], UUID)
    assert isinstance(row["installation_id"], int)
    assert row["status"] == "completed"


def test_review_factory_overrides_replace_defaults() -> None:
    """Overrides win over factory defaults."""
    row = ReviewFactory.create(installation_id=99999, is_eval=True, status="failed")
    assert row["installation_id"] == 99999
    assert row["status"] == "failed"


def test_finding_factory_produces_review_finding() -> None:
    """FindingFactory returns a ReviewFinding with the canonical content_hash."""
    finding = FindingFactory.create()
    assert isinstance(finding, ReviewFinding)
    assert finding.finding_type == FindingType.SQL_INJECTION
    assert finding.evidence_tier == EvidenceTier.JUDGED
    # 64-char lowercase hex per spec §8.5
    assert len(finding.content_hash) == 64


def test_finding_factory_recomputes_hash_on_field_overrides() -> None:
    """Overrides to file_path / line_start / line_end / finding_type recompute hash."""
    a = FindingFactory.create(file_path="src/a.py", line_start=1, line_end=5)
    b = FindingFactory.create(file_path="src/b.py", line_start=1, line_end=5)
    assert a.content_hash != b.content_hash


def test_finding_event_factory_produces_finding_event_with_is_eval_true() -> None:
    """FindingEventFactory.create() returns a FindingEvent with is_eval=True.

    Severity comes from `SEVERITY_POLICY[finding_type]` via `lookup_severity`,
    NOT a hard-coded constant — encoding the policy rule rather than today's
    policy value, so the test catches policy drift.
    """
    event = FindingEventFactory.create()
    assert isinstance(event, FindingEvent)
    assert event.is_eval is True
    assert len(event.finding_content_hash) == 64
    assert event.severity == lookup_severity(event.finding_type)


def test_finding_event_factory_validator_runs_on_construction() -> None:
    """The FindingEvent proof-boundary + canonical-hash validators fire via the factory."""
    # JUDGED is the default tier; query_match_id and trace_path can be None
    event = FindingEventFactory.create()
    assert event.evidence_tier == EvidenceTier.JUDGED
    assert event.query_match_id is None
    assert event.trace_path is None


def test_trace_decision_event_factory_satisfies_three_rule_validator() -> None:
    """Default factory output passes the resolved↔target_file in candidates rule."""
    event = TraceDecisionEventFactory.create()
    assert isinstance(event, TraceDecisionEvent)
    assert event.is_eval is True
    assert event.resolution_status == "resolved"
    assert event.target_file is not None
    assert event.target_file in event.candidates_considered


def test_trace_decision_event_factory_unresolved_override() -> None:
    """Overrides for unresolved + target_file=None construct cleanly."""
    event = TraceDecisionEventFactory.create(
        resolution_status="unresolved",
        target_file=None,
        candidates_considered=("src/foo.py", "src/bar.py"),
    )
    assert event.resolution_status == "unresolved"
    assert event.target_file is None


def test_hitl_request_event_factory_produces_hitl_request_with_is_eval() -> None:
    """HITLRequestEventFactory returns HITLRequestEvent with is_eval=True."""
    event = HITLRequestEventFactory.create()
    assert isinstance(event, HITLRequestEvent)
    assert event.is_eval is True
    assert isinstance(event.findings_requiring_approval, tuple)
    assert isinstance(event.auto_post_findings, tuple)


def test_hitl_decision_event_factory_produces_decision_with_default_approve() -> None:
    """HITLDecisionEventFactory's default decisions tuple has one APPROVE."""
    event = HITLDecisionEventFactory.create()
    assert isinstance(event, HITLDecisionEvent)
    assert event.is_eval is True
    assert isinstance(event.decisions, tuple)
    assert len(event.decisions) == 1
    assert isinstance(event.decisions[0], PerFindingDecision)


def test_hitl_decision_event_factory_admits_decision_overrides() -> None:
    """Caller-supplied decisions tuple replaces the default."""
    from uuid import uuid4

    from outrider.schemas import PerFindingOutcome

    custom = (
        PerFindingDecision(
            finding_id=uuid4(),
            outcome=PerFindingOutcome.REJECT,
            reason="duplicate of prior finding",
        ),
    )
    event = HITLDecisionEventFactory.create(decisions=custom)
    assert event.decisions == custom
