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

from typing import Any
from uuid import UUID

import pytest

from outrider.audit.events import (
    FindingEvent,
    HITLDecisionEvent,
    HITLRequestEvent,
    TraceDecisionEvent,
    compute_finding_content_hash,
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
    """FindingFactory returns a ReviewFinding with the canonical content_hash.

    `ReviewFinding` has no construction-time validator on `content_hash` —
    unlike `FindingEvent`, whose validator enforces hash-equality on
    construction (see `test_finding_event_factory_validator_runs_on_construction`).
    That makes the factory's hash logic the ONLY place the canonical-SHA-256
    contract per spec §8.5 is enforced for `ReviewFinding`. Asserting equality
    against a recomputed hash (not just length) guards against a factory
    hash-logic bug that would otherwise slip through.
    """
    finding = FindingFactory.create()
    assert isinstance(finding, ReviewFinding)
    assert finding.finding_type == FindingType.SQL_INJECTION
    assert finding.evidence_tier == EvidenceTier.JUDGED
    expected_hash = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
    )
    assert finding.content_hash == expected_hash
    assert len(finding.content_hash) == 64  # belt + suspenders for spec §8.5 length


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


# Regression coverage for the discipline guards added during the eval-harness
# Copilot audit chain (rounds 6 + 7 → spec Actual Outcome Items 7 + 11).
# Surfaced by Codex audit: the helpers were correct, but the exact bug paths
# they close had no direct tests, so a future regression would only surface
# the next time someone hit the bug manually. Tests below exercise:
#   - `_normalize_finding_type`: valid str coerces; valid enum stays; invalid
#     str raises at the factory call site (not at a confusing downstream
#     Pydantic ValidationError); the original bug (string input silently
#     skipped severity + content_hash derivation) stays closed.
#   - `_reject_is_eval_false`: any non-True is_eval override is rejected.
#     Strict `is not True` (not `is False`) — Pydantic V2 lenient mode
#     coerces falsy values like 0, "", "false" → False on bool fields, so
#     the narrower `is False` check would let those slip past the gate.


@pytest.mark.parametrize(
    "input_value,expected_enum",
    [
        ("sql_injection", FindingType.SQL_INJECTION),
        (FindingType.SQL_INJECTION, FindingType.SQL_INJECTION),
    ],
)
def test_finding_factory_normalizes_finding_type_str_or_enum(
    input_value: str | FindingType, expected_enum: FindingType
) -> None:
    """FindingFactory accepts str-enum value or enum; both normalize to enum."""
    finding = FindingFactory.create(finding_type=input_value)
    assert finding.finding_type == expected_enum


def test_finding_factory_rejects_invalid_finding_type_string() -> None:
    """An invalid finding_type string raises ValueError at the factory call site.

    Loud-failure pattern: error names the bad value, not a confusing
    downstream Pydantic ValidationError or a placeholder content_hash.
    """
    with pytest.raises(ValueError, match="not a valid FindingType"):
        FindingFactory.create(finding_type="not_a_real_type")


def test_finding_event_factory_normalizes_string_finding_type() -> None:
    """FindingEventFactory shares the same normalization path (preemptive audit)."""
    event = FindingEventFactory.create(finding_type="sql_injection")
    assert event.finding_type == FindingType.SQL_INJECTION


def test_string_finding_type_triggers_severity_and_hash_derivations() -> None:
    """The original bug fix: string finding_type must trigger both derivations.

    Before round 7, the isinstance gate skipped both `lookup_severity()` and
    `compute_finding_content_hash()` when `finding_type` arrived as a string.
    Verifies that severity is derived from policy AND content_hash is the
    canonical recomputed value, not the placeholder.
    """
    finding = FindingFactory.create(finding_type="sql_injection")
    assert finding.severity == lookup_severity(FindingType.SQL_INJECTION)
    expected_hash = compute_finding_content_hash(
        file_path=finding.file_path,
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=finding.finding_type,
    )
    assert finding.content_hash == expected_hash


@pytest.mark.parametrize("bad_value", [False, 0, "false", "False", None, ""])
def test_review_factory_rejects_non_true_is_eval(bad_value: Any) -> None:
    """ReviewFactory raises on any non-True is_eval override (round 6 tightening).

    Strict `is not True` — covers the Pydantic V2 lenient-coercion vector
    (0, "", "false" all coerce to False). Without the strict check, those
    would slip past the construction gate and rely on the teardown integrity
    gate to catch them.
    """
    with pytest.raises(ValueError, match="cannot construct a record with is_eval"):
        ReviewFactory.create(is_eval=bad_value)


@pytest.mark.parametrize("bad_value", [False, 0, "false", None])
def test_finding_event_factory_rejects_non_true_is_eval(bad_value: Any) -> None:
    """FindingEventFactory shares the same is_eval rejection guard."""
    with pytest.raises(ValueError, match="cannot construct a record with is_eval"):
        FindingEventFactory.create(is_eval=bad_value)


def test_review_factory_permits_explicit_true_is_eval() -> None:
    """is_eval=True explicit override is permitted (no-op vs default)."""
    row = ReviewFactory.create(is_eval=True)
    assert row["is_eval"] is True


def test_review_factory_default_is_eval_when_no_override() -> None:
    """No override → factory default is_eval=True applies."""
    row = ReviewFactory.create()
    assert row["is_eval"] is True
