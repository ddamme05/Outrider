"""HITLRequest and HITLDecision shape tests.

Both are decision artifacts (frozen=True). HITLRequest is the agent's gate
envelope at interrupt time; HITLDecision is the reviewer's full submission.
The field on HITLDecision is `decisions: list[PerFindingDecision]` per
spec §7.4 line 290 — NOT `per_finding_decisions`. Round-trip through JSON
matters because audit replay reconstructs HITLDecision from the audit row's
JSONB payload.
"""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.schemas import (
    HITLDecision,
    HITLRequest,
    PerFindingDecision,
    PerFindingOutcome,
)


def _build_request() -> HITLRequest:
    now = datetime.now(UTC)
    return HITLRequest(
        findings_requiring_approval=[uuid4(), uuid4()],
        auto_post_findings=[uuid4()],
        created_at=now,
        expires_at=now + timedelta(minutes=30),
    )


def _build_decision(decisions: list[PerFindingDecision] | None = None) -> HITLDecision:
    return HITLDecision(
        reviewer_id="reviewer@example.com",
        decisions=decisions if decisions is not None else [],
        annotation=None,
        decided_at=datetime.now(UTC),
    )


def test_hitl_request_is_frozen() -> None:
    """Decision-snapshot artifact: assigning a field after construction raises."""
    request = _build_request()
    with pytest.raises(ValidationError):
        request.findings_requiring_approval = []  # type: ignore[misc]


def test_hitl_decision_is_frozen() -> None:
    """Submission record artifact: assigning a field after construction raises."""
    decision = _build_decision()
    with pytest.raises(ValidationError):
        decision.reviewer_id = "someone_else"  # type: ignore[misc]


def test_hitl_decision_decisions_field_required() -> None:
    """Per spec §7.4 line 290: field name is `decisions`, not `per_finding_decisions`.

    Empty list admits (the reviewer can decide on zero findings if all
    auto-approve), but the field itself must be present at construction.
    """
    empty_decisions = HITLDecision(
        reviewer_id="reviewer@example.com",
        decisions=[],
        annotation=None,
        decided_at=datetime.now(UTC),
    )
    assert empty_decisions.decisions == ()

    with pytest.raises(ValidationError):
        HITLDecision(  # type: ignore[call-arg]
            reviewer_id="reviewer@example.com",
            decided_at=datetime.now(UTC),
        )


def test_hitl_decision_serializes_round_trip_through_json() -> None:
    """Pydantic .model_dump_json() + reconstruct → equal.

    Audit replay reconstructs HITLDecision from the audit row's JSONB
    payload; this verifies the round-trip is lossless.
    """
    inner = PerFindingDecision(
        finding_id=uuid4(),
        outcome=PerFindingOutcome.SUPPRESS,
        reason="known false positive on this codebase",
    )
    original = HITLDecision(
        reviewer_id="reviewer@example.com",
        decisions=[inner],
        annotation="reviewed during the morning sweep",
        decided_at=datetime.now(UTC),
    )

    json_payload = original.model_dump_json()
    reconstructed = HITLDecision.model_validate_json(json_payload)

    assert reconstructed == original


def test_hitl_decision_extra_forbid() -> None:
    """Unknown fields raise per docs/conventions.md."""
    with pytest.raises(ValidationError, match="extra"):
        HITLDecision(
            reviewer_id="reviewer@example.com",
            decisions=[],
            annotation=None,
            decided_at=datetime.now(UTC),
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_hitl_request_list_fields_are_immutable_tuples() -> None:
    """Pydantic frozen=True only blocks attribute reassignment; list fields can
    still be .append()'d in place. The field type is `tuple[UUID, ...]` so the
    underlying container is a tuple at runtime — true immutability.
    """
    request = _build_request()
    assert isinstance(request.findings_requiring_approval, tuple)
    assert isinstance(request.auto_post_findings, tuple)
    with pytest.raises(AttributeError):
        request.findings_requiring_approval.append(uuid4())  # type: ignore[attr-defined]


def test_hitl_decision_decisions_field_is_immutable_tuple() -> None:
    """HITLDecision.decisions is `tuple[PerFindingDecision, ...]` — true immutability."""
    decision = _build_decision()
    assert isinstance(decision.decisions, tuple)
    with pytest.raises(AttributeError):
        decision.decisions.append(  # type: ignore[attr-defined]
            PerFindingDecision(
                finding_id=uuid4(),
                outcome=PerFindingOutcome.APPROVE,
                reason="",
            )
        )


def test_hitl_request_extra_forbid() -> None:
    """Unknown fields raise per docs/conventions.md."""
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="extra"):
        HITLRequest(
            findings_requiring_approval=[],
            auto_post_findings=[],
            created_at=now,
            expires_at=now + timedelta(minutes=30),
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_hitl_request_rejects_duplicate_in_findings_requiring_approval() -> None:
    """Set-semantic: each finding listed once."""
    now = datetime.now(UTC)
    finding_id = uuid4()
    with pytest.raises(ValidationError, match="duplicate ids"):
        HITLRequest(
            findings_requiring_approval=[finding_id, finding_id],
            auto_post_findings=[],
            created_at=now,
            expires_at=now + timedelta(minutes=30),
        )


def test_hitl_request_rejects_finding_in_both_tuples() -> None:
    """Each finding is either approval-gated or auto-postable, never both."""
    now = datetime.now(UTC)
    finding_id = uuid4()
    with pytest.raises(ValidationError, match="both"):
        HITLRequest(
            findings_requiring_approval=[finding_id],
            auto_post_findings=[finding_id],
            created_at=now,
            expires_at=now + timedelta(minutes=30),
        )


def test_hitl_decision_rejects_multiple_decisions_for_same_finding() -> None:
    """One decision per finding — duplicates are conflicting verdicts."""
    shared = uuid4()
    decisions = [
        PerFindingDecision(
            finding_id=shared,
            outcome=PerFindingOutcome.APPROVE,
            reason="lgtm",
        ),
        PerFindingDecision(
            finding_id=shared,  # same finding — second verdict
            outcome=PerFindingOutcome.REJECT,
            reason="changed mind",
        ),
    ]
    with pytest.raises(ValidationError, match="multiple decisions"):
        _build_decision(decisions=decisions)
