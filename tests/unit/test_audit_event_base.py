"""AuditEventBase shared-field invariants.

Backs `audit-events-frozen-extra-forbid`, `every-audit-event-has-review-id`,
and `timestamps-are-aware`. Also covers the eval-isolation flag (`is_eval`)
and the audit-identity / content-table-join-key (`event_id`) per
`DECISIONS.md#016`'s `llm_call_content.event_id → audit_events.event_id` FK.
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventBase


class _SampleEvent(AuditEventBase):
    """Minimal subtype for testing base-class invariants."""

    event_type: Literal["sample"] = "sample"


def test_audit_event_base_is_frozen() -> None:
    """frozen=True: assigning to a base-class field raises."""
    event = _SampleEvent(review_id=uuid4())
    with pytest.raises(ValidationError):
        event.review_id = uuid4()  # type: ignore[misc]


def test_audit_event_base_extra_forbid() -> None:
    """Unknown fields raise per audit-events-frozen-extra-forbid."""
    with pytest.raises(ValidationError, match="extra"):
        _SampleEvent(review_id=uuid4(), unknown_field="oops")  # type: ignore[call-arg]


def test_audit_event_base_review_id_required() -> None:
    """review_id is required; missing raises (every-audit-event-has-review-id)."""
    with pytest.raises(ValidationError):
        _SampleEvent()  # type: ignore[call-arg]


def test_audit_event_base_timestamp_must_be_aware() -> None:
    """Naive datetime raises (timestamps-are-aware)."""
    with pytest.raises(ValidationError):
        _SampleEvent(
            review_id=uuid4(),
            timestamp=datetime.utcnow(),  # type: ignore[arg-type]  # noqa: DTZ003
        )


def test_audit_event_base_sequence_number_defaults_to_none() -> None:
    """sequence_number is DB-assigned BIGSERIAL; pre-insert construction is None."""
    event = _SampleEvent(review_id=uuid4())
    assert event.sequence_number is None


def test_audit_event_base_event_id_defaults_to_uuid4() -> None:
    """event_id is the audit-identity + content-table-join-key per DECISIONS.md#016.

    Default-factory wires uuid4 so two back-to-back constructions get distinct
    UUIDs; explicit event_id admits and round-trips through the field.
    """
    event_a = _SampleEvent(review_id=uuid4())
    event_b = _SampleEvent(review_id=uuid4())
    assert isinstance(event_a.event_id, UUID)
    assert event_a.event_id != event_b.event_id

    explicit_id = uuid4()
    event_c = _SampleEvent(review_id=uuid4(), event_id=explicit_id)
    assert event_c.event_id == explicit_id


def test_audit_event_base_is_eval_defaults_to_false() -> None:
    """is_eval is the eval-isolation flag; default False per testing.md."""
    default_event = _SampleEvent(review_id=uuid4())
    assert default_event.is_eval is False

    eval_event = _SampleEvent(review_id=uuid4(), is_eval=True)
    assert eval_event.is_eval is True


def test_audit_event_base_aware_timestamp_admits() -> None:
    """Aware datetime (UTC-tagged) admits cleanly — happy-path counterpart."""
    event = _SampleEvent(review_id=uuid4(), timestamp=datetime.now(UTC))
    assert event.timestamp.tzinfo is not None
