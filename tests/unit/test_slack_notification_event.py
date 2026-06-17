"""SlackNotificationEvent — shape, validators, discriminator routing.

Pins the two message-class `kind`s, the metadata-only field set, frozen +
extra="forbid" (no message body may be stored), and discriminator routing
through `AuditEventAdapter`. See specs/2026-06-15-slack-dashboard-in-slack.md
(Audit events emitted).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import AuditEventAdapter, SlackNotificationEvent


def _kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "review_id": uuid4(),
        "channel_id": "C0123ABC",
        "message_ts": "1718500000.123456",
        "kind": "hitl_pending",
        "posted_at": datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
    }
    base.update(overrides)
    return base


def test_hitl_pending_event_shape() -> None:
    ev = SlackNotificationEvent(**_kwargs())
    assert ev.event_type == "slack_notification"
    assert ev.kind == "hitl_pending"
    assert ev.channel_id == "C0123ABC"
    assert ev.message_ts == "1718500000.123456"
    assert ev.sequence_number is None  # DB-assigned BIGSERIAL at INSERT
    assert ev.is_eval is False


def test_review_posted_kind_accepted() -> None:
    ev = SlackNotificationEvent(**_kwargs(kind="review_posted"))
    assert ev.kind == "review_posted"


def test_unknown_kind_rejected() -> None:
    # Only the two V1 message classes are valid; e.g. "expired" is a mirror
    # state, not a notification kind.
    with pytest.raises(ValidationError):
        SlackNotificationEvent(**_kwargs(kind="expired"))


@pytest.mark.parametrize("field", ["channel_id", "message_ts"])
def test_empty_required_string_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        SlackNotificationEvent(**_kwargs(**{field: ""}))


def test_message_body_cannot_be_stored() -> None:
    # extra="forbid" is the structural guard behind metadata-first: an
    # accidental `text=`/`blocks=` payload fails construction rather than
    # leaking finding prose into the audit row.
    with pytest.raises(ValidationError):
        SlackNotificationEvent(**_kwargs(text="SQL injection in handlers/webhook.py:88"))


def test_frozen() -> None:
    ev = SlackNotificationEvent(**_kwargs())
    with pytest.raises(ValidationError):
        ev.channel_id = "C999"  # type: ignore[misc]


def test_discriminator_routing_through_adapter() -> None:
    ev = SlackNotificationEvent(**_kwargs(kind="review_posted"))
    reconstructed = AuditEventAdapter.validate_python(ev.model_dump(mode="json"))
    assert isinstance(reconstructed, SlackNotificationEvent)
    assert reconstructed.event_type == "slack_notification"
    assert reconstructed.kind == "review_posted"
