"""AuditPersister.emit_slack_notification() — persist + replay the Slack
notification audit event (dashboard-in-Slack V1).

Pins that the event_id-PK row carries channel_id / message_ts / kind / posted_at
in the JSONB payload, reconstructs via `AuditEventAdapter` (replay-equivalence),
is event_id-PK idempotent (re-emit the same event → one row), and threads
`is_eval` to the top-level column. Mirrors test_observed_skip_shadow_event.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import text

from outrider.audit.events import AuditEventAdapter, SlackNotificationEvent

if TYPE_CHECKING:
    from uuid import UUID

    from tests.integration.conftest import PersisterTestSetup  # type: ignore[import-not-found]

_POSTED_AT = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _event(
    review_id: UUID,
    *,
    kind: Literal["hitl_pending", "review_posted"] = "hitl_pending",
    is_eval: bool = False,
) -> SlackNotificationEvent:
    return SlackNotificationEvent(
        review_id=review_id,
        is_eval=is_eval,
        channel_id="C0123ABC",
        message_ts="1718500000.123456",
        kind=kind,
        posted_at=_POSTED_AT,
    )


async def test_emit_slack_notification_persists_metadata(
    persister_setup: PersisterTestSetup,
) -> None:
    """The row carries channel_id / message_ts / kind / posted_at in JSONB and
    no message body (metadata-only); sequence_number is DB-assigned, not in payload."""
    event = _event(persister_setup.review_id, kind="hitl_pending")
    await persister_setup.persister.emit_slack_notification(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT event_type, is_eval, payload FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    assert row.event_type == "slack_notification"
    assert row.is_eval is False
    assert row.payload["channel_id"] == "C0123ABC"
    assert row.payload["message_ts"] == "1718500000.123456"
    assert row.payload["kind"] == "hitl_pending"
    assert row.payload["posted_at"].startswith("2026-06-15")
    assert "sequence_number" not in row.payload


async def test_emit_slack_notification_replays_via_adapter(
    persister_setup: PersisterTestSetup,
) -> None:
    """The row reconstructs via AuditEventAdapter (replay-equivalence) with kind
    + channel_id + message_ts intact."""
    event = _event(persister_setup.review_id, kind="review_posted")
    await persister_setup.persister.emit_slack_notification(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT sequence_number, payload FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    reconstructed = AuditEventAdapter.validate_python(
        {**row.payload, "sequence_number": row.sequence_number}
    )
    assert isinstance(reconstructed, SlackNotificationEvent)
    assert reconstructed.kind == "review_posted"
    assert reconstructed.channel_id == "C0123ABC"
    assert reconstructed.message_ts == "1718500000.123456"


async def test_emit_slack_notification_is_event_id_idempotent(
    persister_setup: PersisterTestSetup,
) -> None:
    """Re-emitting the same event (same event_id, same payload) writes at most one
    row — event_id-PK idempotency per DECISIONS.md#026 (resume re-emission collapses)."""
    event = _event(persister_setup.review_id)
    await persister_setup.persister.emit_slack_notification(event)
    await persister_setup.persister.emit_slack_notification(event)

    async with persister_setup.engine.connect() as conn:
        count = (
            await conn.execute(
                text("SELECT COUNT(*) AS n FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    assert count.n == 1


async def test_emit_slack_notification_is_eval_propagation(
    persister_setup: PersisterTestSetup,
) -> None:
    """is_eval on the event lands in the row's top-level column (eval isolation)."""
    event = _event(persister_setup.review_id, is_eval=True)
    await persister_setup.persister.emit_slack_notification(event)

    async with persister_setup.engine.connect() as conn:
        row = (
            await conn.execute(
                text("SELECT is_eval FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
        ).one()
    assert row.is_eval is True
