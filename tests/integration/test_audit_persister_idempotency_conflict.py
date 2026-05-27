"""AuditPersister idempotency-conflict — same event_id, different payload.

Pins H4 + the metadata-only exception contract:

- Same `event_id` with mismatched payload → `AuditPersisterIdempotencyConflict`.
- Exception carries `event_id`, `mismatched_fields`, and `field_digests`
  (SHA-256 + length per field).
- Exception MUST NOT carry raw `prompt`, `completion`, `existing`,
  `attempted`, or `payload` attrs (regression test for the #016
  logs-stay-metadata-only contract).
- `str(exc)` MUST NOT contain raw content text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy import text

from outrider.audit.persister import AuditPersisterIdempotencyConflict, FieldDigest

if TYPE_CHECKING:
    from tests.integration.conftest import (  # type: ignore[import-not-found]
        LLMCallEventFactory,
        LLMRequestFactory,
        LLMResponseFactory,
        PersisterTestSetup,
        ReviewPhaseEventFactory,
    )


async def test_persist_same_event_id_different_payload_raises_conflict(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Producer bug: re-emit with same event_id, different timestamp. The
    persister catches the payload mismatch on the audit-row conflict
    path and raises AuditPersisterIdempotencyConflict.

    Divergence is on `timestamp` because it's the simplest non-content-
    bearing field that can legitimately differ between two emissions
    with the same event_id (e.g., a retry that picks up a fresh
    `datetime.now(UTC)` clock read). `cost_usd` and `pricing_version`
    would also work post-round-51 (the pricing checks moved to the
    fresh-write branch only; same-event_id re-emits route through the
    audit-conflict path again), but timestamp keeps the test focused
    on the conflict path without introducing pricing-table mechanics.
    """
    from datetime import UTC, datetime, timedelta

    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    event1 = llm_call_event_factory(persister_setup.review_id)
    await persister_setup.persister.persist(event1, request, response)

    # Construct a second event with the SAME event_id but different timestamp.
    later = datetime.now(UTC) + timedelta(seconds=5)
    event2 = event1.model_copy(update={"timestamp": later})
    assert event2.event_id == event1.event_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event2, request, response)

    exc = exc_info.value
    assert exc.event_id == event1.event_id
    assert "timestamp" in exc.mismatched_fields
    digest = exc.field_digests["timestamp"]
    assert isinstance(digest, FieldDigest)
    assert digest.existing_sha256 != digest.attempted_sha256


async def test_persist_idempotent_when_payload_matches(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Same event, called twice → no conflict raised (idempotent no-op)."""
    event_obj = llm_call_event_factory(persister_setup.review_id)
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    await persister_setup.persister.persist(event_obj, request, response)
    # Should NOT raise.
    await persister_setup.persister.persist(event_obj, request, response)


async def test_persist_content_mismatch_raises_conflict(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Same audit payload, different prompt — the content-conflict path raises.

    NOTE: this is reachable only when a concurrent emit lands both rows
    between this persister's audit INSERT and content INSERT. In single-
    threaded test setup, we force this by manually mutating the
    `llm_call_content.prompt` between two emissions — same `event_id`,
    same audit payload (idempotent), different content stored than what
    we're attempting to insert.
    """
    event_obj = llm_call_event_factory(persister_setup.review_id, user_prompt="prompt A")
    request1 = llm_request_factory(persister_setup.review_id, user_prompt="prompt A")
    response = llm_response_factory()
    await persister_setup.persister.persist(event_obj, request1, response)

    # Manually overwrite the content row's prompt to simulate a divergent
    # write that landed before our second persist() attempt. We use raw
    # SQL because the append-only trigger only protects audit_events, not
    # llm_call_content.
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text("UPDATE llm_call_content SET prompt = 'prompt B' WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )

    # Re-emit with the same event + request (both "prompt A"). The
    # audit row matches (same payload); the content path's
    # conflict-verification SELECTs "prompt B" (manually injected
    # above) and detects mismatch against "prompt A" from the request.
    # The content row exists, so the resurrection guard does NOT
    # return; the content INSERT hits ON CONFLICT DO NOTHING;
    # verification reads "prompt B" vs request's "prompt A" and raises.
    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event_obj, request1, response)

    exc = exc_info.value
    assert "prompt" in exc.mismatched_fields


async def test_idempotency_conflict_str_omits_raw_content(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """`str(exc)` flows into `logger.exception(...)` log record's `message`
    field; `RejectLLMContentFilter` is key-based and does NOT pattern-
    match against `message`. The exception's __str__ must therefore omit
    raw payload content. Regression test for #016 + FUP-023."""
    secret_prompt = "operator_secret_prompt_text_xyz123"  # noqa: S105 — test fixture, not credential
    secret_completion = "model_completion_text_abc789"  # noqa: S105 — test fixture, not credential

    event_obj = llm_call_event_factory(persister_setup.review_id, user_prompt=secret_prompt)
    request1 = llm_request_factory(persister_setup.review_id, user_prompt=secret_prompt)
    response = llm_response_factory(text_value=secret_completion)
    await persister_setup.persister.persist(event_obj, request1, response)

    # Manually divert content to simulate a separately-stored value.
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text("UPDATE llm_call_content SET prompt = :alt WHERE event_id = :eid"),
            {"eid": event_obj.event_id, "alt": "different_secret_prompt"},
        )

    request2 = llm_request_factory(persister_setup.review_id, user_prompt=secret_prompt)
    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event_obj, request2, response)

    rendered = str(exc_info.value)
    assert secret_prompt not in rendered
    assert "different_secret_prompt" not in rendered
    assert secret_completion not in rendered
    # vars() also doesn't carry raw content keys.
    exc_vars = set(vars(exc_info.value))
    for forbidden in (
        "existing_payload",
        "attempted_payload",
        "prompt",
        "completion",
        "payload",
    ):
        assert forbidden not in exc_vars


async def test_persist_content_installation_id_mismatch_raises_conflict(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Same prompt/completion text but different `installation_id` →
    `AuditPersisterIdempotencyConflict` with `installation_id` in
    `mismatched_fields`. Pins the round-26 fold: content-row idempotency
    must compare the purge-scope column, not just text. Otherwise a row
    with matching content but wrong `installation_id` (e.g., a producer
    bug crossing review scopes) would silently pass as idempotent.
    """
    event_obj = llm_call_event_factory(persister_setup.review_id, user_prompt="same prompt")
    request = llm_request_factory(persister_setup.review_id, user_prompt="same prompt")
    response = llm_response_factory(text_value="same completion")
    await persister_setup.persister.persist(event_obj, request, response)

    # Divert the stored installation_id directly on the content row so
    # re-emission compares against a mismatched value. (Seed a second
    # installation row first so the FK is satisfied.)
    alternate_installation_id = persister_setup.installation_id + 99999
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install) "
                "VALUES (:id, 'outrider', 88888888, 'alt-org', 'Organization', '{}'::jsonb)"
            ),
            {"id": alternate_installation_id},
        )
        await conn.execute(
            text("UPDATE llm_call_content SET installation_id = :alt WHERE event_id = :eid"),
            {"alt": alternate_installation_id, "eid": event_obj.event_id},
        )

    # Re-emit with the same content + same review (which still resolves
    # to the original installation_id). The content row in DB now has
    # the alternate id; the comparison must raise.
    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event_obj, request, response)

    assert "installation_id" in exc_info.value.mismatched_fields
    # installation_id is a small primitive; no digest entry generated.
    assert "installation_id" not in exc_info.value.field_digests


async def test_persist_content_is_eval_mismatch_raises_conflict(
    persister_setup: PersisterTestSetup,
    llm_call_event_factory: LLMCallEventFactory,
    llm_request_factory: LLMRequestFactory,
    llm_response_factory: LLMResponseFactory,
) -> None:
    """Same prompt/completion text but different `is_eval` →
    `AuditPersisterIdempotencyConflict` with `is_eval` in
    `mismatched_fields`. Pins the round-26 fold: content-row idempotency
    must compare the eval-isolation flag, not just text. Otherwise a
    re-emission with flipped `is_eval` would silently bury production
    review content under the eval flag (or vice versa), defeating the
    `docs/testing.md` eval-isolation contract.
    """
    event_obj = llm_call_event_factory(
        persister_setup.review_id, is_eval=False, user_prompt="same prompt"
    )
    request = llm_request_factory(persister_setup.review_id, user_prompt="same prompt")
    response = llm_response_factory(text_value="same completion")
    await persister_setup.persister.persist(event_obj, request, response)

    # Flip the stored is_eval directly on the content row.
    async with persister_setup.engine.begin() as conn:
        await conn.execute(
            text("UPDATE llm_call_content SET is_eval = TRUE WHERE event_id = :eid"),
            {"eid": event_obj.event_id},
        )

    # Re-emit with is_eval=False (original event's value); the audit row
    # matches (audit payload includes is_eval=False), so we reach the
    # content-comparison path. The stored content row now has
    # is_eval=True; raise.
    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event_obj, request, response)

    assert "is_eval" in exc_info.value.mismatched_fields
    assert "is_eval" not in exc_info.value.field_digests


async def test_emit_phase_idempotent_on_natural_key_no_op_on_resume(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """emit_phase() dedupes on the natural key
    `(review_id, phase_id, COALESCE(phase_key, ''), marker)` per the
    `uq_audit_events_review_phase_natural_key` partial unique index.
    Two emits with the SAME natural key but FRESH `event_id`s — the
    canonical HITL-resume / body-replay scenario, where
    `compute_phase_id` produces the deterministic phase_id and each
    re-run mints a fresh event_id — collapse to a single audit row.

    Previously this test pinned the OLD `on_conflict_do_nothing(
    index_elements=['event_id'])` shape, which raised
    `AuditPersisterIdempotencyConflict` on event_id collision. The
    natural-key index migration (4b9f1c5a7e21) shifted the dedup
    surface to the natural key; resume body re-runs no longer
    accumulate duplicate `start` rows."""
    event1 = review_phase_event_factory(
        persister_setup.review_id, marker="start", phase_key="analyze:src/a.py"
    )
    await persister_setup.persister.emit_phase(event1)

    # Fresh `event_id` (uuid4 default) but identical natural key.
    event2 = event1.model_copy(update={"event_id": uuid4()})
    assert event2.event_id != event1.event_id
    assert event2.phase_id == event1.phase_id
    assert event2.marker == event1.marker
    assert event2.phase_key == event1.phase_key

    # Second emit silently no-ops (no raise; row count stays at 1).
    await persister_setup.persister.emit_phase(event2)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            sa.text(
                "SELECT COUNT(*) FROM audit_events "
                "WHERE review_id = :r AND event_type = 'review_phase'"
            ),
            {"r": persister_setup.review_id},
        )
        assert row.scalar_one() == 1


async def test_emit_phase_distinct_marker_admits_pair(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """`start` and `end` markers for the same `(review_id, phase_id,
    phase_key)` are distinct natural keys — both rows MUST coexist
    for the start/end pair `phase-events-bound-work` invariant to
    hold."""
    start_event = review_phase_event_factory(
        persister_setup.review_id, marker="start", phase_key="analyze:src/a.py"
    )
    end_event = start_event.model_copy(update={"event_id": uuid4(), "marker": "end"})
    await persister_setup.persister.emit_phase(start_event)
    await persister_setup.persister.emit_phase(end_event)

    async with persister_setup.engine.connect() as conn:
        row = await conn.execute(
            sa.text(
                "SELECT COUNT(*) FROM audit_events "
                "WHERE review_id = :r AND event_type = 'review_phase'"
            ),
            {"r": persister_setup.review_id},
        )
        assert row.scalar_one() == 2


async def test_emit_phase_natural_key_conflict_raises_on_node_id_drift(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Per CodeRabbit 2026-05-27: a producer bug that mints a phase_id
    colliding with another node's natural key (phase_id is a `str` —
    no schema-level enforcement that the value came from `compute_phase_id`
    for the right (node_id, attempt_key)) MUST surface as a loud
    `AuditPersisterIdempotencyConflict`, not a silent no-op. Without the
    reload+compare on natural-key conflict, the producer bug would write
    the wrong-node-id row and the index would silently skip — replay
    tooling would never know two distinct logical phases shared a key.
    """
    first_event = review_phase_event_factory(
        persister_setup.review_id,
        marker="start",
        phase_key=None,
    )
    await persister_setup.persister.emit_phase(first_event)

    # Construct a second event with the SAME (review_id, phase_id,
    # phase_key, marker) natural key but a DIFFERENT node_id —
    # simulates a producer bug or replay-injected row. Factory default
    # is node_id="triage"; model_copy flips it to "hitl".
    drifted_event = first_event.model_copy(update={"event_id": uuid4(), "node_id": "hitl"})
    assert drifted_event.phase_id == first_event.phase_id
    assert drifted_event.marker == first_event.marker
    assert drifted_event.phase_key == first_event.phase_key
    assert drifted_event.node_id != first_event.node_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.emit_phase(drifted_event)

    assert "node_id" in exc_info.value.mismatched_fields


async def test_emit_phase_natural_key_conflict_raises_on_is_eval_drift(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """Sibling of the node_id-drift test: same natural key but flipped
    `is_eval` flag MUST raise. The eval-isolation invariant per
    `docs/testing.md` requires that prod and eval rows never collapse
    onto each other; without the is_eval reload-and-compare, a producer
    bug or replay-injected row could silently mix the two."""
    first_event = review_phase_event_factory(
        persister_setup.review_id,
        marker="start",
        phase_key=None,
        is_eval=False,
    )
    await persister_setup.persister.emit_phase(first_event)

    drifted_event = first_event.model_copy(update={"event_id": uuid4(), "is_eval": True})
    assert drifted_event.is_eval is True
    assert drifted_event.phase_id == first_event.phase_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.emit_phase(drifted_event)

    assert "is_eval" in exc_info.value.mismatched_fields
