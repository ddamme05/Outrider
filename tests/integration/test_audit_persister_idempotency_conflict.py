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

import pytest
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
    """Producer bug: re-emit with same event_id, different cost_usd. The
    persister catches the payload mismatch on the audit-row conflict
    path and raises AuditPersisterIdempotencyConflict."""
    request = llm_request_factory(persister_setup.review_id)
    response = llm_response_factory()

    event1 = llm_call_event_factory(persister_setup.review_id, cost_usd=0.001)
    await persister_setup.persister.persist(event1, request, response)

    # Construct a second event with the SAME event_id but different cost_usd.
    event2 = event1.model_copy(update={"cost_usd": 0.999})
    assert event2.event_id == event1.event_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event2, request, response)

    exc = exc_info.value
    assert exc.event_id == event1.event_id
    assert "cost_usd" in exc.mismatched_fields
    digest = exc.field_digests["cost_usd"]
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
    event_obj = llm_call_event_factory(persister_setup.review_id)
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

    # Now re-emit with prompt C (different from both the stored "prompt B"
    # and the original "prompt A"). The audit row matches (same payload);
    # the content path's conflict-verification SELECTs "prompt B" and
    # detects mismatch against "prompt C".
    request2 = llm_request_factory(persister_setup.review_id, user_prompt="prompt C")
    # The content row exists (the resurrection guard would NOT return);
    # the content INSERT hits ON CONFLICT DO NOTHING; verification reads
    # "prompt B" vs "prompt C" and raises.
    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.persist(event_obj, request2, response)

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

    event_obj = llm_call_event_factory(persister_setup.review_id)
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


async def test_emit_phase_payload_mismatch_raises_conflict(
    persister_setup: PersisterTestSetup,
    review_phase_event_factory: ReviewPhaseEventFactory,
) -> None:
    """emit_phase() uses the same idempotency mechanism; same event_id with
    different payload (e.g., different node_id) → AuditPersisterIdempotencyConflict."""
    event1 = review_phase_event_factory(
        persister_setup.review_id, marker="start", phase_key="analyze:src/a.py"
    )
    await persister_setup.persister.emit_phase(event1)

    event2 = event1.model_copy(update={"phase_key": "analyze:src/b.py"})
    assert event2.event_id == event1.event_id

    with pytest.raises(AuditPersisterIdempotencyConflict) as exc_info:
        await persister_setup.persister.emit_phase(event2)

    assert "phase_key" in exc_info.value.mismatched_fields
