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
