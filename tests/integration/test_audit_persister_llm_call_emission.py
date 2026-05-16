"""AuditPersister.persist() — happy path + review-not-found + idempotent re-call.

The integration test FUP-007's exit rule names. Seeds real `installations`
+ `reviews` rows so the `SELECT reviews.installation_id` lookup returns
a live value; then exercises the persister end-to-end against a real
Postgres via the `migrated_db` fixture.

Critical assertions:
  - `llm_call_content.prompt` equals `request.user_prompt` VERBATIM
    (regression against the model-dump-redaction footgun: a naive
    `request.model_dump()["user_prompt"]` would persist
    `"<redacted, N chars>"` instead of the real prompt).
  - `llm_call_content.completion` equals `response.text` VERBATIM.
  - `llm_call_content.installation_id` equals `reviews.installation_id`
    — proves the SELECT lookup path works.
  - Re-call with the same event is a no-op (idempotent on event_id).
  - Calling with an unresolvable review_id raises
    `AuditPersisterReviewNotFoundError`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.events import LLMCallEvent
from outrider.audit.persister import AuditPersister, AuditPersisterReviewNotFoundError
from outrider.llm.base import LLMRequest, LLMResponse

_INSTALLATION_ID = 12345


async def _seed_installation_and_review(engine: AsyncEngine) -> str:
    """Seed `installations` + `reviews` rows; return the review id (UUID str)."""
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
            ),
            {"id": _INSTALLATION_ID},
        )
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                "  total_input_tokens, total_output_tokens, total_cost_usd, "
                "  wall_clock_seconds, retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'running', 0, 0, 0, 0, 0, 0, 0, "
                "  NOW() + INTERVAL '90 days'"
                ") RETURNING id"
            ),
            {"id": _INSTALLATION_ID},
        )
        return str(result.scalar_one())


def _make_persister(engine: AsyncEngine) -> AuditPersister:
    """Construct the persister against a live engine."""
    return AuditPersister(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        retention_settings=RetentionSettings(),
    )


def _make_llm_call_event(review_id_str: str) -> LLMCallEvent:
    """Construct a representative LLMCallEvent fixture."""
    from uuid import UUID

    return LLMCallEvent(
        review_id=UUID(review_id_str),
        model="claude-haiku-4-5",
        node_id="triage",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        cost_usd=0.001,
        pricing_version="1.0.0",
        latency_ms=250,
        prompt_hash="a" * 64,
        cache_hit=False,
        context_summary=(),
        prompt_template_version="triage:1",
        system_prompt_hash="b" * 64,
        degraded_mode=False,
        timestamp=datetime.now(UTC),
    )


def _make_llm_request(review_id_str: str, user_prompt: str = "the user prompt") -> LLMRequest:
    """Construct a representative LLMRequest with non-redacted text."""
    from uuid import UUID

    return LLMRequest(
        system_prompt="the system prompt",
        user_prompt=user_prompt,
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.0,
        review_id=UUID(review_id_str),
        node_id="triage",
        prompt_template_version="triage:1",
        degraded_mode=False,
    )


def _make_llm_response(text_value: str = "the completion text") -> LLMResponse:
    """Construct a representative LLMResponse."""
    return LLMResponse(
        text=text_value,
        model="claude-haiku-4-5",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=250,
    )


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


async def test_persist_writes_both_rows_atomically(migrated_db: str) -> None:
    """Persister writes the audit_events row AND the llm_call_content row
    in one transaction. Asserts every field on both rows."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id_str = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(review_id_str)
        request = _make_llm_request(review_id_str, user_prompt="my secret prompt")
        response = _make_llm_response(text_value="my secret completion")

        persister = _make_persister(engine)
        await persister.persist(event, request, response)

        async with engine.connect() as conn:
            # audit_events row landed with the right shape.
            audit_row = await conn.execute(
                text(
                    "SELECT event_id, review_id, event_type, phase_key, is_eval, payload "
                    "FROM audit_events WHERE event_id = :eid"
                ),
                {"eid": event.event_id},
            )
            audit = audit_row.one()
            assert audit.event_id == event.event_id
            assert str(audit.review_id) == review_id_str
            assert audit.event_type == "llm_call"
            assert audit.phase_key is None  # LLMCallEvent never has phase_key
            assert audit.is_eval is False
            assert audit.payload["model"] == "claude-haiku-4-5"
            assert audit.payload["input_tokens"] == 100
            # Payload excludes sequence_number per events.py docstring.
            assert "sequence_number" not in audit.payload

            # llm_call_content row landed with raw content (NOT redaction marker).
            content_row = await conn.execute(
                text(
                    "SELECT event_id, installation_id, prompt, completion, "
                    "       is_eval, retention_expires_at "
                    "FROM llm_call_content WHERE event_id = :eid"
                ),
                {"eid": event.event_id},
            )
            content = content_row.one()
            assert content.event_id == event.event_id
            assert content.installation_id == _INSTALLATION_ID
            assert content.prompt == "my secret prompt"
            assert content.completion == "my secret completion"
            assert content.is_eval is False
            assert content.retention_expires_at is not None
    finally:
        await engine.dispose()


async def test_persist_persists_raw_prompt_not_redaction_marker(migrated_db: str) -> None:
    """The C2 regression test: the persister MUST persist `request.user_prompt`
    verbatim, NOT the field-serializer's `"<redacted, N chars>"` marker."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id_str = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(review_id_str)
        # Fixture strings deliberately do NOT contain the substring
        # "redacted" so the substring-absence assertion below catches the
        # real failure mode (persisting `"<redacted, 38 chars>"`).
        secret_prompt = "the user's actual untouched prompt"  # noqa: S105 — test fixture
        secret_completion = "the model's actual untouched completion"  # noqa: S105 — test fixture
        request = _make_llm_request(review_id_str, user_prompt=secret_prompt)
        response = _make_llm_response(text_value=secret_completion)

        persister = _make_persister(engine)
        await persister.persist(event, request, response)

        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT prompt, completion FROM llm_call_content WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
            prompt_db, completion_db = row.one()
            # Verbatim — NOT redaction markers like "<redacted, N chars>".
            assert prompt_db == secret_prompt
            assert completion_db == secret_completion
            assert "redacted" not in prompt_db.lower()
            assert "redacted" not in completion_db.lower()
    finally:
        await engine.dispose()


async def test_persist_links_to_correct_installation_id(migrated_db: str) -> None:
    """`llm_call_content.installation_id` equals `reviews.installation_id`
    for the event's `review_id` — proves the SELECT lookup path."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id_str = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(review_id_str)
        request = _make_llm_request(review_id_str)
        response = _make_llm_response()

        persister = _make_persister(engine)
        await persister.persist(event, request, response)

        async with engine.connect() as conn:
            row = await conn.execute(
                text("SELECT installation_id FROM llm_call_content WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
            installation_id_db = row.scalar_one()
            assert installation_id_db == _INSTALLATION_ID
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Idempotent re-call.
# ---------------------------------------------------------------------------


async def test_persist_is_idempotent_on_repeated_same_event(migrated_db: str) -> None:
    """Calling persist() twice with the same event is a no-op on the second
    call. Both INSERT statements hit `ON CONFLICT DO NOTHING`; the payload-
    equality verification passes (same payload); the content-existence
    check sees the existing row; no exception."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id_str = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(review_id_str)
        request = _make_llm_request(review_id_str)
        response = _make_llm_response()

        persister = _make_persister(engine)
        await persister.persist(event, request, response)
        await persister.persist(event, request, response)  # re-call, should no-op

        async with engine.connect() as conn:
            audit_count = await conn.execute(
                text("SELECT COUNT(*) FROM audit_events WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
            content_count = await conn.execute(
                text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :eid"),
                {"eid": event.event_id},
            )
            assert audit_count.scalar_one() == 1
            assert content_count.scalar_one() == 1
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Review-not-found failure.
# ---------------------------------------------------------------------------


async def test_persist_raises_when_event_request_review_ids_mismatch(
    migrated_db: str,
) -> None:
    """Round-10 regression: event.review_id MUST equal request.review_id.

    Without this guard, a future provider/test mock could pass mismatched
    event and request — the persister would look up `installation_id` via
    `event.review_id` but store `request.user_prompt`/`response.text` under
    that installation, mis-attributing the audit trail (Review A's prompt
    stored under Review B's installation scope).

    Metadata-only failure: the raised `AuditPersisterReviewIdMismatchError`
    carries only the two UUIDs, never prompt/completion content.
    """
    import pytest

    from outrider.audit.persister import AuditPersisterReviewIdMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        # Event uses the seeded review_id; request uses a DIFFERENT review_id.
        event = _make_llm_call_event(seeded_review_id)
        phantom_review_id = str(uuid4())
        request = _make_llm_request(phantom_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterReviewIdMismatchError) as exc_info:
            await persister.persist(event, request, response)

        # Metadata-only contract: exception text carries only the two UUIDs.
        rendered = str(exc_info.value)
        assert seeded_review_id in rendered
        assert phantom_review_id in rendered
        # No content from request leaked into the exception.
        assert request.user_prompt not in rendered

        # No rows landed — guard fires BEFORE the transaction opens.
        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_persist_raises_when_review_id_does_not_resolve(migrated_db: str) -> None:
    """`event.review_id` not in `reviews` → AuditPersisterReviewNotFoundError.

    Producer-side bug; reviews row must be created by the webhook handler
    before graph dispatch. Surfacing loud here is preferable to silently
    writing a content row with a fabricated installation_id (which would
    fire `IntegrityError` from the FK regardless)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        # No installations or reviews seeded — event.review_id won't resolve.
        phantom_review_id = str(uuid4())
        event = _make_llm_call_event(phantom_review_id)
        request = _make_llm_request(phantom_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterReviewNotFoundError, match="review_id"):
            await persister.persist(event, request, response)

        # No rows landed.
        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()
