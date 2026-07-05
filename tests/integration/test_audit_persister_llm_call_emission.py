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
                "  retention_expires_at"
                ") VALUES ("
                "  :id, 100, 1, 'sha1', 'running', "
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


_DEFAULT_SYSTEM_PROMPT = "the system prompt"
_DEFAULT_USER_PROMPT = "the user prompt"


def _make_llm_call_event(
    review_id_str: str,
    *,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
    user_prompt: str = _DEFAULT_USER_PROMPT,
    phase_key: str | None = None,
) -> LLMCallEvent:
    """Construct a representative LLMCallEvent fixture. Hashes match the
    canonical hash of the request prompts; cost_usd / pricing_version
    match what the response cross-check recomputes for the matching
    `_make_llm_response()` fixture (100/50 tokens on claude-haiku-4-5)."""
    from uuid import UUID

    from outrider.llm.anthropic_provider import (
        _ANTHROPIC_CONTRACT_DIGEST,
        _ANTHROPIC_PROFILE_ID,
    )
    from outrider.llm.base import _canonical_prompt_hash, _canonical_system_prompt_hash
    from outrider.llm.pricing import PRICING_VERSION, compute_cost_usd

    canonical_cost = float(
        compute_cost_usd(
            _ANTHROPIC_PROFILE_ID,
            "claude-haiku-4-5",
            input_tokens=100,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=50,
        )
    )

    return LLMCallEvent(
        review_id=UUID(review_id_str),
        model="claude-haiku-4-5",
        # Must match `_make_llm_response`'s finish_reason for the persister's
        # response<->event cross-check (DECISIONS.md#016 Amended 2026-06-30).
        finish_reason="end_turn",
        node_id="triage",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        cost_usd=canonical_cost,
        pricing_version=PRICING_VERSION,
        latency_ms=250,
        prompt_hash=_canonical_prompt_hash(system_prompt=system_prompt, user_prompt=user_prompt),
        cache_hit=False,
        context_summary=(),
        prompt_template_version="triage:1",
        system_prompt_hash=_canonical_system_prompt_hash(system_prompt),
        degraded_mode=False,
        timestamp=datetime.now(UTC),
        # Host-qualified per #056 so the persister's event-vs-response triad
        # cross-check has a coherent base (the field-mismatch tests override one
        # triad member to force a divergence).
        profile_id=_ANTHROPIC_PROFILE_ID,
        reasoning_enabled=False,
        profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        phase_key=phase_key,
    )


def _make_llm_request(
    review_id_str: str,
    user_prompt: str = _DEFAULT_USER_PROMPT,
    phase_key: str | None = None,
) -> LLMRequest:
    """Construct a representative LLMRequest with non-redacted text."""
    from uuid import UUID

    return LLMRequest(
        system_prompt=_DEFAULT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model="claude-haiku-4-5",
        max_tokens=1024,
        temperature=0.0,
        review_id=UUID(review_id_str),
        node_id="triage",
        prompt_template_version="triage:1",
        degraded_mode=False,
        phase_key=phase_key,
    )


def _make_llm_response(text_value: str = "the completion text") -> LLMResponse:
    """Construct a representative LLMResponse."""
    from outrider.llm.anthropic_provider import (
        _ANTHROPIC_CONTRACT_DIGEST,
        _ANTHROPIC_PROFILE_ID,
    )

    return LLMResponse(
        text=text_value,
        model="claude-haiku-4-5",
        input_tokens=100,
        output_tokens=50,
        cache_read_tokens=0,
        cache_write_tokens=0,
        finish_reason="end_turn",
        latency_ms=250,
        profile_id=_ANTHROPIC_PROFILE_ID,
        reasoning_enabled=False,
        profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
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
        event = _make_llm_call_event(review_id_str, user_prompt="my secret prompt")
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
        # Fixture strings deliberately do NOT contain the substring
        # "redacted" so the substring-absence assertion below catches the
        # real failure mode (persisting `"<redacted, 38 chars>"`).
        secret_prompt = "the user's actual untouched prompt"  # noqa: S105 — test fixture
        secret_completion = "the model's actual untouched completion"  # noqa: S105 — test fixture
        event = _make_llm_call_event(review_id_str, user_prompt=secret_prompt)
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


async def test_persist_raises_when_event_prompt_hash_disagrees_with_request(
    migrated_db: str,
) -> None:
    """`event.prompt_hash` must equal canonical hash of (request.system_prompt,
    request.user_prompt). A mismatch means audit row would carry hash-of-X
    while content row holds text-Y; after retention purges content, only the
    (wrong) hash survives and replay reconstructs under a false identity.
    """
    from outrider.audit.persister import AuditPersisterEventRequestFieldMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        # Event's hash computed over "different" user prompt; request carries default.
        event = _make_llm_call_event(seeded_review_id, user_prompt="hash-divergent-prompt")
        request = _make_llm_request(seeded_review_id)  # default user_prompt
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterEventRequestFieldMismatchError) as exc_info:
            await persister.persist(event, request, response)
        assert exc_info.value.field_name == "prompt_hash"

        rendered = str(exc_info.value)
        # Sentinel content from either side never leaks into the exception.
        assert "hash-divergent-prompt" not in rendered
        assert request.user_prompt not in rendered
        assert request.system_prompt not in rendered

        # Guard is pre-tx: no rows landed.
        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_persist_raises_when_event_system_prompt_hash_disagrees_with_request(
    migrated_db: str,
) -> None:
    """`event.system_prompt_hash` must equal canonical hash of
    `request.system_prompt`. Same retention-window identity-drift hazard as
    the prompt_hash check, isolated to the system-prompt surface.

    Independent isolation requires a `model_copy` that overrides ONLY
    `system_prompt_hash` (leaving `prompt_hash` consistent with the
    request); otherwise the earlier `prompt_hash` check fires first and
    masks this branch.
    """
    from outrider.audit.persister import AuditPersisterEventRequestFieldMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        consistent_event = _make_llm_call_event(seeded_review_id)
        # Override only system_prompt_hash; prompt_hash stays consistent with request.
        divergent_event = consistent_event.model_copy(update={"system_prompt_hash": "f" * 64})
        request = _make_llm_request(seeded_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterEventRequestFieldMismatchError) as exc_info:
            await persister.persist(divergent_event, request, response)
        assert exc_info.value.field_name == "system_prompt_hash"

        rendered = str(exc_info.value)
        assert request.system_prompt not in rendered
        assert request.user_prompt not in rendered

        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_persist_raises_when_event_response_format_digest_disagrees_with_request(
    migrated_db: str,
) -> None:
    """`event.response_format_digest` must equal the request's derived
    digest (FUP-096). An event claiming a constrained-decoding digest
    for a free-form request would mislabel the output population that
    replay and the cache telemetry split on."""
    from outrider.audit.persister import AuditPersisterEventRequestFieldMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        consistent_event = _make_llm_call_event(seeded_review_id)
        # Event claims a schema rode the call; the request carried none.
        divergent_event = consistent_event.model_copy(update={"response_format_digest": "d" * 64})
        request = _make_llm_request(seeded_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterEventRequestFieldMismatchError) as exc_info:
            await persister.persist(divergent_event, request, response)
        assert exc_info.value.field_name == "response_format_digest"

        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()


@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("model", {"model": "claude-sonnet-4-6"}),
        ("input_tokens", {"input_tokens": 9999}),
        ("output_tokens", {"output_tokens": 9999}),
        ("latency_ms", {"latency_ms": 99999}),
        ("cached_tokens", {"cached_tokens": 9999}),
        ("cache_hit", {"cache_hit": True}),  # response.cache_read_tokens is 0
        # DECISIONS.md#016 Amended 2026-06-30: the same loop catches a finish_reason the
        # provider stamped on the event but not (matching) on the response — e.g. a
        # refusal mislabeled as a success. `_make_llm_response` returns "end_turn".
        ("finish_reason", {"finish_reason": "refusal"}),
        ("cost_usd", {"cost_usd": 0.999}),  # canonical is 0.00035 for the default fixture
        ("pricing_version", {"pricing_version": "v0-pre-release"}),  # PRICING_VERSION is v2
        # Host-identity triad (DECISIONS.md#056): the same cross-check loop catches a triad
        # field the provider stamped on the event but not (matching) on the response — a
        # divergence that would mislabel the cache/replay host-split. (model_copy skips the
        # coherence validator, so a single-field override is sufficient to force the mismatch.)
        ("profile_id", {"profile_id": "baseten"}),
        ("reasoning_enabled", {"reasoning_enabled": True}),
        ("profile_contract_digest", {"profile_contract_digest": "b" * 64}),
    ],
)
async def test_persist_raises_when_event_response_field_disagrees(
    migrated_db: str,
    field_name: str,
    override: dict[str, object],
) -> None:
    """Provider-return-through fields shared between LLMResponse and
    LLMCallEvent must agree. Otherwise the audit row carries stale
    metrics while the content row holds the actual completion text the
    persister stored.
    """
    from outrider.audit.persister import AuditPersisterEventResponseFieldMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        consistent_event = _make_llm_call_event(seeded_review_id)
        divergent_event = consistent_event.model_copy(update=override)
        request = _make_llm_request(seeded_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterEventResponseFieldMismatchError) as exc_info:
            await persister.persist(divergent_event, request, response)
        assert exc_info.value.field_name == field_name

        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            content_count = await conn.execute(text("SELECT COUNT(*) FROM llm_call_content"))
            assert audit_count.scalar_one() == 0
            assert content_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_persist_idempotent_re_emit_survives_pricing_version_bump(
    migrated_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event originally persisted under PRICING_VERSION=vN must remain
    idempotent-re-emittable after a deploy bumps the constant to v(N+1).
    The cost_usd / pricing_version checks run on the fresh-write branch
    only; on conflict, the audit-row payload-equality check trusts the
    historical pricing values stored in the existing row.

    Without the fresh-write-only split, a producer holding a cached
    event from before the bump would receive `EventResponseFieldMismatchError`
    on every retry — across-deploy retries are a normal operational
    surface, not a producer bug.
    """
    from outrider.audit import persister as persister_module

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(seeded_review_id)
        request = _make_llm_request(seeded_review_id)
        response = _make_llm_response()

        persister = _make_persister(engine)
        # First persist: succeeds under the current PRICING_VERSION.
        await persister.persist(event, request, response)

        # Simulate a deploy that bumps PRICING_VERSION. The persister's
        # local binding (imported at module load) is what the in-tx
        # check reads, so patch there.
        monkeypatch.setattr(persister_module, "PRICING_VERSION", "v-future-bump")

        # Re-emit the SAME event (cached at the producer with pre-bump
        # pricing_version). Audit-conflict path runs; payload equality
        # against the stored row holds; no-op. No exception, no extra
        # rows, no resurrection.
        await persister.persist(event, request, response)

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


async def test_persist_raises_when_event_request_phase_keys_mismatch(
    migrated_db: str,
) -> None:
    """V1.5 phase attribution (DECISIONS.md#064): `phase_key` is on the
    request↔event cross-check allowlist — a provider that drops or rewrites
    the key mid-pipeline would attribute the LLM call to the wrong worker
    phase, and replay's strict hybrid grouping would bind it under a phase
    that never made it. Without this guard the mismatched pair persisted
    successfully (the increment-1 review catch)."""
    from outrider.audit.persister import AuditPersisterEventRequestFieldMismatchError

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        # Request carries a worker key; the event lost it (None) — divergence.
        event = _make_llm_call_event(seeded_review_id)
        request = _make_llm_request(seeded_review_id, phase_key="file:src/app.py#0")
        response = _make_llm_response()

        persister = _make_persister(engine)
        with pytest.raises(AuditPersisterEventRequestFieldMismatchError) as exc_info:
            await persister.persist(event, request, response)
        assert exc_info.value.field_name == "phase_key"

        # No rows landed.
        async with engine.connect() as conn:
            audit_count = await conn.execute(text("SELECT COUNT(*) FROM audit_events"))
            assert audit_count.scalar_one() == 0
    finally:
        await engine.dispose()


async def test_persist_carries_phase_key_in_payload_not_denormalized_column(
    migrated_db: str,
) -> None:
    """A matched request/event pair with a worker key persists; the key rides
    the JSONB payload in full while the denormalized `phase_key` column stays
    NULL (ReviewPhaseEvent-only per the `_NO_PHASE_KEY` sentinel rule)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        seeded_review_id = await _seed_installation_and_review(engine)
        event = _make_llm_call_event(seeded_review_id, phase_key="file:src/app.py#0")
        request = _make_llm_request(seeded_review_id, phase_key="file:src/app.py#0")
        response = _make_llm_response()

        persister = _make_persister(engine)
        await persister.persist(event, request, response)

        async with engine.connect() as conn:
            row = (
                await conn.execute(
                    text("SELECT phase_key, payload->>'phase_key' FROM audit_events")
                )
            ).one()
            assert row[0] is None  # denormalized column: ReviewPhaseEvent-only
            assert row[1] == "file:src/app.py#0"  # payload is authoritative
    finally:
        await engine.dispose()
