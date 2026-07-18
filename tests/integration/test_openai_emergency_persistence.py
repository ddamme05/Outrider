"""openai-native-host persistence (specs/2026-07-18-openai-native-host.md).

DB-backed coverage the spec names: (1) persist-before-raise outcomes land the
`LLMCallEvent` + `llm_call_content` rows atomically with the CORRECT shape —
a costable tier mismatch carries the policy cost, an unpriceable echo carries
`cost_usd=NULL` + the typed reason, round-tripped through the JSONB payload;
(2) the fresh-insert guard REJECTS an incomplete OpenAI pricing context or a
cost/reason pair diverging from the canonical outcome, rolling the fresh row
back; (3) a PRE-FIELD historical row (keys absent) re-emitted post-upgrade
(keys null) passes the absent≡null comparator normalization as an idempotent
no-op — the checkpoint-resume-across-deploy case.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from outrider.audit.config import RetentionSettings
from outrider.audit.events import LLMCallEvent
from outrider.audit.persister import (
    AuditPersister,
    AuditPersisterEventResponseFieldMismatchError,
    _serialize_event_payload,
)
from outrider.llm.base import (
    LLMRequest,
    LLMResponse,
    _canonical_prompt_hash,
    _canonical_system_prompt_hash,
)
from outrider.llm.host_profiles import OPENAI_PROFILE
from outrider.llm.pricing import (
    PRICING_VERSION,
    CostUnpricedReason,
    Priced,
    compute_cost_outcome,
)

_INSTALLATION_ID = 54321
_SYSTEM = "the system prompt"
_USER = "the user prompt"
_MODEL = "gpt-5.6-sol"


async def _seed_installation_and_review(engine: AsyncEngine) -> str:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )
        result = await conn.execute(
            text(
                "INSERT INTO reviews ("
                "  installation_id, repo_id, pr_number, head_sha, status, "
                "  retention_expires_at"
                ") VALUES (:id, 100, 1, 'sha1', 'running', NOW() + INTERVAL '90 days') "
                "RETURNING id"
            ),
            {"id": _INSTALLATION_ID},
        )
        return str(result.scalar_one())


def _make_persister(engine: AsyncEngine) -> AuditPersister:
    return AuditPersister(
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        retention_settings=RetentionSettings(),
    )


def _request(review_id: str) -> LLMRequest:
    return LLMRequest(
        system_prompt=_SYSTEM,
        user_prompt=_USER,
        model=_MODEL,
        max_tokens=100,
        temperature=0.0,
        review_id=UUID(review_id),
        node_id="triage",
        prompt_template_version="triage:1",
        degraded_mode=False,
    )


def _response(*, service_tier: str | None, billed: int = 2000, write: int = 400) -> LLMResponse:
    return LLMResponse(
        text="{}",
        model=_MODEL,
        input_tokens=billed - 1500,
        output_tokens=50,
        cache_read_tokens=1500,
        cache_write_tokens=write,
        finish_reason="end_turn",
        latency_ms=250,
        profile_id=OPENAI_PROFILE.host_id,
        reasoning_enabled=False,
        profile_contract_digest=OPENAI_PROFILE.profile_contract_digest,
        billed_prompt_tokens=billed,
        service_tier_actual=service_tier,
    )


def _event(review_id: str, response: LLMResponse, **overrides: Any) -> LLMCallEvent:
    """Event mirroring `response`, costed by the canonical outcome (the same
    derivation the provider performs) unless an override forces divergence."""
    outcome = compute_cost_outcome(
        OPENAI_PROFILE.host_id,
        _MODEL,
        input_tokens=response.input_tokens,
        cache_write_tokens=response.cache_write_tokens,
        cache_read_tokens=response.cache_read_tokens,
        output_tokens=response.output_tokens,
        billed_prompt_tokens=response.billed_prompt_tokens,
        service_tier=response.service_tier_actual,
        expects_tier_echo=True,
    )
    if isinstance(outcome, Priced):
        cost: float | None = float(outcome.cost_usd)
        reason = None
    else:
        cost = None
        reason = outcome.reason
    base: dict[str, Any] = {
        "review_id": UUID(review_id),
        "timestamp": datetime.now(UTC),
        "model": _MODEL,
        "finish_reason": "end_turn",
        "node_id": "triage",
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cached_tokens": response.cache_read_tokens,
        "cost_usd": cost,
        "cost_unpriced_reason": reason,
        "pricing_version": PRICING_VERSION,
        "latency_ms": 250,
        "prompt_hash": _canonical_prompt_hash(system_prompt=_SYSTEM, user_prompt=_USER),
        "cache_hit": response.cache_read_tokens > 0,
        "context_summary": (),
        "prompt_template_version": "triage:1",
        "system_prompt_hash": _canonical_system_prompt_hash(_SYSTEM),
        "degraded_mode": False,
        "profile_id": response.profile_id,
        "reasoning_enabled": response.reasoning_enabled,
        "profile_contract_digest": response.profile_contract_digest,
        "service_tier": response.service_tier_actual,
        "billed_prompt_tokens": response.billed_prompt_tokens,
        "cache_write_tokens": response.cache_write_tokens,
    }
    return LLMCallEvent(**(base | overrides))


async def _fetch_payload_and_content(engine: AsyncEngine, event_id: UUID) -> tuple[Any, int]:
    async with engine.connect() as conn:
        payload = (
            await conn.execute(
                text("SELECT payload FROM audit_events WHERE event_id = :e"), {"e": event_id}
            )
        ).scalar_one_or_none()
        content_count = (
            await conn.execute(
                text("SELECT COUNT(*) FROM llm_call_content WHERE event_id = :e"),
                {"e": event_id},
            )
        ).scalar_one()
    return payload, int(content_count)


@pytest.mark.asyncio
async def test_unpriced_exchange_persists_null_cost_typed_reason(migrated_db: str) -> None:
    """A scale echo persists BOTH rows with cost_usd=NULL + the typed reason —
    the durable half of persist-before-raise; never 0.0, never a dropped row."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_installation_and_review(engine)
        response = _response(service_tier="scale")
        event = _event(review_id, response)
        await _make_persister(engine).persist(event, _request(review_id), response)

        payload, content_count = await _fetch_payload_and_content(engine, event.event_id)
        assert payload is not None and content_count == 1
        assert payload["cost_usd"] is None
        assert payload["cost_unpriced_reason"] == CostUnpricedReason.SCALE_TIER.value
        assert payload["service_tier"] == "scale"
        assert payload["billed_prompt_tokens"] == 2000
        assert payload["cache_write_tokens"] == 400
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_costable_flex_mismatch_persists_policy_cost(migrated_db: str) -> None:
    """A flex echo persists the 0.5x policy cost (the same figure the canonical
    outcome derives), JSONB round-tripped."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_installation_and_review(engine)
        response = _response(service_tier="flex")
        event = _event(review_id, response)
        await _make_persister(engine).persist(event, _request(review_id), response)

        payload, content_count = await _fetch_payload_and_content(engine, event.event_id)
        assert payload is not None and content_count == 1
        assert payload["cost_usd"] == pytest.approx(event.cost_usd)
        assert payload["cost_unpriced_reason"] is None
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fresh_guard_rejects_incomplete_context_and_rolls_back(migrated_db: str) -> None:
    """An echo-expecting host's fresh event without the billed count is
    rejected AND the freshly-inserted audit row rolls back (no orphan)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_installation_and_review(engine)
        response = _response(service_tier="default")
        event = _event(review_id, response, billed_prompt_tokens=None)
        with pytest.raises(AuditPersisterEventResponseFieldMismatchError):
            await _make_persister(engine).persist(event, _request(review_id), response)
        payload, content_count = await _fetch_payload_and_content(engine, event.event_id)
        assert payload is None and content_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_fresh_guard_rejects_cost_reason_divergence(migrated_db: str) -> None:
    """A fabricated cost on an unpriceable echo (valid coupling, WRONG
    classification vs the canonical outcome) is rejected + rolled back."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_installation_and_review(engine)
        response = _response(service_tier="scale")
        event = _event(review_id, response, cost_usd=0.01, cost_unpriced_reason=None)
        with pytest.raises(AuditPersisterEventResponseFieldMismatchError):
            await _make_persister(engine).persist(event, _request(review_id), response)
        payload, content_count = await _fetch_payload_and_content(engine, event.event_id)
        assert payload is None and content_count == 0
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_prefield_row_reemit_passes_absent_vs_null_normalization(migrated_db: str) -> None:
    """The checkpoint-resume-across-upgrade case: a stored PRE-FIELD row (the
    four keys ABSENT) re-emitted by a post-upgrade process (keys serialized as
    null) must be an idempotent no-op, not a spurious conflict."""
    from outrider.llm.anthropic_provider import (
        _ANTHROPIC_CONTRACT_DIGEST,
        _ANTHROPIC_PROFILE_ID,
    )
    from outrider.llm.pricing import compute_cost_usd

    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        review_id = await _seed_installation_and_review(engine)
        cost = float(
            compute_cost_usd(
                _ANTHROPIC_PROFILE_ID,
                "claude-haiku-4-5",
                input_tokens=100,
                cache_write_tokens=0,
                cache_read_tokens=0,
                output_tokens=50,
            )
        )
        response = LLMResponse(
            text="{}",
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
        event = LLMCallEvent(
            review_id=UUID(review_id),
            timestamp=datetime.now(UTC),
            model="claude-haiku-4-5",
            finish_reason="end_turn",
            node_id="triage",
            input_tokens=100,
            output_tokens=50,
            cached_tokens=0,
            cost_usd=cost,
            pricing_version=PRICING_VERSION,
            latency_ms=250,
            prompt_hash=_canonical_prompt_hash(system_prompt=_SYSTEM, user_prompt=_USER),
            cache_hit=False,
            context_summary=(),
            prompt_template_version="triage:1",
            system_prompt_hash=_canonical_system_prompt_hash(_SYSTEM),
            degraded_mode=False,
            profile_id=_ANTHROPIC_PROFILE_ID,
            reasoning_enabled=False,
            profile_contract_digest=_ANTHROPIC_CONTRACT_DIGEST,
        )
        # Simulate the pre-upgrade stored row: serialize, DROP the four keys
        # entirely (a pre-field process never wrote them), insert directly —
        # append-only forbids mutating a persister-written row into shape.
        prefield = {
            k: v
            for k, v in _serialize_event_payload(event).items()
            if k
            not in {
                "service_tier",
                "billed_prompt_tokens",
                "cache_write_tokens",
                "cost_unpriced_reason",
            }
        }
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO audit_events "
                    "(event_id, review_id, event_type, timestamp, payload, is_eval) "
                    "VALUES (:e, :r, 'llm_call', :ts, CAST(:p AS jsonb), false)"
                ),
                {
                    "e": event.event_id,
                    "r": UUID(review_id),
                    "ts": event.timestamp,
                    "p": json.dumps(prefield),
                },
            )
            await conn.execute(
                text(
                    "INSERT INTO llm_call_content "
                    "(event_id, installation_id, prompt, completion, is_eval, "
                    " retention_expires_at) "
                    "VALUES (:e, :i, :pr, :co, false, NOW() + INTERVAL '90 days')"
                ),
                {
                    "e": event.event_id,
                    "i": _INSTALLATION_ID,
                    "pr": _USER,
                    "co": "{}",
                },
            )
        # Post-upgrade re-emit of the SAME event: serializes the four keys as
        # null. Without the targeted normalization this raises
        # AuditPersisterIdempotencyConflict; with it, an idempotent no-op.
        await _make_persister(engine).persist(event, _request(review_id), response)
    finally:
        await engine.dispose()
