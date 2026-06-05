"""Integration tests for `GET /api/reviews/{id}/replay`.

The endpoint is a thin wrapper over `audit/replay.py::AuditReplayer`. A valid
metadata-only stream — a phase-pair wrapping an `llm_call` + a `finding` event,
no review/findings/content rows — reconstructs and asserts cleanly, so the
verdict is `replay_equivalent=True`. Plus 404 (no audit rows) + auth. The
event factories mirror `tests/integration/test_audit_replay.py`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.dashboard import reviews_router
from outrider.audit.events import (
    AuditEventBase,
    FindingEvent,
    LLMCallEvent,
    ReplayVerdictEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingSeverity, FindingType
from outrider.schemas import ReviewDimension

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}


def _finding_event(review_id: UUID) -> FindingEvent:
    return FindingEvent(
        review_id=review_id,
        finding_id=uuid4(),
        finding_type=FindingType.SQL_INJECTION,
        severity=FindingSeverity.CRITICAL,
        file_path="src/app/models.py",
        line_start=10,
        line_end=20,
        dimension=ReviewDimension.SECURITY,
        finding_content_hash=compute_finding_content_hash(
            "src/app/models.py",
            line_start=10,
            line_end=20,
            finding_type=FindingType.SQL_INJECTION,
        ),
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
        policy_version="1.0.0",
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
    )


def _llm_call_event(review_id: UUID) -> LLMCallEvent:
    return LLMCallEvent(
        review_id=review_id,
        model="claude-sonnet-4-5",
        node_id="analyze",
        input_tokens=100,
        output_tokens=50,
        cached_tokens=0,
        cost_usd=0.01,
        pricing_version="v1",
        latency_ms=1200,
        prompt_hash=hashlib.sha256(b"prompt").hexdigest(),
        cache_hit=False,
        context_summary=(),
        prompt_template_version="analyze.v1",
        system_prompt_hash=hashlib.sha256(b"sys").hexdigest(),
        degraded_mode=False,
    )


async def _insert_event(engine: AsyncEngine, event: AuditEventBase) -> None:
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, payload) VALUES (:event_id, :review_id, :event_type, :phase_key, "
                ":timestamp, CAST(:payload AS jsonb))"
            ),
            {
                "event_id": event.event_id,
                "review_id": event.review_id,
                "event_type": event.event_type,
                "phase_key": phase_key,
                "timestamp": event.timestamp,
                "payload": json.dumps(payload),
            },
        )


@pytest_asyncio.fixture
async def replay_client(
    migrated_db: str,
) -> AsyncGenerator[tuple[TestClient, UUID, AsyncEngine]]:
    """Seed a valid metadata-only stream (4 events, 1 finding) + mount the router."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    review_id = uuid4()
    llm_call = _llm_call_event(review_id)
    finding = _finding_event(review_id)
    # analyze phase-pair wrapping the work events => phase-bounded, monotonic,
    # metadata-only (no review/content rows) — reconstructs + asserts cleanly.
    events: list[AuditEventBase] = [
        ReviewPhaseEvent(
            review_id=review_id,
            phase_id="analyze:0",
            node_id="analyze",
            marker="start",
            phase_key=None,
        ),
        llm_call,
        finding,
        ReviewPhaseEvent(
            review_id=review_id,
            phase_id="analyze:0",
            node_id="analyze",
            marker="end",
            phase_key=None,
        ),
    ]
    for event in events:
        await _insert_event(engine, event)

    app = FastAPI()
    app.include_router(reviews_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)

    try:
        yield TestClient(app), review_id, engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_replay_verdict_equivalent(
    replay_client: tuple[TestClient, UUID, AsyncEngine],
) -> None:
    client, review_id, _ = replay_client
    resp = client.get(f"/api/reviews/{review_id}/replay", headers=_AUTH)
    assert resp.status_code == 200
    v = resp.json()
    assert v["replay_equivalent"] is True
    assert v["mode"] == "metadata_only"
    assert v["event_count"] == 4
    assert v["finding_count"] == 1
    assert v["orphan_finding_count"] == 0
    assert v["reason"] is None


@pytest.mark.asyncio
async def test_replay_event_count_excludes_projected_verdict(
    replay_client: tuple[TestClient, UUID, AsyncEngine],
) -> None:
    # Once the background projector appends a `replay_verdict` event, this full-stream
    # reconstruct would otherwise count it (event_count 4 -> 5), diverging from the PERSISTED
    # verdict's count (the judged non-verdict prefix). The verdict is replay METADATA, not
    # review work, so it must be excluded. Revert-the-fold: drop the isinstance filter and the
    # count becomes 5.
    client, review_id, engine = replay_client
    await _insert_event(
        engine,
        ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=True,
            mode="metadata_only",
            event_count=4,
            finding_count=1,
            orphan_finding_count=0,
            target_max_sequence_number=4,
        ),
    )
    resp = client.get(f"/api/reviews/{review_id}/replay", headers=_AUTH)
    assert resp.status_code == 200
    v = resp.json()
    assert v["event_count"] == 4  # the 4 work events, NOT 5 — the verdict row is excluded
    assert v["replay_equivalent"] is True  # verdict is phase-unbounded-exempt; assert still passes


@pytest.mark.asyncio
async def test_replay_unknown_review_404(
    replay_client: tuple[TestClient, UUID, AsyncEngine],
) -> None:
    """No audit rows for the id -> ReplayReviewNotFoundError -> 404."""
    client, _, _ = replay_client
    assert client.get(f"/api/reviews/{uuid4()}/replay", headers=_AUTH).status_code == 404


@pytest.mark.asyncio
async def test_replay_auth_required(
    replay_client: tuple[TestClient, UUID, AsyncEngine],
) -> None:
    client, review_id, _ = replay_client
    assert client.get(f"/api/reviews/{review_id}/replay").status_code == 401
