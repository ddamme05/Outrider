"""Integration tests for `GET /api/reviews/{review_id}/events` (FUP-133).

The events endpoint returns a review's full audit stream as the typed `AuditEvent`
union, ordered by `sequence_number`, reconstructed through the SHARED
`reconstruct_event_from_row` helper — so it inherits replay's historical tolerance
(DECISIONS.md#032) and row-consistency check. These tests seed FULL, valid event
payloads (unlike the partial JSONB-extraction rows in test_dashboard_reviews_api.py)
because the endpoint deserializes each row into a real event.
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
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from outrider.api.dashboard import reviews_router
from outrider.audit.events import (
    AuditEventBase,
    FindingEvent,
    ReviewPhaseEvent,
    compute_finding_content_hash,
)
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.schemas import ReviewDimension

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 7777


def _phase_event(review_id: UUID, marker: str, *, is_eval: bool = False) -> ReviewPhaseEvent:
    return ReviewPhaseEvent(
        review_id=review_id,
        phase_id="analyze:0",
        node_id="analyze",
        marker=marker,  # type: ignore[arg-type]
        phase_key=None,
        is_eval=is_eval,
    )


def _finding_event(review_id: UUID, *, is_eval: bool = False) -> FindingEvent:
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
            file_path="src/app/models.py",
            line_start=10,
            line_end=20,
            finding_type=FindingType.SQL_INJECTION,
        ),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version="1.0.0",
        proposal_hash=hashlib.sha256(b"proposal").hexdigest(),
        is_eval=is_eval,
    )


async def _seed_installation(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations (installation_id, app_slug, account_id, "
                "account_login, account_type, permissions_at_install) "
                "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb) "
                "ON CONFLICT (installation_id) DO NOTHING"
            ),
            {"id": _INSTALLATION_ID},
        )


async def _insert_event(
    engine: AsyncEngine, event: AuditEventBase, *, drop: set[str] | None = None
) -> None:
    """Insert one audit row from a real event. The mirrored base columns are taken
    from the event so they agree with the payload (`_verify_row_consistent`). `drop`
    removes payload keys (to simulate a pre-field historical row).
    """
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    for field in drop or set():
        payload.pop(field, None)
    phase_key = event.phase_key if isinstance(event, ReviewPhaseEvent) else None
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, is_eval, payload) VALUES (:event_id, :review_id, :event_type, "
                ":phase_key, :timestamp, :is_eval, CAST(:payload AS jsonb))"
            ),
            {
                "event_id": event.event_id,
                "review_id": event.review_id,
                "event_type": event.event_type,
                "phase_key": phase_key,
                "timestamp": event.timestamp,
                "is_eval": event.is_eval,
                "payload": json.dumps(payload),
            },
        )


async def _seed_review(engine: AsyncEngine, review_id: UUID, *, is_eval: bool = False) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO reviews (id, installation_id, repo_id, pr_number, head_sha, "
                "status, is_eval, files_examined, files_traced_beyond_diff, llm_calls_made, "
                "total_input_tokens, total_output_tokens, total_cost_usd, wall_clock_seconds, "
                "retention_expires_at) VALUES (:id, :iid, 100, 1, 'sha1', 'completed', :ie, "
                "1, 0, 1, 100, 50, 0.01, 1.5, NOW() + INTERVAL '180 days')"
            ),
            {"id": review_id, "iid": _INSTALLATION_ID, "ie": is_eval},
        )


@pytest_asyncio.fixture
async def client(migrated_db: str) -> AsyncGenerator[tuple[TestClient, AsyncEngine]]:
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    await _seed_installation(engine)
    app = FastAPI()
    app.include_router(reviews_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    try:
        yield TestClient(app), engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_events_returns_ordered_typed_stream(
    client: tuple[TestClient, AsyncEngine],
) -> None:
    api, engine = client
    review_id = uuid4()
    await _seed_review(engine, review_id)
    finding = _finding_event(review_id)
    await _insert_event(engine, _phase_event(review_id, "start"))
    await _insert_event(engine, finding)
    await _insert_event(engine, _phase_event(review_id, "end"))

    resp = api.get(f"/api/reviews/{review_id}/events", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    # Ordered by sequence_number (insert order), typed via the discriminator.
    types = [e["event_type"] for e in body["events"]]
    assert types == ["review_phase", "finding", "review_phase"]
    # Typed finding fields are present (full union exposed, Option A).
    fe = body["events"][1]
    assert fe["finding_type"] == "sql_injection"
    assert fe["evidence_tier"] == "judged"


@pytest.mark.asyncio
async def test_events_404_when_absent(client: tuple[TestClient, AsyncEngine]) -> None:
    api, _ = client
    assert api.get(f"/api/reviews/{uuid4()}/events", headers=_AUTH).status_code == 404


@pytest.mark.asyncio
async def test_events_excludes_divergent_is_eval_event(
    client: tuple[TestClient, AsyncEngine],
) -> None:
    """A divergent `is_eval=True` event on a production (is_eval=False) review is
    excluded from the explorer — FUP-130 read-side `is_eval` predicate. The
    events firehose was the broadest unguarded leak (every event type, no
    is_eval filter); the predicate scopes the stream to the review's own
    is_eval, so a divergent eval event can't surface on a production review."""
    api, engine = client
    review_id = uuid4()
    await _seed_review(engine, review_id, is_eval=False)
    await _insert_event(engine, _phase_event(review_id, "start", is_eval=False))
    # A divergent eval finding event sneaks onto the production review's stream.
    await _insert_event(engine, _finding_event(review_id, is_eval=True))

    resp = api.get(f"/api/reviews/{review_id}/events", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    # Only the is_eval=False phase event; the divergent eval finding is filtered.
    assert body["total"] == 1
    assert [e["event_type"] for e in body["events"]] == ["review_phase"]


@pytest.mark.asyncio
async def test_events_auth_required(client: tuple[TestClient, AsyncEngine]) -> None:
    api, _ = client
    assert api.get(f"/api/reviews/{uuid4()}/events").status_code == 401


@pytest.mark.asyncio
async def test_events_is_eval_review_reachable_by_id(
    client: tuple[TestClient, AsyncEngine],
) -> None:
    """Parity with the detail endpoint: a by-id events fetch is NOT is_eval-filtered."""
    api, engine = client
    review_id = uuid4()
    await _seed_review(engine, review_id, is_eval=True)
    await _insert_event(engine, _finding_event(review_id, is_eval=True))
    resp = api.get(f"/api/reviews/{review_id}/events", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


@pytest.mark.asyncio
async def test_events_historical_finding_missing_proposal_hash_returns_200(
    client: tuple[TestClient, AsyncEngine],
) -> None:
    """FUP-136 regression ON THIS ENDPOINT: a pre-#025 finding row (no
    proposal_hash) must reconstruct via the shared helper, not 500. Proves the
    endpoint uses reconstruct_event_from_row, not raw AuditEventAdapter.
    """
    api, engine = client
    review_id = uuid4()
    await _seed_review(engine, review_id)
    await _insert_event(engine, _finding_event(review_id), drop={"proposal_hash"})

    resp = api.get(f"/api/reviews/{review_id}/events", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["events"][0]["proposal_hash"] == "0" * 64


@pytest.mark.asyncio
async def test_events_row_payload_drift_is_loud(
    client: tuple[TestClient, AsyncEngine],
) -> None:
    """A row whose mirrored review_id column disagrees with its payload must raise
    via _verify_row_consistent (structured 500), never return a silent mismatched
    event. Proves the endpoint runs the row-consistency check.
    """
    api, engine = client
    review_id = uuid4()
    await _seed_review(engine, review_id)
    # Seed ONE drifted row: the review_id COLUMN matches the query surface, but the
    # payload's review_id is a different uuid — exactly the row/payload drift
    # _verify_row_consistent guards.
    finding = _finding_event(review_id)
    payload = finding.model_dump(mode="json", exclude={"sequence_number"})
    payload["review_id"] = str(uuid4())  # drift: payload disagrees with the column
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events (event_id, review_id, event_type, phase_key, "
                "timestamp, is_eval, payload) VALUES (:eid, :rid, :et, NULL, :ts, false, "
                "CAST(:p AS jsonb))"
            ),
            {
                "eid": uuid4(),
                "rid": review_id,  # column = query surface
                "et": "finding",
                "ts": finding.timestamp,
                "p": json.dumps(payload),
            },
        )

    resp = api.get(f"/api/reviews/{review_id}/events", headers=_AUTH)
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == "audit_row_inconsistent"
