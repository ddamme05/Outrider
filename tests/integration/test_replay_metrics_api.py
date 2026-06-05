"""Integration tests for the Replay-% aggregation endpoint (`GET /api/metrics/replay`).

The stage-3 sibling of `/api/metrics` (`DECISIONS.md#039`): reads the PERSISTED
`replay_verdict` events the background projector appends (`sweep/replay_verdict.py`) and
returns `equivalent / total` bucketed by `reviews.completed_at`. Proves, against real seeded
rows:

- equivalent / total per the current window + the period-over-period deltas;
- ONLY reviews carrying a persisted verdict are in the denominator — a completed-but-pending
  review (no verdict yet) is excluded, never assumed equivalent (the projector-lag guard);
- bucketing is by `reviews.completed_at`, NOT the verdict EVENT's timestamp — a review that
  completed in the prior window stays in the prior window even when its verdict was projected
  (event timestamp) inside the current window;
- `is_eval` excluded by default, exposed with `?include_eval=true`, and a verdict whose
  `is_eval` column DISAGREES with its review is drift, rejected in both directions (FUP-130);
- sparse / empty windows render honest zeros; 401 without the admin key.

Verdict payloads are built from the real `ReplayVerdictEvent` (faithful shape, not a
hand-typed dict); only `replay_equivalent` is read by the endpoint, but the envelope
validators keep the fixture honest.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.dashboard import metrics_router
from outrider.audit.events import ReplayVerdictEvent

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 5151


async def _seed_completed_review(
    conn: Any, *, is_eval: bool, completed_age_days: float, repo_id: int
) -> UUID:
    """A `status='completed'` review; `completed_at` (the bucket dimension) is `age` days old."""
    result = await conn.execute(
        text(
            "INSERT INTO reviews ("
            "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
            "  created_at, completed_at, retention_expires_at"
            ") VALUES ("
            "  :iid, :repo, 1, 'sha1', 'completed', :is_eval, "
            "  NOW() - (:age * INTERVAL '1 day'), NOW() - (:age * INTERVAL '1 day'), "
            "  NOW() + INTERVAL '90 days'"
            ") RETURNING id"
        ),
        {"iid": _INSTALLATION_ID, "repo": repo_id, "is_eval": is_eval, "age": completed_age_days},
    )
    return UUID(str(result.scalar_one()))


async def _seed_verdict(
    conn: Any,
    review_id: UUID,
    *,
    replay_equivalent: bool,
    event_age_days: float,
    is_eval: bool,
) -> None:
    """Append a faithful `replay_verdict` event. `event_age_days` controls the EVENT timestamp
    (deliberately decoupled from the review's `completed_at` so the bucket-dimension test can
    prove the endpoint ignores it)."""
    if replay_equivalent:
        event = ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=True,
            mode="full",
            event_count=4,
            finding_count=0,
            orphan_finding_count=0,
            target_max_sequence_number=4,
            is_eval=is_eval,
        )
    else:
        event = ReplayVerdictEvent(
            review_id=review_id,
            replay_equivalent=False,
            reason="unterminated phase",
            target_max_sequence_number=4,
            is_eval=is_eval,
        )
    payload = event.model_dump(mode="json", exclude={"sequence_number"})
    await conn.execute(
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'replay_verdict', NOW() - (:age * INTERVAL '1 day'), :is_eval, "
            "CAST(:payload AS jsonb))"
        ),
        {
            "eid": event.event_id,
            "rid": review_id,
            "is_eval": is_eval,
            "age": event_age_days,
            "payload": json.dumps(payload),
        },
    )


async def _seed_installation(conn: Any) -> None:
    await conn.execute(
        text(
            "INSERT INTO installations "
            "(installation_id, app_slug, account_id, account_login, "
            " account_type, permissions_at_install) "
            "VALUES (:id, 'test-app', 1, 'octocat', 'User', '{}'::jsonb)"
        ),
        {"id": _INSTALLATION_ID},
    )


def _mount(session_factory: Any) -> TestClient:
    app = FastAPI()
    app.include_router(metrics_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    return TestClient(app)


@pytest_asyncio.fixture
async def replay_client(migrated_db: str) -> AsyncGenerator[TestClient]:
    """A 7d-window scenario exercising every contract.

    Current window (is_eval=False default): R1 (equiv) + R2 (inequiv). R3 is completed but
    has NO verdict (projector-pending → excluded). R4 is an eval verdict (excluded by default).
    Previous window: R5 (equiv, completed age 10). R6 completed age 10 (prior) but its verdict
    EVENT is recent (age 1) — it must stay in the PRIOR window (bucket by completed_at).
    """
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await _seed_installation(conn)
        r1 = await _seed_completed_review(conn, is_eval=False, completed_age_days=1, repo_id=100)
        r2 = await _seed_completed_review(conn, is_eval=False, completed_age_days=2, repo_id=200)
        await _seed_completed_review(conn, is_eval=False, completed_age_days=1, repo_id=300)  # R3
        r4 = await _seed_completed_review(conn, is_eval=True, completed_age_days=1, repo_id=400)
        r5 = await _seed_completed_review(conn, is_eval=False, completed_age_days=10, repo_id=500)
        r6 = await _seed_completed_review(conn, is_eval=False, completed_age_days=10, repo_id=600)

        await _seed_verdict(conn, r1, replay_equivalent=True, event_age_days=1, is_eval=False)
        await _seed_verdict(conn, r2, replay_equivalent=False, event_age_days=2, is_eval=False)
        # R3: NO verdict — completed but projector-pending.
        await _seed_verdict(conn, r4, replay_equivalent=True, event_age_days=1, is_eval=True)
        await _seed_verdict(conn, r5, replay_equivalent=True, event_age_days=10, is_eval=False)
        # R6: completed in the PRIOR window, but verdict EVENT projected recently (age 1).
        await _seed_verdict(conn, r6, replay_equivalent=True, event_age_days=1, is_eval=False)
    try:
        yield _mount(session_factory)
    finally:
        await engine.dispose()


def _get(client: TestClient, **params: str) -> dict[str, Any]:
    resp = client.get("/api/metrics/replay", params=params, headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


@pytest.mark.asyncio
async def test_equivalent_and_total_counts(replay_client: TestClient) -> None:
    body = _get(replay_client, window="7d")
    cur = body["deltas"]["current"]
    # R1 (equiv) + R2 (inequiv): equivalent=1, total=2. R3 (no verdict) + R4 (eval) excluded.
    assert cur["equivalent"] == 1
    assert cur["total"] == 2
    # Buckets sum to the totals (the series cannot drift from `current`).
    assert sum(b["equivalent"] for b in body["buckets"]) == 1
    assert sum(b["total"] for b in body["buckets"]) == 2


@pytest.mark.asyncio
async def test_pending_review_excluded_from_denominator(replay_client: TestClient) -> None:
    # R3 is completed but has NO persisted verdict. It must NOT be in the denominator — a
    # projector-pending review is never assumed equivalent. If it leaked, total would be 3.
    body = _get(replay_client, window="7d")
    assert body["deltas"]["current"]["total"] == 2  # R1 + R2 only, NOT R3


@pytest.mark.asyncio
async def test_bucketed_by_completed_at_not_event_timestamp(replay_client: TestClient) -> None:
    # R6 completed in the PRIOR window (age 10) but its verdict EVENT was projected recently
    # (age 1, in the current window). Bucketing is by `reviews.completed_at`, so R6 belongs to
    # PREVIOUS, never CURRENT — even though its audit row is recent. If the endpoint (wrongly)
    # bucketed by the event timestamp, R6 would inflate current.total to 3.
    body = _get(replay_client, window="7d")
    assert body["deltas"]["current"]["total"] == 2  # R6 did NOT leak into current
    # Previous window: R5 (age 10) + R6 (age 10) — both equivalent.
    assert body["deltas"]["previous"]["equivalent"] == 2
    assert body["deltas"]["previous"]["total"] == 2


@pytest.mark.asyncio
async def test_is_eval_excluded_by_default_and_exposed_with_flag(replay_client: TestClient) -> None:
    default = _get(replay_client, window="7d")
    assert default["deltas"]["current"]["total"] == 2  # eval R4 excluded
    assert default["deltas"]["current"]["equivalent"] == 1

    with_eval = _get(replay_client, window="7d", include_eval="true")
    # R4 (eval, equivalent) now included: equivalent 1→2, total 2→3.
    assert with_eval["deltas"]["current"]["equivalent"] == 2
    assert with_eval["deltas"]["current"]["total"] == 3


@pytest.mark.asyncio
async def test_30d_window_pulls_prior_into_current(replay_client: TestClient) -> None:
    # The 30d window pulls the age-10 R5 + R6 into the CURRENT window — proves window selection
    # actually shifts start/prev_start (only 7d is otherwise exercised).
    body = _get(replay_client, window="30d")
    assert body["granularity"] == "day"
    assert body["deltas"]["current"]["total"] == 4  # R1, R2, R5, R6 (eval R4 still excluded)
    assert body["deltas"]["current"]["equivalent"] == 3  # R1, R5, R6 (R2 inequiv)


@pytest.mark.asyncio
async def test_24h_window_uses_hourly_buckets(replay_client: TestClient) -> None:
    body = _get(replay_client, window="24h")
    assert body["granularity"] == "hour"
    assert 24 <= len(body["buckets"]) <= 26


@pytest.mark.asyncio
async def test_sparse_window_renders_zeros(replay_client: TestClient) -> None:
    body = _get(replay_client, window="7d")
    assert len(body["buckets"]) >= 7
    zero_days = [b for b in body["buckets"] if b["total"] == 0]
    assert zero_days, "sparse days must be honest zeros, not omitted"
    for b in zero_days:
        assert b["equivalent"] == 0  # no equivalent without a total


@pytest_asyncio.fixture
async def empty_replay_client(migrated_db: str) -> AsyncGenerator[TestClient]:
    engine = create_async_engine(migrated_db, hide_parameters=True)
    try:
        yield _mount(async_sessionmaker(engine, expire_on_commit=False))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_empty_window_renders_all_zeros(empty_replay_client: TestClient) -> None:
    resp = empty_replay_client.get("/api/metrics/replay", params={"window": "7d"}, headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Zero-row safety: the `.one()` (prev totals) + empty-dict bucket path degrade to zeros.
    assert body["deltas"]["current"] == {"equivalent": 0, "total": 0}
    assert body["deltas"]["previous"] == {"equivalent": 0, "total": 0}
    assert body["buckets"]  # honest zero-fill, not an empty list
    assert all(b["equivalent"] == 0 and b["total"] == 0 for b in body["buckets"])


def test_requires_admin_key(replay_client: TestClient) -> None:
    resp = replay_client.get("/api/metrics/replay")
    assert resp.status_code == 401


@pytest_asyncio.fixture
async def drift_replay_client(migrated_db: str) -> AsyncGenerator[TestClient]:
    """A verdict whose `is_eval` COLUMN disagrees with its review — drift the FUP-130 equality
    predicate must reject in BOTH directions (a one-sided `Review.is_eval` filter would leak the
    prod-review/eval-verdict case)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await _seed_installation(conn)
        # Production review + an AGREEING verdict, and a SECOND review with a DRIFT verdict
        # (verdict is_eval=True on a prod review). One verdict per review (unique index), so the
        # drift case needs its own review.
        prod_ok = await _seed_completed_review(
            conn, is_eval=False, completed_age_days=1, repo_id=100
        )
        prod_drift = await _seed_completed_review(
            conn, is_eval=False, completed_age_days=1, repo_id=200
        )
        await _seed_verdict(conn, prod_ok, replay_equivalent=True, event_age_days=1, is_eval=False)
        await _seed_verdict(
            conn, prod_drift, replay_equivalent=True, event_age_days=1, is_eval=True
        )  # DRIFT
    try:
        yield _mount(session_factory)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_is_eval_drift_verdict_excluded(drift_replay_client: TestClient) -> None:
    # The drift verdict (prod review, eval-labeled event) must NOT count under production scope —
    # `AuditEvent.is_eval == Review.is_eval` rejects it. Only the agreeing prod verdict counts.
    body = _get(drift_replay_client, window="7d")
    assert body["deltas"]["current"]["total"] == 1  # prod_ok only; the drift verdict excluded
    assert body["deltas"]["current"]["equivalent"] == 1
    # Even with include_eval, the drift verdict (review is_eval=False, event is_eval=True) stays
    # excluded — it agrees with NO review.
    full = _get(drift_replay_client, window="7d", include_eval="true")
    assert full["deltas"]["current"]["total"] == 1
