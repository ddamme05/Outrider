"""Integration tests for the dashboard read-API (`api/dashboard/reviews.py`).

Covers the increment-1 hard contract from `specs/2026-05-31-dashboard-v1.md`:
auth (401), `is_eval` default-exclude, the per-metric source contract
(LLM aggregates summed from `llm_call` rows; file/wall-clock read from
`synthesize_completed`; null-not-zero when synthesize never emitted), the
"never read `reviews.*` metric columns" guarantee (seeded with garbage to
prove it), status filtering, 404, and read-only (no mutation).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
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

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 4242


async def _seed_review(
    conn: Any,
    *,
    status: str,
    is_eval: bool,
    repo_id: int,
    llm_events: list[dict[str, Any]],
    synth: dict[str, Any] | None,
) -> UUID:
    """Insert one review + its audit events. The `reviews.*` metric columns
    are seeded to 999 (garbage) to prove the API never reads them.
    """
    result = await conn.execute(
        text(
            "INSERT INTO reviews ("
            "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
            "  files_examined, files_traced_beyond_diff, llm_calls_made, "
            "  total_input_tokens, total_output_tokens, total_cost_usd, "
            "  wall_clock_seconds, retention_expires_at"
            ") VALUES ("
            "  :iid, :repo, 1, 'sha1', :status, :is_eval, "
            "  999, 999, 999, 999, 999, 999, 999, NOW() + INTERVAL '90 days'"
            ") RETURNING id"
        ),
        {"iid": _INSTALLATION_ID, "repo": repo_id, "status": status, "is_eval": is_eval},
    )
    review_id = UUID(str(result.scalar_one()))

    async def _insert_event(event_type: str, payload: dict[str, Any]) -> None:
        await conn.execute(
            text(
                "INSERT INTO audit_events "
                "(event_id, review_id, event_type, timestamp, is_eval, payload) "
                "VALUES (:eid, :rid, :etype, NOW(), :is_eval, CAST(:payload AS jsonb))"
            ),
            {
                "eid": uuid4(),
                "rid": review_id,
                "etype": event_type,
                "is_eval": is_eval,
                "payload": json.dumps(payload),
            },
        )

    for ev in llm_events:
        await _insert_event("llm_call", ev)
    if synth is not None:
        await _insert_event("synthesize_completed", synth)
    return review_id


@pytest_asyncio.fixture
async def dashboard_client(
    migrated_db: str,
) -> AsyncGenerator[tuple[TestClient, dict[str, UUID], AsyncEngine]]:
    """Seed installation + reviews A/B/C and mount the read-API router.

    - A: completed, is_eval=False, 2 llm_call events + synthesize_completed.
    - B: running, is_eval=False, 1 llm_call event, NO synthesize_completed.
    - C: completed, is_eval=True (must be excluded from the default list).
    """
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

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
        review_a = await _seed_review(
            conn,
            status="completed",
            is_eval=False,
            repo_id=100,
            llm_events=[
                {"input_tokens": 100, "output_tokens": 40, "cost_usd": 0.02},
                {"input_tokens": 200, "output_tokens": 60, "cost_usd": 0.03},
            ],
            synth={
                "files_examined": 5,
                "files_traced_beyond_diff": 2,
                "wall_clock_seconds": 42.5,
            },
        )
        review_b = await _seed_review(
            conn,
            status="running",
            is_eval=False,
            repo_id=200,
            llm_events=[{"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.001}],
            synth=None,
        )
        review_c = await _seed_review(
            conn,
            status="completed",
            is_eval=True,
            repo_id=300,
            llm_events=[],
            synth=None,
        )

    app = FastAPI()
    app.include_router(reviews_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)

    ids = {"a": review_a, "b": review_b, "c": review_c}
    try:
        yield TestClient(app), ids, engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_list_excludes_eval_by_default(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    resp = client.get("/api/reviews", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    returned = {r["id"] for r in body["reviews"]}
    assert returned == {str(ids["a"]), str(ids["b"])}
    assert str(ids["c"]) not in returned
    assert body["total"] == 2


@pytest.mark.asyncio
async def test_list_include_eval_exposes_eval_rows(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    resp = client.get("/api/reviews", params={"include_eval": "true"}, headers=_AUTH)
    assert resp.status_code == 200
    returned = {r["id"] for r in resp.json()["reviews"]}
    assert str(ids["c"]) in returned


@pytest.mark.asyncio
async def test_metric_contract_from_audit_stream_not_reviews_columns(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    resp = client.get(f"/api/reviews/{ids['a']}", headers=_AUTH)
    assert resp.status_code == 200
    m = resp.json()["metrics"]
    # LLM aggregates summed from the 2 llm_call rows — NOT the seeded 999s.
    assert m["llm_calls_made"] == 2
    assert m["total_input_tokens"] == 300
    assert m["total_output_tokens"] == 100
    assert m["total_cost_usd"] == pytest.approx(0.05)
    # File/wall-clock read from the synthesize_completed payload.
    assert m["files_examined"] == 5
    assert m["files_traced_beyond_diff"] == 2
    assert m["wall_clock_seconds"] == pytest.approx(42.5)


@pytest.mark.asyncio
async def test_running_review_file_metrics_null_not_zero(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    resp = client.get(f"/api/reviews/{ids['b']}", headers=_AUTH)
    assert resp.status_code == 200
    m = resp.json()["metrics"]
    # No synthesize_completed -> file/wall-clock pending (None), NOT zero.
    assert m["files_examined"] is None
    assert m["files_traced_beyond_diff"] is None
    assert m["wall_clock_seconds"] is None
    # LLM sum still works for a review with >=1 call.
    assert m["llm_calls_made"] == 1
    assert m["total_input_tokens"] == 10


@pytest.mark.asyncio
async def test_status_filter(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    running = client.get("/api/reviews", params={"status": "running"}, headers=_AUTH)
    assert {r["id"] for r in running.json()["reviews"]} == {str(ids["b"])}
    completed = client.get("/api/reviews", params={"status": "completed"}, headers=_AUTH)
    # 'completed' default-excludes the is_eval=True review C -> only A.
    assert {r["id"] for r in completed.json()["reviews"]} == {str(ids["a"])}


@pytest.mark.asyncio
async def test_invalid_status_filter_is_422(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, _, _ = dashboard_client
    resp = client.get("/api/reviews", params={"status": "bogus"}, headers=_AUTH)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_auth_required(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = dashboard_client
    assert client.get("/api/reviews").status_code == 401
    assert client.get("/api/reviews", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get(f"/api/reviews/{ids['a']}").status_code == 401


@pytest.mark.asyncio
async def test_unknown_review_is_404(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, _, _ = dashboard_client
    resp = client.get(f"/api/reviews/{uuid4()}", headers=_AUTH)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_read_is_non_mutating(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A GET must not mutate (audit-append-only / read-only boundary):
    the review's updated_at and the audit-event count are unchanged.
    """
    client, ids, engine = dashboard_client
    async with engine.connect() as conn:
        before_updated = (
            await conn.execute(
                text("SELECT updated_at FROM reviews WHERE id = :id"),
                {"id": ids["a"]},
            )
        ).scalar_one()
        before_events = (
            await conn.execute(
                text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                {"id": ids["a"]},
            )
        ).scalar_one()

    assert client.get(f"/api/reviews/{ids['a']}", headers=_AUTH).status_code == 200
    assert client.get("/api/reviews", headers=_AUTH).status_code == 200

    async with engine.connect() as conn:
        after_updated = (
            await conn.execute(
                text("SELECT updated_at FROM reviews WHERE id = :id"),
                {"id": ids["a"]},
            )
        ).scalar_one()
        after_events = (
            await conn.execute(
                text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                {"id": ids["a"]},
            )
        ).scalar_one()

    assert after_updated == before_updated
    assert after_events == before_events
