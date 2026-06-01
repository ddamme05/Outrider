"""Integration tests for `GET /api/reviews/{id}/findings`.

Covers the source-split contract from `specs/2026-05-31-dashboard-v1.md`:
analyze-time content read from the `findings` table; `publish_destination`
joined from `PublishRoutingEvent` by `finding_id` (NOT the null
`findings.publish_destination` column), latest-routing-wins; auth; 404.
"""

from __future__ import annotations

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

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 5252
_DUMMY_HASH = "a" * 64


async def _insert_finding(
    conn: object,
    *,
    review_id: UUID,
    policy_version: str,
    finding_id: UUID,
    severity: str,
    evidence_tier: str,
    query_match_id: str | None,
    trace_path: list[str] | None,
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO findings ("
            "  finding_id, review_id, installation_id, policy_version, finding_type, "
            "  dimension, severity, evidence_tier, file_path, line_start, line_end, "
            "  title, description, evidence, suggested_fix, query_match_id, trace_path, "
            "  content_hash, retention_expires_at"
            ") VALUES ("
            "  :fid, :rid, :iid, :pv, 'sql_injection', 'security', :sev, :tier, "
            "  'app/db.py', 10, 12, :title, 'desc', 'evidence-snippet', 'fix it', "
            "  :qmid, CAST(:tp AS jsonb), :ch, NOW() + INTERVAL '90 days'"
            ")"
        ),
        {
            "fid": finding_id,
            "rid": review_id,
            "iid": _INSTALLATION_ID,
            "pv": policy_version,
            "sev": severity,
            "tier": evidence_tier,
            "title": f"finding-{severity}",
            "qmid": query_match_id,
            "tp": None if trace_path is None else json.dumps(trace_path),
            "ch": _DUMMY_HASH,
        },
    )


async def _insert_publish_routing(
    conn: object, *, review_id: UUID, finding_id: UUID, destination: str
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'publish_routing', NOW(), false, CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "payload": json.dumps({"finding_id": str(finding_id), "destination": destination}),
        },
    )


@pytest_asyncio.fixture
async def findings_client(
    migrated_db: str,
) -> AsyncGenerator[tuple[TestClient, dict[str, UUID], AsyncEngine]]:
    """Seed a review with two findings (F1 routed inline; F2 unrouted)."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    f1, f2 = uuid4(), uuid4()
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
        review_id = UUID(
            str(
                (
                    await conn.execute(
                        text(
                            "INSERT INTO reviews ("
                            "  installation_id, repo_id, pr_number, head_sha, status, "
                            "  files_examined, files_traced_beyond_diff, llm_calls_made, "
                            "  total_input_tokens, total_output_tokens, total_cost_usd, "
                            "  wall_clock_seconds, retention_expires_at"
                            ") VALUES ("
                            "  :iid, 100, 7, 'sha7', 'completed', 0, 0, 0, 0, 0, 0, 0, "
                            "  NOW() + INTERVAL '90 days'"
                            ") RETURNING id"
                        ),
                        {"iid": _INSTALLATION_ID},
                    )
                ).scalar_one()
            )
        )
        policy_version = (
            await conn.execute(text("SELECT version FROM severity_policies LIMIT 1"))
        ).scalar_one()
        await _insert_finding(
            conn,
            review_id=review_id,
            policy_version=policy_version,
            finding_id=f1,
            severity="high",
            evidence_tier="observed",
            query_match_id="q-match-1",
            trace_path=None,
        )
        await _insert_finding(
            conn,
            review_id=review_id,
            policy_version=policy_version,
            finding_id=f2,
            severity="medium",
            evidence_tier="inferred",
            query_match_id=None,
            trace_path=["app/db.py::run_query"],
        )
        # F1 routed inline; F2 left unrouted (publish_destination -> None).
        await _insert_publish_routing(
            conn, review_id=review_id, finding_id=f1, destination="inline_comment"
        )

    app = FastAPI()
    app.include_router(reviews_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)

    try:
        yield TestClient(app), {"review": review_id, "f1": f1, "f2": f2}, engine
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_findings_content_and_proof(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = findings_client
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    assert resp.status_code == 200
    findings = {f["finding_id"]: f for f in resp.json()["findings"]}
    assert set(findings) == {str(ids["f1"]), str(ids["f2"])}

    f1 = findings[str(ids["f1"])]
    assert f1["severity"] == "high"
    assert f1["evidence_tier"] == "observed"
    assert f1["query_match_id"] == "q-match-1"
    assert f1["trace_path"] is None
    assert f1["file_path"] == "app/db.py"
    assert f1["line_start"] == 10

    f2 = findings[str(ids["f2"])]
    assert f2["evidence_tier"] == "inferred"
    assert f2["query_match_id"] is None
    assert f2["trace_path"] == ["app/db.py::run_query"]


@pytest.mark.asyncio
async def test_publish_destination_joined_from_event_not_column(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = findings_client
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    findings = {f["finding_id"]: f for f in resp.json()["findings"]}
    # F1 has a PublishRoutingEvent -> inline_comment; the findings.* column is null.
    assert findings[str(ids["f1"])]["publish_destination"] == "inline_comment"
    # F2 was never routed -> None (not yet published).
    assert findings[str(ids["f2"])]["publish_destination"] is None


@pytest.mark.asyncio
async def test_publish_destination_latest_routing_wins(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A re-route lands a second PublishRoutingEvent; the latest wins."""
    client, ids, engine = findings_client
    async with engine.begin() as conn:
        await _insert_publish_routing(
            conn, review_id=ids["review"], finding_id=ids["f1"], destination="review_body"
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    findings = {f["finding_id"]: f for f in resp.json()["findings"]}
    assert findings[str(ids["f1"])]["publish_destination"] == "review_body"


@pytest.mark.asyncio
async def test_findings_auth_required(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = findings_client
    assert client.get(f"/api/reviews/{ids['review']}/findings").status_code == 401


@pytest.mark.asyncio
async def test_findings_unknown_review_404(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, _, _ = findings_client
    assert client.get(f"/api/reviews/{uuid4()}/findings", headers=_AUTH).status_code == 404
