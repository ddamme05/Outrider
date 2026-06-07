"""Integration tests for `GET /api/reviews/{id}/agent-view` (feature 3 / S2).

The read-only structured agent channel. Covers: separate-token scope (the agent
key works, the ADMIN key is rejected, no key 401, the surface is disabled when
OUTRIDER_AGENT_API_KEY is unset); the full JSON shape; pr_url derivation;
hitl_gated + reviewer_decision (incl. decided_at); publish_event; the omitted
V1 fields (github_comment_url / suggested_patch); 404. Auth is the load-bearing
part — agents must NEVER hold the admin key (which can POST /decide).
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

from outrider.api.dashboard import agent_view_router

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AGENT_KEY = "test-agent-key"  # noqa: S105
_AGENT_AUTH = {"Authorization": f"Bearer {_AGENT_KEY}"}
_ADMIN_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 6363
_REPO_ID = 100
_REPO_FULL_NAME = "octocat/demo"
_PR_NUMBER = 7
_DUMMY_HASH = "a" * 64


async def _insert_finding(
    conn: object, *, review_id: UUID, policy_version: str, finding_id: UUID, severity: str
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO findings ("
            "  finding_id, review_id, installation_id, policy_version, finding_type, "
            "  dimension, severity, evidence_tier, file_path, line_start, line_end, "
            "  title, description, evidence, suggested_fix, query_match_id, trace_path, "
            "  content_hash, is_eval, retention_expires_at"
            ") VALUES ("
            "  :fid, :rid, :iid, :pv, 'sql_injection', 'security', :sev, 'judged', "
            "  'app/db.py', 10, 12, :title, 'a finding description', 'evidence', 'fix it', "
            "  NULL, NULL, :ch, false, NOW() + INTERVAL '90 days'"
            ")"
        ),
        {
            "fid": finding_id,
            "rid": review_id,
            "iid": _INSTALLATION_ID,
            "pv": policy_version,
            "sev": severity,
            "title": f"finding-{severity}",
            "ch": _DUMMY_HASH,
        },
    )


async def _insert_audit_event(
    conn: object, *, review_id: UUID, event_type: str, payload: dict[str, object]
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, :etype, NOW(), false, CAST(:payload AS jsonb))"
        ),
        {"eid": uuid4(), "rid": review_id, "etype": event_type, "payload": json.dumps(payload)},
    )


@pytest_asyncio.fixture
async def agent_client(
    migrated_db: str,
) -> AsyncGenerator[tuple[TestClient, dict[str, UUID], AsyncEngine]]:
    """Seed a completed review: one gated+approved CRITICAL finding, one non-gated
    MEDIUM finding, an InstallationRepository (for pr_url), a PublishEvent, and a
    HITLDecisionEvent."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    f_crit, f_med = uuid4(), uuid4()
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
        await conn.execute(
            text(
                "INSERT INTO installation_repositories "
                "(installation_id, repo_id, repo_full_name, added_at) "
                "VALUES (:iid, :rid, :name, NOW())"
            ),
            {"iid": _INSTALLATION_ID, "rid": _REPO_ID, "name": _REPO_FULL_NAME},
        )
        review_id = UUID(
            str(
                (
                    await conn.execute(
                        text(
                            "INSERT INTO reviews ("
                            "  installation_id, repo_id, pr_number, head_sha, status, "
                            "  hitl_request, retention_expires_at"
                            ") VALUES ("
                            "  :iid, :repo, :pr, 'sha7', 'completed', "
                            "  CAST(:hitl AS jsonb), NOW() + INTERVAL '90 days'"
                            ") RETURNING id"
                        ),
                        {
                            "iid": _INSTALLATION_ID,
                            "repo": _REPO_ID,
                            "pr": _PR_NUMBER,
                            "hitl": json.dumps({"findings_requiring_approval": [str(f_crit)]}),
                        },
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
            finding_id=f_crit,
            severity="critical",
        )
        await _insert_finding(
            conn,
            review_id=review_id,
            policy_version=policy_version,
            finding_id=f_med,
            severity="medium",
        )
        # A FindingEvent carrying policy_version (the per-review snapshot source,
        # DECISIONS.md#028) — deduped against f_crit's content row, so it does not
        # add a redacted stub; it just supplies the endpoint's policy_version read.
        await _insert_audit_event(
            conn,
            review_id=review_id,
            event_type="finding",
            payload={"finding_id": str(f_crit), "policy_version": policy_version},
        )
        await _insert_audit_event(
            conn,
            review_id=review_id,
            event_type="publish",
            payload={"github_review_id": 4242, "review_status": "COMMENT", "comments_posted": 1},
        )
        await _insert_audit_event(
            conn,
            review_id=review_id,
            event_type="hitl_decision",
            payload={
                "reviewer_id": "admin",
                "decided_at": "2026-06-06T19:10:24+00:00",
                "annotation": None,
                "decisions": [
                    {
                        "finding_id": str(f_crit),
                        "outcome": "approve",
                        "reason": "operator-approved",
                        "original_severity": None,
                        "override_severity": None,
                    }
                ],
            },
        )

    app = FastAPI()
    app.include_router(agent_view_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    app.state.agent_api_key = SecretStr(_AGENT_KEY)

    try:
        yield TestClient(app), {"review": review_id, "crit": f_crit, "med": f_med}, engine
    finally:
        await engine.dispose()


def _finding(body: dict, finding_id: UUID) -> dict:
    return next(f for f in body["findings"] if f["finding_id"] == str(finding_id))


@pytest.mark.asyncio
async def test_agent_view_full_shape(
    agent_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = agent_client
    resp = client.get(f"/api/reviews/{ids['review']}/agent-view", headers=_AGENT_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "1"
    assert body["review_id"] == str(ids["review"])
    assert body["pr_url"] == f"https://github.com/{_REPO_FULL_NAME}/pull/{_PR_NUMBER}"
    assert body["status"] == "completed"
    assert body["policy_version"]  # a real version string
    assert len(body["findings"]) == 2
    assert body["publish_event"] == {
        "github_review_id": 4242,
        "review_status": "COMMENT",
        "comments_posted": 1,
    }


@pytest.mark.asyncio
async def test_agent_view_auth_scope_separation(
    agent_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """The agent key reads; the ADMIN key is REJECTED (agents must not hold a key
    that can POST /decide); no header 401s."""
    client, ids, _ = agent_client
    url = f"/api/reviews/{ids['review']}/agent-view"
    assert client.get(url, headers=_AGENT_AUTH).status_code == 200
    assert client.get(url, headers=_ADMIN_AUTH).status_code == 401  # admin key rejected
    assert client.get(url).status_code == 401  # no header


@pytest.mark.asyncio
async def test_agent_view_disabled_when_agent_key_unset() -> None:
    """With OUTRIDER_AGENT_API_KEY unset (app.state.agent_api_key absent), the
    surface is disabled — uniform 401, even with a Bearer header, never a 500."""
    app = FastAPI()
    app.include_router(agent_view_router)  # no agent_api_key on app.state
    client = TestClient(app)
    url = f"/api/reviews/{uuid4()}/agent-view"
    assert client.get(url, headers=_AGENT_AUTH).status_code == 401
    assert client.get(url).status_code == 401


@pytest.mark.asyncio
async def test_agent_view_hitl_gated_and_decision(
    agent_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, ids, _ = agent_client
    body = client.get(f"/api/reviews/{ids['review']}/agent-view", headers=_AGENT_AUTH).json()

    crit = _finding(body, ids["crit"])
    assert crit["hitl_gated"] is True
    assert crit["reviewer_decision"] == {
        "outcome": "approve",
        "reviewer_id": "admin",
        "reason": "operator-approved",
        "decided_at": "2026-06-06T19:10:24Z",
    }

    med = _finding(body, ids["med"])
    assert med["hitl_gated"] is False
    assert med["reviewer_decision"] is None


@pytest.mark.asyncio
async def test_agent_view_omits_unstored_fields(
    agent_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """github_comment_url (not stored) + suggested_patch (feature-2) are ABSENT
    from the V1 contract, pinning it so a later addition is a deliberate change."""
    client, ids, _ = agent_client
    crit = _finding(
        client.get(f"/api/reviews/{ids['review']}/agent-view", headers=_AGENT_AUTH).json(),
        ids["crit"],
    )
    assert "github_comment_url" not in crit
    assert "suggested_patch" not in crit
    # The contract IS the trust-critical fields straight from decided state.
    assert crit["severity"] == "critical"
    assert crit["evidence_tier"] == "judged"


@pytest.mark.asyncio
async def test_agent_view_unknown_review_404(
    agent_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    client, _, _ = agent_client
    assert client.get(f"/api/reviews/{uuid4()}/agent-view", headers=_AGENT_AUTH).status_code == 404
