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
    is_eval: bool = False,
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO findings ("
            "  finding_id, review_id, installation_id, policy_version, finding_type, "
            "  dimension, severity, evidence_tier, file_path, line_start, line_end, "
            "  title, description, evidence, suggested_fix, query_match_id, trace_path, "
            "  content_hash, is_eval, retention_expires_at"
            ") VALUES ("
            "  :fid, :rid, :iid, :pv, 'sql_injection', 'security', :sev, :tier, "
            "  'app/db.py', 10, 12, :title, 'desc', 'evidence-snippet', 'fix it', "
            "  :qmid, CAST(:tp AS jsonb), :ch, :ie, NOW() + INTERVAL '90 days'"
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
            "ie": is_eval,
        },
    )


async def _insert_publish_routing(
    conn: object, *, review_id: UUID, finding_id: UUID, destination: str, is_eval: bool = False
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'publish_routing', NOW(), :is_eval, CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "is_eval": is_eval,
            "payload": json.dumps({"finding_id": str(finding_id), "destination": destination}),
        },
    )


async def _insert_finding_event(
    conn: object,
    *,
    review_id: UUID,
    finding_id: UUID,
    severity: str,
    evidence_tier: str,
    file_path: str = "app/legacy.py",
) -> None:
    """Insert a FindingEvent (event_type='finding') with NO findings row — the
    dangling / retention-purged case (DECISIONS.md#014 point 3).
    """
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'finding', NOW(), false, CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "payload": json.dumps(
                {
                    "finding_id": str(finding_id),
                    "finding_type": "sql_injection",
                    "dimension": "security",
                    "severity": severity,
                    "evidence_tier": evidence_tier,
                    "file_path": file_path,
                    "line_start": 5,
                    "line_end": 7,
                    "query_match_id": None,
                    "trace_path": None,
                }
            ),
        },
    )


async def _insert_publish_eligibility(
    conn: object,
    *,
    review_id: UUID,
    finding_id: UUID,
    eligibility: str,
    reason: str | None,
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'publish_eligibility', NOW(), false, CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "payload": json.dumps(
                {"finding_id": str(finding_id), "eligibility": eligibility, "reason": reason}
            ),
        },
    )


async def _insert_hitl_decision(
    conn: object,
    *,
    review_id: UUID,
    decisions: list[dict[str, object]],
    reviewer_id: str = "admin",
) -> None:
    """Insert the review's single `HITLDecisionEvent` (event_type=
    'hitl_decision'). The dashboard's `_hitl_decisions` reads `reviewer_id` +
    `decisions[*]` (`finding_id` / `outcome` / `reason` / `original_severity` /
    `override_severity`) — the `decisions_content_hash` validator is bypassed
    because this is a raw-row insert (the read path never recomputes it).
    """
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, 'hitl_decision', NOW(), false, CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "payload": json.dumps({"reviewer_id": reviewer_id, "decisions": decisions}),
        },
    )


async def _insert_purge_audit(
    conn: object, *, installation_id: int, target_table: str, timestamp_iso: str
) -> None:
    await conn.execute(  # type: ignore[attr-defined]
        text(
            "INSERT INTO purge_audit "
            "(installation_id, target_table, rows_affected, purge_role, timestamp) "
            "VALUES (:iid, :tt, 1, 'sweep', :ts)"
        ),
        {"iid": installation_id, "tt": target_table, "ts": timestamp_iso},
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
                            "  retention_expires_at"
                            ") VALUES ("
                            "  :iid, 100, 7, 'sha7', 'completed', "
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
    # Surviving content -> full finding, not redacted; no eligibility/sweep.
    assert f1["content_redacted"] is False
    assert f1["title"] == "finding-high"
    assert f1["eligibility"] is None
    assert f1["redaction_sweep_at"] is None

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


@pytest.mark.asyncio
async def test_dangling_finding_event_renders_redacted(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A FindingEvent whose findings row was purged renders as a redacted stub
    (content None, metadata from the event) — DECISIONS.md#014 point 3.
    """
    client, ids, engine = findings_client
    f3 = uuid4()
    async with engine.begin() as conn:
        await _insert_finding_event(
            conn,
            review_id=ids["review"],
            finding_id=f3,
            severity="critical",
            evidence_tier="judged",
        )
        # The TTL retention sweep (the reachable case) writes purge_audit with
        # the GLOBAL sentinel installation_id 0, NOT the review's install
        # (sweep/purge_expired.py::_GLOBAL_SWEEP_INSTALLATION_ID). The stub's
        # redaction_sweep_at must still resolve from this row.
        await _insert_purge_audit(
            conn,
            installation_id=0,
            target_table="findings",
            timestamp_iso="2026-05-31T12:00:00+00:00",
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    findings = {f["finding_id"]: f for f in resp.json()["findings"]}
    assert str(f3) in findings
    stub = findings[str(f3)]
    assert stub["content_redacted"] is True
    assert stub["title"] is None
    assert stub["description"] is None
    assert stub["evidence"] is None
    # Metadata survives in the audit stream.
    assert stub["severity"] == "critical"
    assert stub["evidence_tier"] == "judged"
    assert stub["file_path"] == "app/legacy.py"
    # The sweep date comes from purge_audit (findings target), not a per-finding
    # delete time. Date-level assertion (tz formatting is incidental).
    assert stub["redaction_sweep_at"] is not None
    assert stub["redaction_sweep_at"].startswith("2026-05-31")
    # The surviving findings still render full (not redacted, no sweep date).
    assert findings[str(ids["f1"])]["content_redacted"] is False
    assert findings[str(ids["f1"])]["redaction_sweep_at"] is None


@pytest.mark.asyncio
async def test_routed_but_withheld_finding_shows_eligibility(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """Routing != eligibility (DECISIONS.md#023): a finding routed inline can
    still be withheld; the endpoint surfaces both, not just the destination.
    """
    client, ids, engine = findings_client
    async with engine.begin() as conn:
        await _insert_publish_eligibility(
            conn,
            review_id=ids["review"],
            finding_id=ids["f1"],
            eligibility="withheld",
            reason="hitl_required_node_absent",
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    f1 = {f["finding_id"]: f for f in resp.json()["findings"]}[str(ids["f1"])]
    # Routed inline (fixture) AND withheld — both visible, not conflated.
    assert f1["publish_destination"] == "inline_comment"
    assert f1["eligibility"] == "withheld"
    assert f1["eligibility_reason"] == "hitl_required_node_absent"


@pytest.mark.asyncio
async def test_findings_content_row_excludes_divergent_is_eval(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A divergent `is_eval=True` Finding CONTENT row on a production
    (is_eval=False) review must NOT appear in the findings list — FUP-130
    read-side `Finding.is_eval` predicate. This is the highest-sensitivity read
    (finding title/description/evidence), so the read filters even though
    `emit_finding` now also enforces the match write-side."""
    client, ids, engine = findings_client
    f_eval = uuid4()
    async with engine.begin() as conn:
        policy_version = (
            await conn.execute(text("SELECT version FROM severity_policies LIMIT 1"))
        ).scalar_one()
        # The fixture review is is_eval=False; inject an is_eval=True content row.
        await _insert_finding(
            conn,
            review_id=ids["review"],
            policy_version=policy_version,
            finding_id=f_eval,
            severity="high",
            evidence_tier="observed",
            query_match_id="q-eval",
            trace_path=None,
            is_eval=True,
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    fids = {f["finding_id"] for f in resp.json()["findings"]}
    # The divergent eval finding is filtered by the review's is_eval=False scope;
    # only the two production findings remain.
    assert str(f_eval) not in fids
    assert fids == {str(ids["f1"]), str(ids["f2"])}


@pytest.mark.asyncio
async def test_publish_lifecycle_excludes_divergent_is_eval_event(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A divergent `is_eval=True` publish_routing event on a production
    (is_eval=False) review must NOT decorate the finding — FUP-130 read-side
    `is_eval` predicate. Without it, the unguarded publish-emit path would leak
    eval lifecycle state onto a production review's findings view."""
    client, ids, engine = findings_client
    # F2 is unrouted in the fixture. Inject an is_eval=True routing for it.
    async with engine.begin() as conn:
        await _insert_publish_routing(
            conn,
            review_id=ids["review"],
            finding_id=ids["f2"],
            destination="inline_comment",
            is_eval=True,
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    f2 = {f["finding_id"]: f for f in resp.json()["findings"]}[str(ids["f2"])]
    # The eval event is filtered out by the review's is_eval=False scope — F2
    # stays unrouted, not decorated with the divergent eval routing.
    assert f2["publish_destination"] is None


@pytest.mark.asyncio
async def test_hitl_override_joined_from_event_not_column(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """Override provenance is read from the `HITLDecisionEvent` stream by
    `finding_id`, NOT the (V1-null) `findings` override columns (FUP-128 /
    DECISIONS.md#034). F1 is overridden high→medium; F2 has no decision.
    """
    client, ids, engine = findings_client
    async with engine.begin() as conn:
        await _insert_hitl_decision(
            conn,
            review_id=ids["review"],
            decisions=[
                {
                    "finding_id": str(ids["f1"]),
                    "outcome": "severity_override",
                    "reason": "policy too strict for this internal tool",
                    "original_severity": "high",
                    "override_severity": "medium",
                }
            ],
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    findings = {f["finding_id"]: f for f in resp.json()["findings"]}

    f1 = findings[str(ids["f1"])]["hitl_decision"]
    assert f1 is not None
    assert f1["outcome"] == "severity_override"
    assert f1["reviewer_id"] == "admin"
    assert f1["reason"] == "policy too strict for this internal tool"
    assert f1["original_severity"] == "high"
    assert f1["override_severity"] == "medium"
    # The findings.severity column stays the pre-override analyze-time snapshot
    # (DECISIONS.md#034): the stream carries the override, the row does not.
    assert findings[str(ids["f1"])]["severity"] == "high"
    # F2 was never decided -> no projection, not a half-populated object.
    assert findings[str(ids["f2"])]["hitl_decision"] is None


@pytest.mark.asyncio
async def test_hitl_non_override_outcome_carries_no_severities(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A non-override outcome (approve/reject/suppress) surfaces with both
    severity fields null — mirrors `PerFindingDecision.enforce_override_fields`.
    """
    client, ids, engine = findings_client
    async with engine.begin() as conn:
        await _insert_hitl_decision(
            conn,
            review_id=ids["review"],
            decisions=[
                {
                    "finding_id": str(ids["f1"]),
                    "outcome": "reject",
                    "reason": "false positive — the sink is parameterized",
                    "original_severity": None,
                    "override_severity": None,
                }
            ],
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    f1 = {f["finding_id"]: f for f in resp.json()["findings"]}[str(ids["f1"])]["hitl_decision"]
    assert f1 is not None
    assert f1["outcome"] == "reject"
    assert f1["reason"] == "false positive — the sink is parameterized"
    assert f1["original_severity"] is None
    assert f1["override_severity"] is None


@pytest.mark.asyncio
async def test_hitl_override_survives_content_redaction(
    findings_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """Override provenance is stream-sourced, so it survives content redaction:
    a `content_redacted` stub (findings row purged, FindingEvent + HITLDecision
    survive) still carries its `hitl_decision` (DECISIONS.md#034).
    """
    client, ids, engine = findings_client
    f3 = uuid4()
    async with engine.begin() as conn:
        await _insert_finding_event(
            conn,
            review_id=ids["review"],
            finding_id=f3,
            severity="critical",
            evidence_tier="judged",
        )
        await _insert_hitl_decision(
            conn,
            review_id=ids["review"],
            decisions=[
                {
                    "finding_id": str(f3),
                    "outcome": "severity_override",
                    "reason": "downgraded after manual review",
                    "original_severity": "critical",
                    "override_severity": "low",
                }
            ],
        )
    resp = client.get(f"/api/reviews/{ids['review']}/findings", headers=_AUTH)
    stub = {f["finding_id"]: f for f in resp.json()["findings"]}[str(f3)]
    # Content gone, but the override provenance (audit stream) survives.
    assert stub["content_redacted"] is True
    assert stub["title"] is None
    decision = stub["hitl_decision"]
    assert decision is not None
    assert decision["outcome"] == "severity_override"
    assert decision["original_severity"] == "critical"
    assert decision["override_severity"] == "low"
