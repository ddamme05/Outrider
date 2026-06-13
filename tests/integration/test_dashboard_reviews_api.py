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
    head_sha: str = "sha1",
) -> UUID:
    """Insert one review + its audit events. The `reviews.*` aggregate-metric
    columns no longer exist (dropped per DECISIONS.md#037); the API computes
    every metric read-through from the audit stream, so there is nothing on the
    row to read. `head_sha` varies the `(repo_id, pr_number, head_sha)` natural
    key so multiple reviews can share one `repo_id`.
    """
    result = await conn.execute(
        text(
            "INSERT INTO reviews ("
            "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
            "  retention_expires_at"
            ") VALUES ("
            "  :iid, :repo, 1, :sha, :status, :is_eval, "
            "  NOW() + INTERVAL '90 days'"
            ") RETURNING id"
        ),
        {
            "iid": _INSTALLATION_ID,
            "repo": repo_id,
            "sha": head_sha,
            "status": status,
            "is_eval": is_eval,
        },
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
                "policy_version": "1.0.0",
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
    # policy_version read from the per-review snapshot on the audit stream
    # (synthesize_completed payload here), not the reviews row.
    assert resp.json()["policy_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_metrics_exclude_divergent_is_eval_synthesize_event(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A divergent `is_eval=True` synthesize_completed on a production
    (is_eval=False) review must NOT surface its file/wall-clock metrics — FUP-130
    read-side `is_eval` predicate. SynthesizeCompletedEvent reaches the row
    through an unguarded emit path (`_persist_non_phase_event`), so the read
    predicate is the actual gate; the persister guard only covers
    persist()/emit_finding(). Without the predicate, this was a real leak."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        review_id = await _seed_review(
            conn, status="completed", is_eval=False, repo_id=500, llm_events=[], synth=None
        )
        # A divergent eval synthesize_completed sneaks onto the production review.
        await conn.execute(
            text(
                "INSERT INTO audit_events "
                "(event_id, review_id, event_type, timestamp, is_eval, payload) "
                "VALUES (:eid, :rid, 'synthesize_completed', NOW(), true, CAST(:payload AS jsonb))"
            ),
            {
                "eid": uuid4(),
                "rid": review_id,
                "payload": json.dumps(
                    {
                        "files_examined": 999,
                        "files_traced_beyond_diff": 999,
                        "wall_clock_seconds": 999.0,
                        "policy_version": "9.9.9",
                    }
                ),
            },
        )
    resp = client.get(f"/api/reviews/{review_id}", headers=_AUTH)
    assert resp.status_code == 200
    m = resp.json()["metrics"]
    # The eval synth is filtered by the review's is_eval=False scope -> file
    # metrics stay None (pending), NOT the divergent eval 999s.
    assert m["files_examined"] is None
    assert m["files_traced_beyond_diff"] is None
    assert m["wall_clock_seconds"] is None
    # policy_version likewise ignores the divergent eval event.
    assert resp.json()["policy_version"] is None


@pytest.mark.asyncio
async def test_findings_requiring_approval_gated_set(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """`findings_requiring_approval` mirrors `reviews.hitl_request` exactly:
    None when no snapshot, [] when the snapshot gates nothing, else the ids.
    """
    client, ids, engine = dashboard_client

    # No HITL request snapshot (review A) -> None, not [].
    assert (
        client.get(f"/api/reviews/{ids['a']}", headers=_AUTH).json()["findings_requiring_approval"]
        is None
    )

    fid_1, fid_2 = str(uuid4()), str(uuid4())
    async with engine.begin() as conn:
        gated = await _seed_review(
            conn, status="awaiting_approval", is_eval=False, repo_id=400, llm_events=[], synth=None
        )
        empty = await _seed_review(
            conn, status="awaiting_approval", is_eval=False, repo_id=401, llm_events=[], synth=None
        )
        for rid, faa in ((gated, [fid_1, fid_2]), (empty, [])):
            await conn.execute(
                text("UPDATE reviews SET hitl_request = CAST(:hr AS jsonb) WHERE id = :id"),
                {
                    "hr": json.dumps(
                        {
                            "findings_requiring_approval": faa,
                            "auto_post_findings": [],
                            "created_at": "2026-06-01T00:00:00Z",
                            "expires_at": "2026-06-01T01:00:00Z",
                        }
                    ),
                    "id": str(rid),
                },
            )

    # Non-empty snapshot -> the exact authoritative id set the decide call must cover.
    gated_resp = client.get(f"/api/reviews/{gated}", headers=_AUTH)
    assert gated_resp.status_code == 200
    assert gated_resp.json()["findings_requiring_approval"] == [fid_1, fid_2]

    # Snapshot present but nothing gated -> [], distinct from None.
    assert (
        client.get(f"/api/reviews/{empty}", headers=_AUTH).json()["findings_requiring_approval"]
        == []
    )


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
    # No policy-version-bearing event yet (only an llm_call) -> None.
    assert resp.json()["policy_version"] is None


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


@pytest.mark.asyncio
async def test_duplicate_synthesize_completed_latest_wins(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """Crash-recovery can land >1 synthesize_completed per review (no V1
    natural-key dedup — event_id-PK only). The latest row (highest
    sequence_number) must win, never an arbitrary stale completion.
    """
    client, ids, engine = dashboard_client
    # Review A already has a synthesize_completed (files_examined=5). Land a
    # LATER one (higher sequence_number) with different metrics — it wins.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO audit_events "
                "(event_id, review_id, event_type, timestamp, is_eval, payload) "
                "VALUES (:eid, :rid, 'synthesize_completed', NOW(), false, "
                "CAST(:payload AS jsonb))"
            ),
            {
                "eid": uuid4(),
                "rid": ids["a"],
                "payload": json.dumps(
                    {
                        "files_examined": 99,
                        "files_traced_beyond_diff": 88,
                        "wall_clock_seconds": 77.7,
                    }
                ),
            },
        )

    resp = client.get(f"/api/reviews/{ids['a']}", headers=_AUTH)
    assert resp.status_code == 200
    m = resp.json()["metrics"]
    assert m["files_examined"] == 99
    assert m["files_traced_beyond_diff"] == 88
    assert m["wall_clock_seconds"] == pytest.approx(77.7)


# --- reviews-page-mockup-restyle: repo name, pr_title, severity tally, status_counts -------

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_SYNTH = {"files_examined": 1, "files_traced_beyond_diff": 0, "wall_clock_seconds": 1.0}


async def _seed_repo_membership(
    conn: Any, *, repo_id: int, repo_full_name: str, removed: bool = False
) -> None:
    # `removed_at` defaults to NULL (active membership); set it for the removed case.
    await conn.execute(
        text(
            "INSERT INTO installation_repositories "
            "(installation_id, repo_id, repo_full_name, added_at) "
            "VALUES (:iid, :repo, :name, NOW())"
        ),
        {"iid": _INSTALLATION_ID, "repo": repo_id, "name": repo_full_name},
    )
    if removed:
        await conn.execute(
            text(
                "UPDATE installation_repositories SET removed_at = NOW() "
                "WHERE installation_id = :iid AND repo_id = :repo"
            ),
            {"iid": _INSTALLATION_ID, "repo": repo_id},
        )


async def _policy_version(conn: Any) -> str:
    return str(
        (await conn.execute(text("SELECT version FROM severity_policies LIMIT 1"))).scalar_one()
    )


async def _insert_finding(
    conn: Any,
    *,
    review_id: UUID,
    policy_version: str,
    severity: str,
    content_hash: str,
    is_eval: bool = False,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO findings ("
            "  finding_id, review_id, installation_id, policy_version, finding_type, "
            "  dimension, severity, evidence_tier, file_path, line_start, line_end, "
            "  title, description, evidence, suggested_fix, query_match_id, trace_path, "
            "  content_hash, is_eval, retention_expires_at"
            ") VALUES ("
            "  :fid, :rid, :iid, :pv, 'sql_injection', 'security', :sev, 'judged', "
            "  'app/db.py', 10, 12, 'finding', 'desc', 'ev', NULL, NULL, NULL, "
            "  :ch, :ie, NOW() + INTERVAL '90 days'"
            ")"
        ),
        {
            "fid": uuid4(),
            "rid": review_id,
            "iid": _INSTALLATION_ID,
            "pv": policy_version,
            "sev": severity,
            "ch": content_hash,
            "ie": is_eval,
        },
    )


@pytest.mark.asyncio
async def test_list_repo_full_name_join_with_fallback(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """repo_full_name comes from the active installation_repositories membership;
    no row (or a removed row) falls back to null (client renders `repo {id}`)."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        active = await _seed_review(
            conn, status="running", is_eval=False, repo_id=700, llm_events=[], synth=None
        )
        await _seed_repo_membership(conn, repo_id=700, repo_full_name="acme/api")
        no_member = await _seed_review(
            conn, status="running", is_eval=False, repo_id=701, llm_events=[], synth=None
        )
        removed = await _seed_review(
            conn, status="running", is_eval=False, repo_id=702, llm_events=[], synth=None
        )
        await _seed_repo_membership(conn, repo_id=702, repo_full_name="acme/old", removed=True)

    by_id = {r["id"]: r for r in client.get("/api/reviews", headers=_AUTH).json()["reviews"]}
    assert by_id[str(active)]["repo_full_name"] == "acme/api"
    assert by_id[str(no_member)]["repo_full_name"] is None
    assert by_id[str(removed)]["repo_full_name"] is None


@pytest.mark.asyncio
async def test_list_pr_title_with_null_fallback(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """pr_title is returned when persisted; a row without it (the pre-migration
    / no-backfill case, like the fixture's review A) returns null."""
    client, ids, engine = dashboard_client
    async with engine.begin() as conn:
        titled = await _seed_review(
            conn, status="running", is_eval=False, repo_id=710, llm_events=[], synth=None
        )
        await conn.execute(
            text("UPDATE reviews SET pr_title = :t WHERE id = :id"),
            {"t": "Add session token storage", "id": str(titled)},
        )
    by_id = {r["id"]: r for r in client.get("/api/reviews", headers=_AUTH).json()["reviews"]}
    assert by_id[str(titled)]["pr_title"] == "Add session token storage"
    assert by_id[str(ids["a"])]["pr_title"] is None


@pytest.mark.asyncio
async def test_severity_counts_are_report_equivalent_deduped(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """Counts are COUNT(DISTINCT content_hash) per severity — the synthesize-
    deduplicated set, NOT raw admitted findings rows. Two HIGH findings sharing
    one content_hash collapse to 1."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        pv = await _policy_version(conn)
        rid = await _seed_review(
            conn, status="completed", is_eval=False, repo_id=720, llm_events=[], synth=_SYNTH
        )
        await _insert_finding(
            conn, review_id=rid, policy_version=pv, severity="high", content_hash=_HASH_A
        )
        await _insert_finding(
            conn, review_id=rid, policy_version=pv, severity="high", content_hash=_HASH_A
        )
        await _insert_finding(
            conn, review_id=rid, policy_version=pv, severity="critical", content_hash=_HASH_B
        )
    body = client.get("/api/reviews", headers=_AUTH).json()
    sc = next(r for r in body["reviews"] if r["id"] == str(rid))["severity_counts"]
    # high=1 (deduped from 2 raw rows), critical=1.
    assert sc == {"critical": 1, "high": 1, "medium": 0, "low": 0, "info": 0}


@pytest.mark.asyncio
async def test_severity_counts_null_before_synthesize(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """A running review with analyze-emitted findings but NO synthesize_completed
    has no report-equivalent set yet -> severity_counts is null, not a raw tally."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        pv = await _policy_version(conn)
        rid = await _seed_review(
            conn, status="running", is_eval=False, repo_id=730, llm_events=[], synth=None
        )
        await _insert_finding(
            conn, review_id=rid, policy_version=pv, severity="high", content_hash=_HASH_A
        )
    body = client.get("/api/reviews", headers=_AUTH).json()
    sc = next(r for r in body["reviews"] if r["id"] == str(rid))["severity_counts"]
    assert sc is None


@pytest.mark.asyncio
async def test_severity_counts_respect_per_row_is_eval(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """The tally matches each review's OWN is_eval (Finding.is_eval ==
    Review.is_eval), never a global predicate: a divergent eval finding on a
    production review does not leak into its tally."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        pv = await _policy_version(conn)
        prod = await _seed_review(
            conn, status="completed", is_eval=False, repo_id=740, llm_events=[], synth=_SYNTH
        )
        await _insert_finding(
            conn,
            review_id=prod,
            policy_version=pv,
            severity="high",
            content_hash=_HASH_A,
            is_eval=False,
        )
        # Divergent eval finding on the production review — must be excluded.
        await _insert_finding(
            conn,
            review_id=prod,
            policy_version=pv,
            severity="critical",
            content_hash=_HASH_B,
            is_eval=True,
        )
    body = client.get("/api/reviews", headers=_AUTH).json()
    sc = next(r for r in body["reviews"] if r["id"] == str(prod))["severity_counts"]
    assert sc == {"critical": 0, "high": 1, "medium": 0, "low": 0, "info": 0}


@pytest.mark.asyncio
async def test_status_counts_real_filter_independent_and_scoped(
    dashboard_client: tuple[TestClient, dict[str, UUID], AsyncEngine],
) -> None:
    """status_counts is a per-status GROUP BY over the BASE filters (include_eval
    + repo_id), independent of the active status filter; sum == "All N"."""
    client, _, engine = dashboard_client
    async with engine.begin() as conn:
        for i, st in enumerate(("running", "running", "completed", "awaiting_approval", "failed")):
            await _seed_review(
                conn,
                status=st,
                is_eval=False,
                repo_id=750,
                llm_events=[],
                synth=None,
                head_sha=f"sha-prod-{i}",
            )
        await _seed_review(
            conn,
            status="completed",
            is_eval=True,
            repo_id=750,
            llm_events=[],
            synth=None,
            head_sha="sha-eval",
        )

    # Scoped to repo 750 + status=running active: the LIST is running-only, but
    # status_counts still reflects every status (filter-independent).
    resp = client.get(
        "/api/reviews", params={"repo_id": 750, "status": "running"}, headers=_AUTH
    ).json()
    sc = resp["status_counts"]
    assert sc["running"] == 2
    assert sc["completed"] == 1
    assert sc["awaiting_approval"] == 1
    assert sc["failed"] == 1
    assert sum(sc.values()) == 5  # "All N" for repo 750, eval excluded
    assert [r["status"] for r in resp["reviews"]] == ["running", "running"]  # list IS filtered

    # include_eval scoping: the eval completed review joins the counts (sum 6).
    sc_eval = client.get(
        "/api/reviews", params={"repo_id": 750, "include_eval": "true"}, headers=_AUTH
    ).json()["status_counts"]
    assert sum(sc_eval.values()) == 6
