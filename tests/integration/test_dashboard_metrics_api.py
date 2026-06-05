"""Integration tests for the dashboard analytics endpoint (`api/dashboard/metrics.py`).

Proves the honest-aggregation contract from `specs/2026-06-04-dashboard-analytics.md`
+ `DECISIONS.md#039` against real seeded `audit_events` / `reviews` rows:

- per-day reviews / cost / failed buckets + the severity AND evidence-tier distributions;
- "findings" = DEDUPED logical findings (distinct `(review_id, finding_content_hash)`),
  so a re-emit of the same hash counts ONCE (the Codex catch);
- cross-window guard: a finding whose FIRST emission predates the window is excluded
  even when re-emitted in-window (min-then-filter, not filter-then-min);
- tier representative: the EARLIEST emission's tier wins (OBSERVED-then-JUDGED → observed);
- `is_eval` excluded by default, exposed with `?include_eval=true`;
- sparse windows render honest zeros;
- 401 without the admin key; read-only.

Seeding controls timestamps via `NOW() - (:age * INTERVAL '1 day')` and emission ORDER
(= `sequence_number`, a BIGINT IDENTITY) by INSERT order — the representative is the
earliest-inserted row of a content_hash group, matching production (earlier emission =
lower sequence_number = earlier timestamp). Finding payloads are minimal (the endpoint
only reads `finding_content_hash` / `severity` / `evidence_tier`).
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
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from outrider.api.dashboard import metrics_router

_ADMIN_KEY = "test-admin-key"  # noqa: S105
_AUTH = {"Authorization": f"Bearer {_ADMIN_KEY}"}
_INSTALLATION_ID = 4242


async def _seed_review(
    conn: Any, *, status: str, is_eval: bool, age_days: float, repo_id: int = 100
) -> UUID:
    result = await conn.execute(
        text(
            "INSERT INTO reviews ("
            "  installation_id, repo_id, pr_number, head_sha, status, is_eval, "
            "  created_at, retention_expires_at"
            ") VALUES ("
            "  :iid, :repo, 1, 'sha1', :status, :is_eval, "
            "  NOW() - (:age * INTERVAL '1 day'), NOW() + INTERVAL '90 days'"
            ") RETURNING id"
        ),
        {
            "iid": _INSTALLATION_ID,
            "repo": repo_id,
            "status": status,
            "is_eval": is_eval,
            "age": age_days,
        },
    )
    return UUID(str(result.scalar_one()))


async def _seed_event(
    conn: Any,
    review_id: UUID,
    event_type: str,
    payload: dict[str, Any],
    *,
    age_days: float,
    is_eval: bool,
) -> None:
    await conn.execute(
        text(
            "INSERT INTO audit_events "
            "(event_id, review_id, event_type, timestamp, is_eval, payload) "
            "VALUES (:eid, :rid, :etype, NOW() - (:age * INTERVAL '1 day'), :is_eval, "
            "CAST(:payload AS jsonb))"
        ),
        {
            "eid": uuid4(),
            "rid": review_id,
            "etype": event_type,
            "is_eval": is_eval,
            "payload": json.dumps(payload),
            "age": age_days,
        },
    )


async def _seed_finding(
    conn: Any,
    review_id: UUID,
    *,
    content_hash: str,
    severity: str,
    tier: str,
    age_days: float,
    is_eval: bool,
) -> None:
    await _seed_event(
        conn,
        review_id,
        "finding",
        {
            "finding_id": str(uuid4()),
            "finding_content_hash": content_hash,
            "severity": severity,
            "evidence_tier": tier,
        },
        age_days=age_days,
        is_eval=is_eval,
    )


async def _seed_llm(
    conn: Any, review_id: UUID, *, cost_usd: float, age_days: float, is_eval: bool
) -> None:
    await _seed_event(
        conn,
        review_id,
        "llm_call",
        {"input_tokens": 1, "output_tokens": 1, "cost_usd": cost_usd},
        age_days=age_days,
        is_eval=is_eval,
    )


@pytest_asyncio.fixture
async def metrics_client(
    migrated_db: str,
) -> AsyncGenerator[TestClient]:
    """Seed a 7d-window scenario exercising every contract, then mount the endpoint.

    Within the 7d window (is_eval=False): R1 (completed, age 1) + R2 (failed, age 2).
    Findings: h1 OBSERVED then JUDGED re-emit (dedup + tier-rep), h2 HIGH/INFERRED,
    h3 first-emitted PRE-window (age 10) then re-emitted in-window (cross-window guard).
    R3 is eval (excluded by default). R4 (age 10) + h5 live in the PRIOR window (deltas).
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
        r1 = await _seed_review(conn, status="completed", is_eval=False, age_days=1, repo_id=100)
        r2 = await _seed_review(conn, status="failed", is_eval=False, age_days=2, repo_id=200)
        r3 = await _seed_review(conn, status="completed", is_eval=True, age_days=1, repo_id=300)
        r4 = await _seed_review(conn, status="completed", is_eval=False, age_days=10, repo_id=400)

        await _seed_llm(conn, r1, cost_usd=0.02, age_days=1, is_eval=False)
        await _seed_llm(conn, r2, cost_usd=0.03, age_days=2, is_eval=False)
        await _seed_llm(conn, r3, cost_usd=0.99, age_days=1, is_eval=True)
        await _seed_llm(conn, r4, cost_usd=0.50, age_days=10, is_eval=False)

        # Emission ORDER = sequence_number. Insert each content_hash's intended
        # representative (earliest emission) FIRST.
        # h1: OBSERVED (rep) then JUDGED re-emit -> dedup to 1, tier=observed.
        await _seed_finding(
            conn,
            r1,
            content_hash="h1",
            severity="medium",
            tier="observed",
            age_days=1,
            is_eval=False,
        )
        await _seed_finding(
            conn, r1, content_hash="h1", severity="medium", tier="judged", age_days=1, is_eval=False
        )
        # h3: first emission PRE-window (age 10), re-emit in-window (age 1) -> excluded.
        await _seed_finding(
            conn, r1, content_hash="h3", severity="low", tier="judged", age_days=10, is_eval=False
        )
        await _seed_finding(
            conn, r1, content_hash="h3", severity="low", tier="observed", age_days=1, is_eval=False
        )
        # h2: single in-window finding -> counted, HIGH / inferred.
        await _seed_finding(
            conn, r2, content_hash="h2", severity="high", tier="inferred", age_days=2, is_eval=False
        )
        # h4: eval finding -> excluded by default.
        await _seed_finding(
            conn,
            r3,
            content_hash="h4",
            severity="critical",
            tier="observed",
            age_days=1,
            is_eval=True,
        )
        # h5: prior-window finding (age 10) -> deltas.previous.
        await _seed_finding(
            conn,
            r4,
            content_hash="h5",
            severity="high",
            tier="inferred",
            age_days=10,
            is_eval=False,
        )

    app = FastAPI()
    app.include_router(metrics_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    try:
        yield TestClient(app)
    finally:
        await engine.dispose()


def _get(client: TestClient, **params: str) -> dict[str, Any]:
    resp = client.get("/api/metrics", params=params, headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body: dict[str, Any] = resp.json()
    return body


@pytest.mark.asyncio
async def test_buckets_and_distributions(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    # Current-window totals (is_eval excluded by default): R1 + R2.
    assert body["deltas"]["current"]["reviews"] == 2
    assert body["deltas"]["current"]["failed"] == 1  # R2 only
    assert body["deltas"]["current"]["cost_usd"] == pytest.approx(
        0.05
    )  # 0.02 + 0.03, eval 0.99 excluded
    # Deduped findings in-window: h1 + h2 (h3 pre-window, h4 eval, h5 prior).
    assert body["deltas"]["current"]["findings"] == 2
    # Severity distribution (zero-filled across the full enum): h1 medium, h2 high.
    assert body["severity_distribution"]["medium"] == 1
    assert body["severity_distribution"]["high"] == 1
    assert body["severity_distribution"]["critical"] == 0  # the eval h4 must NOT leak
    assert body["severity_distribution"]["low"] == 0  # h3 (pre-window) excluded
    # Buckets sum to the totals; cost summed across buckets.
    assert sum(b["reviews"] for b in body["buckets"]) == 2
    assert sum(b["findings"] for b in body["buckets"]) == 2
    assert sum(b["cost_usd"] for b in body["buckets"]) == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_findings_dedup_counts_once(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    # h1 was emitted TWICE (same finding_content_hash) — must count as ONE finding,
    # ONE severity-distribution entry. Raw-counting would give medium == 2.
    assert body["severity_distribution"]["medium"] == 1
    assert body["deltas"]["current"]["findings"] == 2  # h1 + h2, not h1×2 + h2


@pytest.mark.asyncio
async def test_tier_representative_is_earliest_emission(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    tier = body["evidence_tier_distribution"]
    # h1: OBSERVED (earliest) then JUDGED re-emit -> representative tier is OBSERVED.
    # h2: INFERRED. h3 (judged) is cross-window-excluded; the eval h4 (observed) excluded.
    assert tier["observed"] == 1  # h1 representative — NOT the judged re-emit
    assert tier["inferred"] == 1  # h2
    assert tier["judged"] == 0  # h1's judged re-emit is dedup'd away; h3's judged is pre-window


@pytest.mark.asyncio
async def test_cross_window_pre_emit_excluded(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    # h3 was FIRST emitted at age 10 (pre-window) and re-emitted at age 1 (in-window).
    # min-then-filter: its true first emission is pre-window -> excluded from the 7d window.
    # If it leaked, findings would be 3 and low/severity or judged/tier would be 1.
    assert body["deltas"]["current"]["findings"] == 2
    assert body["severity_distribution"]["low"] == 0  # h3's severity


@pytest.mark.asyncio
async def test_is_eval_excluded_by_default_and_exposed_with_flag(
    metrics_client: TestClient,
) -> None:
    default = _get(metrics_client, window="7d")
    assert default["severity_distribution"]["critical"] == 0  # eval h4 excluded
    assert default["deltas"]["current"]["cost_usd"] == pytest.approx(0.05)  # eval 0.99 excluded

    with_eval = _get(metrics_client, window="7d", include_eval="true")
    # R3 (eval) + h4 (critical/observed) + 0.99 cost now included.
    assert with_eval["severity_distribution"]["critical"] == 1
    # include_eval must thread to the TIER distribution too (separate group_by), not only severity:
    assert with_eval["evidence_tier_distribution"]["observed"] == 2  # h1 + eval h4
    assert with_eval["deltas"]["current"]["reviews"] == 3  # R1, R2, R3
    assert with_eval["deltas"]["current"]["cost_usd"] == pytest.approx(1.04)  # 0.05 + 0.99


@pytest.mark.asyncio
async def test_sparse_window_renders_zeros(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    # The 7d window spans 7-8 UTC-day buckets; only 2 carry findings/reviews.
    assert len(body["buckets"]) >= 7
    zero_days = [b for b in body["buckets"] if b["reviews"] == 0 and b["findings"] == 0]
    assert zero_days, "sparse days must be present as honest zeros, not omitted"
    for b in zero_days:
        assert b["cost_usd"] == 0.0  # honest zero, not interpolated


@pytest.mark.asyncio
async def test_deltas_current_vs_previous_window(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="7d")
    # Current window [now-7d, now): R1 + R2. Previous [now-14d, now-7d): R4 (age 10).
    assert body["deltas"]["current"]["reviews"] == 2
    assert body["deltas"]["previous"]["reviews"] == 1  # R4 only
    assert body["deltas"]["previous"]["cost_usd"] == pytest.approx(0.50)  # R4's llm_call
    # Previous-window findings (exercises the prev-window representative path, end=start):
    # h5 (age 10) AND h3's representative (first emission age 10) both fall in the prior window.
    assert body["deltas"]["previous"]["findings"] == 2


@pytest.mark.asyncio
async def test_24h_window_uses_hourly_buckets(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="24h")
    # 24h must bucket HOURLY (not a degenerate 1-2 point daily series) — the mockup sparkline.
    assert body["granularity"] == "hour"
    assert 24 <= len(body["buckets"]) <= 26  # ~25 hourly boundaries (inclusive both ends)


@pytest.mark.asyncio
async def test_30d_window_includes_older_data(metrics_client: TestClient) -> None:
    body = _get(metrics_client, window="30d")
    assert body["granularity"] == "day"
    # The age-10 R4 + h5 + h3's representative now fall INSIDE the 30d window — proves window
    # selection actually shifts start/prev_start (only 7d is otherwise exercised).
    assert body["deltas"]["current"]["reviews"] == 3  # R1, R2, R4 (eval R3 still excluded)
    assert body["deltas"]["current"]["findings"] == 4  # h1, h2, h3, h5


@pytest_asyncio.fixture
async def empty_metrics_client(migrated_db: str) -> AsyncGenerator[TestClient]:
    """Mount the endpoint on a migrated-but-UNSEEDED DB — the zero-row path."""
    engine = create_async_engine(migrated_db, hide_parameters=True)
    app = FastAPI()
    app.include_router(metrics_router)
    app.state.session_factory = async_sessionmaker(engine, expire_on_commit=False)
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    try:
        yield TestClient(app)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_empty_window_renders_all_zeros(empty_metrics_client: TestClient) -> None:
    resp = empty_metrics_client.get("/api/metrics", params={"window": "7d"}, headers=_AUTH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # Zero-row safety: .one()/.scalar_one()/empty-dict-comprehension paths all degrade to zeros.
    assert body["deltas"]["current"] == {"reviews": 0, "cost_usd": 0.0, "findings": 0, "failed": 0}
    assert body["deltas"]["previous"]["findings"] == 0
    assert all(v == 0 for v in body["severity_distribution"].values())
    assert all(v == 0 for v in body["evidence_tier_distribution"].values())
    # Buckets are still present (honest zero-fill), not an empty list.
    assert body["buckets"]
    assert all(
        b["reviews"] == 0 and b["findings"] == 0 and b["cost_usd"] == 0.0 and b["failed"] == 0
        for b in body["buckets"]
    )


def test_requires_admin_key(metrics_client: TestClient) -> None:
    resp = metrics_client.get("/api/metrics")
    assert resp.status_code == 401


@pytest_asyncio.fixture
async def divergent_eval_client(migrated_db: str) -> AsyncGenerator[TestClient]:
    """Seed DIVERGENT `is_eval` rows in BOTH directions — events whose `is_eval` disagrees with
    their review's, the divergence the persister's write-side check forbids (`persister.py`
    AuditPersisterIsEvalMismatchError), reproduced via raw INSERT to prove the read-side equality
    defense (FUP-130 `AuditEvent.is_eval == review_is_eval`). A production review carries an
    agreeing pair + an eval-labeled drift pair; an eval review carries an agreeing pair + a
    prod-labeled drift finding.
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
        # Production review (is_eval=False): an AGREEING finding/cost (hp/cp) + a DRIFT pair
        # mislabeled is_eval=True (hd/cd) — the prod-review->eval-event direction a one-sided
        # Review.is_eval filter would have LEAKED into production.
        prod = await _seed_review(conn, status="completed", is_eval=False, age_days=1, repo_id=100)
        await _seed_llm(conn, prod, cost_usd=0.10, age_days=1, is_eval=False)  # agrees
        await _seed_llm(conn, prod, cost_usd=5.00, age_days=1, is_eval=True)  # DRIFT
        await _seed_finding(
            conn,
            prod,
            content_hash="hp",
            severity="high",
            tier="observed",
            age_days=1,
            is_eval=False,  # agrees
        )
        await _seed_finding(
            conn,
            prod,
            content_hash="hd",
            severity="critical",
            tier="observed",
            age_days=1,
            is_eval=True,  # DRIFT (prod review, eval-labeled event)
        )
        # Eval review (is_eval=True): an AGREEING finding/cost (he/ce) + a DRIFT finding mislabeled
        # is_eval=False (hm) — the eval-review->prod-event direction.
        ev = await _seed_review(conn, status="completed", is_eval=True, age_days=1, repo_id=300)
        await _seed_llm(conn, ev, cost_usd=9.99, age_days=1, is_eval=True)  # agrees
        await _seed_finding(
            conn,
            ev,
            content_hash="he",
            severity="critical",
            tier="inferred",
            age_days=1,
            is_eval=True,  # agrees
        )
        await _seed_finding(
            conn,
            ev,
            content_hash="hm",
            severity="low",
            tier="judged",
            age_days=1,
            is_eval=False,  # DRIFT (eval review, prod-labeled event)
        )

    app = FastAPI()
    app.include_router(metrics_router)
    app.state.session_factory = session_factory
    app.state.admin_api_key = SecretStr(_ADMIN_KEY)
    try:
        yield TestClient(app)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_is_eval_drift_events_excluded_both_directions(
    divergent_eval_client: TestClient,
) -> None:
    """An audit event whose `is_eval` DISAGREES with its review is drift and must not count — in
    EITHER direction — matching the FUP-130 equality predicate (`reviews.py`) + the replay
    consistency check. The endpoint requires `AuditEvent.is_eval == Review.is_eval`, so a
    prod-review eval-labeled event (the case a one-sided `Review.is_eval` filter would have leaked)
    AND an eval-review prod-labeled event are BOTH rejected — under production and include_eval.
    """
    # Production (include_eval defaults False): only the agreeing prod data (hp, $0.10).
    body = _get(divergent_eval_client, window="7d")
    cur = body["deltas"]["current"]
    assert cur["findings"] == 1  # hp only — hd (drift) + he/hm (eval/drift) excluded
    assert cur["cost_usd"] == pytest.approx(0.10)  # cp only — cd (drift) + ce (eval) excluded
    assert body["severity_distribution"]["critical"] == 0  # neither hd (drift) nor he (eval) leaked
    assert body["evidence_tier_distribution"]["observed"] == 1  # only hp

    # include_eval=true: agreeing prod + agreeing eval (hp + he); drift (hd, hm, cd) stays excluded.
    full = _get(divergent_eval_client, window="7d", include_eval="true")
    fcur = full["deltas"]["current"]
    assert fcur["findings"] == 2  # hp + he; hd + hm (drift) still excluded
    assert fcur["cost_usd"] == pytest.approx(10.09)  # cp 0.10 + ce 9.99; cd (drift) excluded
    assert full["severity_distribution"]["critical"] == 1  # he (agreeing eval), NOT hd (drift)
