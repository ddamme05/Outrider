"""Webhook router DB-touching integration tests against real Postgres.

Per the intake-and-webhook spec Test Scenarios:
  - test_webhook_to_triage_happy_path — full slice with mocked GitHub + real DB
  - test_webhook_idempotency_returns_200 — fast-path on existing review row
  - test_webhook_natural_key_conflict_returns_200 — slow-path via IntegrityError;
    branch logic covered at unit tier (tests/unit/
    test_webhook_router_integrity_introspection.py); end-to-end race
    deferred pending a monkey-patch fixture
  - test_webhook_unknown_installation_returns_4xx — fail-closed
  - test_webhook_inactive_repo_membership_returns_4xx — fail-closed
  - test_audit_side_integrity_error_reraises — branch logic covered at
    unit tier (same file as above); end-to-end deferred pending a
    monkey-patch fixture that forces the audit-side INSERT to collide

These tests use the project's `migrated_db` fixture (per-test Postgres
database with the genesis migration applied) and a minimal FastAPI app
that mounts the webhook router with the necessary `app.state` slots.
GitHub client + LLM are mocked; everything else (membership SELECT,
review INSERT, audit_events INSERT, IntegrityError introspection)
exercises real Postgres + psycopg3.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from outrider.api.webhooks.router import router
from outrider.audit.config import RetentionSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SECRET = "test-webhook-secret"  # noqa: S105 — test fixture
_INSTALLATION_ID = 12345
_REPO_ID = 999


def _sign(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={mac}"


def _valid_payload(
    *,
    installation_id: int = _INSTALLATION_ID,
    repo_id: int = _REPO_ID,
    pr_number: int = 42,
    head_sha: str = "h" * 40,
) -> dict[str, Any]:
    return {
        "action": "opened",
        "pull_request": {
            "number": pr_number,
            "title": "Test PR",
            "body": None,
            "user": {"login": "alice", "id": 1},
            "head": {"sha": head_sha, "ref": "feat/x"},
            "base": {"sha": "b" * 40, "ref": "main"},
            "additions": 5,
            "deletions": 2,
        },
        "repository": {
            "id": repo_id,
            "full_name": "acme/widgets",
            "name": "widgets",
            "owner": {"login": "acme", "id": 2},
        },
        "installation": {"id": installation_id},
    }


async def _seed_installation_and_membership(
    engine: AsyncEngine,
    *,
    installation_id: int = _INSTALLATION_ID,
    repo_id: int = _REPO_ID,
    repo_full_name: str = "acme/widgets",
    installation_tombstoned: bool = False,
    membership_removed: bool = False,
) -> None:
    """Seed an Installation + InstallationRepository row for the test."""
    tombstone_clause = "NOW()" if installation_tombstoned else "NULL"
    removed_clause = "NOW()" if membership_removed else "NULL"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "  # noqa: S608 — interpolated values are test-fixture literals (NOW()/NULL), not user input
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install, tombstoned_at) "
                f"VALUES (:iid, 'test-app', 1, 'acme', 'Organization', "
                f" '{{}}'::jsonb, {tombstone_clause})"
            ),
            {"iid": installation_id},
        )
        await conn.execute(
            text(
                "INSERT INTO installation_repositories "  # noqa: S608 — interpolated values are test-fixture literals (NOW()/NULL), not user input
                "(installation_id, repo_id, repo_full_name, added_at, removed_at) "
                f"VALUES (:iid, :rid, :name, NOW(), {removed_clause})"
            ),
            {"iid": installation_id, "rid": repo_id, "name": repo_full_name},
        )


@pytest_asyncio.fixture
async def webhook_app(migrated_db: str) -> AsyncGenerator[tuple[FastAPI, AsyncEngine]]:
    """Mount the webhook router on a FastAPI app wired to the test DB.

    Yields the (app, engine) tuple so individual tests can seed
    installation/repo membership directly via the engine before issuing
    webhook requests via TestClient.
    """
    engine = create_async_engine(migrated_db, hide_parameters=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    app = FastAPI()
    app.include_router(router)

    # Settings stub — webhook secret matches `_SECRET` so signed test
    # bodies verify cleanly. `app_id` deliberately distinct from
    # `_INSTALLATION_ID` so an app-id-vs-installation-id mixup in the
    # webhook wiring fails loudly: a regression that read `app_id` where
    # the router should read `installation.id` from the payload would
    # surface as the wrong-row lookup or a Pydantic validation error.
    app.state.github_app_settings = SimpleNamespace(
        app_id=98765,
        app_private_key=SecretStr("test-private-key"),  # noqa: S106
        webhook_secret=SecretStr(_SECRET),
    )
    app.state.session_factory = session_factory
    app.state.retention_settings = RetentionSettings()

    # run_graph stub — counts invocations so happy-path tests can
    # confirm dispatch happened.
    dispatched: list[Any] = []

    async def _stub_run_graph(state: Any) -> None:
        dispatched.append(state)

    app.state.run_graph = _stub_run_graph
    app.state._dispatched = dispatched  # expose for assertions  # noqa: SLF001

    try:
        yield app, engine
    finally:
        await engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_webhook_unknown_installation_returns_4xx(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Signed valid payload with installation.id not in `installations`
    table → 4xx fail-closed. No review row, no audit row."""
    app, engine = webhook_app
    # Deliberately do NOT seed any installation row.

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 404
    assert "installation or repository not active" in response.json()["detail"]

    # No DB rows, no dispatched task — full fail-closed contract.
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
        n_audit = await conn.scalar(text("SELECT COUNT(*) FROM audit_events"))
    assert n_reviews == 0
    assert n_audit == 0
    assert app.state._dispatched == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_webhook_inactive_repo_membership_returns_4xx(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Installation exists but `installation_repositories.removed_at`
    is non-NULL → 4xx."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine, membership_removed=True)

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 404

    # Full fail-closed contract: no review row, no audit row, no dispatch.
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
        n_audit = await conn.scalar(text("SELECT COUNT(*) FROM audit_events"))
    assert n_reviews == 0
    assert n_audit == 0
    assert app.state._dispatched == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_webhook_tombstoned_installation_returns_4xx(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """`installations.tombstoned_at` non-NULL → 4xx even if membership
    row is otherwise active."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine, installation_tombstoned=True)

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 404

    # Full fail-closed contract: no review row, no audit row, no dispatch.
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
        n_audit = await conn.scalar(text("SELECT COUNT(*) FROM audit_events"))
    assert n_reviews == 0
    assert n_audit == 0
    assert app.state._dispatched == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_webhook_to_triage_happy_path(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Full slice: signed valid payload + active membership →
    - Review row inserted with status='running'
    - Direct-SQL AgentTransitionEvent(from_node='webhook', to_node='intake') inserted
    - Same event_id on row + payload (replay-equivalence)
    - run_graph callable invoked via BackgroundTasks
    - 202 Accepted with review_id."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()

    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "test-delivery-1",
        },
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["status"] == "running"
    review_id_str = payload["review_id"]

    # Review row exists with status='running' + correct natural key
    async with engine.connect() as conn:
        row = await conn.execute(
            text(
                "SELECT id, status, repo_id, pr_number, head_sha, installation_id "
                "FROM reviews WHERE id = :id"
            ),
            {"id": review_id_str},
        )
        review = row.first()
    assert review is not None
    assert review.status == "running"
    assert review.repo_id == _REPO_ID
    assert review.pr_number == 42

    # AgentTransitionEvent row exists with matching event_id in row + payload
    async with engine.connect() as conn:
        ae_row = await conn.execute(
            text("SELECT event_id, event_type, payload FROM audit_events WHERE review_id = :rid"),
            {"rid": review_id_str},
        )
        ae = ae_row.first()
    assert ae is not None
    assert ae.event_type == "agent_transition"
    payload_dict = ae.payload
    assert payload_dict["from_node"] == "webhook"
    assert payload_dict["to_node"] == "intake"
    # CRITICAL: row event_id matches payload event_id (replay-equivalence).
    assert str(ae.event_id) == payload_dict["event_id"]

    # Dispatch happened (run_graph stub was called via BackgroundTasks)
    dispatched = app.state._dispatched  # noqa: SLF001
    assert len(dispatched) == 1
    assert str(dispatched[0].review_id) == review_id_str


@pytest.mark.asyncio
async def test_webhook_idempotency_fast_path_returns_200(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Second valid delivery for the same `(repo_id, pr_number, head_sha)`
    → 200 with existing review_id via the SELECT fast-path. No new
    review or audit rows."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()
    headers = {
        "X-Hub-Signature-256": _sign(_SECRET, body),
        "X-GitHub-Event": "pull_request",
    }

    # First delivery — creates the row.
    first = client.post("/webhooks/github", content=body, headers=headers)
    assert first.status_code == 202
    first_review_id = first.json()["review_id"]

    # Second delivery (same body) — fast-path SELECT finds existing row.
    second = client.post("/webhooks/github", content=body, headers=headers)
    assert second.status_code == 200
    assert second.json()["review_id"] == first_review_id

    # Exactly one row in reviews + exactly one audit row.
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
        n_audit = await conn.scalar(text("SELECT COUNT(*) FROM audit_events"))
    assert n_reviews == 1
    assert n_audit == 1


@pytest.mark.asyncio
async def test_webhook_retention_uses_settings_field(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """`reviews.retention_expires_at` is `received_at +
    RetentionSettings.review_retention_ttl`, not a hardcoded constant.

    Pins the Codex finding fix that removed `_DEFAULT_REVIEW_RETENTION`
    in favor of operator-overridable settings per `DECISIONS.md#012/#014`.
    """
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)
    # Override the settings to a non-default value so we can verify the
    # webhook reads from settings, not a constant.
    app.state.retention_settings = RetentionSettings(
        review_retention_ttl=timedelta(days=7),
    )

    client = TestClient(app)
    body = json.dumps(_valid_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 202
    review_id_str = response.json()["review_id"]

    async with engine.connect() as conn:
        row = await conn.execute(
            text("SELECT created_at, retention_expires_at FROM reviews WHERE id = :id"),
            {"id": review_id_str},
        )
        review = row.first()
    assert review is not None
    # retention_expires_at - created_at is approximately 7 days (allow
    # small skew between received_at and server-default created_at).
    delta = review.retention_expires_at - review.created_at
    assert timedelta(days=6, hours=23) < delta < timedelta(days=7, hours=1), (
        f"Expected ~7 days between created_at and retention_expires_at; "
        f"got {delta} (settings should have provided 7-day TTL)"
    )


# ---------------------------------------------------------------------------
# Deferred — tracked at FOLLOWUPS.md#FUP-028
#
# The load-bearing branches the spec mandates — `uq_review_natural_key` →
# duplicate (200) vs any other constraint → re-raise — are covered at the
# unit tier in tests/unit/test_webhook_router_integrity_introspection.py
# by fabricating SQLAlchemyIntegrityError objects with the relevant
# `exc.orig.diag.constraint_name` values and asserting the classification.
#
# End-to-end transactional evidence (real psycopg, real `uq_review_natural_key`
# collision, real session rollback) is tracked under FUP-028; closure
# requires a fixture that deterministically forces either constraint-name
# branch to fire. Spec line 105's `test_webhook_idempotency_race.py` is
# also rolled into the same FUP — same orchestration prerequisite.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Tracked at FOLLOWUPS.md#FUP-028. End-to-end race needs a fixture "
        "to deterministically interleave two transactions between fast-path "
        "SELECT and slow-path INSERT. The branch logic — "
        "constraint_name == 'uq_review_natural_key' → 200 with existing "
        "review_id — is covered at the unit tier in "
        "tests/unit/test_webhook_router_integrity_introspection.py; this "
        "test would add round-trip evidence but the predicate is not "
        "untested."
    )
)
@pytest.mark.asyncio
async def test_webhook_natural_key_conflict_via_integrity_error_returns_200() -> None:
    """Concurrent race where the fast-path SELECT misses but the INSERT
    hits the UNIQUE on `uq_review_natural_key` → IntegrityError caught,
    constraint-name introspected, 200 with existing review_id."""
    # Fail-loud body: if someone removes the `@pytest.mark.skip` above
    # without implementing the test, this raises rather than silently
    # passing as an empty function. The skip-marker is the only thing
    # making this an acceptable not-yet-implemented placeholder.
    pytest.fail("FUP-028 placeholder body — implement before removing skip marker.")


@pytest.mark.skip(
    reason=(
        "Tracked at FOLLOWUPS.md#FUP-028. End-to-end audit-side "
        "IntegrityError needs a fixture to force the AgentTransitionEvent "
        "INSERT to collide (e.g., pre-insert a row with the pre-minted "
        "event_id). The branch logic — any constraint name other than "
        "'uq_review_natural_key' → re-raise — is covered at the unit tier "
        "in tests/unit/test_webhook_router_integrity_introspection.py."
    )
)
@pytest.mark.asyncio
async def test_webhook_audit_side_integrity_error_reraises() -> None:
    """Force the AgentTransitionEvent INSERT to raise (e.g., pre-insert
    a row with the same event_id pre-minted); the constraint-name
    introspection at step 10 should re-raise the IntegrityError (not
    misclassify as a natural-key duplicate). Returns 5xx."""
    # Fail-loud body: if someone removes the `@pytest.mark.skip` above
    # without implementing the test, this raises rather than silently
    # passing as an empty function.
    pytest.fail("FUP-028 placeholder body — implement before removing skip marker.")
