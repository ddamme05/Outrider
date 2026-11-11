"""Check webhook authorization and transaction outcomes against PostgreSQL.

Use a minimal app to exercise membership checks, review and audit writes,
duplicate delivery, concurrent inserts and audit collisions. The graph runner
records dispatches without calling GitHub or a model."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import outrider.api.webhooks.router as webhook_router
from outrider.api.webhooks.router import router
from outrider.audit.config import RetentionSettings

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from httpx import Response
    from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SECRET = "test-webhook-secret"  # noqa: S105 — test fixture
_INSTALLATION_ID = 12345
_REPO_ID = 999
_LOGGER = logging.getLogger(__name__)


def _sign(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={mac}"


def _valid_payload(
    *,
    installation_id: int = _INSTALLATION_ID,
    repo_id: int = _REPO_ID,
    pr_number: int = 42,
    head_sha: str = "a" * 40,
    title: str = "Test PR",
) -> dict[str, Any]:
    return {
        "action": "opened",
        "pull_request": {
            "number": pr_number,
            "title": title,
            "body": None,
            "user": {"login": "alice", "id": 1},
            "head": {"sha": head_sha, "ref": "feat/x"},
            "base": {"sha": "b" * 40, "ref": "main"},
            "additions": 5,
            "deletions": 2,
            "draft": False,
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
    installation_suspended: bool = False,
    repository_selection: str = "selected",
    seed_membership: bool = True,
    membership_removed: bool = False,
) -> None:
    """Seed an Installation (+ optional InstallationRepository) row for the test."""
    tombstone_clause = "NOW()" if installation_tombstoned else "NULL"
    suspended_clause = "NOW()" if installation_suspended else "NULL"
    removed_clause = "NOW()" if membership_removed else "NULL"

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO installations "  # noqa: S608 — interpolated values are test-fixture literals (NOW()/NULL), not user input
                "(installation_id, app_slug, account_id, account_login, "
                " account_type, permissions_at_install, tombstoned_at, "
                " suspended_at, repository_selection) "
                f"VALUES (:iid, 'test-app', 1, 'acme', 'Organization', "
                f" '{{}}'::jsonb, {tombstone_clause}, {suspended_clause}, :sel)"
            ),
            {"iid": installation_id, "sel": repository_selection},
        )
        if seed_membership:
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

    # Credential-provider stub (`DECISIONS.md#070`): the router `await`s
    # `credential_provider.current()` per request for the webhook secret. The
    # snapshot's secret matches `_SECRET` so signed test bodies verify cleanly.
    # `app_id` deliberately distinct from `_INSTALLATION_ID` so an
    # app-id-vs-installation-id mixup in the webhook wiring fails loudly: a
    # regression that read `app_id` where the router should read
    # `installation.id` from the payload would surface as the wrong-row lookup
    # or a Pydantic validation error.
    _creds_stub = SimpleNamespace(
        app_id=98765,
        app_private_key=SecretStr("test-private-key"),  # noqa: S106
        webhook_secret=SecretStr(_SECRET),
    )

    class _StubProvider:
        async def current(self) -> Any:
            return _creds_stub

        async def is_configured(self) -> bool:
            return True

    app.state.credential_provider = _StubProvider()
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


@pytest.fixture
def webhook_sql_trace(
    webhook_app: tuple[FastAPI, AsyncEngine], caplog: pytest.LogCaptureFixture
) -> Iterator[list[dict[str, Any]]]:
    """Observe successful SQL, actual driver errors and transaction rollbacks."""
    _, engine = webhook_app
    records: list[dict[str, Any]] = []
    # Alembic's fileConfig disables existing loggers during the DB fixture.
    logger_was_disabled = _LOGGER.disabled
    _LOGGER.disabled = False
    caplog.set_level(logging.INFO, logger=__name__)
    _LOGGER.info("webhook SQL capture restored logger, previously disabled=%s", logger_was_disabled)

    def record(item: dict[str, Any]) -> None:
        records.append(item)
        _LOGGER.info("webhook SQL %s", json.dumps(item, sort_keys=True))

    def after_statement(
        connection: Any,
        _cursor: Any,
        statement: str,
        _parameters: Any,
        _context: Any,
        _executemany: bool,
    ) -> None:
        normalized = " ".join(statement.split())
        if " reviews" in normalized or " audit_events" in normalized:
            record({"kind": "statement", "connection": id(connection), "sql": normalized})

    def handle_error(context: Any) -> None:
        original = context.original_exception
        statement = context.statement
        record(
            {
                "kind": "error",
                "connection": id(context.connection),
                "sql": None if statement is None else " ".join(statement.split()),
                "driver": f"{type(original).__module__}.{type(original).__name__}",
                "sqlstate": getattr(original, "sqlstate", None),
                "constraint": getattr(getattr(original, "diag", None), "constraint_name", None),
            }
        )

    def rollback(connection: Any) -> None:
        record({"kind": "rollback", "connection": id(connection)})

    listeners = (
        ("after_cursor_execute", after_statement),
        ("handle_error", handle_error),
        ("rollback", rollback),
    )
    for name, listener in listeners:
        event.listen(engine.sync_engine, name, listener)
    try:
        yield records
    finally:
        for name, listener in listeners:
            event.remove(engine.sync_engine, name, listener)
        _LOGGER.disabled = logger_was_disabled


async def _post_signed_webhook(client: AsyncClient, body: bytes, delivery: str) -> Response:
    headers = {
        "X-Hub-Signature-256": _sign(_SECRET, body),
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": delivery,
    }
    _LOGGER.info(
        "webhook request %s",
        json.dumps({"path": "/webhooks/github", "headers": headers, "body": body.decode()}),
    )
    response = await client.post("/webhooks/github", content=body, headers=headers)
    _LOGGER.info(
        "webhook response %s",
        json.dumps({"delivery": delivery, "status": response.status_code, "body": response.text}),
    )
    return response


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
async def test_webhook_suspended_installation_returns_4xx(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Arc B2: `installations.suspended_at` non-NULL → 4xx (fail-closed) even
    with an otherwise-active membership row. Suspension is a reversible pause."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine, installation_suspended=True)

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
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
    assert n_reviews == 0
    assert app.state._dispatched == []  # noqa: SLF001


@pytest.mark.asyncio
async def test_webhook_all_selection_authorizes_without_repo_row(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """Arc B2: an `repository_selection='all'` install authorizes at the install
    level — a PR is accepted even with NO per-repo `installation_repositories` row."""
    app, engine = webhook_app
    await _seed_installation_and_membership(
        engine, repository_selection="all", seed_membership=False
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
    # Accepted: review created + dispatched (no per-repo row needed for `all`).
    assert response.status_code == 202
    async with engine.connect() as conn:
        n_reviews = await conn.scalar(text("SELECT COUNT(*) FROM reviews"))
    assert n_reviews == 1
    assert len(app.state._dispatched) == 1  # noqa: SLF001


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
    # Natural-key fields pinned: the SELECT already retrieves these,
    # the test was just missing the assertions. A regression in either
    # head_sha (idempotency key) or installation_id (token-minting
    # scope) would otherwise slip past the happy-path guard.
    assert review.head_sha == "a" * 40
    assert review.installation_id == _INSTALLATION_ID

    # AgentTransitionEvent row exists with matching event_id in row + payload.
    # COUNT first, then row inspection: a duplicate agent_transition row for
    # the same review_id would slip past `.first()`-only inspection.
    async with engine.connect() as conn:
        n_for_review = await conn.scalar(
            text("SELECT COUNT(*) FROM audit_events WHERE review_id = :rid"),
            {"rid": review_id_str},
        )
        ae_row = await conn.execute(
            text("SELECT event_id, event_type, payload FROM audit_events WHERE review_id = :rid"),
            {"rid": review_id_str},
        )
        ae = ae_row.first()
    assert n_for_review == 1, f"Expected exactly 1 audit row for review; got {n_for_review}."
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
async def test_webhook_persists_pr_title_immutable_on_redelivery(
    webhook_app: tuple[FastAPI, AsyncEngine],
) -> None:
    """The webhook persists `pull_request.title` to `reviews.pr_title` at
    creation, and a duplicate delivery (same `(repo_id, pr_number, head_sha)`
    natural key) with a DIFFERENT title does NOT mutate the captured value —
    the title is immutable-at-creation, riding the idempotency fast-path.
    Exercises the real persist path (router.py), not a direct column UPDATE.
    """
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)
    client = TestClient(app)

    # First delivery — captures the title at creation.
    body = json.dumps(_valid_payload(title="Add session token storage")).encode()
    first = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-Hub-Signature-256": _sign(_SECRET, body), "X-GitHub-Event": "pull_request"},
    )
    assert first.status_code == 202
    review_id = first.json()["review_id"]

    async with engine.connect() as conn:
        persisted = await conn.scalar(
            text("SELECT pr_title FROM reviews WHERE id = :id"), {"id": review_id}
        )
    assert persisted == "Add session token storage"

    # Re-delivery with the SAME natural key but an EDITED title → idempotent
    # 200, same review, title unchanged (immutable at creation).
    edited = json.dumps(_valid_payload(title="EDITED after the fact")).encode()
    second = client.post(
        "/webhooks/github",
        content=edited,
        headers={"X-Hub-Signature-256": _sign(_SECRET, edited), "X-GitHub-Event": "pull_request"},
    )
    assert second.status_code == 200
    assert second.json()["review_id"] == review_id

    async with engine.connect() as conn:
        after = await conn.scalar(
            text("SELECT pr_title FROM reviews WHERE id = :id"), {"id": review_id}
        )
    assert after == "Add session token storage"  # not mutated by the re-delivery


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
    # Override the settings to non-default values. Distinct values that
    # satisfy the llm_content <= findings <= review ordering validator make it
    # clear the webhook reads review_retention_ttl (7d) specifically, not a
    # constant or one of the sibling TTLs. The assertion below pins the 7-day
    # delta, so it stays unchanged.
    app.state.retention_settings = RetentionSettings(
        llm_content_retention_ttl=timedelta(days=5),
        findings_retention_ttl=timedelta(days=6),
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


@pytest.mark.parametrize(
    "authored_false",
    [False, True],
    ids=["ordinary-driver-classification", "authored-false-classification"],
)
@pytest.mark.asyncio
async def test_webhook_natural_key_conflict_via_integrity_error_returns_200(
    webhook_app: tuple[FastAPI, AsyncEngine],
    webhook_sql_trace: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    authored_false: bool,
) -> None:
    """Two real lookup misses produce one accepted review and one duplicate."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)
    real_session_factory = app.state.session_factory
    session_entries = 0
    lookup_exits = 0
    both_lookups_finished = asyncio.Event()

    @asynccontextmanager
    async def interleaved_sessions() -> AsyncGenerator[AsyncSession]:
        nonlocal session_entries, lookup_exits
        entry = session_entries
        session_entries += 1
        async with real_session_factory() as session:
            yield session
        if entry < 2:
            lookup_exits += 1
            _LOGGER.info("webhook lookup session exited %s", lookup_exits)
            if lookup_exits == 2:
                both_lookups_finished.set()
            await asyncio.wait_for(both_lookups_finished.wait(), timeout=10)

    monkeypatch.setattr(app.state, "session_factory", interleaved_sessions)
    body = json.dumps(_valid_payload()).encode()
    real_classifier = webhook_router._is_reviews_natural_key_conflict  # noqa: SLF001
    classification_results: list[bool] = []

    def authored_false_classifier(error: SQLAlchemyIntegrityError) -> bool:
        real_result = real_classifier(error)
        classification_results.append(real_result)
        _LOGGER.info(
            "authored classifier override %s",
            json.dumps(
                {
                    "real_result": real_result,
                    "authored_result": False,
                    "sqlstate": error.orig.sqlstate,
                    "constraint": error.orig.diag.constraint_name,
                }
            ),
        )
        assert real_result is True
        return False

    with monkeypatch.context() as classifier_patch:
        classifier_patch.setattr(webhook_router.logger, "disabled", False)
        caplog.set_level(logging.WARNING, logger=webhook_router.logger.name)
        if authored_false:
            classifier_patch.setattr(
                webhook_router, "_is_reviews_natural_key_conflict", authored_false_classifier
            )
        async with (
            AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
            asyncio.timeout(20),
            asyncio.TaskGroup() as tasks,
        ):
            first = tasks.create_task(_post_signed_webhook(client, body, "natural-key-first"))
            second = tasks.create_task(_post_signed_webhook(client, body, "natural-key-second"))
    responses = [first.result(), second.result()]
    assert lookup_exits == 2
    assert sorted(response.status_code for response in responses) == [200, 202]
    review_id = responses[0].json()["review_id"]
    assert all(
        response.json() == {"status": "running", "review_id": review_id} for response in responses
    )
    fallback_warnings = [
        record
        for record in caplog.records
        if record.name == webhook_router.logger.name
        and record.levelno == logging.WARNING
        and record.getMessage().startswith("webhook IntegrityError: natural-key row exists but ")
    ]
    if authored_false:
        assert classification_results == [True]
        assert len(fallback_warnings) == 1
        warning = fallback_warnings[0]
        assert warning.review_id == review_id
        assert warning.x_github_delivery in {"natural-key-first", "natural-key-second"}
        _LOGGER.info(
            "actual router fallback warning %s",
            json.dumps(
                {
                    "logger": warning.name,
                    "level": warning.levelname,
                    "message": warning.getMessage(),
                    "review_id": warning.review_id,
                    "x_github_delivery": warning.x_github_delivery,
                }
            ),
        )
    else:
        assert classification_results == []
        assert fallback_warnings == []

    errors = [item for item in webhook_sql_trace if item["kind"] == "error"]
    assert len(errors) == 1
    conflict = errors[0]
    assert (conflict["sqlstate"], conflict["constraint"]) == ("23505", "uq_review_natural_key")
    assert conflict["driver"] == "psycopg.errors.UniqueViolation"
    assert conflict["sql"].startswith("INSERT INTO reviews ")
    assert any(
        item["kind"] == "rollback" and item["connection"] == conflict["connection"]
        for item in webhook_sql_trace[webhook_sql_trace.index(conflict) + 1 :]
    )

    async with engine.connect() as connection:
        reviews = (
            await connection.execute(
                text("SELECT id, status, repo_id, pr_number, head_sha FROM reviews")
            )
        ).all()
        audit_rows = (
            await connection.execute(
                text("SELECT event_id, review_id, event_type, payload FROM audit_events")
            )
        ).all()
    assert len(reviews) == 1
    review = reviews[0]
    assert (str(review.id), review.status, review.repo_id, review.pr_number, review.head_sha) == (
        review_id,
        "running",
        _REPO_ID,
        42,
        "a" * 40,
    )
    assert len(audit_rows) == 1
    transition = audit_rows[0]
    assert str(transition.review_id) == review_id
    assert transition.event_type == "agent_transition"
    assert transition.payload["from_node"] == "webhook"
    assert transition.payload["to_node"] == "intake"
    assert transition.payload["event_id"] == str(transition.event_id)
    assert [str(state.review_id) for state in app.state._dispatched] == [review_id]  # noqa: SLF001


@pytest.mark.asyncio
async def test_webhook_audit_side_integrity_error_reraises(
    webhook_app: tuple[FastAPI, AsyncEngine],
    webhook_sql_trace: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real audit collision leaves the new natural key absent and undispatched."""
    app, engine = webhook_app
    await _seed_installation_and_membership(engine)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        original = await _post_signed_webhook(
            client, json.dumps(_valid_payload()).encode(), "audit-original"
        )
        assert original.status_code == 202
        async with engine.connect() as connection:
            original_review = dict(
                (await connection.execute(text("SELECT * FROM reviews"))).mappings().one()
            )
            original_audit = dict(
                (await connection.execute(text("SELECT * FROM audit_events"))).mappings().one()
            )
            assert (
                await connection.scalar(text("SELECT count(*) FROM reviews WHERE pr_number = 43"))
                == 0
            )
        app.state._dispatched.clear()  # noqa: SLF001
        webhook_sql_trace.clear()
        candidate_review_id = uuid4()
        _LOGGER.info(
            "webhook audit collision identities %s",
            json.dumps(
                {"review_id": str(candidate_review_id), "event_id": str(original_audit["event_id"])}
            ),
        )
        attempted_ids = iter((candidate_review_id, original_audit["event_id"]))
        body = json.dumps(_valid_payload(pr_number=43)).encode()
        with monkeypatch.context() as patch:
            patch.setattr(webhook_router, "uuid4", lambda: next(attempted_ids))
            rejected = await _post_signed_webhook(client, body, "audit-collision")
        assert rejected.status_code == 500

        errors = [item for item in webhook_sql_trace if item["kind"] == "error"]
        assert len(errors) == 1
        conflict = errors[0]
        assert (conflict["sqlstate"], conflict["constraint"]) == ("23505", "audit_events_pkey")
        assert conflict["driver"] == "psycopg.errors.UniqueViolation"
        assert conflict["sql"].startswith("INSERT INTO audit_events ")
        assert any(
            item["kind"] == "rollback" and item["connection"] == conflict["connection"]
            for item in webhook_sql_trace[webhook_sql_trace.index(conflict) + 1 :]
        )
        async with engine.connect() as connection:
            assert (
                await connection.scalar(
                    text(
                        "SELECT count(*) FROM reviews WHERE id = :id OR "
                        "(repo_id = :repo AND pr_number = 43 AND head_sha = :sha)"
                    ),
                    {"id": candidate_review_id, "repo": _REPO_ID, "sha": "a" * 40},
                )
                == 0
            )
            assert (
                await connection.scalar(
                    text("SELECT count(*) FROM audit_events WHERE review_id = :id"),
                    {"id": candidate_review_id},
                )
                == 0
            )
            reviews = (await connection.execute(text("SELECT * FROM reviews"))).mappings().all()
            audits = (await connection.execute(text("SELECT * FROM audit_events"))).mappings().all()
        assert [dict(row) for row in reviews] == [original_review]
        assert [dict(row) for row in audits] == [original_audit]
        assert app.state._dispatched == []  # noqa: SLF001

        redelivery = await _post_signed_webhook(client, body, "audit-redelivery")
        assert redelivery.status_code == 202
        recovered_review_id = redelivery.json()["review_id"]
        assert recovered_review_id != str(candidate_review_id)
        assert [str(state.review_id) for state in app.state._dispatched] == [recovered_review_id]  # noqa: SLF001
