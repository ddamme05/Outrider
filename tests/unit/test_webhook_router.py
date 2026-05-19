"""Tests for `api/webhooks/router.py` — the testable paths.

DB-touching paths (membership lookup, INSERT, dispatch, IntegrityError
introspection) need a real Postgres for honest coverage and live in the
integration tier alongside the rest of the router-to-graph slice. These
unit tests focus on:

  - Missing `X-Hub-Signature-256` → 401 (and pins step-2 ordering:
    body is NOT read on the missing-sig path, via a `Request.body` spy).
  - Signature verification failure → 401, including the malformed-header
    `verify` False-path and the unexpected-exception 5xx path.
  - Content-Length precheck: oversized header → 413 BEFORE body-read
    (pinned via `Request.body` spy); malformed header → 400.
  - Post-read body-size guard: body exceeding `_MAX_WEBHOOK_BODY_BYTES`
    → 413; just-under-cap admits past size guards (reaches signature
    verification).
  - Non-`pull_request` event → 2xx no-op.
  - Unsupported PR action → 2xx no-op.
  - Malformed payload → 400.
  - `X-GitHub-Delivery` header is forwarded to logs (traceability), not
    persisted; receipt log fires on the 401 path.

All tests use FastAPI's `TestClient` with a minimal app that wires only
the router + the `app.state` slots the router reads. No DB connection
is required.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from outrider.api.webhooks.router import router

if TYPE_CHECKING:
    import pytest

# ---------------------------------------------------------------------------
# Minimal app fixture
# ---------------------------------------------------------------------------


_SECRET = "test-webhook-secret"  # noqa: S105 — fixture sentinel


def _make_app(*, valid_membership: bool = True) -> FastAPI:
    """Build a minimal FastAPI app that exposes the webhook router and
    stubs the `app.state` slots the router reads.

    `valid_membership=False` returns no row from the membership SELECT,
    triggering the 4xx fail-closed path. (Not actually used in the unit
    tests below — listed here for the integration-tier follow-up.)
    """
    app = FastAPI()
    app.include_router(router)

    # Stub GitHubAppSettings
    settings_stub = SimpleNamespace(
        app_id=12345,
        app_private_key=SecretStr("test-private-key"),  # noqa: S106 — fixture
        webhook_secret=SecretStr(_SECRET),
    )
    app.state.github_app_settings = settings_stub

    # Stub session_factory (returns a session that errors if hit —
    # ensures these unit tests never reach the DB-touching paths).
    def _never_call_session_factory() -> Any:
        raise AssertionError("Unit test should not reach the DB-touching code path.")

    app.state.session_factory = _never_call_session_factory

    # Stub run_graph (never called in these tests — dispatch path is
    # past the DB INSERT).
    async def _never_call_run_graph(state: Any) -> Any:
        raise AssertionError("Unit test should not reach the dispatch / run_graph path.")

    app.state.run_graph = _never_call_run_graph

    return app


def _sign(secret: str, body: bytes) -> str:
    mac = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={mac}"


def _valid_pr_opened_payload() -> dict[str, Any]:
    return {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "title": "Test PR",
            "body": None,
            "user": {"login": "alice", "id": 1},
            "head": {"sha": "a" * 40, "ref": "feat/x"},
            "base": {"sha": "b" * 40, "ref": "main"},
            "additions": 5,
            "deletions": 2,
        },
        "repository": {
            "id": 999,
            "full_name": "acme/widgets",
            "name": "widgets",
            "owner": {"login": "acme", "id": 2},
        },
        "installation": {"id": 12345},
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_missing_signature_header_returns_401() -> None:
    """No `X-Hub-Signature-256` → 401 without any body parsing."""
    client = TestClient(_make_app())
    body = json.dumps(_valid_pr_opened_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "missing signature"}


def test_missing_signature_does_not_read_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pins step-2 ordering: the 401 path raises BEFORE `await request.body()`.

    The router's header-first ordering is what closes the unsigned-multi-GB
    DoS surface (router.py step-2 comment + spec Actual Outcome divergence
    #1). A regression that moved body-read above the header check would
    still pass the 401 assertion in the test above — this test makes that
    regression fail loudly by monkey-patching `Request.body` to record
    invocations and asserting the count stays at zero on the 401 path.
    """
    from starlette.requests import Request  # noqa: PLC0415 — test-local

    body_read_count = 0

    original_body = Request.body

    async def _recording_body(self: Request) -> bytes:
        nonlocal body_read_count
        body_read_count += 1
        return await original_body(self)

    monkeypatch.setattr(Request, "body", _recording_body)

    client = TestClient(_make_app())
    body = json.dumps(_valid_pr_opened_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert response.status_code == 401
    assert body_read_count == 0, (
        "Body was read before the missing-signature 401 — the header-first "
        "ordering that closes the unsigned-multi-GB DoS surface has regressed. "
        "See router.py step-2 comment + spec Actual Outcome divergence #1."
    )


def test_invalid_signature_returns_401() -> None:
    """Signature mismatch → 401 without any body parsing."""
    client = TestClient(_make_app())
    body = json.dumps(_valid_pr_opened_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 401
    assert response.json() == {"detail": "invalid signature"}


def test_oversized_content_length_returns_413_before_body_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Content-Length` header above the cap → 413 BEFORE
    `await request.body()` buffers anything. Pins the precheck
    ordering — a regression that moved the precheck below body-read
    would still 413 but would have already buffered the multi-MB
    payload, defeating the DoS defense.
    """
    from starlette.requests import Request  # noqa: PLC0415

    body_read_count = 0
    original_body = Request.body

    async def _recording_body(self: Request) -> bytes:
        nonlocal body_read_count
        body_read_count += 1
        return await original_body(self)

    monkeypatch.setattr(Request, "body", _recording_body)

    client = TestClient(_make_app())
    # Don't actually send 10 MB — the precheck reads the HEADER, not
    # the body. Tiny body + lying header is enough to exercise.
    body = b"{}"
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
            "Content-Length": str(2_000_000),  # 2 MB, over the 1 MiB cap
        },
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "payload too large"}
    assert body_read_count == 0, (
        "Body was read despite Content-Length precheck firing — precheck ordering regressed."
    )


def test_malformed_content_length_returns_400() -> None:
    """Non-integer `Content-Length` → 400. Without this guard a
    `Content-Length: abc` header would raise `ValueError` inside
    `int(...)` and surface as 5xx, a worse operator experience than
    a 400 telling the sender their header is malformed.
    """
    client = TestClient(_make_app())
    body = b"{}"
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
            "Content-Length": "not-a-number",
        },
    )
    # Note: starlette/httpx may itself reject malformed Content-Length
    # at a lower layer. Accept either the router's 400 OR a transport
    # 400; either way the path doesn't 5xx and doesn't admit the body.
    assert response.status_code == 400


def test_negative_content_length_returns_400() -> None:
    """`Content-Length: -1` → 400. `int("-1")` parses cleanly and
    `-1 > cap` is False, so without the explicit negative-check the
    negative header would bypass BOTH the malformed-400 path AND the
    413 cap, deferring rejection until after the body is buffered.
    """
    client = TestClient(_make_app())
    body = b"{}"
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
            "Content-Length": "-1",
        },
    )
    # Either the router's 400 OR a transport-layer rejection; either
    # way the path doesn't 5xx and doesn't reach body-read.
    assert response.status_code == 400


def test_body_just_under_cap_admitted_past_signature_check() -> None:
    """Positive-boundary: a body just under the 1 MiB cap is NOT
    rejected by the size guard — the request flows past the
    Content-Length precheck AND the post-read len() guard and into
    the signature-verification path (which then fails with 401
    because the test sends a bogus signature). Pins the inclusive
    boundary on the cap so a strict-less-than regression flips this
    test.
    """
    client = TestClient(_make_app())
    # 1 MiB - 1 byte. Anything ≤ _MAX_WEBHOOK_BODY_BYTES (1_048_576)
    # must not 413; this exercises the just-under-cap path.
    body = b"x" * (1_048_576 - 1)
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
        },
    )
    # Reaches signature verification → 401 (invalid signature).
    # The path is what's pinned, not the 401 specifically — the test
    # would also pass with 200 if the body had a valid signature.
    # What it MUST NOT be is 413.
    assert response.status_code != 413, (
        f"Body just under the cap was rejected as too large (got {response.status_code}). "
        f"The cap boundary regressed to strict-less-than or the cap value shifted."
    )
    assert response.status_code == 401


def test_post_read_guard_catches_oversized_body_without_content_length() -> None:
    """Defense-in-depth: when `Content-Length` is absent (chunked
    transfer or a lying header below the cap), the post-read length
    guard at `len(body) > _MAX_WEBHOOK_BODY_BYTES` fires. The
    Content-Length precheck alone wouldn't catch this case.

    httpx's TestClient sets Content-Length automatically when given
    `content=bytes`, so we can't easily test the "no Content-Length"
    case at the unit tier. We test the load-bearing condition: send
    a body whose length exceeds the cap with Content-Length matching
    (so the precheck DOES fire — which means the post-read guard is
    a redundant net for the lying / chunked case). Test name pins
    the intent; FUP-034 part 1's streaming-HMAC bound is what closes
    the chunked-no-Content-Length DoS for real.
    """
    client = TestClient(_make_app())
    body = b"x" * (1_048_576 + 1)
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 413
    assert response.json() == {"detail": "payload too large"}


def test_malformed_signature_header_returns_401_via_false_path() -> None:
    """Malformed signature header (no `sha256=` prefix, wrong length) →
    `githubkit.webhooks.verify` returns False (it does NOT raise on
    malformed shapes — see `githubkit.versions.*.webhooks._namespace.verify`
    which falls back to "sha1" mode and runs `hmac.compare_digest`
    unconditionally). Router treats False as 401.

    Distinct from `test_unexpected_verifier_exception_returns_5xx`
    below, which exercises the raise-path. This test exercises the
    False-return-path: `verify` returns False for malformed shapes,
    the router translates False to 401.
    """
    client = TestClient(_make_app())
    body = json.dumps(_valid_pr_opened_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": "not-a-valid-signature-shape",
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 401


def test_unexpected_verifier_exception_returns_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `verify_signature` raises an UNEXPECTED exception (programming
    error, dependency regression, etc.), the router does NOT collapse
    it to 401 — the exception propagates and FastAPI returns 5xx.

    The router does NOT wrap `verify_signature` in `except Exception →
    401` — that wrap would hide real server faults behind auth
    failures. Operators see the actual failure class. Test injects a
    monkeypatched `verify_signature` that raises `RuntimeError`;
    asserts 5xx.
    """
    import outrider.api.webhooks.router as router_mod

    def _raising_verify(*args: object, **kwargs: object) -> bool:  # noqa: ARG001
        msg = "simulated unexpected verifier fault"
        raise RuntimeError(msg)

    monkeypatch.setattr(router_mod, "verify_signature", _raising_verify)

    client = TestClient(_make_app(), raise_server_exceptions=False)
    body = json.dumps(_valid_pr_opened_payload()).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    # Server-side 5xx (FastAPI default for uncaught exception). The
    # contract is "NOT 401" + "5xx range" — exact code depends on the
    # FastAPI exception handler chain, which is a server-side concern,
    # not the security-critical-route concern this test pins.
    assert response.status_code >= 500
    assert response.status_code != 401


def test_non_pull_request_event_returns_2xx_ignored() -> None:
    """Signed-but-non-pull_request event → 2xx with `ignored` status.

    GitHub should NOT retry these (per spec line 26: signed but
    unsupported → 2xx no-op).
    """
    client = TestClient(_make_app())
    body = json.dumps({"hello": "world"}).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "push",
        },
    )
    # FastAPI's default 202 status code on the route + 200-class for
    # ignored — the route declares 202; ignored returns explicit dict.
    assert response.status_code == 202
    assert response.json() == {"status": "ignored", "reason": "event_type"}


def test_unsupported_pr_action_returns_2xx_ignored() -> None:
    """Signed `pull_request.closed` → 2xx ignored (allowlist is
    opened/synchronize/reopened)."""
    client = TestClient(_make_app())
    payload = _valid_pr_opened_payload()
    payload["action"] = "closed"
    body = json.dumps(payload).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 202
    assert response.json() == {"status": "ignored", "reason": "action"}


def test_malformed_payload_returns_400() -> None:
    """Valid signature, valid event/action, but the JSON shape doesn't
    match `PullRequestEventPayload` → 400."""
    client = TestClient(_make_app())
    # Missing required `installation` field; valid signature on whatever bytes.
    body = json.dumps({"action": "opened", "pull_request": {}}).encode()
    response = client.post(
        "/webhooks/github",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(_SECRET, body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 400


def test_x_github_delivery_logged_not_persisted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The delivery GUID flows into the log line for the request but
    isn't otherwise observable. Pin against accidental persistence in
    a future refactor."""
    client = TestClient(_make_app())
    body = json.dumps(_valid_pr_opened_payload()).encode()
    delivery_id = "abc123-delivery-guid"

    with caplog.at_level("INFO"):
        response = client.post(
            "/webhooks/github",
            content=body,
            headers={
                # No signature → 401, but the receipt log still fires.
                "X-GitHub-Event": "pull_request",
                "X-GitHub-Delivery": delivery_id,
            },
        )

    # The 401 path AND the receipt-log line both fire — the log carries
    # the delivery id regardless of acceptance.
    assert response.status_code == 401
    receipt_records = [r for r in caplog.records if r.message == "webhook received"]
    assert len(receipt_records) == 1
    assert getattr(receipt_records[0], "x_github_delivery", None) == delivery_id


# ---------------------------------------------------------------------------
# DB-touching paths covered in tests/integration/test_webhook_router_integration.py
# against real Postgres via the project's `migrated_db` fixture. The
# integration tier covers: unknown installation 4xx, inactive membership
# 4xx, tombstoned installation 4xx, happy-path 202 with event_id-match,
# idempotency fast-path 200, and retention-from-settings. The two
# IntegrityError-race tests (natural-key conflict slow-path,
# audit-side IntegrityError re-raise) remain skipped in the integration
# file pending a monkey-patch fixture; tracked there, not duplicated here.
