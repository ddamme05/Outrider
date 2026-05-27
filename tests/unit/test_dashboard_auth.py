"""Unit tests for the dashboard `require_admin_api_key` dependency.

Covers the M12 step-order contract's first slot (auth fires FIRST) and
the failure-uniform 401 response shape (no leak via response variation).
HMAC `compare_digest` is the timing-oracle defense.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from outrider.api.dashboard.auth import require_admin_api_key


def _app_with_key(api_key: str | None) -> TestClient:
    """Build a tiny FastAPI app with one auth-protected endpoint."""
    from fastapi import Depends

    app = FastAPI()
    if api_key is not None:
        app.state.admin_api_key = SecretStr(api_key)

    @app.get("/protected", dependencies=[Depends(require_admin_api_key)])
    async def protected() -> dict[str, str]:
        return {"ok": "yes"}

    return TestClient(app)


def test_missing_authorization_header_returns_401() -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


def test_wrong_prefix_returns_401() -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": "Basic correct-key"})
    assert resp.status_code == 401
    assert resp.json() == {"detail": "unauthorized"}


# Canonical 401 response body — used by every unauthorized-branch test
# below. Pinning both status AND body ensures every failure path emits
# the SAME uniform response shape (no leak via response variation).
_UNAUTHORIZED_BODY = {"detail": "unauthorized"}


def test_wrong_key_returns_401() -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": "Bearer wrong-key"})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


def test_correct_key_returns_200() -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": "Bearer correct-key"})
    assert resp.status_code == 200
    assert resp.json() == {"ok": "yes"}


def test_unwired_admin_key_returns_401() -> None:
    """If lifespan didn't install `app.state.admin_api_key`, auth fails 401
    (not 500) so the response surface is uniform."""
    client = _app_with_key(None)
    resp = client.get("/protected", headers={"Authorization": "Bearer anything"})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


def test_empty_token_after_bearer_returns_401() -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


def test_longer_submitted_key_returns_401() -> None:
    """An attacker submitting a longer-than-expected ASCII key is
    rejected via `hmac.compare_digest`'s length-mismatch (early-return
    without leaking via timing)."""
    client = _app_with_key("short")
    resp = client.get("/protected", headers={"Authorization": "Bearer shortlongersuffix"})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


def test_shorter_submitted_key_returns_401() -> None:
    """Mirror: a shorter-than-expected key fails the length check first."""
    client = _app_with_key("longer-key-text")
    resp = client.get("/protected", headers={"Authorization": "Bearer longer"})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


@pytest.mark.parametrize(
    "header_value",
    [
        "",
        "Bearer",  # no space, no token
        "BeArEr correct-key",  # case-sensitive prefix per the spec
        "Token correct-key",
    ],
)
def test_malformed_authorization_header_returns_401(header_value: str) -> None:
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": header_value})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


def test_app_with_partial_authorization_prefix_returns_401() -> None:
    """A header that LOOKS like Bearer but isn't (`Bearertoken`) fails."""
    client = _app_with_key("correct-key")
    resp = client.get("/protected", headers={"Authorization": "Bearertoken"})
    assert resp.status_code == 401
    assert resp.json() == _UNAUTHORIZED_BODY


# Defensive: confirm the module exports.
def test_require_admin_api_key_is_async_callable() -> None:
    import inspect

    assert inspect.iscoroutinefunction(require_admin_api_key)


def test_module_imports_clean() -> None:
    """Structural import check: `outrider.api.dashboard.auth` loads
    cleanly and re-exports `require_admin_api_key`. A breakage here
    would surface at collection time, not in this assert."""
    from outrider.api.dashboard import auth as _auth

    assert _auth.require_admin_api_key is require_admin_api_key
