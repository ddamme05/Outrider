"""Unit tests for the #065 live-authorization check (`github/authz.py`).

Every case is driven through a fake `arequest` client — no real GitHub. Confirms the two
ordered checks (GET installation → POST repo-scoped token), the exact request shapes #065
prescribes, and the fail-closed collapse: only active-install + repo-accessible returns
`authorized`; suspended / uninstalled / repo-removed / network-error all fail closed.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from outrider.github.authz import (
    LiveAuthOutcome,
    check_installation_authorization,
    make_installation_authorizer,
)

_INSTALLATION_ID = 12345
_OWNER = "acme"
_REPO = "widgets"


def _resp(text: str) -> types.SimpleNamespace:
    """A 2xx response stand-in exposing `.text` (githubkit returns raw text; the helper
    does `json.loads(response.text)`)."""
    return types.SimpleNamespace(text=text)


def _active_installation() -> types.SimpleNamespace:
    return _resp(json.dumps({"id": _INSTALLATION_ID, "suspended_at": None}))


def _http_error(status: int | None, text: str = "") -> Exception:
    """A githubkit-`RequestFailed`-shaped exception: `.response.status_code` present for an
    HTTP error, absent entirely for a network/pre-response failure (status → None)."""
    exc = Exception(f"HTTP {status}")
    if status is not None:
        exc.response = types.SimpleNamespace(status_code=status, text=text)  # type: ignore[attr-defined]
    return exc


class _FakeAppClient:
    """Dispatches `arequest` by method: GET → the install-check result, POST → the
    token-mint result. Each result is either a response (returned) or an Exception
    (raised). Records calls for request-shape assertions."""

    def __init__(self, *, get_result: Any, post_result: Any = None) -> None:
        self._get = get_result
        self._post = post_result
        self.calls: list[tuple[str, str, Any]] = []

    async def arequest(
        self, method: str, path: str, *, json: Any = None, headers: Any = None
    ) -> Any:
        self.calls.append((method, path, json))
        result = self._get if method == "GET" else self._post
        if isinstance(result, Exception):
            raise result
        return result


async def _check(client: _FakeAppClient):
    return await check_installation_authorization(
        client, installation_id=_INSTALLATION_ID, owner=_OWNER, repo=_REPO
    )


# ---------------------------------------------------------------------------
# Happy path + request shapes.
# ---------------------------------------------------------------------------
async def test_active_install_accessible_repo_is_authorized() -> None:
    client = _FakeAppClient(
        get_result=_active_installation(), post_result=_resp('{"token":"ghs_x"}')
    )
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.AUTHORIZED
    assert result.authorized is True
    # #065's two ordered calls, exact shapes.
    assert client.calls[0] == ("GET", f"/app/installations/{_INSTALLATION_ID}", None)
    assert client.calls[1] == (
        "POST",
        f"/app/installations/{_INSTALLATION_ID}/access_tokens",
        {"repositories": [_REPO]},
    )


async def test_minted_token_never_appears_in_result_detail() -> None:
    client = _FakeAppClient(
        get_result=_active_installation(), post_result=_resp('{"token":"ghs_supersecret"}')
    )
    result = await _check(client)
    assert "ghs_supersecret" not in result.detail


# ---------------------------------------------------------------------------
# Install-check (GET) failures — all fail closed, POST never fires.
# ---------------------------------------------------------------------------
async def test_suspended_via_body_is_denied_without_minting() -> None:
    client = _FakeAppClient(
        get_result=_resp(json.dumps({"suspended_at": "2026-07-07T00:00:00Z"})),
    )
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.SUSPENDED
    assert result.authorized is False
    # Check 2 must NOT run once suspended is known.
    assert [c[0] for c in client.calls] == ["GET"]


async def test_uninstalled_via_get_404() -> None:
    client = _FakeAppClient(get_result=_http_error(404))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNINSTALLED
    assert [c[0] for c in client.calls] == ["GET"]


async def test_get_5xx_is_uncertain() -> None:
    client = _FakeAppClient(get_result=_http_error(503))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN


async def test_get_network_error_no_response_is_uncertain() -> None:
    client = _FakeAppClient(get_result=_http_error(None))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN


async def test_unparseable_install_body_is_uncertain() -> None:
    client = _FakeAppClient(get_result=_resp("not json{"))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN


# ---------------------------------------------------------------------------
# Token-mint (POST) failures — install active, repo probe fails → fail closed.
# ---------------------------------------------------------------------------
async def test_repo_removed_422_is_repo_inaccessible() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(422))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.REPO_INACCESSIBLE
    assert result.authorized is False
    assert [c[0] for c in client.calls] == ["GET", "POST"]


async def test_mint_403_is_suspended() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(403))
    assert (await _check(client)).outcome is LiveAuthOutcome.SUSPENDED


async def test_mint_404_is_uninstalled() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(404))
    assert (await _check(client)).outcome is LiveAuthOutcome.UNINSTALLED


async def test_mint_network_error_is_uncertain() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(None))
    assert (await _check(client)).outcome is LiveAuthOutcome.UNCERTAIN


async def test_mint_401_expired_jwt_fails_closed() -> None:
    # 401 = our own JWT problem, not the install's fault — but #065 fails closed on ANY
    # non-affirmative, so it must not authorize.
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(401))
    assert (await _check(client)).authorized is False


# ---------------------------------------------------------------------------
# The injected closure.
# ---------------------------------------------------------------------------
async def test_make_installation_authorizer_binds_client() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_resp("{}"))
    authorize = make_installation_authorizer(client)
    result = await authorize(_INSTALLATION_ID, _OWNER, _REPO)
    assert result.authorized is True
    assert client.calls[0][1] == f"/app/installations/{_INSTALLATION_ID}"


@pytest.mark.parametrize(
    "outcome",
    [
        LiveAuthOutcome.SUSPENDED,
        LiveAuthOutcome.UNINSTALLED,
        LiveAuthOutcome.REPO_INACCESSIBLE,
        LiveAuthOutcome.UNCERTAIN,
    ],
)
def test_only_authorized_is_authorized(outcome: LiveAuthOutcome) -> None:
    from outrider.github.authz import LiveAuthResult

    assert LiveAuthResult(outcome, "x").authorized is False
    assert LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "x").authorized is True
