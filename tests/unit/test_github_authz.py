"""Unit tests for the #065 live-authorization check (`github/authz.py`).

Every case is driven through a fake `arequest` client — no real GitHub. Confirms the two
ordered checks (GET installation → POST repo-scoped token by IMMUTABLE repo_id), the exact
request shapes #065 prescribes, and the fail-closed classification: only active-install +
repo-accessible returns `authorized`; suspended / uninstalled / repo-removed all fail closed,
and 401 / 429 / 5xx / network / missing-field are UNCERTAIN (fail closed but not mislabelled).
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
_REPO_ID = 999


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
        client, installation_id=_INSTALLATION_ID, repo_id=_REPO_ID
    )


# ---------------------------------------------------------------------------
# Happy path + request shapes (immutable repo_id scoping).
# ---------------------------------------------------------------------------
async def test_active_install_accessible_repo_is_authorized() -> None:
    client = _FakeAppClient(
        get_result=_active_installation(), post_result=_resp('{"token":"ghs_x"}')
    )
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.AUTHORIZED
    assert result.authorized is True
    # #065's two ordered calls; the token mint scopes by IMMUTABLE repository_ids.
    assert client.calls[0] == ("GET", f"/app/installations/{_INSTALLATION_ID}", None)
    assert client.calls[1] == (
        "POST",
        f"/app/installations/{_INSTALLATION_ID}/access_tokens",
        {"repository_ids": [_REPO_ID]},
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
    client = _FakeAppClient(get_result=_resp(json.dumps({"suspended_at": "2026-07-07T00:00:00Z"})))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.SUSPENDED
    assert result.authorized is False
    assert [c[0] for c in client.calls] == ["GET"]  # check 2 must NOT run


async def test_missing_suspended_at_field_is_uncertain() -> None:
    # The affirmative signal (an explicit suspended_at) is absent → cannot conclude active.
    client = _FakeAppClient(get_result=_resp(json.dumps({"id": _INSTALLATION_ID})))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN
    assert [c[0] for c in client.calls] == ["GET"]


async def test_non_object_install_body_is_uncertain() -> None:
    client = _FakeAppClient(get_result=_resp("[]"))  # valid JSON, wrong shape
    assert (await _check(client)).outcome is LiveAuthOutcome.UNCERTAIN


async def test_uninstalled_via_get_404() -> None:
    client = _FakeAppClient(get_result=_http_error(404))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNINSTALLED
    assert [c[0] for c in client.calls] == ["GET"]


async def test_get_5xx_is_uncertain() -> None:
    assert (await _check(_FakeAppClient(get_result=_http_error(503)))).outcome is (
        LiveAuthOutcome.UNCERTAIN
    )


async def test_get_network_error_no_response_is_uncertain() -> None:
    assert (await _check(_FakeAppClient(get_result=_http_error(None)))).outcome is (
        LiveAuthOutcome.UNCERTAIN
    )


async def test_unparseable_install_body_is_uncertain() -> None:
    assert (await _check(_FakeAppClient(get_result=_resp("not json{")))).outcome is (
        LiveAuthOutcome.UNCERTAIN
    )


# ---------------------------------------------------------------------------
# Token-mint (POST) failures. The GET already settled existence + suspension reliably;
# the pinned 2026-03-10 REST contract attributes no reliable cause to ANY mint-failure
# status (403 "Forbidden" / 404 / 422 "validation or spam" / 401 / 429 / 5xx), so every
# one fails closed as UNCERTAIN — never mislabelled suspended / repo-inaccessible.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status", [401, 403, 404, 422, 429, 500, 503, None])
async def test_any_token_mint_failure_is_uncertain(status: int | None) -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_http_error(status))
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN
    assert result.authorized is False
    assert [c[0] for c in client.calls] == ["GET", "POST"]  # both checks ran


# ---------------------------------------------------------------------------
# The injected closure.
# ---------------------------------------------------------------------------
async def test_make_installation_authorizer_binds_client() -> None:
    client = _FakeAppClient(get_result=_active_installation(), post_result=_resp("{}"))
    authorize = make_installation_authorizer(client)
    result = await authorize(_INSTALLATION_ID, _REPO_ID)
    assert result.authorized is True
    assert client.calls[1][2] == {"repository_ids": [_REPO_ID]}


@pytest.mark.parametrize(
    "outcome",
    [
        LiveAuthOutcome.SUSPENDED,
        LiveAuthOutcome.UNINSTALLED,
        LiveAuthOutcome.UNCERTAIN,
    ],
)
def test_only_authorized_is_authorized(outcome: LiveAuthOutcome) -> None:
    from outrider.github.authz import LiveAuthResult

    assert LiveAuthResult(outcome, "x").authorized is False
    assert LiveAuthResult(LiveAuthOutcome.AUTHORIZED, "x").authorized is True
