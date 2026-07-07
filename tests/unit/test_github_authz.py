"""Unit tests for the #065 live-authorization check (`github/authz.py`).

Every case is driven through a fake `arequest` client — no real GitHub. Confirms the two
ordered checks (GET installation → POST repo-scoped token by IMMUTABLE repo_id), the exact
request shapes #065 prescribes, and the fail-closed classification: only active-install +
repo-accessible returns `authorized`; suspended / uninstalled / repo-removed all fail closed,
and 401 / 429 / 5xx / network / missing-field are UNCERTAIN (fail closed but not mislabelled).
"""

from __future__ import annotations

import asyncio
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
_SETTINGS_SENTINEL: Any = object()  # make_app_client is monkeypatched, so settings is opaque


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


class _ExplodingResponse:
    """A 2xx-shaped response whose `.text` raises an UNEXPECTED error — simulates a probe-body
    defect (the probe only guards json/Attribute/Type errors, so a RuntimeError escapes)."""

    @property
    def text(self) -> str:
        raise RuntimeError("unexpected probe defect")


class _FakeAppClient:
    """Async-context-manager fake (check_installation_authorization enters it under
    `async with` so the GET+POST share one client). Dispatches `arequest` by method: GET →
    the install-check result, POST → the token-mint result. Each result is either a response
    (returned) or an Exception (raised). Records calls + enter/exit for lifecycle assertions."""

    def __init__(
        self,
        *,
        get_result: Any = None,
        post_result: Any = None,
        enter_error: BaseException | None = None,
        exit_error: BaseException | None = None,
    ) -> None:
        self._get = get_result
        self._post = post_result
        self._enter_error = enter_error
        self._exit_error = exit_error
        self.calls: list[tuple[str, str, Any]] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> _FakeAppClient:
        self.entered += 1
        if self._enter_error is not None:
            raise self._enter_error
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1
        if self._exit_error is not None:
            raise self._exit_error

    async def arequest(
        self, method: str, path: str, *, json: Any = None, headers: Any = None
    ) -> Any:
        self.calls.append((method, path, json))
        result = self._get if method == "GET" else self._post
        if isinstance(result, BaseException):  # BaseException, so a CancelledError raises too
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
async def test_check_context_manages_the_client_once() -> None:
    """check_installation_authorization async-with-scopes the client so its GET + POST share
    one httpx client (githubkit's reusing-client guidance) — enter/exit exactly once."""
    client = _FakeAppClient(get_result=_active_installation(), post_result=_resp("{}"))
    await check_installation_authorization(
        client, installation_id=_INSTALLATION_ID, repo_id=_REPO_ID
    )
    assert client.entered == 1
    assert client.exited == 1
    assert [c[0] for c in client.calls] == ["GET", "POST"]  # both inside the one context


async def test_authorizer_builds_and_scopes_a_fresh_client_per_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_installation_authorizer takes SETTINGS and constructs a FRESH client per call via
    make_app_client (a githubkit CM cannot be entered twice; the un-entered path leaks). Two
    authorizations build two clients, each context-managed exactly once."""
    clients: list[_FakeAppClient] = []
    seen_settings: list[Any] = []

    def _fake_make_app_client(settings: Any) -> _FakeAppClient:
        seen_settings.append(settings)
        c = _FakeAppClient(get_result=_active_installation(), post_result=_resp("{}"))
        clients.append(c)
        return c

    monkeypatch.setattr("outrider.github.authz.make_app_client", _fake_make_app_client)
    authorize = make_installation_authorizer(_SETTINGS_SENTINEL)

    r1 = await authorize(_INSTALLATION_ID, _REPO_ID)
    r2 = await authorize(_INSTALLATION_ID, _REPO_ID)

    assert r1.authorized is True and r2.authorized is True
    assert len(clients) == 2  # a fresh client per authorization, not one shared instance
    assert seen_settings == [_SETTINGS_SENTINEL, _SETTINGS_SENTINEL]
    assert all(c.entered == 1 and c.exited == 1 for c in clients)
    assert clients[0].calls[1][2] == {"repository_ids": [_REPO_ID]}


# ---------------------------------------------------------------------------
# Client construction / lifecycle failures → UNCERTAIN (fail closed → intake `skipped`),
# NOT intake's `failed` path. Cancellation must still propagate (never swallowed).
# ---------------------------------------------------------------------------
async def test_client_enter_failure_is_uncertain() -> None:
    client = _FakeAppClient(
        get_result=_active_installation(), enter_error=RuntimeError("httpx client init failed")
    )
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN
    assert result.authorized is False
    assert client.calls == []  # the probe never ran — __aenter__ failed first


async def test_client_exit_failure_is_uncertain() -> None:
    # The GET+POST succeed, but closing the client raises on __aexit__ — still fail closed.
    client = _FakeAppClient(
        get_result=_active_installation(),
        post_result=_resp("{}"),
        exit_error=RuntimeError("client close failed"),
    )
    result = await _check(client)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN


async def test_client_construction_failure_is_uncertain(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(settings: Any) -> Any:
        raise RuntimeError("malformed App private key")

    monkeypatch.setattr("outrider.github.authz.make_app_client", _boom)
    authorize = make_installation_authorizer(_SETTINGS_SENTINEL)
    result = await authorize(_INSTALLATION_ID, _REPO_ID)
    assert result.outcome is LiveAuthOutcome.UNCERTAIN
    assert result.authorized is False


async def test_cancellation_propagates_from_client_enter() -> None:
    client = _FakeAppClient(get_result=_active_installation(), enter_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _check(client)


async def test_cancellation_propagates_from_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    def _cancel(settings: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr("outrider.github.authz.make_app_client", _cancel)
    authorize = make_installation_authorizer(_SETTINGS_SENTINEL)
    with pytest.raises(asyncio.CancelledError):
        await authorize(_INSTALLATION_ID, _REPO_ID)


async def test_unexpected_probe_defect_propagates_not_uncertain() -> None:
    """An UNEXPECTED exception from the probe body (a defect — the probe returns every
    expected failure as a result) must PROPAGATE to intake's `failed` path, NOT be masked as
    UNCERTAIN → skipped. The client is still context-managed on the way out."""
    client = _FakeAppClient(get_result=_ExplodingResponse())
    with pytest.raises(RuntimeError, match="unexpected probe defect"):
        await _check(client)
    # The lifecycle handler did NOT swallow it, and the client was still entered + closed.
    assert client.entered == 1
    assert client.exited == 1


async def test_probe_defect_wins_over_exit_failure() -> None:
    """Compound unwind: probe defect AND __aexit__ failure → the load-bearing BODY defect
    is re-raised (→ intake `failed`), NOT masked as UNCERTAIN by the close failure."""
    client = _FakeAppClient(
        get_result=_ExplodingResponse(), exit_error=RuntimeError("client close also failed")
    )
    with pytest.raises(
        RuntimeError, match="unexpected probe defect"
    ):  # body wins, not the close error
        await _check(client)
    assert client.exited == 1  # close was still attempted


async def test_probe_cancellation_wins_over_exit_failure() -> None:
    """Compound unwind: probe CancelledError AND __aexit__ failure → cancellation propagates
    (never swallowed by the close failure)."""
    client = _FakeAppClient(
        get_result=asyncio.CancelledError(), exit_error=RuntimeError("client close also failed")
    )
    with pytest.raises(asyncio.CancelledError):
        await _check(client)
    assert client.exited == 1  # close attempted; cancellation still won


async def test_standalone_exit_cancellation_propagates() -> None:
    """Precedence rule 1: the probe SUCCEEDS but `__aexit__` raises CancelledError → the
    cancellation propagates (a cancelled task must not resolve to a result)."""
    client = _FakeAppClient(
        get_result=_active_installation(),
        post_result=_resp("{}"),
        exit_error=asyncio.CancelledError(),
    )
    with pytest.raises(asyncio.CancelledError):
        await _check(client)


async def test_exit_cancellation_wins_over_probe_defect() -> None:
    """Precedence rule 1 over rule 2: a probe defect AND an exit `CancelledError` → the
    cancellation propagates (not the RuntimeError defect) — a cancelled task must not resolve."""
    client = _FakeAppClient(get_result=_ExplodingResponse(), exit_error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _check(client)


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
