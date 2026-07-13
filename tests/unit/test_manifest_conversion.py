"""Unit tests for `github/manifest_conversion` — the App Manifest conversion wrapper (#070).

Drives `convert_manifest_code` with a FAKE githubkit client (no network): the happy path parses +
discards `client_secret`, the `code` is percent-encoded into the path, and malformed / null / wrong-
type / non-201 responses fail closed (never coerce a null into a persisted credential).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from outrider.github.manifest_conversion import ManifestConversionError, convert_manifest_code

_FULL_BODY: dict[str, Any] = {
    "id": 4242,
    "slug": "acme-outrider",
    "owner": {"login": "acme"},
    "permissions": {"metadata": "read", "contents": "read", "pull_requests": "write"},
    "events": ["pull_request"],
    "client_id": "Iv1.dead",
    "client_secret": "SUPER-SECRET-NEVER-STORE",
    "pem": "-----BEGIN RSA PRIVATE KEY-----\nX\n-----END RSA PRIVATE KEY-----",
    "webhook_secret": "wh-secret",
}


class _Resp:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeGitHub:
    """Minimal stand-in for the githubkit client — only `arequest` is called."""

    def __init__(self, *, response: _Resp | None = None, exc: Exception | None = None) -> None:
        self._response = response
        self._exc = exc
        self.last_url: str | None = None

    async def arequest(self, method: str, url: str, headers: dict[str, str] | None = None) -> _Resp:
        self.last_url = url
        if self._exc is not None:
            raise self._exc
        assert self._response is not None
        return self._response


async def test_happy_path_parses_and_discards_client_secret() -> None:
    gh = _FakeGitHub(response=_Resp(201, json.dumps(_FULL_BODY)))
    conv = await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]
    assert conv.app_id == 4242
    assert conv.slug == "acme-outrider"
    assert conv.owner_login == "acme"
    assert conv.events == ["pull_request"]
    assert conv.permissions == {"metadata": "read", "contents": "read", "pull_requests": "write"}
    assert conv.pem.get_secret_value().startswith("-----BEGIN")
    assert conv.webhook_secret.get_secret_value() == "wh-secret"
    assert not hasattr(conv, "client_secret")  # discarded at the boundary (minimization)


async def test_code_percent_encoded_into_path() -> None:
    gh = _FakeGitHub(response=_Resp(201, json.dumps(_FULL_BODY)))
    await convert_manifest_code("a/b?c#d", github=gh)  # type: ignore[arg-type]
    assert gh.last_url == "/app-manifests/a%2Fb%3Fc%23d/conversions"


async def test_null_pem_rejected_not_coerced() -> None:
    """A `pem: null` must NOT coerce to the non-blank string "None" and get persisted."""
    gh = _FakeGitHub(response=_Resp(201, json.dumps({**_FULL_BODY, "pem": None})))
    with pytest.raises(ManifestConversionError, match="pem"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]


async def test_bool_id_rejected() -> None:
    gh = _FakeGitHub(response=_Resp(201, json.dumps({**_FULL_BODY, "id": True})))
    with pytest.raises(ManifestConversionError, match="'id'"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]


async def test_non_201_rejected() -> None:
    gh = _FakeGitHub(response=_Resp(200, json.dumps(_FULL_BODY)))
    with pytest.raises(ManifestConversionError, match="status 200"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]


async def test_malformed_json_rejected() -> None:
    gh = _FakeGitHub(response=_Resp(201, "<html>not json</html>"))
    with pytest.raises(ManifestConversionError, match="JSON"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]


async def test_missing_owner_rejected() -> None:
    body = {k: v for k, v in _FULL_BODY.items() if k != "owner"}
    gh = _FakeGitHub(response=_Resp(201, json.dumps(body)))
    with pytest.raises(ManifestConversionError, match="owner"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]


async def test_request_failure_maps_to_orphan_message() -> None:
    class _RequestFailedError(Exception):
        def __init__(self) -> None:
            self.response = _Resp(422, "")

    gh = _FakeGitHub(exc=_RequestFailedError())
    with pytest.raises(ManifestConversionError, match="already exists"):
        await convert_manifest_code("CODE", github=gh)  # type: ignore[arg-type]
