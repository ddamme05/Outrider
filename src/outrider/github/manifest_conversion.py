# See DECISIONS.md#070 — the GitHub App Manifest conversion (code → App credentials).
"""The App-Manifest handshake: `POST /app-manifests/{code}/conversions` (`DECISIONS.md#070`).

GitHub redirects the operator to `redirect_url?code=&state=` after they create the App from
our manifest. This wrapper exchanges that temporary `code` for the App's `id`, `pem`, and
`webhook_secret` (plus non-secret `slug`/`client_id` and the response-verifiable `owner`/
`permissions`/`events`). The endpoint is **unauthenticated** — the `code` IS the credential; GitHub
rejects tokens on it — so we use an unauthenticated `GitHub()` client, but NOT a default one:
`auto_retry=False` (the `code` is single-use — a retried POST would re-send a spent credential; a
failed conversion must orphan, never retry) and a bounded `timeout` (githubkit defaults to no
timeout; the state machine orphans a stale `CONVERTING` after 5 min ASSUMING the request is bounded
well below that — see `_CONVERSION_TIMEOUT_SECONDS`).

Boundary duties (githubkit confined here, `vendor-sdks-only-in-wrappers`):
- The `code` is external input → percent-encoded into the path segment (no path injection).
- `client_secret` is present in the response but **NOT read into the domain model** — OAuth
  user-tokens are a `#070` non-goal, so it is never persisted or surfaced (credential minimization).
- `pem` / `webhook_secret` come back as `SecretStr` and are never logged; a non-201 / malformed
  response raises `ManifestConversionError` with the status only (no body — it may echo attacker
  input), and the caller routes to `ORPHANED` (the App likely already exists).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import quote

from githubkit import GitHub
from pydantic import SecretStr

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["ManifestConversion", "ManifestConversionError", "convert_manifest_code"]

_API_VERSION_HEADER: Final[dict[str, str]] = {"X-GitHub-Api-Version": "2026-03-10"}

# Bounded conversion request timeout (seconds). githubkit defaults to no timeout, but the setup
# state machine orphans a stale `CONVERTING` after `state_machine._STALE_CONVERTING_AFTER` (5 min)
# on the assumption that a real conversion request resolves well below that — an unbounded request
# would let a genuinely in-flight conversion be false-orphaned. Kept far below 5 min; a hung request
# raises (→ the callback orphans) rather than lingering. auto_retry is off, so this is one attempt.
_CONVERSION_TIMEOUT_SECONDS: Final[float] = 30.0


class ManifestConversionError(RuntimeError):
    """The manifest conversion failed — a non-201 (4xx/422/timeout), a malformed body, or missing
    required fields. GitHub creates the App BEFORE redirecting with the code, so any failure means
    the App likely already exists: the caller routes to `ORPHANED`, never reuses the spent code."""


@dataclass(frozen=True)
class ManifestConversion:
    """The response-derived App credentials + the response-verifiable contract. `client_secret` is
    deliberately absent (minimization). `pem`/`webhook_secret` are `SecretStr` (never logged)."""

    app_id: int
    slug: str
    client_id: str | None
    pem: SecretStr
    webhook_secret: SecretStr
    owner_login: str
    permissions: Mapping[str, str]
    events: list[str]


def _req_str(value: object, field: str) -> str:
    # Reject a null/number/blank where a non-empty string is required — a bare `str(value)` would
    # turn `null` into "None" (non-blank), which would then encrypt + persist as a valid credential.
    if not isinstance(value, str) or not value.strip():
        raise ManifestConversionError(
            f"conversion response field {field!r} is missing or not a non-empty string"
        )
    return value


def _req_pos_int(value: object, field: str) -> int:
    # `bool` is an `int` subclass, so exclude it explicitly (`int(True)` would silently become 1).
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestConversionError(
            f"conversion response field {field!r} is not a positive integer"
        )
    return value


async def convert_manifest_code(
    code: str, *, github: GitHub[Any] | None = None
) -> ManifestConversion:
    """Exchange a manifest `code` for `ManifestConversion`. Unauthenticated (`GitHub()`; an
    explicit client may be injected for tests). Raises `ManifestConversionError` on any failure."""
    gh = (
        github
        if github is not None
        else GitHub(auto_retry=False, timeout=_CONVERSION_TIMEOUT_SECONDS)
    )
    path = f"/app-manifests/{quote(code, safe='')}/conversions"
    try:
        response = await gh.arequest("POST", path, headers=_API_VERSION_HEADER)
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise ManifestConversionError(
            f"manifest conversion request failed (status={status}); the App likely already exists"
        ) from exc
    # Require exactly 201 — githubkit raises on non-2xx, but an unexpected 2xx (200/204) must not
    # be parsed as a successful conversion.
    status_code = getattr(response, "status_code", None)
    if status_code != 201:
        raise ManifestConversionError(
            f"manifest conversion returned status {status_code}, expected 201"
        )
    try:
        data = json.loads(response.text)
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ManifestConversionError("conversion response body was not valid JSON") from exc
    if not isinstance(data, dict):
        raise ManifestConversionError("conversion response was not a JSON object")
    owner = data.get("owner")
    permissions = data.get("permissions")
    events = data.get("events")
    if not isinstance(owner, dict):
        raise ManifestConversionError("conversion response 'owner' is missing or not an object")
    if not isinstance(permissions, dict) or not isinstance(events, list):
        raise ManifestConversionError("conversion response permissions/events are malformed")
    client_id_raw = data.get("client_id")
    # Each field is TYPE-validated (never coerced) so a null/number never becomes a credential;
    # `client_secret` is deliberately never read (minimization, #070).
    return ManifestConversion(
        app_id=_req_pos_int(data.get("id"), "id"),
        slug=_req_str(data.get("slug"), "slug"),
        client_id=(_req_str(client_id_raw, "client_id") if client_id_raw is not None else None),
        pem=SecretStr(_req_str(data.get("pem"), "pem")),
        webhook_secret=SecretStr(_req_str(data.get("webhook_secret"), "webhook_secret")),
        owner_login=_req_str(owner.get("login"), "owner.login"),
        permissions={str(k): str(v) for k, v in permissions.items()},
        events=[str(e) for e in events],
    )
