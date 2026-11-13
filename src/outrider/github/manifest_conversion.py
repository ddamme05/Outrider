"""Exchange a one-use GitHub manifest code for App credentials.

The request has a timeout and is not automatically retried. Returned secrets
must stay out of logs; errors expose status metadata rather than response
content. The client secret is not part of the stored App credentials.

See DECISIONS.md#070 and vendor-payloads-normalized-at-boundary.
"""

from __future__ import annotations

import asyncio
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

# httpx per-phase timeout (seconds): bounds connect / read / write / pool INACTIVITY, not total
# wall-clock. githubkit defaults to none, so this alone stops a dead connection from hanging.
_CONVERSION_TIMEOUT_SECONDS: Final[float] = 30.0

# TOTAL wall-clock deadline (seconds) for the whole conversion await, enforced by `asyncio.timeout`.
# The per-phase httpx timeout above does NOT bound total duration — a slow-drip response (data
# arriving within each read interval) could run past `state_machine._STALE_CONVERTING_AFTER` (5 min)
# and be false-orphaned. This hard total keeps the request FAR below that threshold; a breach raises
# `TimeoutError` → `ManifestConversionError` → the callback orphans. auto_retry off (one attempt).
_CONVERSION_TOTAL_DEADLINE_SECONDS: Final[float] = 60.0


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
        # Hard TOTAL deadline (`asyncio.timeout`) around the whole await — the httpx per-phase
        # timeout can't bound total wall-clock. A breach raises TimeoutError, caught below → orphan.
        async with asyncio.timeout(_CONVERSION_TOTAL_DEADLINE_SECONDS):
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
