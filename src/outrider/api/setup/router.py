# See DECISIONS.md#070 — the self-service onboarding HTTP surface.
"""The App-Manifest onboarding router (`DECISIONS.md#070`).

Four endpoints composing the Slice-2 engine + the manifest builder + the conversion wrapper:

- **`POST /setup`** (admin) — CAS-start the state machine, mint a signed+nonce `state`, build the
  manifest, and return it + the org's GitHub target URL for the dashboard to auto-submit.
- **`GET /setup/callback`** (public) — verify the signed `state`, atomically consume the nonce, run
  the conversion, verify the response binding, encrypt + activate, and redirect to the install
  screen. Any conversion/binding failure orphans the attempt.
- **`GET /setup/status`** (public, metadata-minimal) — the state-machine status; no credentials.
- **`POST /setup/reset`** (admin) — `ORPHANED → UNCONFIGURED` after operator cleans up on GitHub.

The router is built by `build_setup_router` with its runtime deps injected (the state machine, the
onboarding settings, and the conversion callable — test-faked). Admin auth reuses the dashboard's
`require_admin_api_key` (reads `app.state.admin_api_key`). Credentials are encrypted via
`credential_crypto` and never logged; the callback reads its trusted values from the VERIFIED state,
never the raw query.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, field_validator

from outrider.api.dashboard.auth import require_admin_api_key
from outrider.api.setup.binding import BindingMismatchError, verify_conversion_binding
from outrider.api.setup.manifest import EXPECTED_EVENTS, EXPECTED_PERMISSIONS, build_manifest
from outrider.api.setup.nonce import new_nonce
from outrider.api.setup.state_machine import (
    NONCE_TTL_SECONDS,
    SetupBinding,
    SetupConflictError,
    SetupIntegrityError,
    SetupNonceError,
    SetupStateMachine,
)
from outrider.api.setup.state_token import SetupStateError, sign_state, verify_state
from outrider.github.credential_crypto import encrypt_credential

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from outrider.api.setup.config import SetupSettings
    from outrider.github.manifest_conversion import ManifestConversion

__all__ = ["build_setup_router"]

_log = logging.getLogger(__name__)

# GitHub org/user login shape: 1–39 chars, alphanumeric with single interior hyphens, no leading or
# trailing hyphen. A shape gate before the login is placed in the GitHub target URL (GitHub is the
# real authority; this stops a malformed value from breaking / injecting into the URL).
_ORG_LOGIN_RE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")


def _default_app_name(org: str) -> str:
    """The default (operator-editable) App name for an org — deterministic so the callback can
    re-derive the exact manifest THIS attempt submitted and verify its digest."""
    return f"Outrider {org}"


def _verify_attempt_digest(settings: SetupSettings, binding: SetupBinding) -> None:
    """Deployment-continuity guard (`#070`) — the single-use nonce is the PRIMARY binding of the
    callback to the attempt; this re-derives the manifest from the stored org + CURRENT config and
    confirms its digest matches the one recorded at Start. A changed `OUTRIDER_PUBLIC_BASE_URL` (or
    app-name policy) yields a different digest — the App was created with URLs that no longer point
    here — so reject. Raises `BindingMismatchError` on drift (routed to `orphan()` by the saga)."""
    if binding.expected_org_login is None:
        raise BindingMismatchError("setup attempt has no bound org")
    _, current_digest = build_manifest(
        base_url=settings.base_url, name=_default_app_name(binding.expected_org_login)
    )
    if binding.manifest_contract_digest != current_digest:
        raise BindingMismatchError(
            "manifest digest mismatch — OUTRIDER_PUBLIC_BASE_URL changed since setup started"
        )


class SetupStartRequest(BaseModel):
    """`POST /setup` body: the operator's GitHub **org** login — the App is created org-owned so a
    private App installs only on the org whose repos Outrider reviews (`#066` one-org model)."""

    model_config = ConfigDict(extra="forbid")

    org: str

    @field_validator("org", mode="after")
    @classmethod
    def _validate_org(cls, v: str) -> str:
        org = v.strip()
        if not _ORG_LOGIN_RE.match(org):
            raise ValueError(
                "org must be a valid GitHub org login (1–39 chars, alphanumeric with interior "
                "hyphens)."
            )
        return org


class SetupStartResponse(BaseModel):
    """The manifest + the GitHub target URL (carrying the signed `state`) the dashboard auto-submits
    as an HTML form (`manifest` field POSTed to `target_url`)."""

    target_url: str
    manifest: dict[str, Any]


class SetupStatusResponse(BaseModel):
    """Public, metadata-minimal status — the state-machine state and a configured flag. No
    credentials, no attempt binding."""

    status: str
    configured: bool


def build_setup_router(
    *,
    machine: SetupStateMachine,
    settings: SetupSettings,
    convert: Callable[[str], Awaitable[ManifestConversion]],
) -> APIRouter:
    """Construct the `/setup` router with deps injected. `convert` is the manifest-conversion
    callable (`github.manifest_conversion.convert_manifest_code` in production, a fake in tests)."""
    router = APIRouter(prefix="/setup", tags=["setup"])

    @router.post(
        "",
        response_model=SetupStartResponse,
        dependencies=[Depends(require_admin_api_key)],
    )
    async def start_setup(body: SetupStartRequest) -> SetupStartResponse:
        raw_nonce, nonce_hash = new_nonce()
        state = sign_state(nonce=raw_nonce, ttl_seconds=NONCE_TTL_SECONDS)
        manifest, digest = build_manifest(
            base_url=settings.base_url, name=_default_app_name(body.org)
        )
        try:
            await machine.begin_setup(
                expected_org_login=body.org,
                expected_permissions=dict(EXPECTED_PERMISSIONS),
                expected_events=list(EXPECTED_EVENTS),
                manifest_contract_digest=digest,
                nonce_hash=nonce_hash,
            )
        except SetupConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"setup cannot start: the instance is {exc.actual} "
                    "(POST /setup/reset first if it is ORPHANED; re-onboarding a CONFIGURED "
                    "instance is not supported in V1)."
                ),
            ) from exc
        except SetupIntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="setup state corrupted"
            ) from exc
        target_url = (
            f"https://github.com/organizations/{quote(body.org, safe='')}/settings/apps/new"
            f"?{urlencode({'state': state})}"
        )
        return SetupStartResponse(target_url=target_url, manifest=manifest)

    @router.get("/callback")
    async def setup_callback(code: str, state: str) -> RedirectResponse:
        # 1. Verify the signed state; read the nonce from the VERIFIED state, never the raw query.
        try:
            token = verify_state(state)
        except SetupStateError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="invalid or expired setup state"
            ) from exc
        # 2. Atomically consume the nonce + AWAITING_CALLBACK → CONVERTING, returning the binding.
        try:
            binding = await machine.consume_callback(raw_nonce=token.nonce)
        except SetupNonceError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="setup nonce is invalid, expired, or already used",
            ) from exc
        except SetupConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"setup is not awaiting a callback (state is {exc.actual})",
            ) from exc
        # 3–5. Convert + verify binding + encrypt/activate, as ONE saga. GitHub creates the App
        # BEFORE redirecting, so ANY post-consume failure routes to orphan (spec §Failure →
        # ORPHANED): conversion 4xx/timeout/malformed, binding mismatch, encrypt failure, persist
        # crash, OR a SetupConflictError (the nonce is single-use so a concurrent activation can't
        # happen — a conflict here is an insert/CAS failure that left the attempt stuck CONVERTING).
        # orphan() is a CAS `WHERE status=CONVERTING`: a SAFE NO-OP if the state is already
        # CONFIGURED or ORPHANED, and it correctly orphans a stuck CONVERTING. Never persisted; the
        # operator lands on the status page to clean up on GitHub + reset.
        try:
            _verify_attempt_digest(settings, binding)
            conversion = await convert(code)
            verify_conversion_binding(
                owner_login=conversion.owner_login,
                permissions=conversion.permissions,
                events=conversion.events,
                binding=binding,
            )
            await machine.mark_configured(
                app_id=conversion.app_id,
                slug=conversion.slug,
                client_id=conversion.client_id,
                pem_ciphertext=encrypt_credential(conversion.pem),
                webhook_secret_ciphertext=encrypt_credential(conversion.webhook_secret),
            )
            return RedirectResponse(
                f"https://github.com/apps/{quote(conversion.slug, safe='')}/installations/new",
                status_code=302,
            )
        except Exception as exc:  # noqa: BLE001 — saga: any post-consume failure orphans
            # The onboarding observability trail is structured logs (spec §Trust boundary: no
            # audit_events). Log the failure TYPE only — never the message, body, or secrets.
            _log.warning("setup callback failed; orphaning attempt: %s", type(exc).__name__)
            try:
                await machine.orphan()
            except SetupIntegrityError:
                _log.error("setup_state singleton missing while orphaning a failed setup callback")
            return RedirectResponse(f"{settings.base_url}/setup/status", status_code=302)

    @router.get("/status", response_model=SetupStatusResponse)
    async def setup_status() -> SetupStatusResponse:
        try:
            current = await machine.current_status()
        except SetupIntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="setup state corrupted"
            ) from exc
        return SetupStatusResponse(status=current, configured=current == "CONFIGURED")

    @router.post("/reset", dependencies=[Depends(require_admin_api_key)])
    async def reset_setup() -> SetupStatusResponse:
        try:
            await machine.reset()
        except SetupConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"setup is not in a resettable state (state is {exc.actual}; reset needs "
                "ORPHANED)",
            ) from exc
        except SetupIntegrityError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="setup state corrupted"
            ) from exc
        return SetupStatusResponse(status="UNCONFIGURED", configured=False)

    return router
