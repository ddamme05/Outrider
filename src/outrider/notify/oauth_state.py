"""Slack OAuth install `state` — a stateless, HMAC-signed CSRF token.

The `state` round-trips through Slack's authorize → callback redirect, so it MUST
be unforgeable AND bound to the originating admin's choices. It carries the
`installation_id`, the admin principal, the chosen `channel_id`, a random nonce,
and an expiry, HMAC-signed with `OUTRIDER_SLACK_STATE_SECRET`. The callback
verifies the signature + expiry and reads installation_id / admin / channel FROM
THE VERIFIED STATE — never from the (attacker-controllable) callback query params.

Stateless by design (dashboard-in-Slack spec): no pending-state table; the nonce +
expiry bound OAuth CSRF replay for this phase. A missing secret fails CLOSED
(raises) so a misconfigured deploy can't mint/accept unsigned state.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256

__all__ = [
    "STATE_SECRET_ENV",
    "SlackInstallState",
    "SlackStateError",
    "sign_state",
    "validate_channel_id",
    "verify_state",
]

# Env var NAME (not a secret value) — the HMAC key for OAuth state.
STATE_SECRET_ENV = "OUTRIDER_SLACK_STATE_SECRET"  # noqa: S105
_DEFAULT_TTL_SECONDS = 600  # 10 minutes — ample for the OAuth round-trip
# Slack channel id shape: C… (public) / G… (private) — uppercase alphanumerics.
_CHANNEL_ID_RE = re.compile(r"\A[A-Z0-9]{6,}\Z")


class SlackStateError(ValueError):
    """The OAuth `state` is missing, malformed, expired, or badly signed — OR the
    signing secret is unset. Fail-closed: the install-start / callback rejects."""


@dataclass(frozen=True)
class SlackInstallState:
    """The verified contents of a signed OAuth `state` (read by the callback)."""

    installation_id: int
    admin_id: str
    channel_id: str
    nonce: str
    exp: int


def _state_secret() -> bytes:
    """Read the HMAC key from env, fail-closed if absent. Read fresh per call (test
    monkeypatch + restart-free rotation), like the truncation-marker secret."""
    raw = os.environ.get(STATE_SECRET_ENV, "")
    if not raw:
        raise SlackStateError(
            f"{STATE_SECRET_ENV} must be set (non-empty) to sign/verify Slack OAuth state. "
            "The Slack install flow fails closed without it."
        )
    return raw.encode("utf-8")


def _sign(body: str) -> str:
    return base64.urlsafe_b64encode(
        hmac.new(_state_secret(), body.encode("utf-8"), sha256).digest()
    ).decode("ascii")


def validate_channel_id(channel_id: str) -> str:
    """Shape-gate a Slack channel id (non-empty, uppercase-alnum, ≥6 chars). Applied
    BEFORE signing into state so a malformed channel never round-trips, and again on
    verify. Returns the stripped, validated id; raises `SlackStateError` otherwise."""
    cid = channel_id.strip()
    if not _CHANNEL_ID_RE.match(cid):
        raise SlackStateError(f"invalid Slack channel_id shape: {channel_id!r}")
    return cid


def sign_state(
    *,
    installation_id: int,
    admin_id: str,
    channel_id: str,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> str:
    """Mint a signed `state` for the install-start redirect. `channel_id` is
    validated before signing; the returned token is `base64url(payload).hmac`."""
    channel = validate_channel_id(channel_id)
    payload = {
        "iid": installation_id,
        "admin": admin_id,
        "chan": channel,
        "nonce": secrets.token_urlsafe(16),
        "exp": int(time.time()) + ttl_seconds,
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"{body}.{_sign(body)}"


def verify_state(state: str) -> SlackInstallState:
    """Verify a callback `state`: signature (constant-time), then expiry, then field
    shape. Raises `SlackStateError` on any failure. The returned values — NOT the
    callback query params — are the trusted installation_id / admin / channel."""
    if not state or "." not in state:
        raise SlackStateError("missing or malformed state")
    body, _, sig = state.partition(".")
    if not hmac.compare_digest(sig, _sign(body)):
        raise SlackStateError("state signature mismatch")
    try:
        payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        iid = int(payload["iid"])
        admin_id = str(payload["admin"])
        channel_id = str(payload["chan"])
        nonce = str(payload["nonce"])
        exp = int(payload["exp"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SlackStateError("state payload malformed or missing fields") from exc
    if int(time.time()) >= exp:
        raise SlackStateError("state expired")
    return SlackInstallState(
        installation_id=iid,
        admin_id=admin_id,
        channel_id=validate_channel_id(channel_id),
        nonce=nonce,
        exp=exp,
    )
