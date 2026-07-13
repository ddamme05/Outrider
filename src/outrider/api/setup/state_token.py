# See DECISIONS.md#070 — the signed setup `state` (CSRF + single-use-nonce carrier).
"""The onboarding `state` — an HMAC-signed, expiring nonce carrier (`DECISIONS.md#070`).

Mirrors `notify/oauth_state.py` (Slack OAuth state): the `state` round-trips through GitHub's App
registration → callback redirect, so it MUST be unforgeable AND bound to the originating attempt. It
carries the **raw single-use nonce** (whose sha256 is stored in `setup_nonce`) and an expiry,
HMAC-signed with `OUTRIDER_SETUP_STATE_SECRET`. The callback verifies signature + expiry and reads
the nonce FROM THE VERIFIED STATE — never from the attacker-controllable callback query.

A signed+expiring-only state is NOT enough: it is replayable with a *different* attacker `code`,
which would inject an attacker's App as the root identity. The single-use nonce (consumed atomically
at callback, `state_machine.consume_callback`) is what closes that; this module only signs/verifies
the carrier. A missing/weak secret fails CLOSED (raises) so a misconfigured deploy can't mint or
accept unsigned state.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256

__all__ = [
    "SETUP_STATE_SECRET_ENV",
    "SetupStateError",
    "SetupStateToken",
    "sign_state",
    "validate_setup_state_secret",
    "verify_state",
]

# Env var NAME (not a secret value) — the HMAC key for the setup state.
SETUP_STATE_SECRET_ENV = "OUTRIDER_SETUP_STATE_SECRET"  # noqa: S105
# Every sibling secret root the setup-state secret must be DISTINCT from (DECISIONS.md#070): reusing
# one value across roles means a leak of one compromises the others. Checked in the validator —
# mirrors credential_crypto's key separation, widened to ALL signing / auth / encryption roots in
# the codebase (the enc keys are comma-separated MultiFernet sets, compared componentwise).
_SIBLING_SECRET_ENVS: tuple[str, ...] = (
    "OUTRIDER_ADMIN_API_KEY",
    "OUTRIDER_AGENT_API_KEY",
    "OUTRIDER_GITHUB_APP_PRIVATE_KEY",
    "OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY",
    "OUTRIDER_GITHUB_WEBHOOK_SECRET",
    "OUTRIDER_SLACK_CLIENT_SECRET",
    "OUTRIDER_SLACK_STATE_SECRET",
    "OUTRIDER_TOKEN_ENC_KEY",
    "OUTRIDER_TRUNCATION_HMAC_SECRET",
)
# Known placeholders shipped in .env.example (+ usual suspects); reject a verbatim copy. Mirrors
# github/config.py / notify/oauth_state.py — keep in sync.
_PLACEHOLDER_SECRETS: frozenset[str] = frozenset(
    {
        "replace-me",
        "replace-me-with-a-long-random-secret",
        "change-me",
        "changeme",
        "secret",
        "password",
        "your-secret-here",
    }
)
# Min length (chars). A short non-placeholder secret slips the placeholder set but has too little
# entropy to be a credible HMAC root; 32 is the floor a `secrets.token_urlsafe(32)` (~43 chars)
# clears comfortably. Mirrors notify/oauth_state._MIN_STATE_SECRET_LEN.
_MIN_STATE_SECRET_LEN = 32


class SetupStateError(ValueError):
    """The setup `state` is missing, malformed, expired, or badly signed — OR the signing secret is
    unset/weak. Fail-closed: `POST /setup` (mint) and `GET /setup/callback` (verify) reject."""


@dataclass(frozen=True)
class SetupStateToken:
    """The verified contents of a signed setup `state` (read by the callback): the raw single-use
    nonce (hashed + matched against `setup_nonce` for atomic delete-on-consume) and its expiry."""

    nonce: str
    exp: int


def _state_secret() -> bytes:
    """Read the HMAC key from env, fail-closed if absent, a known placeholder, OR too short. Read
    fresh per call (test monkeypatch + restart-free rotation). This key is the CSRF unforgeability
    root, so a weak/default/low-entropy value is a real forgery risk."""
    raw = os.environ.get(SETUP_STATE_SECRET_ENV, "").strip()
    if not raw:
        raise SetupStateError(
            f"{SETUP_STATE_SECRET_ENV} must be set (non-empty) to sign/verify the setup state. "
            "The manifest onboarding flow fails closed without it."
        )
    if raw.lower() in _PLACEHOLDER_SECRETS:
        raise SetupStateError(
            f"{SETUP_STATE_SECRET_ENV} is a known placeholder ({raw!r}); it is the CSRF "
            "unforgeability root for the setup state — set a real random secret."
        )
    if len(raw) < _MIN_STATE_SECRET_LEN:
        raise SetupStateError(
            f"{SETUP_STATE_SECRET_ENV} is too short ({len(raw)} chars); it signs the setup state "
            f"CSRF token, so use at least {_MIN_STATE_SECRET_LEN} chars — e.g. "
            '`python -c "import secrets; print(secrets.token_urlsafe(32))"`.'
        )
    return raw.encode("utf-8")


def _sign(body: str) -> str:
    return base64.urlsafe_b64encode(
        hmac.new(_state_secret(), body.encode("utf-8"), sha256).digest()
    ).decode("ascii")


def sign_state(*, nonce: str, ttl_seconds: int) -> str:
    """Mint a signed `state` carrying the raw single-use `nonce` and an expiry `ttl_seconds` out.
    The returned token is `base64url(payload).hmac`. The caller stores `sha256(nonce)` +
    `now + ttl_seconds` in `setup_nonce` under the same transaction that CAS-starts the attempt."""
    if not nonce:
        raise SetupStateError("refusing to sign an empty setup nonce")
    payload = {"nonce": nonce, "exp": int(time.time()) + ttl_seconds}
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    return f"{body}.{_sign(body)}"


def verify_state(state: str) -> SetupStateToken:
    """Verify a callback `state`: signature (constant-time), then expiry, then field shape. Raises
    `SetupStateError` on any failure. The returned nonce — NOT the callback query — is the trusted
    single-use token the caller hashes and atomically consumes."""
    if not state or "." not in state:
        raise SetupStateError("missing or malformed state")
    body, _, sig = state.partition(".")
    if not hmac.compare_digest(sig, _sign(body)):
        raise SetupStateError("state signature mismatch")
    try:
        payload = json.loads(base64.urlsafe_b64decode(body.encode("ascii")))
        nonce = payload["nonce"]
        exp = payload["exp"]
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise SetupStateError("state payload malformed or missing fields") from exc
    # Type-check BEFORE use — a JSON null/number nonce must not slip through `str()` coercion (which
    # would turn null into the truthy "None" and defeat the empty-nonce guard). Unreachable via the
    # real signer, but the state is post-signature attacker input, so keep the guard exact.
    if not isinstance(nonce, str) or not nonce:
        raise SetupStateError("state carries an empty or non-string nonce")
    if not isinstance(exp, int):
        raise SetupStateError("state carries a non-integer expiry")
    if int(time.time()) >= exp:
        raise SetupStateError("state expired")
    return SetupStateToken(nonce=nonce, exp=exp)


def _comma_key_set(raw: str) -> frozenset[str]:
    """The set of individual key strings in a (possibly comma-separated) secret env value — so a
    single HMAC secret and a MultiFernet key set are both reduced to their components for reuse
    checks. Mirrors `credential_crypto._key_values`."""
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def validate_setup_state_secret() -> None:
    """Eager boot validation for `database` mode: assert `OUTRIDER_SETUP_STATE_SECRET` is present,
    non-placeholder, long enough, AND distinct from every sibling secret root (`DECISIONS.md#070`) —
    so a missing/weak/reused key surfaces at boot, not at the first `POST /setup`. Raises
    `SetupStateError` on any failure; returns None. Mirrors `validate_credential_enc_key`.
    """
    _state_secret()  # present + non-placeholder + length
    ours = _comma_key_set(os.environ.get(SETUP_STATE_SECRET_ENV, ""))
    for sibling in _SIBLING_SECRET_ENVS:
        # Componentwise: split each on comma so a match against ONE key in a comma-separated
        # MultiFernet key set (the enc keys) is caught, not just whole-string equality.
        if ours & _comma_key_set(os.environ.get(sibling, "")):
            raise SetupStateError(
                f"{SETUP_STATE_SECRET_ENV} reuses a value from {sibling}; #070 requires "
                "a dedicated secret distinct from every sibling signing/auth/encryption root — "
                "generate a separate one."
            )
