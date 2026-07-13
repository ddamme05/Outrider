"""Unit tests for `api/setup/state_token` — the signed setup `state` (#070).

Parallels `test_slack_oauth_state` (the sibling it mirrors): round-trip, tamper/expiry/malformed
rejection, secret fail-closed (missing / placeholder / too-short), and the #070 distinct-secret
requirement. All run for real against monkeypatched env secrets.
"""

from __future__ import annotations

import base64
import hmac
import json
import secrets
import time
from hashlib import sha256
from typing import Any

import pytest

from outrider.api.setup.state_token import (
    SETUP_STATE_SECRET_ENV,
    SetupStateError,
    sign_state,
    validate_setup_state_secret,
    verify_state,
)

_GOOD_SECRET = secrets.token_urlsafe(32)  # ~43 chars, clears the 32-char floor


def _forge(payload: dict[str, Any], secret: str) -> str:
    """Mint a VALIDLY-signed state with an arbitrary payload (the real signer can't emit these) —
    to prove the post-signature field guards are exact, not relying on the signer's typing."""
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")
    sig = base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), body.encode("utf-8"), sha256).digest()
    ).decode("ascii")
    return f"{body}.{sig}"


@pytest.fixture
def _secret(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, _GOOD_SECRET)
    return _GOOD_SECRET


def test_round_trip(_secret: str) -> None:
    token = verify_state(sign_state(nonce="abc123", ttl_seconds=600))
    assert token.nonce == "abc123"
    assert token.exp > 0


def test_tampered_body_rejected(_secret: str) -> None:
    body, _, sig = sign_state(nonce="abc123", ttl_seconds=600).partition(".")
    forged = body[:-1] + ("A" if body[-1] != "A" else "B")
    with pytest.raises(SetupStateError, match="signature mismatch"):
        verify_state(f"{forged}.{sig}")


def test_tampered_signature_rejected(_secret: str) -> None:
    body, _, _ = sign_state(nonce="abc123", ttl_seconds=600).partition(".")
    with pytest.raises(SetupStateError, match="signature mismatch"):
        verify_state(f"{body}.deadbeef")


def test_expired_rejected(_secret: str) -> None:
    with pytest.raises(SetupStateError, match="expired"):
        verify_state(sign_state(nonce="abc123", ttl_seconds=-1))


def test_malformed_rejected(_secret: str) -> None:
    with pytest.raises(SetupStateError, match="missing or malformed"):
        verify_state("no-dot-here")


def test_null_nonce_rejected(_secret: str) -> None:
    """A validly-signed but null nonce must NOT slip through as the truthy string "None"."""
    state = _forge({"nonce": None, "exp": int(time.time()) + 600}, _secret)
    with pytest.raises(SetupStateError, match="non-string nonce"):
        verify_state(state)


def test_non_int_exp_rejected(_secret: str) -> None:
    state = _forge({"nonce": "abc", "exp": "later"}, _secret)
    with pytest.raises(SetupStateError, match="non-integer expiry"):
        verify_state(state)


def test_cross_secret_does_not_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state signed under secret A must not verify under secret B (rotation isolation)."""
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, secrets.token_urlsafe(32))
    state = sign_state(nonce="abc123", ttl_seconds=600)
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, secrets.token_urlsafe(32))
    with pytest.raises(SetupStateError, match="signature mismatch"):
        verify_state(state)


def test_empty_nonce_sign_rejected(_secret: str) -> None:
    with pytest.raises(SetupStateError, match="empty setup nonce"):
        sign_state(nonce="", ttl_seconds=600)


def test_missing_secret_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SETUP_STATE_SECRET_ENV, raising=False)
    with pytest.raises(SetupStateError, match="must be set"):
        sign_state(nonce="abc123", ttl_seconds=600)


@pytest.mark.parametrize("placeholder", ["replace-me", "change-me", "secret", "your-secret-here"])
def test_placeholder_secret_rejected(monkeypatch: pytest.MonkeyPatch, placeholder: str) -> None:
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, placeholder)
    with pytest.raises(SetupStateError, match="placeholder"):
        validate_setup_state_secret()


def test_short_secret_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, "hunter2")  # non-placeholder but < 32 chars
    with pytest.raises(SetupStateError, match="too short"):
        validate_setup_state_secret()


def test_validate_passes_on_good_secret(_secret: str) -> None:
    validate_setup_state_secret()  # no raise


@pytest.mark.parametrize(
    "sibling",
    [
        "OUTRIDER_SLACK_STATE_SECRET",
        "OUTRIDER_ADMIN_API_KEY",
        "OUTRIDER_GITHUB_WEBHOOK_SECRET",
        "OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY",
        "OUTRIDER_TOKEN_ENC_KEY",
    ],
)
def test_reused_sibling_secret_rejected(monkeypatch: pytest.MonkeyPatch, sibling: str) -> None:
    """#070 dedicated-secret: the setup state secret must be DISTINCT from every sibling root."""
    shared = secrets.token_urlsafe(32)
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, shared)
    monkeypatch.setenv(sibling, shared)
    with pytest.raises(SetupStateError, match="reuses the value"):
        validate_setup_state_secret()


def test_distinct_sibling_secrets_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SETUP_STATE_SECRET_ENV, secrets.token_urlsafe(32))
    monkeypatch.setenv("OUTRIDER_ADMIN_API_KEY", secrets.token_urlsafe(32))
    validate_setup_state_secret()  # distinct → no raise
