"""Slack OAuth signed-`state` CSRF token (commit 6.3a).

Pins: round-trip, tampered body/signature rejection, expiry, fail-closed on a
missing secret, malformed-state rejection, and channel-id shape validation.
"""

from __future__ import annotations

import base64
import json

import pytest

from outrider.notify.oauth_state import (
    STATE_SECRET_ENV,
    SlackStateError,
    sign_state,
    validate_channel_id,
    verify_state,
)


@pytest.fixture
def state_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STATE_SECRET_ENV, "test-slack-state-secret-value-0123456789")


@pytest.mark.usefixtures("state_secret")
def test_round_trip() -> None:
    st = verify_state(sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE"))
    assert st.installation_id == 42
    assert st.admin_id == "admin"
    assert st.channel_id == "C0ABCDE"
    assert st.nonce


@pytest.mark.usefixtures("state_secret")
def test_tampered_body_rejected() -> None:
    body, _, sig = sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE").partition(
        "."
    )
    # Forge a different installation_id but keep the original signature.
    payload = json.loads(base64.urlsafe_b64decode(body.encode()))
    payload["iid"] = 999
    forged = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    with pytest.raises(SlackStateError, match="signature mismatch"):
        verify_state(f"{forged}.{sig}")


@pytest.mark.usefixtures("state_secret")
def test_tampered_signature_rejected() -> None:
    body, _, _ = sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE").partition(
        "."
    )
    with pytest.raises(SlackStateError, match="signature mismatch"):
        verify_state(f"{body}.deadbeef")


@pytest.mark.usefixtures("state_secret")
def test_expired_state_rejected() -> None:
    expired = sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE", ttl_seconds=-1)
    with pytest.raises(SlackStateError, match="expired"):
        verify_state(expired)


@pytest.mark.usefixtures("state_secret")
@pytest.mark.parametrize("bad", ["", "no-dot-here"])
def test_malformed_state_rejected(bad: str) -> None:
    with pytest.raises(SlackStateError, match="missing or malformed"):
        verify_state(bad)


def test_sign_fails_closed_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STATE_SECRET_ENV, raising=False)
    with pytest.raises(SlackStateError, match="must be set"):
        sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE")


def test_verify_fails_closed_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STATE_SECRET_ENV, raising=False)
    with pytest.raises(SlackStateError, match="must be set"):
        verify_state("a.b")


@pytest.mark.parametrize("placeholder", ["changeme", "replace-me", "secret", "CHANGEME"])
def test_secret_placeholder_rejected(monkeypatch: pytest.MonkeyPatch, placeholder: str) -> None:
    """The state secret is the CSRF root — a known placeholder fails closed (matches
    the auth-secret discipline), not just empty."""
    monkeypatch.setenv(STATE_SECRET_ENV, placeholder)
    with pytest.raises(SlackStateError, match="placeholder"):
        sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE")


@pytest.mark.parametrize("weak", ["hunter2", "short", "x" * 31])  # non-placeholder but < 32 chars
def test_secret_too_short_rejected(monkeypatch: pytest.MonkeyPatch, weak: str) -> None:
    """A short, non-placeholder secret slips past the placeholder set but has too little
    entropy to be a credible HMAC/CSRF root — fail closed below the 32-char floor."""
    monkeypatch.setenv(STATE_SECRET_ENV, weak)
    with pytest.raises(SlackStateError, match="too short"):
        sign_state(installation_id=42, admin_id="admin", channel_id="C0ABCDE")


@pytest.mark.usefixtures("state_secret")
@pytest.mark.parametrize(
    "bad",
    ["", "   ", "lower", "C 0DEF", "C@1DEF", "ab", "X0ABCDE", "ABCDEF"],  # last two: non-C/G prefix
)
def test_invalid_channel_rejected_at_sign(bad: str) -> None:
    with pytest.raises(SlackStateError, match="invalid Slack channel_id"):
        sign_state(installation_id=42, admin_id="admin", channel_id=bad)


def test_validate_channel_id_strips_and_accepts() -> None:
    assert validate_channel_id("  C0ABCDE  ") == "C0ABCDE"
