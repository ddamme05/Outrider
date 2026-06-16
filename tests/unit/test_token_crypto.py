"""Slack bot-token encryption boundary (DECISIONS.md#051).

Pins: round-trip, ciphertext != plaintext, fail-closed on a missing/malformed key,
tamper detection (Fernet auth), empty-token rejection, and MultiFernet key rotation.
"""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from outrider.notify.token_crypto import (
    TOKEN_ENC_KEY_ENV,
    TokenCryptoError,
    decrypt_token,
    encrypt_token,
)

_TOKEN = "xoxb-1234567890-abcdefghijklmnop"  # noqa: S105  (test fixture, not a real token)


@pytest.fixture
def enc_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid single Fernet key in the env for the duration of the test."""
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, Fernet.generate_key().decode())


@pytest.mark.usefixtures("enc_key")
def test_round_trip() -> None:
    ct = encrypt_token(SecretStr(_TOKEN))
    assert decrypt_token(ct).get_secret_value() == _TOKEN


@pytest.mark.usefixtures("enc_key")
def test_ciphertext_is_not_plaintext() -> None:
    ct = encrypt_token(SecretStr(_TOKEN))
    assert _TOKEN.encode() not in ct  # the token is not stored in the clear


def test_encrypt_fails_closed_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENC_KEY_ENV, raising=False)
    with pytest.raises(TokenCryptoError, match="unset or empty"):
        encrypt_token(SecretStr(_TOKEN))


def test_decrypt_fails_closed_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TOKEN_ENC_KEY_ENV, raising=False)
    with pytest.raises(TokenCryptoError, match="unset or empty"):
        decrypt_token(b"whatever")


@pytest.mark.usefixtures("enc_key")
def test_encrypt_rejects_empty_token() -> None:
    with pytest.raises(TokenCryptoError, match="empty token"):
        encrypt_token(SecretStr(""))


@pytest.mark.usefixtures("enc_key")
def test_decrypt_rejects_tampered_ciphertext() -> None:
    ct = bytearray(encrypt_token(SecretStr(_TOKEN)))
    ct[-1] ^= 0x01  # flip a bit → Fernet authentication fails
    with pytest.raises(TokenCryptoError, match="failed authentication"):
        decrypt_token(bytes(ct))


def test_malformed_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, "not-a-valid-fernet-key")
    with pytest.raises(TokenCryptoError, match="not a valid Fernet key"):
        encrypt_token(SecretStr(_TOKEN))


def test_key_rotation_decrypts_under_old_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """MultiFernet rotation: encrypt under key A, prepend a new key B; the old
    ciphertext still decrypts (B encrypts going forward, A still decrypts)."""
    key_a = Fernet.generate_key().decode()
    key_b = Fernet.generate_key().decode()
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, key_a)
    ct = encrypt_token(SecretStr(_TOKEN))
    monkeypatch.setenv(TOKEN_ENC_KEY_ENV, f"{key_b},{key_a}")
    assert decrypt_token(ct).get_secret_value() == _TOKEN
