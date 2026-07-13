"""Unit tests for `github.credential_crypto` — the GitHub App credential at-rest boundary (#070).

Parallels `test_token_crypto` (Slack), but a dedicated key. Covers the round-trip, fail-closed on
missing/placeholder/malformed key, empty-secret refusal, and tamper rejection.
"""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr

from outrider.github.credential_crypto import (
    CREDENTIAL_ENC_KEY_ENV,
    CredentialCryptoError,
    decrypt_credential,
    encrypt_credential,
    validate_credential_enc_key,
)


@pytest.fixture
def _key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, key)
    return key


def test_round_trip(_key: str) -> None:
    secret = SecretStr("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----")
    ct = encrypt_credential(secret)
    assert isinstance(ct, bytes)
    assert secret.get_secret_value().encode() not in ct  # ciphertext, not plaintext
    assert decrypt_credential(ct).get_secret_value() == secret.get_secret_value()


def test_missing_key_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CREDENTIAL_ENC_KEY_ENV, raising=False)
    with pytest.raises(CredentialCryptoError, match="unset or empty"):
        encrypt_credential(SecretStr("x"))


@pytest.mark.parametrize("placeholder", ["replace-me", "change-me", "secret", "your-secret-here"])
def test_placeholder_key_rejected(monkeypatch: pytest.MonkeyPatch, placeholder: str) -> None:
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, placeholder)
    with pytest.raises(CredentialCryptoError, match="placeholder"):
        validate_credential_enc_key()


def test_malformed_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, "not-a-valid-fernet-key")
    with pytest.raises(CredentialCryptoError, match="not a valid Fernet key"):
        validate_credential_enc_key()


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_empty_or_blank_secret_refused(_key: str, blank: str) -> None:
    with pytest.raises(CredentialCryptoError, match="empty/blank"):
        encrypt_credential(SecretStr(blank))


def test_tampered_ciphertext_rejected(_key: str) -> None:
    ct = bytearray(encrypt_credential(SecretStr("pem")))
    ct[-1] ^= 0x01  # flip a bit
    with pytest.raises(CredentialCryptoError, match="authentication"):
        decrypt_credential(bytes(ct))


def test_wrong_key_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii"))
    ct = encrypt_credential(SecretStr("pem"))
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, Fernet.generate_key().decode("ascii"))  # rotate away
    with pytest.raises(CredentialCryptoError, match="authentication"):
        decrypt_credential(ct)


def test_key_rotation_decrypts_old(monkeypatch: pytest.MonkeyPatch) -> None:
    """MultiFernet: a new key prepended still decrypts ciphertext from the old key."""
    old = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, old)
    ct = encrypt_credential(SecretStr("pem"))
    new = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv(CREDENTIAL_ENC_KEY_ENV, f"{new},{old}")  # first encrypts, all decrypt
    assert decrypt_credential(ct).get_secret_value() == "pem"


def test_validate_passes_on_good_key(_key: str) -> None:
    validate_credential_enc_key()  # no raise
