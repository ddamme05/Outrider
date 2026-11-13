"""Encrypt and decrypt saved GitHub App keys and webhook secrets.

The dedicated encryption key is separate from Slack token encryption.
Comma-separated keys allow rotation; reads try the configured keys and writes
use the first. Missing keys and invalid ciphertext raise, with no plaintext
fallback. See DECISIONS.md#051, #070.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from pydantic import SecretStr

from outrider.policy.secret_placeholders import SECRET_PLACEHOLDERS as _PLACEHOLDER_SECRETS

__all__ = [
    "CREDENTIAL_ENC_KEY_ENV",
    "CredentialCryptoError",
    "decrypt_credential",
    "encrypt_credential",
    "validate_credential_enc_key",
]

# Env var NAME (not a secret value) — the at-rest encryption key(s) for App credentials.
CREDENTIAL_ENC_KEY_ENV = "OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY"  # noqa: S105
# Slack's at-rest key (notify/token_crypto.TOKEN_ENC_KEY_ENV). DECISIONS.md#070 requires the GitHub
# credential key be DISTINCT from it — enforced in validate_credential_enc_key so a compromise of
# one credential class's key doesn't compromise the other.
_SLACK_ENC_KEY_ENV = "OUTRIDER_TOKEN_ENC_KEY"  # noqa: S105


class CredentialCryptoError(RuntimeError):
    """At-rest App-credential encryption/decryption failed — a missing/malformed key, or (on
    decrypt) a tampered / forged / wrong-key ciphertext. Fail-closed by contract: never returns or
    persists plaintext on error."""


def _multifernet() -> MultiFernet:
    """Build the `MultiFernet` from `OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY`, fail-closed.

    Read fresh on every call (not cached) so a test's env monkeypatch takes effect and a
    deploy-time key rotation is picked up without a restart. The env carries comma-separated
    urlsafe-base64 Fernet keys: the first encrypts, all decrypt.
    """
    raw = os.environ.get(CREDENTIAL_ENC_KEY_ENV, "").strip()
    if not raw:
        raise CredentialCryptoError(
            f"{CREDENTIAL_ENC_KEY_ENV} is unset or empty. `database` credential mode requires an "
            "at-rest encryption key — a urlsafe-base64 32-byte Fernet key, separate from "
            "OUTRIDER_TOKEN_ENC_KEY. Generate one with cryptography.fernet.Fernet.generate_key()."
        )
    parts = [p.strip() for p in raw.split(",")]
    for p in parts:
        if p and p.lower() in _PLACEHOLDER_SECRETS:
            raise CredentialCryptoError(
                f"{CREDENTIAL_ENC_KEY_ENV} is a known placeholder ({p!r}); it is the at-rest "
                "encryption key for GitHub App credentials — generate a real key with "
                "cryptography.fernet.Fernet.generate_key()."
            )
    try:
        keys: list[Fernet] = []
        for key_part in parts:
            if key_part:
                keys.append(Fernet(key_part.encode("utf-8")))
    except (ValueError, TypeError) as exc:
        raise CredentialCryptoError(
            f"{CREDENTIAL_ENC_KEY_ENV} is not a valid Fernet key set"
        ) from exc
    if not keys:
        raise CredentialCryptoError(f"{CREDENTIAL_ENC_KEY_ENV} contained no usable keys")
    return MultiFernet(keys)


def encrypt_credential(secret: SecretStr) -> bytes:
    """Encrypt a credential (PEM or webhook secret) for at-rest storage → Fernet ciphertext
    (authenticated; self-describing version + timestamp). Fails closed (raises
    `CredentialCryptoError`) on a missing/malformed key or an empty secret."""
    plaintext = secret.get_secret_value()
    if not plaintext.strip():
        raise CredentialCryptoError("refusing to encrypt an empty/blank credential")
    return _multifernet().encrypt(plaintext.encode("utf-8"))


def decrypt_credential(ciphertext: bytes) -> SecretStr:
    """Decrypt at-rest ciphertext → the credential as `SecretStr`. Raises `CredentialCryptoError`
    on a missing/malformed key OR a tampered / forged / wrong-key ciphertext (Fernet
    authentication) — never returns plaintext on failure."""
    try:
        plaintext = _multifernet().decrypt(ciphertext)
    except InvalidToken as exc:
        raise CredentialCryptoError(
            "credential ciphertext failed authentication (tampered, forged, or wrong key)"
        ) from exc
    return SecretStr(plaintext.decode("utf-8"))


def _key_values(raw: str) -> frozenset[str]:
    """The set of individual key strings in a comma-separated key env value."""
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def validate_credential_enc_key() -> None:
    """Assert `OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY` is present, well-formed, AND distinct from
    Slack's key — eager startup validation for `database` mode, so a missing/placeholder/malformed/
    reused key surfaces at boot, not lazily at the first onboarding persist or first decrypt.
    Raises `CredentialCryptoError` on any failure; returns None when the key set is valid."""
    _multifernet()  # structural: present, non-placeholder, valid Fernet key set
    # DECISIONS.md#070 dedicated-key requirement: reject any key shared with OUTRIDER_TOKEN_ENC_KEY.
    slack_raw = os.environ.get(_SLACK_ENC_KEY_ENV, "").strip()
    if slack_raw and _key_values(os.environ.get(CREDENTIAL_ENC_KEY_ENV, "")) & _key_values(
        slack_raw
    ):
        raise CredentialCryptoError(
            f"{CREDENTIAL_ENC_KEY_ENV} reuses a key from {_SLACK_ENC_KEY_ENV}; DECISIONS.md#070 "
            "requires a dedicated key distinct from Slack's — generate a separate Fernet key."
        )
