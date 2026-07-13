# See DECISIONS.md#070 — GitHub App credential encryption boundary.
"""At-rest encryption for manifest-onboarded GitHub App credentials (`DECISIONS.md#070`).

Parallel to `notify/token_crypto.py` (`#051`, Slack bot tokens), but a **dedicated key** and a
**dedicated boundary module** for a different credential class. Under `database` credential mode,
the App `pem` (private key) and `webhook_secret` GitHub returns from the manifest conversion are
delivered at *runtime* and must be STORED to survive restarts — unlike the `env`-mode PEM, which
lives only as an env `SecretStr`. So they are encrypted at rest: the `github_app_credentials` row
stores ciphertext, never plaintext, and decryption is confined to this module plus the
`GitHubCredentialProvider` construction site.

All `cryptography` use here is confined to this file (mirrored in `notify/token_crypto.py`; the
import lint allowlists exactly these two). The key is `OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY` — one or
more comma-separated urlsafe-base64 Fernet keys (the FIRST encrypts, ALL decrypt: `MultiFernet`
rotation), **separate from Slack's `OUTRIDER_TOKEN_ENC_KEY`**. A missing / empty / malformed /
placeholder key FAILS CLOSED (raises) so a misconfigured deploy can never silently persist
plaintext credentials. Decrypted material is returned as `SecretStr` and held only at the use site
— never in audit events, graph state, logs, or API responses.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from pydantic import SecretStr

__all__ = [
    "CREDENTIAL_ENC_KEY_ENV",
    "CredentialCryptoError",
    "decrypt_credential",
    "encrypt_credential",
    "validate_credential_enc_key",
]

# Env var NAME (not a secret value) — the at-rest encryption key(s) for App credentials.
CREDENTIAL_ENC_KEY_ENV = "OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY"  # noqa: S105

# Known placeholders shipped in .env.example (+ the usual suspects); reject a verbatim copy so a
# deploy can't treat a non-key as its at-rest encryption key. Mirrors github/config.py /
# notify/token_crypto.py — keep in sync. A placeholder is also not a valid Fernet key, so it would
# fail construction regardless; this check names the misconfiguration.
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
        keys = [Fernet(p.encode("utf-8")) for p in parts if p]
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


def validate_credential_enc_key() -> None:
    """Assert `OUTRIDER_GITHUB_CREDENTIAL_ENC_KEY` is present AND well-formed — for eager startup
    validation in `database` mode, so a missing/placeholder/malformed key surfaces at boot rather
    than lazily at the first onboarding persist or the first decrypt. Raises `CredentialCryptoError`
    on any failure; returns None when the key set is present and well-formed."""
    _multifernet()
