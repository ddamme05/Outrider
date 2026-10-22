# See DECISIONS.md#051-slack-bot-tokens-are-encrypted-at-rest
"""Encrypt persisted Slack bot tokens and decrypt at notifier construction.

installations stores ciphertext, never plaintext (DECISIONS.md#051). Read
OUTRIDER_TOKEN_ENC_KEY as comma-separated Fernet keys: the first encrypts and all
decrypt, allowing rotation. Missing or malformed keys raise; there is no plaintext
fallback. Return decrypted tokens as SecretStr and keep them out of state, audit,
logs, and API responses.

GitHub App credentials have a separate encryption boundary and dedicated key in
github/credential_crypto.py (DECISIONS.md#070).
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from pydantic import SecretStr

__all__ = [
    "TOKEN_ENC_KEY_ENV",
    "TokenCryptoError",
    "decrypt_token",
    "encrypt_token",
    "validate_token_enc_key",
]

# Env var NAME (not a secret value) — the at-rest encryption key(s).
TOKEN_ENC_KEY_ENV = "OUTRIDER_TOKEN_ENC_KEY"  # noqa: S105

# Known placeholders shipped in .env.example (+ the usual suspects); reject a verbatim
# copy so a deploy can't treat a non-key as its at-rest encryption key. Mirrors
# notify/config.py / api/dashboard/config.py / notify/oauth_state.py — keep in sync. A
# placeholder is also not a valid Fernet key, so it would fail construction regardless;
# this check names the misconfiguration instead of a generic "not a valid Fernet key".
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


class TokenCryptoError(RuntimeError):
    """At-rest token encryption/decryption failed — a missing/malformed key, or (on
    decrypt) a tampered / forged / wrong-key ciphertext. Fail-closed by contract:
    never returns or persists a plaintext token on error."""


def _multifernet() -> MultiFernet:
    """Build the `MultiFernet` from `OUTRIDER_TOKEN_ENC_KEY`, fail-closed.

    Read fresh on every call (not cached) so a test's env monkeypatch takes effect
    and a deploy-time key rotation is picked up without a restart; Fernet
    construction is microsecond-cheap and these calls are rare (the OAuth callback +
    notifier construction). The env carries comma-separated urlsafe-base64 Fernet
    keys: the first encrypts, all decrypt (prepend a new key, re-encrypt, drop the
    old).
    """
    raw = os.environ.get(TOKEN_ENC_KEY_ENV, "").strip()
    if not raw:
        raise TokenCryptoError(
            f"{TOKEN_ENC_KEY_ENV} is unset or empty. Slack persistent config requires an "
            "at-rest encryption key — a urlsafe-base64 32-byte Fernet key. Generate one "
            "with cryptography.fernet.Fernet.generate_key()."
        )
    parts = [p.strip() for p in raw.split(",")]
    for p in parts:
        if p and p.lower() in _PLACEHOLDER_SECRETS:
            raise TokenCryptoError(
                f"{TOKEN_ENC_KEY_ENV} is a known placeholder ({p!r}); it is the at-rest "
                "encryption key for Slack bot tokens — generate a real key with "
                "cryptography.fernet.Fernet.generate_key()."
            )
    try:
        keys: list[Fernet] = []
        for key_part in parts:
            if key_part:
                keys.append(Fernet(key_part.encode("utf-8")))
    except (ValueError, TypeError) as exc:
        raise TokenCryptoError(f"{TOKEN_ENC_KEY_ENV} is not a valid Fernet key set") from exc
    if not keys:
        raise TokenCryptoError(f"{TOKEN_ENC_KEY_ENV} contained no usable keys")
    return MultiFernet(keys)


def encrypt_token(token: SecretStr) -> bytes:
    """Encrypt a Slack bot token for at-rest storage → Fernet ciphertext
    (authenticated; self-describing version + timestamp). Fails closed (raises
    `TokenCryptoError`) on a missing/malformed key or an empty token."""
    plaintext = token.get_secret_value()
    if not plaintext:
        raise TokenCryptoError("refusing to encrypt an empty token")
    return _multifernet().encrypt(plaintext.encode("utf-8"))


def decrypt_token(ciphertext: bytes) -> SecretStr:
    """Decrypt at-rest ciphertext → the bot token as `SecretStr`. Raises
    `TokenCryptoError` on a missing/malformed key OR a tampered / forged / wrong-key
    ciphertext (Fernet authentication) — never returns plaintext on failure."""
    try:
        plaintext = _multifernet().decrypt(ciphertext)
    except InvalidToken as exc:
        raise TokenCryptoError(
            "token ciphertext failed authentication (tampered, forged, or wrong key)"
        ) from exc
    return SecretStr(plaintext.decode("utf-8"))


def validate_token_enc_key() -> None:
    """Assert ``OUTRIDER_TOKEN_ENC_KEY`` is present AND well-formed — for eager startup
    validation. Builds the ``MultiFernet`` once and discards it, so a missing /
    placeholder / malformed key surfaces at boot (the lifespan logs + disables Slack)
    instead of lazily at the first OAuth callback (a 500 after the admin already
    authorized) or the first decrypt (a swallowed post). Raises ``TokenCryptoError`` on
    any failure; returns None when the key set is present and well-formed."""
    _multifernet()
