# See DECISIONS.md#051-slack-bot-tokens-encrypted-at-rest
"""Slack bot-token encryption boundary (DECISIONS.md#051).

A Slack bot token is a long-lived bearer credential. Unlike the GitHub
installation token — minted on demand from the App private key and never persisted
(`github/auth.py`; the private key lives only as an env `SecretStr`) — a Slack bot
token has no short-lived-mint equivalent: it must be STORED to post across process
restarts. So it is encrypted at rest — the `installations` row stores ciphertext,
never plaintext, and decryption is confined to this module plus the Slack-notifier
construction site.

All `cryptography` use is confined here (the one boundary module, enforced by the
import lint). The key is `OUTRIDER_TOKEN_ENC_KEY` — one or more comma-separated
urlsafe-base64 Fernet keys (the FIRST encrypts, ALL decrypt: `MultiFernet`
rotation). A missing / empty / malformed key FAILS CLOSED (raises) so a
misconfigured deploy can never silently persist plaintext or post. The decrypted
token is returned as `SecretStr` and held only as a local at the use site — never
in audit events, graph state, logs, or API responses.
"""

from __future__ import annotations

import os

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from pydantic import SecretStr

__all__ = ["TOKEN_ENC_KEY_ENV", "TokenCryptoError", "decrypt_token", "encrypt_token"]

# Env var NAME (not a secret value) — the at-rest encryption key(s).
TOKEN_ENC_KEY_ENV = "OUTRIDER_TOKEN_ENC_KEY"  # noqa: S105


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
    try:
        keys = [Fernet(p.encode("utf-8")) for p in parts if p]
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
