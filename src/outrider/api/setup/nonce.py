"""Create and hash the one-use value that binds a setup callback.

The state_machine stores only its hash and expiry. It consumes an unexpired
row atomically, so the same callback cannot use it twice. The next setup request
removes expired rows; there is no periodic cleanup task here.

See DECISIONS.md#070.
"""

from __future__ import annotations

import secrets
from hashlib import sha256

__all__ = ["hash_nonce", "new_nonce"]

# Raw-nonce entropy in bytes (token_urlsafe → ~1.33x chars). 32 bytes ≈ 43 chars, ~256 bits.
_NONCE_BYTES = 32


def new_nonce() -> tuple[str, str]:
    """Mint a fresh single-use nonce → `(raw, hash)`. The `raw` value is signed into the `state`
    (carried to the operator's browser and back); the `hash` is stored in `setup_nonce`. The raw
    value never touches the database."""
    raw = secrets.token_urlsafe(_NONCE_BYTES)
    return raw, hash_nonce(raw)


def hash_nonce(raw: str) -> str:
    """sha256 hex of a raw nonce — the stored/looked-up form. Applied at mint (to store) and at
    callback (to match for delete-on-consume). Deterministic, so the same raw nonce always maps to
    the same row."""
    return sha256(raw.encode("utf-8")).hexdigest()
