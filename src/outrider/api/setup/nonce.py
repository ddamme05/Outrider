# See DECISIONS.md#070 — the single-use setup nonce (hashed at rest, delete-on-consume).
"""The setup callback nonce (`DECISIONS.md#070`).

The onboarding `state` carries a **raw** high-entropy nonce; `setup_nonce` stores only its
**sha256** (never the raw value), with an `expires_at`. Consumption is an atomic delete-on-consume
(`state_machine.consume_callback`): the row is deleted `WHERE nonce_hash=$1 AND expires_at>now()
RETURNING`, so a replay finds no row and an expired nonce never validates. No periodic sweep —
expired rows are removed by the lazy `POST /setup` repair.

sha256 (not a slow KDF) is correct here: the nonce is a `secrets.token_urlsafe(32)` value with ~256
bits of entropy, so there is nothing to brute-force from the stored hash — the hash only prevents a
DB-read attacker from replaying the raw nonce.
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
