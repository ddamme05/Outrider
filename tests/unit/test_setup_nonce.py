"""Unit tests for `api/setup/nonce` — the single-use callback nonce (#070)."""

from __future__ import annotations

from hashlib import sha256

from outrider.api.setup.nonce import hash_nonce, new_nonce


def test_new_nonce_pairs_raw_and_hash() -> None:
    raw, digest = new_nonce()
    assert digest == hash_nonce(raw)
    assert digest == sha256(raw.encode("utf-8")).hexdigest()


def test_hash_is_deterministic() -> None:
    assert hash_nonce("abc") == hash_nonce("abc")


def test_distinct_nonces_differ() -> None:
    raw1, h1 = new_nonce()
    raw2, h2 = new_nonce()
    assert raw1 != raw2
    assert h1 != h2


def test_raw_is_high_entropy() -> None:
    raw, _ = new_nonce()
    # token_urlsafe(32) → ~43 chars; comfortably long enough that the stored hash has nothing to
    # brute-force.
    assert len(raw) >= 40
