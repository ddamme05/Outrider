# See specs/2026-05-19-analyze-foundation.md §1.
"""Canonical-encoding chokepoint tests.

Per §1: define the canonical JSON serialization recipe in ONE module so
all four identity-hash recipes downstream (round_id, candidate_id,
proposal_hash, response_hash) share encoding. Without this chokepoint,
the round-1-crazy-audit C1 finding (`\\n`-separator collisions)
recurs per-call-site.

Tests pin the BYTE OUTPUT, not just the digest, so a future
`json.dumps` semantics change surfaces with a visible diff.
"""

from __future__ import annotations

import hashlib

import pytest

from outrider.policy.canonical import canonicalize_for_hash, compute_identity_hash


def test_canonical_bytes_for_known_fixture() -> None:
    """Pin the exact byte output for a known payload.

    A future json.dumps semantics change (separator default, key
    sorting, encoding) breaks this test loudly rather than silently
    drifting downstream identity hashes.
    """
    payload = {"b": 2, "a": 1, "c": ["x", "y"]}
    out = canonicalize_for_hash(payload)
    assert out == b'{"a":1,"b":2,"c":["x","y"]}'


def test_canonical_bytes_sort_keys_eliminates_field_order() -> None:
    """Same content, different insertion order, identical bytes."""
    payload_a = {"a": 1, "b": 2, "c": 3}
    payload_b = {"c": 3, "b": 2, "a": 1}
    assert canonicalize_for_hash(payload_a) == canonicalize_for_hash(payload_b)


def test_canonical_bytes_compact_separators() -> None:
    """No spaces around `:` or `,` — defends against json.dumps default
    `", "` separator drift.
    """
    out = canonicalize_for_hash({"a": 1, "b": 2})
    assert b", " not in out
    assert b": " not in out


def test_canonical_bytes_preserves_multibyte_utf8() -> None:
    """`ensure_ascii=False` + UTF-8 encode preserves multibyte chars.

    Without this, `ensure_ascii=True` would emit `\\uXXXX` escapes and
    a content-derived hash would silently differ between an ASCII-only
    payload and an equivalent multibyte one.
    """
    payload = {"text": "café — résumé"}
    out = canonicalize_for_hash(payload)
    assert "café".encode() in out
    assert b"\\u" not in out


def test_canonical_bytes_rejects_lone_surrogate() -> None:
    """A lone surrogate raises `UnicodeEncodeError` at hash time.

    Fail-loud at hash time, not at DB insert time. Without this, an
    LLM that emitted a malformed string (e.g., split-encoding a Unicode
    char across two messages) would surface as a deep DB error.
    """
    payload = {"text": "\ud800"}  # lone high surrogate
    with pytest.raises(UnicodeEncodeError):
        canonicalize_for_hash(payload)


def test_compute_identity_hash_returns_lowercase_hex_sha256() -> None:
    """Output is 64 lowercase-hex chars matching `_SHA256_HEX_PATTERN`."""
    payload = {"a": 1}
    digest = compute_identity_hash(payload)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_compute_identity_hash_matches_manual_sha256() -> None:
    """Identity-hash is exactly `sha256(canonicalize_for_hash(payload))`."""
    payload = {"file_path": "src/foo.py", "line_start": 10, "line_end": 12}
    expected = hashlib.sha256(canonicalize_for_hash(payload)).hexdigest()
    assert compute_identity_hash(payload) == expected


def test_compute_identity_hash_field_order_invariance() -> None:
    """Same content, different field order, identical digest."""
    a = compute_identity_hash({"x": 1, "y": 2, "z": [3, 4]})
    b = compute_identity_hash({"z": [3, 4], "y": 2, "x": 1})
    assert a == b


def test_compute_identity_hash_value_sensitivity() -> None:
    """Different content produces a different digest."""
    a = compute_identity_hash({"x": 1})
    b = compute_identity_hash({"x": 2})
    assert a != b


# ---------------------------------------------------------------------------
# §1 crazy-audit folds: fail-loud on NaN/Inf + non-str keys + non-JSON-native.
# ---------------------------------------------------------------------------


def test_canonical_bytes_rejects_nan() -> None:
    """HIGH fold: NaN != NaN — a payload containing NaN would produce a
    digest no semantically-equal payload can reproduce. `allow_nan=False`
    fails loud at hash time, not at downstream JSON-consumer time."""
    with pytest.raises(ValueError, match="Out of range float"):
        canonicalize_for_hash({"x": float("nan")})


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf")])
def test_canonical_bytes_rejects_infinity(bad_value: float) -> None:
    """HIGH fold: Infinity tokens are non-RFC-8259 — downstream consumers
    (V1.5 JS dashboard, V2 archival pipelines) can't re-parse them."""
    with pytest.raises(ValueError, match="Out of range float"):
        canonicalize_for_hash({"x": bad_value})


def test_canonical_bytes_rejects_int_keys() -> None:
    """MEDIUM-2 fold: `{1: "a"}` and `{"1": "a"}` would hash identically
    under sort_keys=True's silent int→str coercion. Fail loud instead."""
    with pytest.raises(TypeError, match="requires str keys"):
        canonicalize_for_hash({1: "a"})  # type: ignore[dict-item]


def test_canonical_bytes_rejects_mixed_keys() -> None:
    """Same fold: catch non-str keys even when most keys ARE str."""
    with pytest.raises(TypeError, match="requires str keys"):
        canonicalize_for_hash({"a": 1, 2: "b"})  # type: ignore[dict-item]


def test_canonical_bytes_rejects_datetime_value() -> None:
    """MEDIUM-1 fold: `datetime` values raise TypeError from json.dumps.
    Caller contract: pre-convert via `.isoformat()` or `mode='json'`
    dump. Test pins the failure at hash time so the contract is visible."""
    from datetime import UTC, datetime

    with pytest.raises(TypeError, match="not JSON serializable"):
        canonicalize_for_hash({"ts": datetime.now(UTC)})  # type: ignore[dict-item]


def test_canonical_bytes_rejects_uuid_value() -> None:
    """Same fold: UUIDs raise. Convert via `str(uuid)` before hashing."""
    from uuid import uuid4

    with pytest.raises(TypeError, match="not JSON serializable"):
        canonicalize_for_hash({"id": uuid4()})  # type: ignore[dict-item]


def test_canonical_bytes_admits_pre_converted_iso_datetime() -> None:
    """The supported caller pattern: convert datetime → isoformat str first."""
    from datetime import UTC, datetime

    ts = datetime.now(UTC)
    out = canonicalize_for_hash({"ts": ts.isoformat()})
    assert b'"ts":"' in out
