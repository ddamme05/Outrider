# See specs/2026-05-19-analyze-foundation.md §1.
"""Single chokepoint for canonical identity-bearing hash inputs.

Per the spec's "Canonical identity encoding" recipe, all identity hashes
use canonical JSON serialization (sort_keys=True, compact separators,
ensure_ascii=False) encoded as UTF-8. Defining the recipe in ONE module
prevents per-call-site drift — the round-1-crazy-audit C1 finding
(\\n-separator collisions) recurred because the recipe was repeated
in each schema's prose. This module is the durable fix.

Consumed by `AnalysisRound.round_id`, `TraceCandidate.candidate_id`,
and downstream proposal/response hashes in the analyze-implementation
sister spec. Callers chain `compute_identity_hash(payload)` for a hex
digest; `_canonicalize_for_hash(payload)` is exposed for tests that
pin the BYTE OUTPUT (not just digest) so a future `json.dumps`
semantics change fails loudly with a visible diff.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

# Lowercase-hex SHA-256 string pattern. Lives here (not in `audit/events.py`)
# so both `schemas/` and `audit/` can import it without a circular dependency.
# Producers go through `compute_identity_hash` below; consumers validate
# stored strings against this pattern.
SHA256_HEX_PATTERN: Final[str] = r"^[a-f0-9]{64}$"


def _canonicalize_for_hash(payload: dict[str, Any]) -> bytes:
    """Canonical UTF-8 bytes for a hash-input payload.

    Encodes via `json.dumps` with `sort_keys=True` (eliminates field-order
    dependence), `separators=(",", ":")` (eliminates whitespace ambiguity),
    `ensure_ascii=False` (preserves multibyte content; paired with
    `.encode("utf-8")`), `allow_nan=False` (rejects NaN/Infinity — the
    §1 crazy-audit HIGH finding: `NaN != NaN` breaks idempotent identity
    hashes AND emits non-RFC-8259 tokens that downstream consumers can't
    re-parse). Returns bytes, NOT a digest — callers chain
    `compute_identity_hash` (or call `hashlib.sha256(...).hexdigest()`)
    when they need a hex string.

    A `payload` containing a lone surrogate like `"\\ud800"` raises
    `UnicodeEncodeError` from `.encode("utf-8")` — fail-loud at hash
    time, not at DB insert time. Tests pin this.

    **Caller contract (§1 crazy-audit MEDIUM-2):** `payload` must be
    pre-serialized to JSON-native values — `str` keys only (int keys
    silently coerce under `sort_keys=True`, producing collisions across
    `{1: "a"}` and `{"1": "a"}`), and values must be `str`/`int`/`bool`/
    `None`/`list`/`dict` only. Callers who hold `datetime` / `UUID` /
    `Decimal` must convert via `.isoformat()` / `str(...)` BEFORE handing
    the payload here (or via `model.model_dump(mode="json")` which
    handles the conversion). This function fails loud on contract
    violations rather than coercing silently — a coercion contract would
    create the same encoding-collision class.
    """
    if not all(isinstance(k, str) for k in payload):
        bad_keys = [k for k in payload if not isinstance(k, str)]
        raise TypeError(
            f"_canonicalize_for_hash requires str keys; got {len(bad_keys)} "
            f"non-str: {bad_keys[:5]!r}. Convert keys via `str(...)` before "
            f"hashing (silent int→str coercion under sort_keys=True is the "
            f"encoding-collision class the canonical recipe is designed to "
            f"prevent — §1 crazy-audit MEDIUM-2)."
        )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_identity_hash(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical UTF-8 bytes for `payload`.

    The hex output is the standard 64-char lowercase shape matching
    `_SHA256_HEX_PATTERN` validators on `AnalysisRound.round_id`,
    `TraceCandidate.candidate_id`, and downstream proposal hashes.
    """
    return hashlib.sha256(_canonicalize_for_hash(payload)).hexdigest()
