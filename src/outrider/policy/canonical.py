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
digest; `canonicalize_for_hash(payload)` is exposed for tests that
pin the BYTE OUTPUT (not just digest) so a future `json.dumps`
semantics change fails loudly with a visible diff.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

__all__ = [
    "SHA256_HEX_PATTERN",
    "SHA256_HEX_PATTERN_SHORT",
    "canonicalize_for_hash",
    "compute_candidate_id",
    "compute_identity_hash",
    "compute_proposal_hash",
    "compute_response_hash",
    "compute_round_id",
]


# Lowercase-hex SHA-256 string pattern (full 64-char digest). Lives here
# (not in `audit/events.py`) so both `schemas/` and `audit/` can import
# it without a circular dependency. Producers go through
# `compute_identity_hash` below; consumers validate stored strings
# against this pattern.
SHA256_HEX_PATTERN: Final[str] = r"^[a-f0-9]{64}$"

# Short SHA-256-hex prefix pattern (16 chars). Used for hostile-string
# fingerprinting per `DECISIONS.md#014` point 1: store
# `sha256(raw_value)[:16]` instead of the raw model-controlled value, so
# audit consumers can dedup + detect length-class anomalies without
# leaking content. Lifted from `audit/events.py` per the foundation-wide
# sharp-edges audit I-2 — same chokepoint discipline as the full
# pattern above; per-call-site `_SHA256_HEX_PATTERN_PREFIX_N` literals
# are the same drift class the canonical recipe is designed to prevent.
SHA256_HEX_PATTERN_SHORT: Final[str] = r"^[a-f0-9]{16}$"


# Pre-existing canonical encodings NOT routed through this module
# (foundation-wide sharp-edges audit I-6, accepted-asymmetry option):
#
# - `outrider.audit.events.compute_finding_content_hash` — JSON-array
#   payload `[file_path, line_start, line_end, finding_type.value]`,
#   SHA-256 hex. Predates this module; stored hash values live in
#   `audit_events.payload` rows. Re-canonicalizing would break those.
# - `outrider.llm.base._canonical_prompt_hash` — `\x1e`-delimited
#   two-string concatenation, SHA-256 hex. Predates this module;
#   stored hash values live in `llm_call_content.prompt_hash` rows
#   under retention. Re-canonicalizing would break those.
#
# Both stay independent of this module to preserve wire-format
# compatibility on historical rows. New STRUCTURED identity-bearing
# hashes added from the foundation onward (round_id, candidate_id,
# proposal_hash) go through `compute_identity_hash` below — avoiding
# per-call-site recipe drift on structured payloads is the load-bearing
# property.
#
# `response_hash` (on `AnalyzeResponseRejectedEvent`) is a separate
# case: it hashes the FULL raw response BYTES (UTF-8 encoded) directly
# via `sha256(text.encode("utf-8")).hexdigest()`, NOT through
# `canonicalize_for_hash` — the input shape isn't a structured dict but
# a single text blob, so JSON canonicalization doesn't apply. The
# `compute_response_hash` wrapper below implements that recipe. Post-PR
# review fold: prior comment claimed `response_hash` used
# `compute_identity_hash`; that was prose drift — the implementation is
# correct, the comment is being corrected here.


def canonicalize_for_hash(payload: dict[str, Any]) -> bytes:
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
    # Recursive validation: nested dict keys can still hit the int-to-str
    # collision class, and nested values can still smuggle a BaseModel
    # past the chokepoint. Walk the entire payload tree once before
    # `json.dumps`, surfacing typed errors that name the offending path
    # rather than letting json.dumps's default raise from deep in the
    # stack. Copilot/CodeRabbit/Codex review convergent fold.
    _validate_hash_payload(payload, "$")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_hash_payload(value: Any, path: str) -> None:
    """Recursively enforce the canonical-hash payload contract.

    Per `canonicalize_for_hash`'s docstring contract: values must be
    JSON-native primitives (`str` / `int` / `bool` / `None` / `list` /
    `dict`), `dict` keys at EVERY level must be `str`, and Pydantic
    BaseModel instances are rejected with a typed pointer at the
    `model_dump(mode='json')` escape hatch.

    Non-JSON-native containers like `set` / `frozenset` are rejected
    explicitly here even though `json.dumps` would also reject them:
    the diagnostic message naming the offending path is more actionable
    than the stack trace `json.dumps` produces, AND `set` carries a
    silent reordering risk because Python's set iteration order is
    insertion-derived but not stable across processes. Treating sets as
    canonical inputs would defeat the deterministic-encoding promise.
    """
    # Local import keeps the module load light — see `_validate_hash_payload`
    # invocation site for rationale.
    from pydantic import BaseModel  # noqa: PLC0415

    if isinstance(value, BaseModel):
        raise TypeError(
            f"canonicalize_for_hash got a Pydantic BaseModel at {path}: "
            f"{type(value).__name__}. Convert via `model.model_dump(mode='json')` "
            f"before hashing — the canonical recipe requires JSON-native "
            f"primitives (str/int/bool/None/list/dict)."
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(
            f"canonicalize_for_hash got a {type(value).__name__} at {path}; "
            f"set iteration is insertion-derived and not stable across "
            f"processes, so set values defeat the deterministic-encoding "
            f"promise. Pass `sorted(...)` as a list to make the order "
            f"explicit + reproducible."
        )
    if isinstance(value, tuple):
        # Tuples are not part of the JSON-native value contract this
        # module enforces. `json.dumps` silently serializes them as
        # arrays, which is exactly the implicit shape coercion this
        # module is trying to prevent. Callers MUST convert via
        # `list(...)` (or `sorted(...)` for set-derived input) so the
        # shape decision is explicit at the call site. Post-PR review
        # fold (CodeRabbit + Copilot convergent).
        raise TypeError(
            f"canonicalize_for_hash got a tuple at {path}; tuples are not "
            f"part of the canonical payload contract (JSON-native: "
            f"str/int/bool/None/list/dict). Convert to list explicitly "
            f"so shape decisions are intentional — json.dumps would "
            f"silently serialize tuples as arrays."
        )
    if isinstance(value, dict):
        bad_keys = [k for k in value if not isinstance(k, str)]
        if bad_keys:
            raise TypeError(
                f"canonicalize_for_hash requires str keys at every dict "
                f"level; found {len(bad_keys)} non-str at {path}: "
                f"{bad_keys[:5]!r}. Convert keys via `str(...)` before "
                f"hashing (silent int→str coercion under sort_keys=True "
                f"is the encoding-collision class the canonical recipe is "
                f"designed to prevent)."
            )
        for k, v in value.items():
            _validate_hash_payload(v, f"{path}.{k}")
        return
    if isinstance(value, list):
        for i, item in enumerate(value):
            _validate_hash_payload(item, f"{path}[{i}]")
        return
    # Reject `float` even though it's JSON-native: float serialization
    # is implementation-dependent in subtle ways (trailing zero
    # truncation, exponent vs decimal notation, locale-sensitive
    # libraries) and `0.1 + 0.2 != 0.3` means content-derived hashes
    # over float arithmetic results are unstable across producers.
    # `allow_nan=False` catches NaN/Inf but not finite floats. The
    # contract says str/int/bool/None/list/dict only; callers with a
    # float MUST convert via `str(...)` (or `Decimal(str(value))`) so
    # the encoding decision is explicit at the call site. Post-PR
    # review fold (high confidence, runtime-contract issue).
    #
    # `bool` is a subclass of `int` in Python; `isinstance(True, float)`
    # is False — so this rejection doesn't accidentally reject booleans.
    if isinstance(value, float):
        raise TypeError(
            f"canonicalize_for_hash got a float at {path}: {value!r}. "
            f"Floats are NOT part of the JSON-native value contract "
            f"(str/int/bool/None/list/dict) because float serialization "
            f"is implementation-dependent (trailing zero truncation, "
            f"exponent notation, 0.1 + 0.2 vs 0.3 representations). "
            f"Convert to a stable string representation (e.g., "
            f"`str(value)` or `Decimal(str(value))`) before hashing so "
            f"the encoding choice is explicit at the call site."
        )
    # Leaf: str/int/bool/None — allowed.


def compute_identity_hash(payload: dict[str, Any]) -> str:
    """SHA-256 hex digest of the canonical UTF-8 bytes for `payload`.

    The hex output is the standard 64-char lowercase shape matching
    `SHA256_HEX_PATTERN` validators on `AnalysisRound.round_id`,
    `TraceCandidate.candidate_id`, and downstream proposal hashes.

    Prefer the typed wrappers below (`compute_proposal_hash`,
    `compute_response_hash`, `compute_round_id`, `compute_candidate_id`)
    for the four foundation-defined identity hashes — they build the
    canonical payload internally so callers can't typo a key or forget
    a field. This bare entrypoint is for ad-hoc test fixtures and
    future identity-hash recipes the wrappers don't yet cover.
    """
    return hashlib.sha256(canonicalize_for_hash(payload)).hexdigest()


# ---------------------------------------------------------------------------
# Typed wrappers for the four foundation-defined identity hashes.
# Per foundation-wide DevEx audit F3: the proposal_hash / response_hash /
# round_id / candidate_id payload shapes are in spec prose only — without
# typed wrappers, a sister-spec parser author can typo a field name,
# forget a key, or stringify a Span incorrectly and the resulting hash
# still matches `SHA256_HEX_PATTERN`. The wrappers move each recipe into
# code so mypy catches missing kwargs and the canonical encoding lives
# in ONE place per hash type.
# ---------------------------------------------------------------------------


def compute_proposal_hash(
    *,
    source_file_path: str,
    finding_type: str,
    evidence_tier: str,
    query_match_id: str | None,
    trace_path: tuple[str, ...] | None,
    title: str,
    description: str,
    evidence: str,
    byte_start: int,
    byte_end: int,
) -> str:
    """SHA-256 hex of an `AnalyzeFindingProposalRaw`'s file-scoped identity.

    Per `DECISIONS.md#022` (Accepted 2026-05-20): proposal identity is
    PR/file-scoped, not raw-proposal-shape-global. The recipe folds 9
    keys: `source_file_path` (the file the proposal came from), plus
    the 8 raw-proposal keys (finding_type, evidence_tier, query_match_id,
    trace_path, title, description, evidence, span as
    `{byte_start, byte_end}`). `trace_candidates` is deliberately
    excluded (model child-output, not parent identity).

    All keyword-only — finding_type/evidence_tier are both raw `str`
    (not enum-coerced) because the hash is computed AT the raw layer,
    BEFORE admission.

    `source_file_path` runs through `coordinates.validate_diff_path`
    BEFORE entering the hash payload. Per the round-2-crazy-audit DI-H1
    path-canonicalization rule (spec.md §1: "All `files_examined`,
    `files_skipped`, and `candidate_path` strings entering identity
    hashes MUST be in canonical form"), `source_file_path` is now the
    fourth path-bearing input to a hash recipe and inherits the same
    rule. Without this gate, `"src/foo.py"` and `"./src/foo.py"` and
    `"src//foo.py"` (the same file under different aliases) would
    produce distinct `proposal_hash` values — reopening exactly the
    dedup false-negative Codex round-7 caught. Canonicalization here
    + at the carrier-schema layers means alias paths produce a SINGLE
    canonical hash. The pedagogical placement of `source_file_path`
    as the first dict key remains — `canonicalize_for_hash`'s
    `sort_keys=True` makes position hash-irrelevant.

    Used by `FindingProposalRejectedEvent.proposal_hash` and by
    `TraceCandidate.source_proposal_hash`. Two analyze passes over
    DIFFERENT source files emitting logically-identical proposals now
    produce DISTINCT hashes (the old recipe collapsed them — a real
    Codex round-6 audit finding). The trace node still dedups actual
    fetches by `candidate_path` at execution time; the candidate-identity
    model preserves the per-source-file causal edges either way.
    """
    from outrider.coordinates import validate_diff_path  # noqa: PLC0415

    canonical_source_file_path = validate_diff_path(source_file_path)
    return compute_identity_hash(
        {
            "source_file_path": canonical_source_file_path,
            "finding_type": finding_type,
            "evidence_tier": evidence_tier,
            "query_match_id": query_match_id,
            "trace_path": list(trace_path) if trace_path is not None else None,
            "title": title,
            "description": description,
            "evidence": evidence,
            "span": {"byte_start": byte_start, "byte_end": byte_end},
        }
    )


def compute_response_hash(response_text: str) -> str:
    """SHA-256 hex of the FULL raw analyze response text, UTF-8 encoded.

    Per spec §5 + post-split S11: full text, NO 8 KiB prefix. The
    SHA-256 output is 64 hex chars carrying no recoverable completion
    text — hash-only, no content leak per `DECISIONS.md#014`.

    Used by `AnalyzeResponseRejectedEvent.response_hash`.

    Note: this hash is over the raw response BYTES, not a structured
    payload, so it doesn't go through `canonicalize_for_hash`. Distinct
    encoding from the other three wrappers because the input shape is
    distinct (bytes, not a dict).
    """
    return hashlib.sha256(response_text.encode("utf-8")).hexdigest()


def compute_round_id(
    *,
    pass_index: int,
    files_examined: tuple[str, ...],
    files_skipped: tuple[str, ...],
    finding_content_hashes: tuple[str, ...],
) -> str:
    """SHA-256 hex identifying an `AnalysisRound`.

    Per spec §1: content-derived from the round's payload so re-emission
    of the same logical round (e.g., from a checkpoint replay) produces
    the same id and collapses on the dedup-by-round_id reducer.

    **Inputs sorted internally** per spec §1 ("hashed inputs are sorted
    for cross-process determinism") so two producers that enumerate
    files / findings in different orders still produce the same id.
    Without this, the dedup-by-round_id reducer would admit both
    orderings as distinct rounds and double-accumulate state on replay.

    `finding_content_hashes` is the sequence of `ReviewFinding.content_hash`
    values from this round's findings (not the full finding payloads —
    the content_hash already captures finding identity).
    `files_examined`/`files_skipped` MUST be the canonical
    `validate_diff_path` output per `AnalysisRound._enforce_canonical_paths`.
    """
    return compute_identity_hash(
        {
            "pass_index": pass_index,
            "files_examined": sorted(files_examined),
            "files_skipped": sorted(files_skipped),
            "finding_content_hashes": sorted(finding_content_hashes),
        }
    )


def compute_candidate_id(
    *,
    source_proposal_hash: str,
    candidate_path: str,
    reason: str,
) -> str:
    """SHA-256 hex identifying a `TraceCandidate`.

    Per spec §1: content-derived from the candidate's payload so
    re-emission of the same logical candidate produces the same id and
    collapses on the dedup-by-candidate_id reducer.

    `candidate_path` MUST be the canonical `validate_diff_path` output
    per `TraceCandidate._enforce_canonical_path`. `source_proposal_hash`
    matches `FindingProposalRejectedEvent.proposal_hash` for the audit
    join — caller passes the same string that landed on the rejection
    event (or would land, if the proposal were rejected).

    The payload dict's key order is IRRELEVANT — `canonicalize_for_hash`
    applies `sort_keys=True` so any ordering produces identical canonical
    bytes. A future refactor reordering the literal below cannot drift
    the digest.
    """
    return compute_identity_hash(
        {
            "source_proposal_hash": source_proposal_hash,
            "candidate_path": candidate_path,
            "reason": reason,
        }
    )
