"""Build stable finding identifiers from validated, canonical JSON data.

Hashes reject non-JSON values and non-finite numbers. Historical identifier
recipes remain available so replay can verify saved reviews.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import (
    Sequence,  # noqa: TC003  (runtime: function-signature annotation in non-TYPE_CHECKING-only path)
)
from typing import Any, Final, Literal
from uuid import UUID, uuid5

__all__ = [
    "SHA256_HEX_PATTERN",
    "SHA256_HEX_PATTERN_SHORT",
    "canonicalize_for_hash",
    "compute_candidate_id",
    "compute_hitl_decision_content_hash",
    "compute_identity_hash",
    "compute_phase_id",
    "compute_proposal_hash",
    "compute_response_hash",
    "compute_round_id",
    "compute_served_finding_id",
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
# leaking content. Single-sourced here so per-call-site
# `_SHA256_HEX_PATTERN_PREFIX_N` literals can't drift.
SHA256_HEX_PATTERN_SHORT: Final[str] = r"^[a-f0-9]{16}$"


# Persisted hashes retain distinct SHA-256 recipes for read compatibility:
# - audit.events.compute_finding_content_hash hashes the JSON array
#   [file_path, line_start, line_end, finding_type.value] in audit payloads;
# - llm.base._canonical_prompt_hash length-prefixes both UTF-8 strings for
#   llm_call_content.prompt_hash;
# - compute_response_hash hashes the full raw UTF-8 response directly.
# Re-encoding these through canonicalize_for_hash would break stored identities.
# Structured identities such as round_id, candidate_id and proposal_hash use
# compute_identity_hash below so their callers share one encoding.


def canonicalize_for_hash(payload: dict[str, Any]) -> bytes:
    """Canonical UTF-8 bytes for a hash-input payload.

    Sort keys, remove formatting whitespace and preserve Unicode. Return bytes;
    compute_identity_hash supplies the digest. Lone surrogates fail UTF-8 encoding.

    Payloads must contain string keys and str/int/bool/None/list/dict values.
    Convert datetime, UUID and Decimal explicitly or use model_dump(mode="json").
    Rejecting other types avoids coercion collisions such as integer key 1 and
    string key "1"; rejecting NaN/Infinity keeps identities stable and JSON valid.
    """
    # Recursive validation: nested dict keys can still hit the int-to-str
    # collision class, and nested values can still smuggle a BaseModel
    # past the chokepoint. Walk the entire payload tree once before
    # `json.dumps`, surfacing typed errors that name the offending path
    # rather than letting json.dumps's default raise from deep in the
    # stack.
    _validate_hash_payload(payload, "$")
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_hash_payload(value: Any, path: str) -> None:
    """Reject unsupported containers, keys, and values before JSON encoding.

    Check every nested key for str, and require explicit conversion of
    BaseModel, tuple, set/frozenset, and float inputs. Sets have no stable
    cross-process order. Errors identify the offending payload path;
    Pydantic callers should use model_dump(mode='json') first."""
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
        # shape decision is explicit at the call site.
        raise TypeError(
            f"canonicalize_for_hash got a tuple at {path}; tuples are not "
            f"part of the canonical payload contract (JSON-native: "
            f"str/int/bool/None/list/dict). Convert to list explicitly "
            f"so shape decisions are intentional — json.dumps would "
            f"silently serialize tuples as arrays."
        )
    if isinstance(value, dict):
        bad_keys: list[Any] = []
        for payload_key in value:
            if not isinstance(payload_key, str):
                bad_keys.append(payload_key)
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
    # Finite floats also require an explicit stable string representation;
    # allow_nan=False alone only rejects NaN/Infinity. This check leaves
    # bool values intact despite bool being an int subclass.
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
# Typed wrappers make each identity's required fields explicit. A valid
# hex digest alone cannot reveal a missing field or a different payload shape.
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
    line_start: int,
    line_end: int,
    source_side: Literal["head", "base"] = "head",
) -> str:
    """Hash a raw proposal together with its canonical source file.

    Per `DECISIONS.md#022` (span-key amendment FUP-126), the payload has
    source_file_path plus the eight proposal keys, with location represented
    by line_range. trace_candidates are child output and excluded. Empty
    trace_path and None represent no trace and hash alike.

    Validate source_file_path before hashing so path aliases share an identity;
    identical proposals from different files remain distinct. Raw type/tier
    strings are hashed before admission. Used by rejection events and
    TraceCandidate.source_proposal_hash for provenance joins.
    HEAD keeps its original payload; BASE adds an explicit source_side key.

    Trace candidates use import strings per `DECISIONS.md#024`. V1 resolves
    them with API probes in agent/nodes/trace.py (FUP-209); the filesystem-aware
    coordinates.resolve_candidate_paths is a separate future caller path."""
    if line_end < line_start:
        # Fail loud on a transposed range. The sole V1 caller passes a
        # raw proposal already validated by `_line_end_not_before_start`,
        # but this is a public recipe other producers reuse; a silent
        # transposition would mint a different-but-valid hash and break
        # the `TraceCandidate.source_proposal_hash` join with no error.
        msg = f"compute_proposal_hash: line_end ({line_end}) must be >= line_start ({line_start})"
        raise ValueError(msg)
    from outrider.coordinates import validate_diff_path  # noqa: PLC0415

    canonical_source_file_path = validate_diff_path(source_file_path)
    # No trace must hash alike whether the raw carrier uses () or None.
    normalized_trace_path = list(trace_path) if trace_path else None
    if source_side not in {"head", "base"}:
        raise ValueError("proposal source_side must be head or base")
    payload = {
        "source_file_path": canonical_source_file_path,
        "finding_type": finding_type,
        "evidence_tier": evidence_tier,
        "query_match_id": query_match_id,
        "trace_path": normalized_trace_path,
        "title": title,
        "description": description,
        "evidence": evidence,
        "line_range": {"line_start": line_start, "line_end": line_end},
    }
    if source_side == "base":
        payload["source_side"] = "base"
    return compute_identity_hash(payload)


def compute_response_hash(response_text: str) -> str:
    """SHA-256 hex of the FULL raw analyze response text, UTF-8 encoded.

    Per spec §5: full text, NO 8 KiB prefix. The
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
    For `TraceCandidate.import_string`, the canonical form comes from
    `coordinates.is_valid_trace_import_string` per
    `TraceCandidate._enforce_canonical_import_string` (different
    validator, different shape — two-form dotted-module / relative-
    specifier string vs. repo-relative POSIX path).
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
    import_string: str,
    reason: str,
) -> str:
    """Hash a trace candidate's proposal reference, import string, and reason.

    The import string must be canonical coordinates.is_valid_trace_import_string
    output; source_proposal_hash must match the proposal's rejection/audit hash.
    Per `DECISIONS.md#024` (two-form amendment), syntax distinguishes dotted
    Python imports from JS/TS relative specifiers without another recipe key.
    Canonical JSON makes literal field order irrelevant."""
    return compute_identity_hash(
        {
            "source_proposal_hash": source_proposal_hash,
            "import_string": import_string,
            "reason": reason,
        }
    )


# Namespace for deterministic served-finding ids (Stage B serve flip,
# specs/2026-06-13-analyze-cache-serve-flip.md). A fixed UUID literal, never
# regenerated: changing it re-mints every served finding and breaks the
# no-resurrection content-row guard across an upgrade.
_SERVED_FINDING_ID_NAMESPACE: Final = UUID("a3f1c7e2-5b94-4d6a-8e0f-1c2d3b4a5e6f")


def compute_served_finding_id(*, review_id: UUID, content_hash: str) -> UUID:
    """Mint a cache-served finding ID stable across retries and cache refreshes.

    Key only on review_id and content_hash (FUP-177), with a fixed namespace.
    Including proposal_hash would change the ID when a refresh changes model
    phrasing, bypassing the content row's no-resurrection guard. Conflicting
    content under the same ID instead raises at persistence. _serve_cache_hit
    checks uniqueness before emission; round validation is a later backstop.
    """
    return uuid5(_SERVED_FINDING_ID_NAMESPACE, f"{review_id}:{content_hash}")


def compute_hitl_decision_content_hash(
    *,
    decisions: Sequence[Any],
    annotation: str | None,
) -> str:
    """SHA-256 hex of a `HITLDecision`'s content-derived identity hash.

    The persister uses this hash to reject divergent decisions for one review.
    Require BaseModel items and serialize with model_dump(mode="json") before
    canonical encoding. Loose annotations avoid a circular schemas/hitl import;
    invalid item types raise TypeError rather than producing another recipe.

    Sort by stringified finding_id so submission order does not change identity.
    Include annotation as forensic content; None remains distinct from "".
    """
    from pydantic import BaseModel  # noqa: PLC0415

    serialized: list[dict[str, Any]] = []
    for i, d in enumerate(decisions):
        if not isinstance(d, BaseModel):
            raise TypeError(
                f"compute_hitl_decision_content_hash got a "
                f"{type(d).__name__} at decisions[{i}]; every item must be "
                f"a Pydantic BaseModel (PerFindingDecision). A dict-shaped "
                f"caller would silently hash differently from the Pydantic "
                f"shape — pass tuple[PerFindingDecision, ...]."
            )
        payload = d.model_dump(mode="json")
        if "finding_id" not in payload:
            raise TypeError(
                f"compute_hitl_decision_content_hash got a BaseModel at "
                f"decisions[{i}] whose model_dump output lacks `finding_id`; "
                f"every PerFindingDecision must carry a finding_id."
            )
        serialized.append(payload)
    # Sort by stringified finding_id so submission order doesn't change
    # the hash for the same logical decision set.
    serialized.sort(key=lambda payload: str(payload["finding_id"]))
    return compute_identity_hash(
        {
            "annotation": annotation,
            "decisions": serialized,
        }
    )


def compute_phase_id(
    *,
    review_id: str,
    node_id: str,
    attempt_key: str,
) -> str:
    """SHA-256 hex digest identifying one logical phase span.

    PhaseEventSink keys on (review_id, phase_id, COALESCE(phase_key, ''), marker).
    Derive the phase ID from review, node and attempt so checkpoint replay
    re-emits the same start/end pair rather than creating another phase.

    `attempt_key` derivation per node:
      - `intake`: `"intake"`.
      - `triage`: `"triage"`.
      - `analyze`, pass 0 (the fan-out; specs/2026-07-05-parallel-analyze.md):
        THREE keyed phases per pass, each phase's `attempt_key` equal to its
        `ReviewPhaseEvent.phase_key` VERBATIM — planner `f"plan#{pass}"`,
        worker `f"file:{path}#{pass}"` (injective even for paths containing
        `#`: the pass index is an integer final segment), aggregate
        `f"aggregate#{pass}"` — so `phase_id` inherits the key recipes'
        retry stability and collision-freedom. All derive from pre-merge
        state (`pass = len(state.analysis_rounds)` before the round
        appends).
      - `analyze`, pass 1 (sequential trace re-entry):
        `f"analyze-pass-{len(state.analysis_rounds)}"` BEFORE appending
        the round — the legacy un-keyed envelope.
      - `trace`: `f"trace-pass-{len(state.analysis_rounds)}"`.
      - `publish`: `"publish"`.
      - `hitl`: `"hitl"`.
    """
    return compute_identity_hash(
        {
            "review_id": review_id,
            "node_id": node_id,
            "attempt_key": attempt_key,
        }
    )
