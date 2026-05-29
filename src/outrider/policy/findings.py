# Proof-boundary validator per docs/trust-boundaries.md §1
"""Proof-boundary admission rules and the EvidenceTier classification.

Every ReviewFinding carries an `evidence_tier` that classifies the
nature of the claim. OBSERVED findings must carry a `query_match_id`
pointing to a real entry in the queries registry. INFERRED findings
must carry a `trace_path` listing the scope units walked. JUDGED
findings carry neither — they are model interpretations that don't
claim structural evidence.

This module exposes:

  - ``EvidenceTier`` enum: the three valid classifications
  - ``ProofBoundaryViolationError``: typed exception with the missing
    field name in the message
  - ``enforce_proof_boundary``: validator function that raises on
    OBSERVED-without-query_match_id or INFERRED-without-trace_path

Replay-time verification of these artifacts is application-layer work in
``audit/replay.py``: V1 re-verifies that ``query_match_id`` resolves in the
queries registry and recomputes ``finding_content_hash`` (verify-only). The
stronger check — that the query actually matches the stored evidence span,
a tree-sitter re-run against source bytes — is future scope per
``DECISIONS.md#031`` (it needs a durable source store, not checkpoint
state). This module covers the admission gate; replay covers the integrity
gate.
"""

from enum import StrEnum


class EvidenceTier(StrEnum):
    """How a finding's claim is justified.

    OBSERVED: structural — a tree-sitter query in the registry matched
              the location, and the match id is recorded.
    INFERRED: structural-by-reference — the agent walked the ast_facts
              trace from a known scope to the claim site, and the
              traversal path is recorded.
    JUDGED:   model interpretation only. No structural evidence is
              claimed. Lower confidence by construction.

    Values are lowercase per docs/spec.md §7.3, matching the convention
    of FindingType and FindingSeverity. The Python member names are
    uppercase (PEP 8); only the serialized string values are lowercase.
    """

    OBSERVED = "observed"
    INFERRED = "inferred"
    JUDGED = "judged"


class ProofBoundaryViolationError(ValueError):
    """Raised when a finding's evidence_tier and proof artifacts disagree.

    Subclasses ValueError so Pydantic's field validators surface it as
    a normal validation error in the model's `ValidationError` accumulator
    (Pydantic distinguishes ValueError-derived from arbitrary exceptions
    in field-validator failure paths).
    """


def _trace_path_is_valid(trace_path: object) -> bool:
    """A trace_path is valid iff it's a non-empty list-or-tuple of non-empty strs.

    The list-or-tuple check rejects strings (sequences but neither list
    nor tuple) and other sequence-likes (dict, set, etc.); the non-empty
    check rejects sequences with no walked-scope-unit identifiers; the
    per-element check rejects `[42]`, `[None]`, `[""]`, etc. — anything
    that isn't a real scope-unit identifier string. Each gate is
    necessary; no single gate is sufficient.

    Accepting both list and tuple matches the schemas-layer canonical
    type for ReviewFinding.trace_path (`tuple[str, ...] | None` for
    true post-construction immutability) while preserving back-compat
    with direct callers passing `list[str]`.
    """
    if not isinstance(trace_path, (list, tuple)) or not trace_path:
        return False
    return all(isinstance(item, str) and item for item in trace_path)


def enforce_proof_boundary(
    evidence_tier: EvidenceTier,
    query_match_id: str | None,
    trace_path: list[str] | tuple[str, ...] | None,
) -> None:
    """Validate that ``evidence_tier`` admits given the supplied proof artifacts.

    Raises ``ProofBoundaryViolationError`` with the failing condition
    named. JUDGED admits regardless — it's the explicit no-structural-
    claim path. Invalid tiers (anything not in the EvidenceTier enum)
    raise; the previous "implicit admission for unrecognized values"
    behavior was a real correctness gap surfaced by the audit pass on
    commit c632a53.

    Per docs/trust-boundaries.md §1 + spec §7.3:
      - evidence_tier MUST be an EvidenceTier member.
      - OBSERVED requires a non-empty `str` query_match_id pointing
        into the queries registry.
      - INFERRED requires a non-empty `list[str]` or `tuple[str, ...]`
        trace_path where every element is a non-empty str (a scope-unit
        identifier). Tuple admits because schemas-layer ReviewFinding
        types the field as `tuple[str, ...] | None` for true immutability;
        direct callers passing `list[str]` continue to work.
      - JUDGED admits without either artifact; the tier itself signals
        that no structural claim is made.

    The runtime checks matter because the validator is exposed publicly
    (any caller, not just Pydantic via ReviewFinding, can invoke it).
    Type hints alone don't enforce at runtime; the validator must
    defend itself.
    """
    if not isinstance(evidence_tier, EvidenceTier):
        raise ProofBoundaryViolationError(
            f"evidence_tier must be an EvidenceTier member; got "
            f"{evidence_tier!r} (type={type(evidence_tier).__name__}). "
            "Valid values are EvidenceTier.OBSERVED, .INFERRED, .JUDGED. "
            "Unrecognized tiers must not slip through to a default-admit "
            "path; the boundary's gate is closed by default."
        )
    if evidence_tier == EvidenceTier.OBSERVED and (
        not isinstance(query_match_id, str) or not query_match_id
    ):
        raise ProofBoundaryViolationError(
            f"OBSERVED finding must carry a non-empty str query_match_id; "
            f"got {query_match_id!r} (type={type(query_match_id).__name__}). "
            "OBSERVED is the structural tier — it requires a tree-sitter "
            "query match identifier in the queries registry. None, an "
            "empty string, or a non-string value all fail the boundary; "
            "if the LLM produced OBSERVED without a real query_match_id, "
            "the right path is to downgrade the tier to JUDGED, not to "
            "admit the finding."
        )
    if evidence_tier == EvidenceTier.INFERRED and not _trace_path_is_valid(trace_path):
        raise ProofBoundaryViolationError(
            f"INFERRED finding must carry a non-empty list[str] or "
            f"tuple[str, ...] trace_path with non-empty string elements; "
            f"got {trace_path!r} (type={type(trace_path).__name__}). "
            "INFERRED is the structural-by-reference tier — it requires "
            "a recorded traversal through ast_facts that lists the scope "
            "units walked. None, an empty sequence, a non-list/non-tuple "
            "value (string, dict, set, etc.), or a sequence containing "
            "non-string / empty-string elements all fail the boundary; "
            "if the LLM produced INFERRED without a real trace, the "
            "right path is to downgrade the tier to JUDGED, not to admit "
            "the finding."
        )
    # JUDGED admits with neither artifact; isinstance check above
    # already gated on tier validity, so falling through here means
    # the tier is JUDGED specifically, not an unrecognized value.
