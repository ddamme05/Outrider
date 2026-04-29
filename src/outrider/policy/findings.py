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

Replay-time verification (that ``query_match_id`` corresponds to a real
query in ``queries/python/*.scm`` and that the query actually matches
the stored evidence span) is application-layer work in
``audit/replay.py`` (when written), not part of this module. This module
covers the admission gate; replay covers the integrity gate.
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


def enforce_proof_boundary(
    evidence_tier: EvidenceTier,
    query_match_id: str | None,
    trace_path: list[str] | None,
) -> None:
    """Validate that ``evidence_tier`` admits given the supplied proof artifacts.

    Raises ``ProofBoundaryViolationError`` with the failing condition
    named if the tier's proof artifact is missing, empty, or the wrong
    shape. JUDGED admits regardless — it's the explicit no-structural-
    claim path.

    Per docs/trust-boundaries.md §1 + spec §7.3: OBSERVED requires a
    non-empty `str` query_match_id pointing into the queries registry;
    INFERRED requires a non-empty `list[str]` trace_path listing the
    scope units walked. The runtime ``isinstance`` checks matter
    because the validator is exposed publicly (any caller, not just
    Pydantic via ReviewFinding, can invoke it). A truthy non-string
    query_match_id (e.g., an int) or a truthy non-list trace_path (e.g.,
    a string — strings are sequences, which would have passed an
    earlier ``Sequence[Any]`` type) would slip through a truthy-only
    check. The validator is the boundary; admitting unstructured proof
    artifacts is the same class of bug as admitting empty ones.
    """
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
    if evidence_tier == EvidenceTier.INFERRED and (
        not isinstance(trace_path, list) or not trace_path
    ):
        raise ProofBoundaryViolationError(
            f"INFERRED finding must carry a non-empty list trace_path; "
            f"got {trace_path!r} (type={type(trace_path).__name__}). "
            "INFERRED is the structural-by-reference tier — it requires "
            "a recorded traversal through ast_facts that lists the scope "
            "units walked. None, an empty list, or a non-list value (a "
            "string, a tuple, etc.) all fail the boundary; if the LLM "
            "produced INFERRED without a real trace, the right path is "
            "to downgrade the tier to JUDGED, not to admit the finding."
        )
    # JUDGED admits with neither, by design.
