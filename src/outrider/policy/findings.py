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

from collections.abc import Sequence
from enum import StrEnum
from typing import Any


class EvidenceTier(StrEnum):
    """How a finding's claim is justified.

    OBSERVED: structural — a tree-sitter query in the registry matched
              the location, and the match id is recorded.
    INFERRED: structural-by-reference — the agent walked the ast_facts
              trace from a known scope to the claim site, and the
              traversal path is recorded.
    JUDGED:   model interpretation only. No structural evidence is
              claimed. Lower confidence by construction.
    """

    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    JUDGED = "JUDGED"


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
    trace_path: Sequence[Any] | None,
) -> None:
    """Validate that ``evidence_tier`` admits given the supplied proof artifacts.

    Raises ``ProofBoundaryViolationError`` with the missing field named
    if the tier requires a proof artifact that's None. JUDGED admits
    regardless — it's the explicit no-structural-claim path.

    Per docs/trust-boundaries.md §1: this is the schema-layer gate. The
    LLM cannot claim structural evidence it didn't produce; if the model
    returns OBSERVED for a finding where no tree-sitter query fired, the
    analyze node tries to construct a ReviewFinding and Pydantic raises
    via this validator. The tier is not self-reported in any
    consequential way — the proof boundary disposes of model claims that
    don't meet it.
    """
    if evidence_tier == EvidenceTier.OBSERVED and not query_match_id:
        raise ProofBoundaryViolationError(
            f"OBSERVED finding must carry a non-empty query_match_id; "
            f"got {query_match_id!r}. OBSERVED is the structural tier — "
            "it requires a tree-sitter query match identifier in the "
            "queries registry. None or an empty string fails the boundary; "
            "if the LLM produced OBSERVED without a real query_match_id, "
            "the right path is to downgrade the tier to JUDGED, not to "
            "admit the finding."
        )
    if evidence_tier == EvidenceTier.INFERRED and not trace_path:
        raise ProofBoundaryViolationError(
            f"INFERRED finding must carry a non-empty trace_path; "
            f"got {trace_path!r}. INFERRED is the structural-by-reference "
            "tier — it requires a recorded traversal through ast_facts "
            "that lists the scope units walked. None or an empty list "
            "fails the boundary (an empty list lists no scope units); "
            "if the LLM produced INFERRED without a real trace, the right "
            "path is to downgrade the tier to JUDGED, not to admit the "
            "finding."
        )
    # JUDGED admits with neither, by design.
