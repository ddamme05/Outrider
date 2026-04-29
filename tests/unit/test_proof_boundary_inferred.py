"""enforce_proof_boundary admits INFERRED only with a non-empty trace_path.

Backs ``evidence-tier-schema-enforced``. The INFERRED tier is the
structural-by-reference claim — the agent walked the ast_facts trace
from a known scope to the claim site, and the traversal path is
recorded. Without a non-empty ``trace_path``, the LLM is claiming a
traversal it didn't perform; the validator rejects it.
"""

import pytest

from outrider.policy import (
    EvidenceTier,
    ProofBoundaryViolationError,
    enforce_proof_boundary,
)


def test_inferred_with_trace_path_admits() -> None:
    """INFERRED + non-empty trace_path is the happy path."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.INFERRED,
        query_match_id=None,
        trace_path=["module:foo", "function:get_user", "call_site:line_42"],
    )


def test_inferred_without_trace_path_raises() -> None:
    """INFERRED + None trace_path raises ProofBoundaryViolationError."""
    with pytest.raises(ProofBoundaryViolationError, match="non-empty trace_path"):
        enforce_proof_boundary(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=None,
        )


def test_inferred_with_empty_trace_path_raises() -> None:
    """INFERRED + empty list raises — an empty list lists no scope units.

    Per docs/trust-boundaries.md §1: INFERRED requires a trace_path
    that "lists the scope units walked." An empty list fails that
    requirement. The validator IS the contract gate; deferring this
    check to a downstream Pydantic field constraint would let the
    boundary admit findings that violate the rule between admission
    and the constraint firing.

    (Earlier draft of this test admitted empty list and deferred to the
    ReviewFinding field constraint. The audit pass on commit 131f8d6
    flagged the deferral as a real correctness gap — the empty list
    case is the boundary's job, not the model field's.)
    """
    with pytest.raises(ProofBoundaryViolationError, match="non-empty trace_path"):
        enforce_proof_boundary(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=[],
        )


def test_inferred_admits_even_with_extraneous_query_match_id() -> None:
    """INFERRED with both trace_path and query_match_id admits."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.INFERRED,
        query_match_id="py.security.placeholder",
        trace_path=["scope_a", "scope_b"],
    )
