"""enforce_proof_boundary admits INFERRED only with a trace_path.

Backs ``evidence-tier-schema-enforced``. The INFERRED tier is the
structural-by-reference claim — the agent walked the ast_facts trace
from a known scope to the claim site, and the traversal path is
recorded. Without ``trace_path``, the LLM is claiming a traversal it
didn't perform; the validator rejects it.
"""

import pytest

from outrider.policy import (
    EvidenceTier,
    ProofBoundaryViolationError,
    enforce_proof_boundary,
)


def test_inferred_with_trace_path_admits() -> None:
    """INFERRED + non-None trace_path is the happy path."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.INFERRED,
        query_match_id=None,
        trace_path=["module:foo", "function:get_user", "call_site:line_42"],
    )


def test_inferred_without_trace_path_raises() -> None:
    """INFERRED + None trace_path raises ProofBoundaryViolationError."""
    with pytest.raises(ProofBoundaryViolationError, match="trace_path"):
        enforce_proof_boundary(
            evidence_tier=EvidenceTier.INFERRED,
            query_match_id=None,
            trace_path=None,
        )


def test_inferred_with_empty_trace_path_admits() -> None:
    """INFERRED + empty list admits — empty is not None.

    The validator gates on None vs not-None for the artifact existence.
    Whether an empty trace_path is semantically meaningful is the
    ReviewFinding's own field constraint (a min_length on the list, etc.)
    rather than this validator's responsibility.
    """
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
