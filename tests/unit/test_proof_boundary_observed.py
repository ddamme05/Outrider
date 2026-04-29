"""enforce_proof_boundary admits OBSERVED only with a query_match_id.

Backs ``evidence-tier-schema-enforced``. The OBSERVED tier is the
structural-evidence claim — a tree-sitter query in the registry fired
on the finding's location, and its id is recorded. Without
``query_match_id``, the LLM is claiming structural evidence it didn't
produce; the validator rejects it.
"""

import pytest

from outrider.policy import (
    EvidenceTier,
    ProofBoundaryViolationError,
    enforce_proof_boundary,
)


def test_observed_with_query_match_id_admits() -> None:
    """OBSERVED + non-None query_match_id is the happy path."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.sql_injection.string_concat",
        trace_path=None,
    )


def test_observed_without_query_match_id_raises() -> None:
    """OBSERVED + None query_match_id raises ProofBoundaryViolationError."""
    with pytest.raises(ProofBoundaryViolationError, match="query_match_id"):
        enforce_proof_boundary(
            evidence_tier=EvidenceTier.OBSERVED,
            query_match_id=None,
            trace_path=None,
        )


def test_observed_admits_even_with_extraneous_trace_path() -> None:
    """OBSERVED with both query_match_id and trace_path admits.

    The validator gates on the REQUIRED artifact for the tier; carrying
    an extra trace_path is not forbidden (it might be informational).
    Only the missing-required case is a violation.
    """
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.OBSERVED,
        query_match_id="py.security.sql_injection.string_concat",
        trace_path=["module", "function", "call_site"],
    )


def test_proof_boundary_violation_is_value_error_subclass() -> None:
    """Pydantic surfaces ValueError-derived errors via field validators.

    If the exception base ever changes, Pydantic's validation chain will
    swallow the error differently — guard the contract.
    """
    assert issubclass(ProofBoundaryViolationError, ValueError)
