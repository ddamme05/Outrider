"""enforce_proof_boundary rejects values that aren't EvidenceTier members.

Audit-driven: the previous validator implicitly admitted any tier value
that didn't compare equal to OBSERVED or INFERRED — the "JUDGED admits"
path was actually "anything-not-OBSERVED-or-INFERRED admits" because no
isinstance(EvidenceTier) check ran first. A typo'd tier value or a
type-confused argument (e.g., a raw string passed before Pydantic
validates) would silently admit.

The fix added an explicit isinstance check at the top of the validator;
this test file verifies the rejection path. The boundary's gate is
closed by default — only EvidenceTier members proceed past this gate.
"""

import pytest

from outrider.policy import (
    EvidenceTier,
    ProofBoundaryViolationError,
    enforce_proof_boundary,
)


def test_string_tier_value_raises() -> None:
    """A raw string (even one matching an EvidenceTier value) raises."""
    with pytest.raises(ProofBoundaryViolationError, match="must be an EvidenceTier"):
        enforce_proof_boundary(
            evidence_tier="observed",  # type: ignore[arg-type]
            query_match_id="py.security.placeholder",
            trace_path=None,
        )


def test_typo_string_tier_raises() -> None:
    """A typo'd string that doesn't match any EvidenceTier value raises."""
    with pytest.raises(ProofBoundaryViolationError, match="must be an EvidenceTier"):
        enforce_proof_boundary(
            evidence_tier="OBESERVED",  # type: ignore[arg-type]  (typo)
            query_match_id="py.security.placeholder",
            trace_path=None,
        )


def test_int_tier_raises() -> None:
    """A non-string non-enum (int) raises."""
    with pytest.raises(ProofBoundaryViolationError, match="must be an EvidenceTier"):
        enforce_proof_boundary(
            evidence_tier=42,  # type: ignore[arg-type]
            query_match_id="py.security.placeholder",
            trace_path=None,
        )


def test_none_tier_raises() -> None:
    """None raises (the previous implementation would have admitted)."""
    with pytest.raises(ProofBoundaryViolationError, match="must be an EvidenceTier"):
        enforce_proof_boundary(
            evidence_tier=None,  # type: ignore[arg-type]
            query_match_id="py.security.placeholder",
            trace_path=None,
        )


def test_valid_tier_with_no_artifacts_for_judged_admits() -> None:
    """Sanity: a real EvidenceTier.JUDGED still admits (regression guard).

    The new isinstance check must not break the JUDGED admission path.
    """
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
    )
