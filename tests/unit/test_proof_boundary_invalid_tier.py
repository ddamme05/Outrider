"""Only EvidenceTier members may enter enforce_proof_boundary.

Reject raw strings, misspellings, integers and None before applying the tier's
artifact requirements. JUDGED admission is covered by test_proof_boundary_judged.py."""

import pytest

from outrider.policy import (
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
    """None is not an EvidenceTier and must be rejected."""
    with pytest.raises(ProofBoundaryViolationError, match="must be an EvidenceTier"):
        enforce_proof_boundary(
            evidence_tier=None,  # type: ignore[arg-type]
            query_match_id="py.security.placeholder",
            trace_path=None,
        )
