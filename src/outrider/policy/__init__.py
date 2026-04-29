"""Severity policy, proof-boundary validator, and replay-time policy loader.

Re-exports for ergonomic ``from outrider.policy import ...`` use. Each
submodule has its own focused docstring; this package is the namespace.
"""

from outrider.policy.findings import (
    EvidenceTier,
    ProofBoundaryViolationError,
    enforce_proof_boundary,
)
from outrider.policy.severity import (
    SEVERITY_POLICY,
    FindingSeverity,
    FindingType,
    lookup_severity,
)

__all__ = [
    "SEVERITY_POLICY",
    "EvidenceTier",
    "FindingSeverity",
    "FindingType",
    "ProofBoundaryViolationError",
    "enforce_proof_boundary",
    "lookup_severity",
]
