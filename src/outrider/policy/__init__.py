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
from outrider.policy.versions import (
    PolicyVersionShapeError,
    UnknownPolicyVersionError,
    load_policy_for_version,
)

# Import canonical and dimensions from their own modules. Re-exporting
# dimensions here would create a cycle through schemas.review_finding;
# canonical keeps one public location for identity-hash recipes.

__all__ = [
    "SEVERITY_POLICY",
    "EvidenceTier",
    "FindingSeverity",
    "FindingType",
    "PolicyVersionShapeError",
    "ProofBoundaryViolationError",
    "UnknownPolicyVersionError",
    "enforce_proof_boundary",
    "load_policy_for_version",
    "lookup_severity",
]
