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

# Deliberately NOT re-exported (per §1 crazy-audit LOW): `outrider.policy.canonical`
# is consumed via the deep import path (`from outrider.policy.canonical import
# SHA256_HEX_PATTERN`, `compute_identity_hash`). Adding it to `__all__` would
# create two coexisting public paths (`outrider.policy.SHA256_HEX_PATTERN` AND
# `outrider.policy.canonical.SHA256_HEX_PATTERN`) and the resulting drift would
# defeat the "single chokepoint for identity-hash encoding" property the module
# exists to enforce. The deep-import path is canonical.

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
