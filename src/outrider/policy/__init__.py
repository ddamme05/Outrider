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

# Deliberately NOT re-exported (per §1 LOW + §6 implementation
# discovery): `outrider.policy.canonical` and `outrider.policy.dimensions`
# are consumed via deep import paths (`from outrider.policy.canonical
# import SHA256_HEX_PATTERN`, `from outrider.policy.dimensions import
# FINDING_TYPE_TO_DIMENSION`). Re-exporting from this `__init__.py`
# creates a circular import for `dimensions` (which imports
# `ReviewDimension` from `outrider.schemas.review_finding`, which
# imports back into `outrider.policy` for `EvidenceTier`), and would
# create two coexisting public paths for `canonical` — defeating the
# "single chokepoint for identity-hash encoding" property the module
# exists to enforce. Deep-import is canonical for both.

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
