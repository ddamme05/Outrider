# See specs/2026-05-19-analyze-foundation.md §6.
"""Map each FindingType to its policy-assigned review dimension.

The model proposes a constrained finding type; the parser derives its
dimension from the immutable `FINDING_TYPE_TO_DIMENSION` table. Import-time
`verify_lockstep` requires the FindingType enum, severity keys, and dimension
keys to agree before analysis runs."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

from outrider.policy.severity import SEVERITY_POLICY, FindingType
from outrider.schemas.review_finding import ReviewDimension

# Immutable like SEVERITY_POLICY: a caller cannot change classification
# for later findings by mutating this process-wide table.
FINDING_TYPE_TO_DIMENSION: Final[Mapping[FindingType, ReviewDimension]] = MappingProxyType(
    {
        FindingType.SQL_INJECTION: ReviewDimension.SECURITY,
        FindingType.XSS: ReviewDimension.SECURITY,
        FindingType.HARDCODED_SECRET: ReviewDimension.SECURITY,
        FindingType.AUTH_BYPASS: ReviewDimension.SECURITY,
        FindingType.PATH_TRAVERSAL: ReviewDimension.SECURITY,
        FindingType.MISSING_INPUT_VALIDATION: ReviewDimension.SECURITY,
        FindingType.N_PLUS_ONE_QUERY: ReviewDimension.PERFORMANCE,
        FindingType.BLOCKING_CALL_IN_ASYNC: ReviewDimension.PERFORMANCE,
        FindingType.UNUSED_IMPORT: ReviewDimension.CODE_QUALITY,
        FindingType.MISSING_ERROR_HANDLING: ReviewDimension.CODE_QUALITY,
        FindingType.MISSING_TEST: ReviewDimension.TEST_COVERAGE,
        FindingType.DEPRECATED_API: ReviewDimension.BEST_PRACTICES,
        # OBSERVED-tier security types (policy 1.1.0, DECISIONS.md#048 + #021).
        FindingType.COMMAND_INJECTION: ReviewDimension.SECURITY,
        FindingType.UNSAFE_DESERIALIZATION: ReviewDimension.SECURITY,
        FindingType.TLS_VERIFY_DISABLED: ReviewDimension.SECURITY,
        # Contextual security types (policy 1.2.0, DECISIONS.md#053 + #021).
        FindingType.WEAK_CRYPTO: ReviewDimension.SECURITY,
        FindingType.WEAK_PASSWORD_HASH: ReviewDimension.SECURITY,
        FindingType.INSECURE_RANDOMNESS: ReviewDimension.SECURITY,
        FindingType.SSRF: ReviewDimension.SECURITY,
        FindingType.SSRF_METADATA: ReviewDimension.SECURITY,
        FindingType.OPEN_REDIRECT: ReviewDimension.SECURITY,
        FindingType.OPEN_REDIRECT_AUTHED: ReviewDimension.SECURITY,
    }
)


def verify_lockstep() -> None:
    """Require matching finding-type, severity, and dimension keys at import.

    `outrider/__init__.py` also calls this public guard at package load;
    tests can invoke it directly when checking a changed taxonomy."""
    finding_type_set = set(FindingType)
    severity_set = set(SEVERITY_POLICY)
    dimension_set = set(FINDING_TYPE_TO_DIMENSION)
    if not (finding_type_set == severity_set == dimension_set):
        raise AssertionError(
            "Policy lockstep violation:\n"
            f"  FindingType members:              {sorted(t.value for t in finding_type_set)}\n"
            f"  SEVERITY_POLICY keys:             {sorted(t.value for t in severity_set)}\n"
            f"  FINDING_TYPE_TO_DIMENSION keys:   {sorted(t.value for t in dimension_set)}\n"
            "All three must be identical. A new FindingType requires "
            "matching entries in SEVERITY_POLICY (severity assignment) and "
            "FINDING_TYPE_TO_DIMENSION (dimension assignment) in the same commit.\n"
            "After fixing the lockstep, lifespan startup will additionally "
            "verify the live SEVERITY_POLICY mapping against the DB-stored row "
            "at ACTIVE_POLICY_VERSION (api/lifespan.py, the severity-policy "
            "fingerprint check) — bump "
            "ACTIVE_POLICY_VERSION and add a severity_policies migration row if "
            "the policy values changed. "
        )


verify_lockstep()


def lookup_dimension(finding_type: FindingType) -> ReviewDimension:
    """Return the policy-assigned dimension; missing keys fail loudly.

    `verify_lockstep` requires an entry for every FindingType at import.
    A missing dimension must not become a null field on a ReviewFinding."""
    return FINDING_TYPE_TO_DIMENSION[finding_type]
