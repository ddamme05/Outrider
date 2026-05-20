# See specs/2026-05-19-analyze-foundation.md §6.
"""`FINDING_TYPE_TO_DIMENSION` mapping + module-load lockstep guard.

Same shape as `SEVERITY_POLICY` (also `MappingProxyType` so runtime
mutation raises). The sister analyze-implementation spec uses this
mapping to derive `ReviewFinding.dimension` deterministically; the
model proposes `finding_type` (constrained to the `FindingType` enum)
and the parser looks up the dimension here.

The model never proposes a dimension directly — same architectural
pattern as severity: deterministic systems map LLM identification to
review-pipeline values; the model only identifies, never sets policy.

**Module-load lockstep assertion.** Test-only totality enforcement
can be bypassed when CI is skipped or tests are not run locally. The
module-import assertion below fails-loud at app startup or test
collection if the three sets (`FindingType` members, `SEVERITY_POLICY`
keys, `FINDING_TYPE_TO_DIMENSION` keys) ever drift apart.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

from outrider.policy.severity import SEVERITY_POLICY, FindingType
from outrider.schemas.review_finding import ReviewDimension

# Wrapped in `MappingProxyType` so runtime mutation raises TypeError.
# Same defense-in-depth shape as `SEVERITY_POLICY` — without the proxy,
# a test fixture or buggy caller could `FINDING_TYPE_TO_DIMENSION[X] = Y`
# and silently change dimension classification for the rest of the
# process. Per `feedback_test_stubs_match_wire_format`: the canonical
# mapping must be tamper-resistant in the same way severity is.
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
    }
)


# Module-load lockstep guard: fail-loud at import time if `FindingType`,
# `SEVERITY_POLICY`, and `FINDING_TYPE_TO_DIMENSION` ever drift out of
# lockstep. Catches the "added a new FindingType + updated SEVERITY_POLICY
# but forgot FINDING_TYPE_TO_DIMENSION" case at app startup or test
# collection, BEFORE any analyze code runs. CI tests cover the
# set-equality assertion at the unit-test layer; this guard is the
# deterministic floor that fires even when `git commit --no-verify`
# bypasses CI.
def verify_lockstep() -> None:
    """Assert lockstep across the three sets at import time.

    Wrapped in a function (not module-level top-level statement) so a
    future contributor adding test-side guard logic has a single
    surface to import + call, and so the subprocess-isolated CI test
    pinning the import-time failure can call this wrapper directly.
    The name is public (no leading underscore) because
    `outrider/__init__.py` imports + calls it as a load-bearing entry
    point.
    """
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
            "at ACTIVE_POLICY_VERSION (api/lifespan.py Step 1b) — bump "
            "ACTIVE_POLICY_VERSION and add a severity_policies migration row if "
            "the policy values changed. "
        )


verify_lockstep()


def lookup_dimension(finding_type: FindingType) -> ReviewDimension:
    """Return the policy-assigned dimension for a finding type.

    The lockstep guard above ensures every `FindingType` has a
    `FINDING_TYPE_TO_DIMENSION` entry at module load; a `KeyError`
    here would mean the lockstep was bypassed (private attribute
    rebinding via something stronger than `MappingProxyType` blocks).
    Fail loud — a missing dimension would silently produce a
    null-dimension `ReviewFinding` downstream, breaking the proof
    boundary's claim that every finding is dimensioned.
    """
    return FINDING_TYPE_TO_DIMENSION[finding_type]
