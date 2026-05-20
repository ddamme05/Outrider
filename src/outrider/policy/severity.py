# Severity policy per docs/trust-boundaries.md §2 (severity-set-by-policy)
"""FindingType + FindingSeverity enums + SEVERITY_POLICY dict.

The LLM identifies a FindingType from a constrained enum; SEVERITY_POLICY
(a static dict keyed by FindingType) assigns severity. Reviewers can
override via PerFindingDecision.outcome = SEVERITY_OVERRIDE with a
required reason. Model output does NOT determine severity anywhere in
the pipeline — every read of severity goes through SEVERITY_POLICY (or
the override path), never through model JSON.

Policy versions are stored in the severity_policies table keyed by
version string per `severity-policy-versioned-for-replay`; replay at
time T uses the version in effect at T, not the current one. The
versioned loader lives in `policy/versions.py`. Mutating SEVERITY_POLICY
in place would invalidate every historical review's replay; mapping
changes ship as new policy versions seeded by new migrations.

`ACTIVE_POLICY_VERSION` is the write-time source: it identifies which
row in `severity_policies` the live `SEVERITY_POLICY` mapping
corresponds to. Whenever `SEVERITY_POLICY` changes, `ACTIVE_POLICY_VERSION`
bumps in the same commit AND a migration seeds the new row. The startup
fingerprint check in `api/lifespan.py` compares the live mapping to the
DB row keyed by this constant at lifespan; mismatches fail loud before
accepting webhooks.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class FindingType(StrEnum):
    """The constrained enum the LLM classifies findings under.

    The retry/normalize/anomaly path that turns a model-produced string
    into a FindingType enum value (or anomalies if no safe match) is the
    analyze node's responsibility per `finding-type-enum-constrained`.
    This enum is the type system; the analyze node is what bridges from
    untrusted model output to it.
    """

    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    HARDCODED_SECRET = "hardcoded_secret"  # noqa: S105 (finding-type label, not a password)
    AUTH_BYPASS = "auth_bypass"
    PATH_TRAVERSAL = "path_traversal"
    MISSING_INPUT_VALIDATION = "missing_input_validation"
    N_PLUS_ONE_QUERY = "n_plus_one_query"
    BLOCKING_CALL_IN_ASYNC = "blocking_call_in_async"
    UNUSED_IMPORT = "unused_import"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    MISSING_TEST = "missing_test"
    DEPRECATED_API = "deprecated_api"


class FindingSeverity(StrEnum):
    """Severity tier assigned by SEVERITY_POLICY, never by the model."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Wrapped in MappingProxyType so runtime mutation raises TypeError.
# Same defense-in-depth shape as `outrider.llm.pricing.RATE_TABLE` and
# `outrider.schemas.review_finding._CONFIDENCE_BY_TIER` — without the
# proxy, a test fixture or buggy caller could `SEVERITY_POLICY[X] = Y`
# and silently change classification for the rest of the process.
# Inlined into the proxy call directly so there's no importable
# bare-dict back-door. Per-version replay reads from
# `policy/versions.py::load_policy_for_version`, not this constant —
# this is the LIVE policy; historical policies live in the DB.
SEVERITY_POLICY: Final[Mapping[FindingType, FindingSeverity]] = MappingProxyType(
    {
        FindingType.SQL_INJECTION: FindingSeverity.CRITICAL,
        FindingType.AUTH_BYPASS: FindingSeverity.CRITICAL,
        FindingType.HARDCODED_SECRET: FindingSeverity.HIGH,
        FindingType.XSS: FindingSeverity.HIGH,
        FindingType.PATH_TRAVERSAL: FindingSeverity.HIGH,
        FindingType.MISSING_INPUT_VALIDATION: FindingSeverity.MEDIUM,
        FindingType.N_PLUS_ONE_QUERY: FindingSeverity.MEDIUM,
        FindingType.BLOCKING_CALL_IN_ASYNC: FindingSeverity.MEDIUM,
        FindingType.MISSING_ERROR_HANDLING: FindingSeverity.LOW,
        FindingType.MISSING_TEST: FindingSeverity.LOW,
        FindingType.UNUSED_IMPORT: FindingSeverity.INFO,
        FindingType.DEPRECATED_API: FindingSeverity.INFO,
    }
)


# `re.ASCII` is load-bearing: without it, `\d` matches Unicode digits
# (Arabic-Indic, etc.), and the Python guard would accept values that the
# DB's CHECK constraint (in migration `3d03bca7f2be`) rejects — silent
# divergence between the two halves of the belt-and-suspenders. Leading
# zeros are disallowed (semver §2 forbids `01.0.0`) so future numeric
# ordering doesn't desync from string-eq lookups.
BARE_SEMVER_PATTERN: Final[str] = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
"""Strict bare-semver regex string.

Single source of truth. Used by the runtime `_SEMVER_RE` check below
AND by every Pydantic `Field(pattern=...)` that gates a
`policy_version` (audit events, ReviewFinding, etc.). Defining the
pattern once stops the catalog-self-drift between `policy/severity.py`,
`audit/events.py`, and the DB CHECK constraint in
`db/migrations/versions/3d03bca7f2be_severity_policies_protections.py`
from re-emerging the way it did pre-PR-review-
"""

_SEMVER_RE: Final[re.Pattern[str]] = re.compile(BARE_SEMVER_PATTERN, re.ASCII)

ACTIVE_POLICY_VERSION: Final[str] = "1.0.0"
if not _SEMVER_RE.fullmatch(ACTIVE_POLICY_VERSION):
    raise RuntimeError(
        f"ACTIVE_POLICY_VERSION must be bare ASCII semver (no v prefix, no "
        f"leading zeros, no pre-release/build suffix); got {ACTIVE_POLICY_VERSION!r}"
    )


_DEFAULT_SEVERITY: Final[FindingSeverity] = FindingSeverity.MEDIUM


def lookup_severity(finding_type: FindingType) -> FindingSeverity:
    """Return the policy-assigned severity for a finding type.

    Returns FindingSeverity.MEDIUM as the documented fallback for an
    unmapped FindingType per spec §7.4. The fallback is a defense-in-depth
    safety net for the case where a future FindingType value is added
    without a corresponding SEVERITY_POLICY entry — a developer bug.
    The unit tests catch that condition (test_severity_policy_dict
    asserts no FindingType is missing); this fallback is the runtime
    behavior if the unit test is ever bypassed.

    Per spec §7.4: "Unknown finding types default to MEDIUM — not
    CRITICAL like an incident response system would, because a code
    review false positive is annoying, not dangerous."
    """
    return SEVERITY_POLICY.get(finding_type, _DEFAULT_SEVERITY)
