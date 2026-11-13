"""Assign severity from finding type using the versioned policy.

The model does not choose severity. Human overrides are validated separately.
A rule change requires a new policy version and matching migration/startup
checks so replay can interpret saved reviews.

See severity-set-by-policy and severity-policy-versioned-for-replay.
"""

import re
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final


class FindingType(StrEnum):
    """Categories admitted by the analyzer and used by the severity policy.

    The analyzer records rejections before applying the severity policy.
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
    COMMAND_INJECTION = "command_injection"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    TLS_VERIFY_DISABLED = "tls_verify_disabled"
    # Contextual security types (policy 1.2.0, DECISIONS.md#053 + #021).
    # Dual-mode: the model picks the finer type as JUDGED; the policy
    # assigns the fixed severity. The _METADATA / _AUTHED / _PASSWORD_HASH
    # splits are the escalated variants, split so context selects a fixed
    # severity without a runtime modifier (severity-set-by-policy).
    WEAK_CRYPTO = "weak_crypto"
    WEAK_PASSWORD_HASH = "weak_password_hash"  # noqa: S105 (finding-type label, not a password)
    INSECURE_RANDOMNESS = "insecure_randomness"
    SSRF = "ssrf"
    SSRF_METADATA = "ssrf_metadata"
    OPEN_REDIRECT = "open_redirect"
    OPEN_REDIRECT_AUTHED = "open_redirect_authed"
    CORRECTNESS_BUG = "correctness_bug"


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
        # OBSERVED-tier security types (policy 1.1.0, DECISIONS.md#048).
        FindingType.COMMAND_INJECTION: FindingSeverity.CRITICAL,
        FindingType.UNSAFE_DESERIALIZATION: FindingSeverity.HIGH,
        FindingType.TLS_VERIFY_DISABLED: FindingSeverity.HIGH,
        # Contextual security types (policy 1.2.0, DECISIONS.md#053).
        FindingType.WEAK_CRYPTO: FindingSeverity.HIGH,
        FindingType.WEAK_PASSWORD_HASH: FindingSeverity.CRITICAL,
        FindingType.INSECURE_RANDOMNESS: FindingSeverity.HIGH,
        FindingType.SSRF: FindingSeverity.HIGH,
        FindingType.SSRF_METADATA: FindingSeverity.CRITICAL,
        FindingType.OPEN_REDIRECT: FindingSeverity.MEDIUM,
        FindingType.OPEN_REDIRECT_AUTHED: FindingSeverity.HIGH,
        FindingType.CORRECTNESS_BUG: FindingSeverity.MEDIUM,
    }
)


# `re.ASCII` is load-bearing: without it, `\d` matches Unicode digits
# (Arabic-Indic, etc.), and the Python guard would accept values that the
# DB's CHECK constraint (in migration `3d03bca7f2be`) rejects — silent
# divergence between the two halves of the belt-and-suspenders. Leading
# zeros are disallowed (semver §2 forbids `01.0.0`) so future numeric
# ordering doesn't desync from string-eq lookups.
BARE_SEMVER_PATTERN: Final[str] = r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
"""Shared bare-semver pattern for runtime and Pydantic policy-version guards.

The DB CHECK constraint in migration 3d03bca7f2be uses the same shape;
keep it aligned when changing policy-version validation.
"""

_SEMVER_RE: Final[re.Pattern[str]] = re.compile(BARE_SEMVER_PATTERN, re.ASCII)

# 1.2.0 adds the contextual security types per DECISIONS.md#053 (seeded by
# the severity_policies 1.2.0 migration). Additive — 1.1.0 reviews replay
# untouched per `severity-policy-versioned-for-replay`.
ACTIVE_POLICY_VERSION: Final[str] = "1.3.0"
if not _SEMVER_RE.fullmatch(ACTIVE_POLICY_VERSION):
    raise RuntimeError(
        f"ACTIVE_POLICY_VERSION must be bare ASCII semver (no v prefix, no "
        f"leading zeros, no pre-release/build suffix); got {ACTIVE_POLICY_VERSION!r}"
    )


_DEFAULT_SEVERITY: Final[FindingSeverity] = FindingSeverity.MEDIUM


def lookup_severity(finding_type: FindingType) -> FindingSeverity:
    """Return the policy severity, with MEDIUM as the spec §7.4 fallback.

    Normal startup requires a total severity/dimension mapping through
    policy.dimensions.verify_lockstep. The fallback remains for direct callers
    with an unmapped value; it does not replace that startup guard."""
    return SEVERITY_POLICY.get(finding_type, _DEFAULT_SEVERITY)
