"""SEVERITY_POLICY maps every FindingType to a FindingSeverity per spec §7.4.

Backs `severity-set-by-policy`: baseline severity comes from this
mapping keyed by finding type, never from model output. The test_severity
_policies_seeded.py integration test verifies the DB side; this unit
test verifies the in-memory dict side. If they drift, both fail loud —
one source of truth per layer is simpler than a shared fixture file
(audit-recommendation pattern).
"""

import pytest

from outrider.policy import (
    SEVERITY_POLICY,
    FindingSeverity,
    FindingType,
    lookup_severity,
)
from outrider.policy import severity as severity_module

# The active v1.1.0 mapping, inlined. DECISIONS.md#048 extended the v1.0.0
# spec §7.4 mapping with three OBSERVED-tier security types (Cost Lever 3).
EXPECTED_ACTIVE_POLICY: dict[FindingType, FindingSeverity] = {
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
    FindingType.COMMAND_INJECTION: FindingSeverity.CRITICAL,
    FindingType.UNSAFE_DESERIALIZATION: FindingSeverity.HIGH,
    FindingType.TLS_VERIFY_DISABLED: FindingSeverity.HIGH,
}


def test_severity_policy_has_no_missing_keys() -> None:
    """Every FindingType has a SEVERITY_POLICY entry.

    Per docs/trust-boundaries.md §2: a new FindingType added without a
    corresponding SEVERITY_POLICY entry is a developer bug. The runtime
    fallback handles this case; this test makes the bug fail loud at
    unit-test time before it ever ships.
    """
    missing = set(FindingType) - set(SEVERITY_POLICY.keys())
    assert missing == set(), f"FindingType values without SEVERITY_POLICY entry: {missing}"


def test_severity_policy_has_no_extra_keys() -> None:
    """SEVERITY_POLICY contains no keys that aren't FindingType members."""
    extras = set(SEVERITY_POLICY.keys()) - set(FindingType)
    assert extras == set(), f"SEVERITY_POLICY keys that aren't FindingType: {extras}"


def test_severity_policy_matches_active_mapping() -> None:
    """In-memory SEVERITY_POLICY equals the active v1.1.0 mapping verbatim."""
    assert SEVERITY_POLICY == EXPECTED_ACTIVE_POLICY


@pytest.mark.parametrize(
    ("finding_type", "expected_severity"),
    [
        (FindingType.SQL_INJECTION, FindingSeverity.CRITICAL),
        (FindingType.AUTH_BYPASS, FindingSeverity.CRITICAL),
        (FindingType.HARDCODED_SECRET, FindingSeverity.HIGH),
        (FindingType.MISSING_TEST, FindingSeverity.LOW),
        (FindingType.UNUSED_IMPORT, FindingSeverity.INFO),
        (FindingType.COMMAND_INJECTION, FindingSeverity.CRITICAL),
        (FindingType.UNSAFE_DESERIALIZATION, FindingSeverity.HIGH),
        (FindingType.TLS_VERIFY_DISABLED, FindingSeverity.HIGH),
    ],
)
def test_lookup_severity_returns_canonical_value(
    finding_type: FindingType, expected_severity: FindingSeverity
) -> None:
    """lookup_severity returns the SEVERITY_POLICY value for known types."""
    assert lookup_severity(finding_type) == expected_severity


def test_lookup_severity_unknown_type_falls_back_to_medium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: an unmapped FindingType returns MEDIUM per spec §7.4.

    This test fires only if SEVERITY_POLICY is missing a FindingType
    entry, which test_severity_policy_has_no_missing_keys catches earlier.
    The fallback is the documented runtime safety net; this test is the
    regression check that the safety net itself works, not a normal-path
    verification.
    """
    fake_policy = dict(severity_module.SEVERITY_POLICY)
    del fake_policy[FindingType.UNUSED_IMPORT]
    monkeypatch.setattr(severity_module, "SEVERITY_POLICY", fake_policy)

    assert lookup_severity(FindingType.UNUSED_IMPORT) == FindingSeverity.MEDIUM
