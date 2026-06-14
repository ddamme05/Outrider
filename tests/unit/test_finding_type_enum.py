"""FindingType enum has the 12 spec §7.4 values plus the 3 OBSERVED-tier
security types (DECISIONS.md#048).

Backs `finding-type-enum-constrained`. The value strings must match the
keys in the active (v1.1.0) seed JSON byte-for-byte; if they drift,
test_v1_1_0_seeded_with_active_mapping (integration) catches one side
and these unit tests catch the other. (v1.0.0 remains the frozen
12-entry subset, pinned by test_v1_0_0_seeded_with_canonical_mapping.)
"""

from outrider.policy import FindingType

EXPECTED_FINDING_TYPE_VALUES = {
    "sql_injection",
    "xss",
    "hardcoded_secret",
    "auth_bypass",
    "path_traversal",
    "missing_input_validation",
    "n_plus_one_query",
    "blocking_call_in_async",
    "unused_import",
    "missing_error_handling",
    "missing_test",
    "deprecated_api",
    "command_injection",
    "unsafe_deserialization",
    "tls_verify_disabled",
}


def test_finding_type_has_exact_15_values() -> None:
    """No extras, no missing — the FindingType enum matches spec §7.4 plus
    the three OBSERVED-tier security types (DECISIONS.md#048)."""
    actual = {t.value for t in FindingType}
    assert actual == EXPECTED_FINDING_TYPE_VALUES, (
        f"diff: extra={actual - EXPECTED_FINDING_TYPE_VALUES} "
        f"missing={EXPECTED_FINDING_TYPE_VALUES - actual}"
    )


def test_finding_type_count_is_15() -> None:
    """12 spec §7.4 types + 3 OBSERVED-tier security types (DECISIONS.md#048)."""
    assert len(list(FindingType)) == 15
