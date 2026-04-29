"""FindingType enum has the 12 values from spec §7.4 verbatim.

Backs `finding-type-enum-constrained`. The value strings must match the
keys in the v1.0.0 seed JSON byte-for-byte; if they drift,
test_v1_0_0_seeded_with_canonical_mapping (integration) catches one
side and these unit tests catch the other.
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
}


def test_finding_type_has_exact_12_values() -> None:
    """No extras, no missing — the FindingType enum matches spec §7.4."""
    actual = {t.value for t in FindingType}
    assert actual == EXPECTED_FINDING_TYPE_VALUES, (
        f"diff: extra={actual - EXPECTED_FINDING_TYPE_VALUES} "
        f"missing={EXPECTED_FINDING_TYPE_VALUES - actual}"
    )


def test_finding_type_count_is_12() -> None:
    """Per spec §7.4: exactly 12 values."""
    assert len(list(FindingType)) == 12
