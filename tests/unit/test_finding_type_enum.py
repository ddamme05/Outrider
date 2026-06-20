"""FindingType enum has the 12 spec §7.4 values, the 3 OBSERVED-tier
security types (DECISIONS.md#048), and the 7 contextual security types
(DECISIONS.md#053).

Backs `finding-type-enum-constrained`. The value strings must match the
keys in the active (v1.2.0) seed JSON byte-for-byte; if they drift,
test_v1_2_0_seeded_with_active_mapping (integration) catches one side
and these unit tests catch the other. (v1.0.0 remains the frozen
12-entry subset, pinned by test_v1_0_0_seeded_with_canonical_mapping;
v1.1.0 the frozen 15-entry subset.)
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
    # Contextual security types (policy 1.2.0, DECISIONS.md#053).
    "weak_crypto",
    "weak_password_hash",
    "insecure_randomness",
    "ssrf",
    "ssrf_metadata",
    "open_redirect",
    "open_redirect_authed",
}


def test_finding_type_has_exact_22_values() -> None:
    """No extras, no missing — the FindingType enum matches spec §7.4 plus
    the 3 OBSERVED-tier (DECISIONS.md#048) + 7 contextual (DECISIONS.md#053)
    security types."""
    actual = {t.value for t in FindingType}
    assert actual == EXPECTED_FINDING_TYPE_VALUES, (
        f"diff: extra={actual - EXPECTED_FINDING_TYPE_VALUES} "
        f"missing={EXPECTED_FINDING_TYPE_VALUES - actual}"
    )


def test_finding_type_count_is_22() -> None:
    """12 spec §7.4 + 3 OBSERVED-tier (DECISIONS.md#048) + 7 contextual
    (DECISIONS.md#053) security types."""
    assert len(list(FindingType)) == 22
