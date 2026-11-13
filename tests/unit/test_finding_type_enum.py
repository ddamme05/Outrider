"""Check the finding-type vocabulary against the active policy.

See spec §7.4, DECISIONS.md#048 and #053, and ``finding-type-enum-constrained``."""

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
    "correctness_bug",
}


def test_finding_type_matches_active_policy_vocabulary() -> None:
    """No extras, no missing — the FindingType enum matches spec §7.4 plus
    the 3 OBSERVED-tier (DECISIONS.md#048) + 7 contextual (DECISIONS.md#053)
    security types and the policy 1.3.0 correctness category."""
    actual = {t.value for t in FindingType}
    assert actual == EXPECTED_FINDING_TYPE_VALUES, (
        f"diff: extra={actual - EXPECTED_FINDING_TYPE_VALUES} "
        f"missing={EXPECTED_FINDING_TYPE_VALUES - actual}"
    )
