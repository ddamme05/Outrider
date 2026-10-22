"""FindingSeverity enum has the 5 values from spec §7.4.

The value strings must match the values in the v1.0.0 seed JSON
byte-for-byte; the integration test asserts the seed side, this asserts
the enum side.
"""

from outrider.policy import FindingSeverity

EXPECTED_SEVERITY_VALUES = {"critical", "high", "medium", "low", "info"}


def test_finding_severity_has_exact_5_values() -> None:
    """No extras, no missing — the FindingSeverity enum matches spec §7.4."""
    actual = {s.value for s in FindingSeverity}
    assert actual == EXPECTED_SEVERITY_VALUES
