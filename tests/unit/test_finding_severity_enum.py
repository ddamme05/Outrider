"""Check severity values against spec §7.4 and the initial policy's vocabulary.

The integration suite checks the stored seed; this file checks the enum."""

from outrider.policy import FindingSeverity

EXPECTED_SEVERITY_VALUES = {"critical", "high", "medium", "low", "info"}


def test_finding_severity_has_exact_5_values() -> None:
    """No extras, no missing — the FindingSeverity enum matches spec §7.4."""
    actual = {s.value for s in FindingSeverity}
    assert actual == EXPECTED_SEVERITY_VALUES
