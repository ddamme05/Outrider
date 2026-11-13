"""Command-execution syntax retains model judgments without proving hostile input.

Drive both the judged and empty-model paths through the scripted graph."""

from outrider.policy import EvidenceTier, FindingSeverity, FindingType, lookup_severity

_COLLISION_FIXTURE = "tests/eval/fixtures/mock_github/dual_mode_command_injection_collision.json"
_OBSERVED_ONLY_FIXTURE = (
    "tests/eval/fixtures/mock_github/dual_mode_command_injection_observed_only.json"
)


def test_command_signal_preserves_model_judged_finding() -> None:
    """The syntax signal cannot replace the contextual model judgment."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_COLLISION_FIXTURE)
    ci = [f for f in findings if f.finding_type == FindingType.COMMAND_INJECTION]
    assert len(ci) == 1, f"dual-mode collision must collapse to one finding, got {len(ci)}"
    finding = ci[0]
    assert finding.evidence_tier == EvidenceTier.JUDGED
    assert finding.query_match_id is None
    assert finding.severity == lookup_severity(FindingType.COMMAND_INJECTION)
    assert finding.severity == FindingSeverity.CRITICAL
    assert finding.line_start == 5 and finding.line_end == 5


def test_empty_model_does_not_turn_command_signal_into_vulnerability() -> None:
    """An execution sink without contextual judgment is not an automatic finding."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_OBSERVED_ONLY_FIXTURE)
    assert not [f for f in findings if f.finding_type == FindingType.COMMAND_INJECTION]
