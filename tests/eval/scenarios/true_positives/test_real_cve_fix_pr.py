"""Placeholder for reviewing a real CVE-fix PR.

This scenario remains skipped until a project, CVE and verified fixture are chosen."""

import pytest

pytestmark = pytest.mark.skip(
    reason="requires eval graph driver + CVE fixture selection (which CVE / "
    "sourcing / ground-truth); analyze node shipped"
)

# Specific CVE + repo pinned at analyze-node spec time; eval ground truth
# defines the canonical finding shape (type, tier, severity, file_path).
EXPECTED_HAS_FINDING = True


def test_real_cve_fix_pr_detected() -> None:
    """Agent identifies the CVE-fixed vulnerability on a real public-project PR."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/real_cve_fix_pr.json")
    assert (len(findings) > 0) is EXPECTED_HAS_FINDING
