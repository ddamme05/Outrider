"""True-positive eval scenario: real CVE-fix PR from a public Python project.

Per spec §11.2: agent identifies the CVE-fixed vulnerability with the
correct finding type + tier on a real PR taken from a public project's
git history.

V1: scaffolded; assertions wire up when analyze node lands. The specific
CVE + project pair is pinned when the analyze node spec ships
ground-truth fixtures.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires analyze node + CVE fixture selection")

# Specific CVE + repo pinned at analyze-node spec time; eval ground truth
# defines the canonical finding shape (type, tier, severity, file_path).
EXPECTED_HAS_FINDING = True


def test_real_cve_fix_pr_detected() -> None:
    """Agent identifies the CVE-fixed vulnerability on a real public-project PR."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/real_cve_fix_pr.json")
    assert (len(findings) > 0) is EXPECTED_HAS_FINDING
