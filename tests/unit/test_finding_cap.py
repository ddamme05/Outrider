# Tests for agent/nodes/finding_cap.py (FUP-180).
"""Pin the gated-aware, severity-ordered finding-cap helper.

The load-bearing property (FUP-180 review design call): a HITL-gated
(CRITICAL/HIGH) finding is NEVER dropped to fit the soft cap — only non-gated
findings drop down to `soft_cap`, and only the `hard_cap` runaway backstop can
drop a gated finding. Selection is content-deterministic (replay-stable).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from outrider.agent.nodes.finding_cap import FindingCapOverflowError, cap_findings_by_severity
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import ReviewFinding

_TYPE_BY_SEVERITY = {
    FindingSeverity.CRITICAL: FindingType.SQL_INJECTION,
    FindingSeverity.HIGH: FindingType.HARDCODED_SECRET,
    FindingSeverity.MEDIUM: FindingType.MISSING_INPUT_VALIDATION,
    FindingSeverity.LOW: FindingType.MISSING_ERROR_HANDLING,
    FindingSeverity.INFO: FindingType.UNUSED_IMPORT,
}


def _finding(severity: FindingSeverity, line: int) -> ReviewFinding:
    """A minimal ReviewFinding of the given severity. `line` varies the span so
    each finding gets a DISTINCT content_hash (the cap's tiebreak)."""
    finding_type = _TYPE_BY_SEVERITY[severity]
    file_path = "src/foo.py"
    ls = line + 1  # line_start is 1-indexed (Field ge=1); keep spans distinct.
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=42,
        finding_type=finding_type,
        severity=severity,
        file_path=file_path,
        line_start=ls,
        line_end=ls,
        title="t",
        description="d",
        evidence="e",
        dimension=lookup_dimension(finding_type),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path, line_start=ls, line_end=ls, finding_type=finding_type
        ),
        proposal_hash="a" * 64,
    )


def test_within_soft_cap_keeps_all() -> None:
    """<= soft_cap: nothing dropped."""
    findings = [_finding(FindingSeverity.MEDIUM, line) for line in range(5)]
    kept, dropped = cap_findings_by_severity(findings, soft_cap=5, hard_cap=100)
    assert dropped == []
    assert {f.content_hash for f in kept} == {f.content_hash for f in findings}


def test_empty_input() -> None:
    kept, dropped = cap_findings_by_severity([], soft_cap=10, hard_cap=100)
    assert kept == []
    assert dropped == []


def test_drops_only_non_gated_to_soft_cap() -> None:
    """Over soft_cap: gated kept, non-gated dropped down to the budget. Keeps the 2
    CRITICAL + the budget (1) of the 3 LOW; drops 2 LOW. No gated dropped."""
    crit = [_finding(FindingSeverity.CRITICAL, line) for line in range(2)]
    low = [_finding(FindingSeverity.LOW, line) for line in range(10, 13)]
    kept, dropped = cap_findings_by_severity([*low, *crit], soft_cap=3, hard_cap=100)
    assert len(kept) == 3
    assert {f.severity for f in kept if f.severity is FindingSeverity.CRITICAL}  # crit kept
    assert sum(1 for f in kept if f.severity is FindingSeverity.CRITICAL) == 2
    assert all(f.severity is FindingSeverity.LOW for f in dropped)
    assert len(dropped) == 2


def test_never_drops_gated_even_over_soft_cap() -> None:
    """The load-bearing case: 4 CRITICAL with soft_cap=2 → ALL 4 kept (kept exceeds
    soft_cap), nothing dropped. A gated finding is never dropped to fit the soft cap."""
    crit = [_finding(FindingSeverity.CRITICAL, line) for line in range(4)]
    kept, dropped = cap_findings_by_severity(crit, soft_cap=2, hard_cap=100)
    assert len(kept) == 4  # exceeds soft_cap — all gated kept
    assert dropped == []


def test_gated_over_cap_drops_all_non_gated() -> None:
    """3 HIGH (gated) + 2 INFO (non-gated), soft_cap=2 → keep all 3 HIGH (exceeds the
    soft cap), drop both INFO. Caller detects len(kept) > soft_cap → loud anomaly."""
    high = [_finding(FindingSeverity.HIGH, line) for line in range(3)]
    info = [_finding(FindingSeverity.INFO, line) for line in range(10, 12)]
    kept, dropped = cap_findings_by_severity([*info, *high], soft_cap=2, hard_cap=100)
    assert len(kept) == 3
    assert all(f.severity is FindingSeverity.HIGH for f in kept)
    assert {f.severity for f in dropped} == {FindingSeverity.INFO}


def test_hard_cap_fails_loud_on_gated_overflow() -> None:
    """The runaway ceiling FAILS LOUD — it never drops a gated finding (FUP-180). More
    than hard_cap gated findings raise FindingCapOverflowError rather than silently
    dropping a CRITICAL below HITL. 4 CRITICAL with hard_cap=3 → raises."""
    crit = [_finding(FindingSeverity.CRITICAL, line) for line in range(4)]
    with pytest.raises(FindingCapOverflowError):
        cap_findings_by_severity(crit, soft_cap=2, hard_cap=3)


def test_gated_at_hard_cap_exactly_is_kept() -> None:
    """At EXACTLY hard_cap gated, all are kept — the raise is on STRICTLY exceeding."""
    crit = [_finding(FindingSeverity.CRITICAL, line) for line in range(3)]
    kept, dropped = cap_findings_by_severity(crit, soft_cap=2, hard_cap=3)
    assert len(kept) == 3
    assert dropped == []


def test_deterministic_correct_subset_under_shuffle() -> None:
    """Selection is content-deterministic AND the CORRECT subset: with 10 same-severity
    findings and soft_cap=4, the kept set is exactly the 4 lowest-content_hash findings,
    regardless of input order."""
    findings = [_finding(FindingSeverity.MEDIUM, line) for line in range(10)]
    expected = {f.content_hash for f in sorted(findings, key=lambda x: x.content_hash)[:4]}
    kept_a, _ = cap_findings_by_severity(findings, soft_cap=4, hard_cap=100)
    kept_b, _ = cap_findings_by_severity(list(reversed(findings)), soft_cap=4, hard_cap=100)
    assert {f.content_hash for f in kept_a} == expected
    assert {f.content_hash for f in kept_b} == expected
