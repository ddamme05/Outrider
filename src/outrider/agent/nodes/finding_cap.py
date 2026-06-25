# Severity-ordered finding cap (FUP-180): keep the N highest-severity
# findings BEFORE any FindingEvent/round side-effect can strand the audit
# stream. Shared by analyze (per-round) and synthesize (per-report).
"""Deterministic, severity-ordered finding truncation.

The per-round (`AnalysisRound.findings`) and per-report
(`ReviewReport.findings`) caps are output-boundary truncation policy: when
the admitted set exceeds the bound, keep the highest-severity findings and
drop the rest. The keep is severity-ordered specifically so a drop can only
ever remove a finding LOWER in severity than every kept one — so
`hitl-gates-high-severity` is never weakened by truncation (a CRITICAL/HIGH
is never dropped to keep a lower-severity finding).

This is a different concern from `review_report._SEVERITY_SORT_KEY` (a
presentation sort, deliberately private per its own comment) and
`policy.subsumption._severity_rank` (a collision tiebreak): same severity
ordering, different purpose. The rank here derives from the
`FindingSeverity` enum definition order (CRITICAL first … INFO last) — the
single source of truth — rather than a hardcoded copy, so a future enum
reorder cannot silently desync it.

Selection is content-deterministic (replay-stable): within a severity tier
the tiebreak is `content_hash`, which is unique within a round/report
(enforced by `AnalysisRound._enforce_findings_unique` + synthesize's dedup),
so the kept set is a pure function of the input findings — the same review
replayed drops the same findings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.review_finding import ReviewFinding

# Rank follows the FindingSeverity definition order = severity order
# (CRITICAL=0 … INFO=4). Derived, not hardcoded.
_SEVERITY_RANK: Final[dict[FindingSeverity, int]] = {s: i for i, s in enumerate(FindingSeverity)}


def cap_findings_by_severity(
    findings: Sequence[ReviewFinding], cap: int
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Return ``(kept, dropped)``: the ``cap`` highest-severity findings
    kept, the rest dropped, deterministically.

    ``kept`` has at most ``cap`` entries, and every ``dropped`` finding is
    no higher in severity than every ``kept`` finding (the HITL-safety
    property — a drop can only remove a strictly-lower-or-equal-severity
    finding). Ordering within the returned lists is the selection order
    (severity rank, then ``content_hash``); callers needing a display order
    re-sort. When ``len(findings) <= cap``, ``dropped`` is empty and
    ``kept`` preserves input order (no truncation occurred).
    """
    if len(findings) <= cap:
        return list(findings), []
    ordered = sorted(findings, key=lambda f: (_SEVERITY_RANK[f.severity], f.content_hash))
    return list(ordered[:cap]), list(ordered[cap:])


__all__ = ["cap_findings_by_severity"]
