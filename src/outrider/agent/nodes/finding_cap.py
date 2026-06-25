# Severity-ordered finding cap (FUP-180): keep findings BEFORE any FindingEvent/
# round side-effect can strand the audit stream, and NEVER silently drop a
# HITL-gated (CRITICAL/HIGH) finding. Shared by analyze (per-round) + synthesize
# (per-report).
"""Gated-aware, deterministic finding truncation.

The per-round (`AnalysisRound.findings`) and per-report (`ReviewReport.findings`)
caps are output-boundary truncation policy: when the admitted set exceeds the
bound, drop findings — but never a gated one.

The contract has two tiers (FUP-180 review design call: "never drop gated"):

- **Gated findings (CRITICAL/HIGH, `is_hitl_gated_severity`) are never dropped to
  fit the soft cap.** They bypass HITL only by reaching it; silently dropping one
  at the report boundary would weaken `hitl-gates-high-severity`. So the cap drops
  ONLY non-gated (MEDIUM/LOW/INFO) down to `soft_cap`; if gated findings alone
  exceed `soft_cap`, ALL of them are kept (the result exceeds `soft_cap`) and the
  caller emits a loud anomaly.
- **`hard_cap` is the runaway backstop.** Only when gated findings exceed
  `hard_cap` (a truly pathological / adversarial input) are gated findings dropped,
  down to `hard_cap` — the schema `max_length` equals `hard_cap`, so the kept set
  always satisfies it.

Selection within each tier is content-deterministic (replay-stable): severity
rank then `content_hash` (unique within a round/report). The severity rank derives
from the `FindingSeverity` enum definition order (CRITICAL first … INFO last) — the
single source of truth — not a hardcoded copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from outrider.policy.publish_eligibility import is_hitl_gated_severity
from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.review_finding import ReviewFinding

# Rank follows the FindingSeverity definition order = severity order
# (CRITICAL=0 … INFO=4). Derived, not hardcoded.
_SEVERITY_RANK: Final[dict[FindingSeverity, int]] = {s: i for i, s in enumerate(FindingSeverity)}


def _sort_key(finding: ReviewFinding) -> tuple[int, str]:
    return (_SEVERITY_RANK[finding.severity], finding.content_hash)


def cap_findings_by_severity(
    findings: Sequence[ReviewFinding], *, soft_cap: int, hard_cap: int
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Return ``(kept, dropped)`` enforcing the gated-aware cap.

    Never drops a HITL-gated (CRITICAL/HIGH) finding to fit ``soft_cap``: only
    non-gated findings are dropped down to ``soft_cap``. If gated findings alone
    exceed ``soft_cap``, ALL gated are kept (``len(kept) > soft_cap`` — the caller
    detects this and emits a loud anomaly) and every non-gated finding is dropped.
    Only ``hard_cap`` (the runaway backstop, == the schema ``max_length``) can drop
    gated findings, and only when they exceed it.

    Both returned lists are in selection order (severity rank, then
    ``content_hash``). When nothing is dropped, ``dropped`` is empty.
    """
    gated = sorted((f for f in findings if is_hitl_gated_severity(f.severity)), key=_sort_key)
    non_gated = sorted(
        (f for f in findings if not is_hitl_gated_severity(f.severity)), key=_sort_key
    )

    if len(gated) >= hard_cap:
        # Runaway backstop: even gated findings exceed the hard ceiling. Keep the
        # top hard_cap gated; drop the rest of gated AND all non-gated.
        return gated[:hard_cap], [*gated[hard_cap:], *non_gated]

    # Keep ALL gated; fill the remaining soft-cap budget with the top non-gated.
    # When len(gated) >= soft_cap the budget is 0, so all non-gated drop and the
    # kept set is exactly the gated set (which exceeds soft_cap).
    budget = max(0, soft_cap - len(gated))
    return [*gated, *non_gated[:budget]], list(non_gated[budget:])


__all__ = ["cap_findings_by_severity"]
