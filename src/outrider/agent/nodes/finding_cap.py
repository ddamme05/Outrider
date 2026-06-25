# Severity-ordered finding cap (FUP-180): keep findings BEFORE any FindingEvent/
# round side-effect can strand the audit stream, and NEVER drop a HITL-gated
# (CRITICAL/HIGH) finding. Shared by analyze (per-round) + synthesize (per-report).
"""Gated-aware, deterministic finding truncation.

The per-round (`AnalysisRound.findings`) and per-report (`ReviewReport.findings`)
caps are output-boundary truncation policy: when the admitted set exceeds the
bound, drop findings — but never a gated one.

The contract (FUP-180 + its review design call: "never drop gated; a telemetry
counter is not a HITL gate"):

- **Gated findings (CRITICAL/HIGH, `is_hitl_gated_severity`) are never dropped.**
  Only non-gated (MEDIUM/LOW/INFO) are dropped, down to `soft_cap`. If gated
  findings alone exceed `soft_cap`, ALL of them are kept (the result exceeds
  `soft_cap`) and the caller emits a loud anomaly.
- **`hard_cap` fails LOUD, it does not drop gated.** `hard_cap` is aligned to
  `HITL_MAX_GATED_FINDINGS` (the most gated findings the HITL request can carry).
  A review with MORE gated findings than that can't have them all reach HITL, so
  rather than silently dropping a CRITICAL below the approval gate it raises
  `FindingCapOverflowError` BEFORE any FINDING/round/completion side effect — so no
  `FindingEvent` / `AnalysisRound` / `AnalyzeCompletedEvent` rows are stranded. (The
  per-file observable events that fire earlier in the analyze loop —
  `FileExaminationEvent`, `LLMCallEvent`, cache writes, the cost-starvation anomaly —
  are complete records of work that genuinely happened, not strands.) Reachable only
  on adversarial / degenerate input (>`HITL_MAX_GATED_FINDINGS` gated findings).

Selection within each tier is content-deterministic (replay-stable): severity
rank then `content_hash`. The rank derives from the `FindingSeverity` enum
definition order (CRITICAL first … INFO last) — the single source of truth.
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


class FindingCapOverflowError(RuntimeError):
    """Raised when HITL-gated (CRITICAL/HIGH) findings exceed the hard ceiling
    (`MAX_FINDINGS_HARD_CAP`, aligned to `HITL_MAX_GATED_FINDINGS`).

    Gated findings are never silently dropped (FUP-180): a review with more gated
    findings than HITL can carry fails LOUD rather than dropping a CRITICAL below the
    approval gate. Raised by `cap_findings_by_severity` BEFORE any FINDING/round/
    completion side effect (FindingEvents, AnalysisRound, AnalyzeCompletedEvent), so
    none of those are stranded; per-file observable events earlier in the loop are
    complete records, not strands. Reachable only on adversarial / degenerate input."""

    def __init__(self, n_gated: int, hard_cap: int) -> None:
        self.n_gated = n_gated
        self.hard_cap = hard_cap
        super().__init__(
            f"{n_gated} HITL-gated findings exceed the hard ceiling {hard_cap} "
            f"(== HITL_MAX_GATED_FINDINGS). Gated findings are never dropped below HITL; "
            f"a review this large fails loud (likely an adversarial / degenerate input)."
        )


def _sort_key(finding: ReviewFinding) -> tuple[int, str]:
    return (_SEVERITY_RANK[finding.severity], finding.content_hash)


def cap_findings_by_severity(
    findings: Sequence[ReviewFinding], *, soft_cap: int, hard_cap: int
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Return ``(kept, dropped)`` enforcing the gated-aware cap.

    Never drops a HITL-gated (CRITICAL/HIGH) finding: only non-gated findings are
    dropped, down to ``soft_cap``. If gated findings alone exceed ``soft_cap``, ALL
    gated are kept (``len(kept) > soft_cap`` — the caller detects this and emits a
    loud anomaly). If gated findings exceed ``hard_cap`` (aligned to
    ``HITL_MAX_GATED_FINDINGS``) the function raises ``FindingCapOverflowError``
    rather than dropping a gated finding. ``dropped`` therefore only ever contains
    non-gated findings. Both lists are in selection order (severity rank, then
    ``content_hash``)."""
    gated = sorted((f for f in findings if is_hitl_gated_severity(f.severity)), key=_sort_key)
    non_gated = sorted(
        (f for f in findings if not is_hitl_gated_severity(f.severity)), key=_sort_key
    )

    if len(gated) > hard_cap:
        # The runaway ceiling: more gated findings than HITL can carry. Fail loud rather
        # than silently drop a CRITICAL/HIGH below the approval gate (FUP-180).
        raise FindingCapOverflowError(len(gated), hard_cap)

    # Keep ALL gated; fill the remaining soft-cap budget with the top non-gated. When
    # len(gated) >= soft_cap the budget is 0, so all non-gated drop and the kept set is
    # exactly the gated set (which exceeds soft_cap).
    budget = max(0, soft_cap - len(gated))
    return [*gated, *non_gated[:budget]], list(non_gated[budget:])


__all__ = ["FindingCapOverflowError", "cap_findings_by_severity"]
