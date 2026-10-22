# Severity-ordered finding cap (FUP-180): keep findings BEFORE any FindingEvent/
# round side-effect can strand the audit stream, and NEVER drop a HITL-gated
# (CRITICAL/HIGH) finding. Shared by analyze (per-round) + synthesize (per-report).
"""Keep gated findings while capping lower-severity output (FUP-180).

Analyze applies the cap per round; synthesis applies it per report. CRITICAL/HIGH
findings are never dropped. They may exceed soft_cap, leaving the caller to emit
an anomaly, but exceeding hard_cap raises before the caller emits that result.
HITL_MAX_GATED_FINDINGS sets the hard ceiling.

Selection sorts by FindingSeverity order, then content_hash for replay stability.
Earlier examination, model-call and cache records still describe completed work."""

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
    """Gated findings exceed the hard ceiling aligned to HITL_MAX_GATED_FINDINGS.

    The caller must fail before emitting the capped round/report, rather than drop
    a CRITICAL/HIGH finding below the approval gate (FUP-180)."""

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
    gated = sorted(
        (finding for finding in findings if is_hitl_gated_severity(finding.severity)), key=_sort_key
    )
    non_gated = sorted(
        (finding for finding in findings if not is_hitl_gated_severity(finding.severity)),
        key=_sort_key,
    )

    if len(gated) > hard_cap:
        # The runaway ceiling: more gated findings than HITL can carry. Fail loud rather
        # than silently drop a CRITICAL/HIGH below the approval gate (FUP-180).
        raise FindingCapOverflowError(len(gated), hard_cap)

    # Fill only the slots left after gated findings. At or above the soft cap,
    # keep the gated set and drop every non-gated finding.
    non_gated_slots = max(0, soft_cap - len(gated))
    return [*gated, *non_gated[:non_gated_slots]], list(non_gated[non_gated_slots:])


__all__ = ["FindingCapOverflowError", "cap_findings_by_severity"]


def admit_with_pair_dedup(
    findings: Sequence[ReviewFinding],
    admitted: list[ReviewFinding],
    keys_seen: set[tuple[str, str]],
) -> None:
    """Append unseen (content_hash, proposal_hash) pairs (FUP-178).

    Sequential analysis and the fan-out fold share this admission rule. Final
    content-hash uniqueness is enforced after their tier-aware merge or collapse."""
    for finding in findings:
        key = (finding.content_hash, finding.proposal_hash)
        if key in keys_seen:
            continue
        keys_seen.add(key)
        admitted.append(finding)


def _collapse_by_content_hash(findings: Sequence[ReviewFinding]) -> list[ReviewFinding]:
    """Keep the first finding per `content_hash`, preserving input order.

    `AnalysisRound._enforce_findings_unique` requires content_hash uniqueness,
    but admission keys on the `(content_hash, proposal_hash)` PAIR, and
    content_hash excludes evidence_tier — so two findings can legitimately reach
    a round sharing a content_hash and would trip that validator. Every plain
    first-wins collapse uses this.

    Not for the prefer-OBSERVED merge in `analyze.py`, which replaces a
    colliding JUDGED incumbent and counts the supersessions rather than dropping
    the newcomer.
    """
    collapsed: list[ReviewFinding] = []
    seen_hashes: set[str] = set()
    for finding in findings:
        if finding.content_hash not in seen_hashes:
            seen_hashes.add(finding.content_hash)
            collapsed.append(finding)
    return collapsed
