"""Limit findings without discarding issues that need human approval.

Analysis applies the limit per round; synthesis applies it to the final report.
Critical, high and verification-held findings can exceed the soft limit. If they
exceed the hard approval limit, raise before recording the retained result.

Selection order stays stable for replay. See FUP-180.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from outrider.policy.publish_eligibility import is_hitl_gated_severity
from outrider.policy.severity import FindingSeverity

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from uuid import UUID

    from outrider.schemas.review_finding import ReviewFinding

# Rank follows the FindingSeverity definition order = severity order
# (CRITICAL=0 … INFO=4). Derived, not hardcoded.
_SEVERITY_RANK: Final[dict[FindingSeverity, int]] = {s: i for i, s in enumerate(FindingSeverity)}


class FindingCapOverflowError(RuntimeError):
    """Approval-required findings exceed the hard HITL capacity.

    The caller must fail before emitting the capped round/report, rather than drop
    a protected finding below the approval gate (FUP-180)."""

    def __init__(self, n_gated: int, hard_cap: int) -> None:
        self.n_gated = n_gated
        self.hard_cap = hard_cap
        super().__init__(
            f"{n_gated} approval-required findings exceed the hard ceiling {hard_cap} "
            f"(== HITL_MAX_GATED_FINDINGS). Protected findings are never dropped below HITL; "
            f"a review this large fails loud (likely an adversarial / degenerate input)."
        )


class FindingEvidenceConflictError(RuntimeError):
    """The same logical finding carries conflicting severity or policy metadata."""


def _sort_key(finding: ReviewFinding) -> tuple[int, str]:
    return (_SEVERITY_RANK[finding.severity], finding.content_hash)


def cap_findings_by_severity(
    findings: Sequence[ReviewFinding],
    *,
    soft_cap: int,
    hard_cap: int,
    protected_finding_ids: Collection[UUID] = (),
) -> tuple[list[ReviewFinding], list[ReviewFinding]]:
    """Keep severity-gated findings and explicit holds before filling other slots.

    Protected findings may exceed the soft cap, but never the hard approval
    capacity. Both partitions retain severity/hash order for stable replay.
    """
    protected_ids = frozenset(protected_finding_ids)
    protected = []
    remaining = []
    for finding in sorted(findings, key=_sort_key):
        if is_hitl_gated_severity(finding.severity) or finding.finding_id in protected_ids:
            protected.append(finding)
        else:
            remaining.append(finding)

    if len(protected) > hard_cap:
        raise FindingCapOverflowError(len(protected), hard_cap)

    remaining_slots = max(0, soft_cap - len(protected))
    return [*protected, *remaining[:remaining_slots]], remaining[remaining_slots:]


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
    """Keep the strongest admitted evidence per hash, with stable ties and group order.

    Confidence is computed from the schema-validated evidence tier, not assigned
    by a model. Choosing a representative does not establish that the claim is true.
    Every member must agree on severity and policy before it can be represented.
    The original chosen object and its provenance remain intact.
    """
    by_hash: dict[str, ReviewFinding] = {}
    for finding in findings:
        incumbent = by_hash.get(finding.content_hash)
        if incumbent is not None:
            if (
                incumbent.severity != finding.severity
                or incumbent.policy_version != finding.policy_version
            ):
                raise FindingEvidenceConflictError(
                    f"conflicting severity or policy for content_hash={finding.content_hash}"
                )
            if finding.confidence <= incumbent.confidence:
                continue
        by_hash[finding.content_hash] = finding
    return list(by_hash.values())
