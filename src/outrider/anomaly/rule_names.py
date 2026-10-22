"""Typed rule names and severities for the text-valued anomaly columns.

Enums catch producer typos without a database enum migration. Each new rule
also needs its matching partial unique index in `AnomalyPersister` and a
migration, which enforce one anomaly per review and rule.
The divergence rule flags violations of `severity-set-by-policy`.
"""

from enum import StrEnum


class AnomalyRuleName(StrEnum):
    """The four operator signals per `docs/spec.md` §16.

    - `HITL_TIMEOUT`: an undecided approval request expired (medium).
    - `CROSS_ROUND_SEVERITY_DIVERGENCE`: the same content hash has different
      severity or policy versions across rounds (high). Synthesis stops to
      investigate policy resolution; this is not model variance.
    - `COST_BUDGET_STARVATION`: an analyze pass reaches the budget-skip
      threshold (medium; FUP-044).
    - `GATED_FINDINGS_OVER_CAP`: HIGH/CRITICAL findings exceed a round or
      report's soft cap (high; FUP-180). They are retained for HITL;
      exceeding the hard ceiling raises instead of dropping a gated finding.

    Each rule has a partial unique index for `(review_id, rule_name)`.
    """

    HITL_TIMEOUT = "hitl_timeout"
    CROSS_ROUND_SEVERITY_DIVERGENCE = "cross_round_severity_divergence"
    COST_BUDGET_STARVATION = "cost_budget_starvation"
    GATED_FINDINGS_OVER_CAP = "gated_findings_over_cap"


class AnomalySeverity(StrEnum):
    """Canonical anomaly severity tiers per `docs/spec.md` §9.9.

    Mirrors the operations-dashboard ordering tiers. The DB column
    is `Text` so the enum is the Python-side contract only — bare
    strings would compile but fail mypy at the Protocol surface.
    Distinct from `FindingSeverity` (which carries `INFO` as the
    lowest tier for finding triage); anomalies do not surface
    informational rows.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


__all__ = ["AnomalyRuleName", "AnomalySeverity"]
