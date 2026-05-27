"""AnomalyRuleName + AnomalySeverity — typed registries for the
canonical `anomalies.(rule_name, severity)` column values.

Both DB columns are `Text` (per the existing schema) to preserve
forward-compat — new rules and severity tiers don't force a
migration. The StrEnums are the Python-side contract: producers
MUST emit values from these sets, and the persister's partial
unique index keys on the canonical string `rule_name` value.

V1 ships one rule. Future detectors extend the enums.
"""

from enum import StrEnum


class AnomalyRuleName(StrEnum):
    """Canonical anomaly rule names per `docs/spec.md` §16.

    `HITL_TIMEOUT` — emitted by `sweep/hitl_expiry.py` when an
    `HITLRequest` expires without a reviewer decision. Severity
    `medium` per canonical record. Idempotent on `(review_id,
    rule_name='hitl_timeout')` via the partial unique index from
    Group 3's HITL migration.
    """

    HITL_TIMEOUT = "hitl_timeout"


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
