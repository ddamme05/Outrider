"""AnomalyRuleName + AnomalySeverity — typed registries for the
canonical `anomalies.(rule_name, severity)` column values.

Both DB columns are `Text` (per the existing schema) to preserve
forward-compat — new rules and severity tiers don't force a
migration. The StrEnums are the Python-side contract: producers
MUST emit values from these sets, and the persister's partial
unique index keys on the canonical string `rule_name` value.

V1 ships two rules: HITL_TIMEOUT (sweep-emitted) and
CROSS_ROUND_SEVERITY_DIVERGENCE (graph-emitted by synthesize).
Future detectors extend the enums.
"""

from enum import StrEnum


class AnomalyRuleName(StrEnum):
    """Canonical anomaly rule names per `docs/spec.md` §16.

    `HITL_TIMEOUT` — emitted by `sweep/hitl_expiry.py` when an
    `HITLRequest` expires without a reviewer decision. Severity
    `medium` per canonical record. Idempotent on `(review_id,
    rule_name='hitl_timeout')` via the partial unique index from
    Group 3's HITL migration.

    `CROSS_ROUND_SEVERITY_DIVERGENCE` — emitted by
    `agent/nodes/synthesize.py` when two findings sharing a
    `content_hash` across different analysis rounds carry different
    `severity` values. Per the `severity-set-by-policy` invariant +
    `compute_finding_content_hash` recipe (keyed over `finding_type`)
    + `ReviewFinding._verify_baseline_severity` (forces severity =
    SEVERITY_POLICY[finding_type]), same content_hash within a single
    review under a single policy_version MUST have identical severity
    by construction — divergence indicates corruption (validator
    bypass, hash-recipe drift, mid-review policy-version change), not
    "model variance." Severity `high` per pre-spec gate #7. Idempotent
    on `(review_id, rule_name='cross_round_severity_divergence')` via
    the partial unique index from the synthesize-node migration.
    """

    HITL_TIMEOUT = "hitl_timeout"
    CROSS_ROUND_SEVERITY_DIVERGENCE = "cross_round_severity_divergence"


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
