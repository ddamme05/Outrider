"""AnomalyRuleName + AnomalySeverity — typed registries for the
canonical `anomalies.(rule_name, severity)` column values.

Both DB columns are `Text` (per the existing schema) to preserve
forward-compat — new rules and severity tiers don't force a
migration. The StrEnums are the Python-side contract: producers
MUST emit values from these sets, and the persister's partial
unique index keys on the canonical string `rule_name` value.

V1 ships three rules: HITL_TIMEOUT (sweep-emitted),
CROSS_ROUND_SEVERITY_DIVERGENCE (graph-emitted by synthesize), and
COST_BUDGET_STARVATION (graph-emitted by analyze). Future detectors
extend the enums.
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
    `agent/nodes/synthesize.py::_detect_and_report_divergence` when
    two findings sharing a `content_hash` across analysis rounds
    carry divergent values on EITHER axis: `severity` OR
    `policy_version`. Per `severity-set-by-policy` +
    `compute_finding_content_hash` recipe (keyed over `finding_type`)
    + `ReviewFinding._verify_baseline_severity` (forces severity =
    SEVERITY_POLICY[finding_type]) + `severity-policy-versioned-for-
    replay`, same content_hash within a single review under a single
    policy_version MUST have identical severity by construction —
    divergence on EITHER axis indicates corruption (validator bypass,
    hash-recipe drift, mid-review policy-version change), not "model
    variance." The recovery action is identical regardless of axis:
    stop the review, investigate the upstream policy-resolution
    layer. Same anomaly rule + same partial unique index serve both
    axes. Severity `high` per pre-spec gate #7. Idempotent on
    `(review_id, rule_name='cross_round_severity_divergence')` via
    the partial unique index from the synthesize-node migration. The
    rule's name is kept narrow (`SEVERITY_DIVERGENCE`) because the
    severity axis was the originally specified trigger; the
    policy_version axis was added in the Pass-3 audit fold and is
    structurally co-emitted, not a separate concept.

    `COST_BUDGET_STARVATION` — emitted by `agent/nodes/analyze.py` when an
    analyze pass skips at least `COST_BUDGET_STARVATION_THRESHOLD` files with
    `skip_reason=COST_BUDGET_EXHAUSTED` (FUP-044 extension 3). Makes
    budget-starvation a first-class operator signal instead of something
    inferred by counting individual `FileExaminationEvent` skips. Severity
    `medium`. Idempotent on `(review_id, rule_name='cost_budget_starvation')`
    via the partial unique index from the analyze-cost-fairness migration. The
    reserve (Stage 1) reduces starvation of high-risk files; this anomaly
    surfaces the residual starvation pattern operators care about.

    `GATED_FINDINGS_OVER_CAP` — emitted by `agent/nodes/analyze.py` (per round) and
    `agent/nodes/synthesize.py` (per report) when HITL-gated (CRITICAL/HIGH) findings
    ALONE exceed the soft finding cap (FUP-180). Gated findings are never dropped to
    fit the soft cap, so they are all kept and reach HITL — but a review with >200
    gated findings is a loud capacity signal worth an operator's attention (the
    telemetry counter `n_findings_dropped_over_cap` only records NON-gated drops).
    Severity `high`. Idempotent on `(review_id,
    rule_name='gated_findings_over_cap')` via the partial unique index from the
    finding-cap migration.
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
