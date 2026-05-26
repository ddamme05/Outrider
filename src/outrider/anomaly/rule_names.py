"""AnomalyRuleName — typed registry of canonical anomaly rule names.

The `anomalies.rule_name` DB column is `Text` (per the existing
schema) to preserve forward-compat — new rules don't force a
migration. This StrEnum is the Python-side contract: producers MUST
emit a `rule_name` value from this set, and the persister's partial
unique index keys on the canonical string value.

V1 ships one rule. Future detectors extend the enum.
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


__all__ = ["AnomalyRuleName"]
