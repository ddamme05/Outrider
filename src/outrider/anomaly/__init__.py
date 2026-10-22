"""Operator anomalies: rule names, sink contract, and durable persistence.

The four rules cover HITL timeouts, cross-round severity/policy divergence,
cost-budget starvation, and gated findings over the soft cap. Sweeps record
timeouts before changing review status; graph nodes report the other rules.
`AnomalyPersister` deduplicates each `(review_id, rule_name)` through a
per-rule partial unique index. Sweep callers also hold `SWEEP_LOCK_ID` for
their surrounding status changes; graph callers need no sweep lock.
This is `sweep-jobs-use-advisory-locks`. Severity divergence protects
`severity-set-by-policy` and `severity-policy-versioned-for-replay`.

Rule/severity columns are text, with Python enums at the producer boundary.
Every new rule still needs its matching partial-index migration.
"""

from outrider.anomaly.persister import (
    AnomalyPersister,
    AnomalyPersisterConfigError,
)
from outrider.anomaly.rule_names import AnomalyRuleName, AnomalySeverity
from outrider.anomaly.sinks import AnomalySink

__all__ = [
    "AnomalyPersister",
    "AnomalyPersisterConfigError",
    "AnomalyRuleName",
    "AnomalySeverity",
    "AnomalySink",
]
