"""Record operator alerts for review problems and approval timeouts.

AnomalyPersister keeps one alert per review and rule. A new rule needs a
matching database index. Alerts about severity disagreement help protect
severity-set-by-policy and severity-policy-versioned-for-replay.

Sweeps hold their status-update lock when emitting alerts
(sweep-jobs-use-advisory-locks). Graph nodes do not need that sweep lock.
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
