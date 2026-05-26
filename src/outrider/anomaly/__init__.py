"""Anomaly subsystem — typed rule-name registry, Protocol seam, durable
persister.

The `anomalies` table records cross-system invariant violations the
sweep job (or future background detectors) surface for operator
triage. Per `docs/schema.md`, the column is `rule_name: Text` (not an
enum at the DB layer); the Python-side `AnomalyRuleName` StrEnum
gives type safety + a grep target without forcing a DB migration when
new rules land.

V1 ships one rule: `hitl_timeout` (severity=medium per `docs/spec.md`
§16). Group 8's HITL-expiry sweep emits via `AnomalySink.emit_anomaly`
under the anomaly-first ordering contract (emit anomaly BEFORE
flipping `reviews.status` so a partial-failure scenario is
recoverable).
"""

from outrider.anomaly.persister import (
    AnomalyPersister,
    AnomalyPersisterConfigError,
)
from outrider.anomaly.rule_names import AnomalyRuleName
from outrider.anomaly.sinks import AnomalySink

__all__ = [
    "AnomalyPersister",
    "AnomalyPersisterConfigError",
    "AnomalyRuleName",
    "AnomalySink",
]
