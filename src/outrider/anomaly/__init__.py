"""Anomaly subsystem — typed rule-name registry, Protocol seam, durable
persister.

The `anomalies` table records cross-system invariant violations the
sweep job (or future background detectors) surface for operator
triage. Per `docs/schema.md`, the column is `rule_name: Text` (not an
enum at the DB layer); the Python-side `AnomalyRuleName` StrEnum
gives type safety + a grep target without forcing a DB migration when
new rules land.

V1 ships two rules:

- `hitl_timeout` (severity=medium per `docs/spec.md` §16) — sweep-
  emitted from `sweep/hitl_expiry.py` under the anomaly-first ordering
  contract (emit anomaly BEFORE flipping `reviews.status` so a
  partial-failure scenario is recoverable). Sweep callers acquire
  `SWEEP_LOCK_ID` per the `sweep-jobs-use-advisory-locks` invariant.

- `cross_round_severity_divergence` (severity=high) — graph-emitted
  from `agent/nodes/synthesize.py` when same-`content_hash` findings
  carry divergent severity across analysis rounds (corruption per
  `severity-set-by-policy` + `compute_finding_content_hash` recipe +
  `ReviewFinding._verify_baseline_severity`). Graph callers do NOT
  acquire any advisory lock — anomaly emission has no non-idempotent
  external side effect, and the per-rule partial unique index +
  `on_conflict_do_nothing` provides DB-layer idempotency regardless
  of ordering. See `AnomalySink` Protocol docstring for the
  two-caller-class contract.
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
