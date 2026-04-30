"""Eval-harness fixture factories.

Re-exports the public factory surface. Each factory produces a
schema-valid instance of its target type with `is_eval=True` set
(loud-failure pattern: a factory that omits the flag is a bug, caught
by the `is_eval_injection` autouse gate in `tests/eval/conftest.py`).
"""

from tests.eval.fixtures.factories import (
    FindingEventFactory,
    FindingFactory,
    HITLDecisionEventFactory,
    HITLRequestEventFactory,
    ReviewFactory,
    TraceDecisionEventFactory,
)

__all__ = [
    "FindingEventFactory",
    "FindingFactory",
    "HITLDecisionEventFactory",
    "HITLRequestEventFactory",
    "ReviewFactory",
    "TraceDecisionEventFactory",
]
