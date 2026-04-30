"""Eval-harness fixture factories.

Re-exports the public factory surface. Each factory produces a
schema-valid instance of its target type with `is_eval=True` set on
every surface that carries the column (loud-failure pattern: a factory
that omits the flag is a bug, caught by the `eval_db` fixture's
teardown integrity gate in `tests/eval/conftest.py`). `ReviewFinding`
is a cross-boundary type with no `is_eval` field; the flag lives on
the corresponding `findings` row, not on the type itself.
"""

from .factories import (
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
