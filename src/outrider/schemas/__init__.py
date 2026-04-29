"""Cross-boundary Pydantic models per docs/conventions.md.

Re-exports the public symbols of the schemas package. Each submodule
has its own focused docstring; this file is the namespace.
"""

from outrider.schemas.hitl import (
    HITLDecision,
    HITLRequest,
    PerFindingDecision,
    PerFindingOutcome,
)
from outrider.schemas.review_finding import (
    PublishDestination,
    ReviewDimension,
    ReviewFinding,
)

__all__ = [
    "HITLDecision",
    "HITLRequest",
    "PerFindingDecision",
    "PerFindingOutcome",
    "PublishDestination",
    "ReviewDimension",
    "ReviewFinding",
]
