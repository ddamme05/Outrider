"""Cross-boundary Pydantic models per docs/conventions.md.

Re-exports the public symbols of the schemas package. Each submodule
has its own focused docstring; this file is the namespace.
"""

from outrider.schemas.analysis_round import AnalysisRound
from outrider.schemas.hitl import (
    HITLDecision,
    HITLRequest,
    PerFindingDecision,
    PerFindingOutcome,
)
from outrider.schemas.pr_context import (
    ChangedFile,
    PRContext,
)
from outrider.schemas.publish import (
    GitHubReviewCreated,
    InlineComment,
    PublishResult,
)
from outrider.schemas.review_finding import (
    PublishDestination,
    ReviewDimension,
    ReviewFinding,
)
from outrider.schemas.review_state import ReviewState
from outrider.schemas.trace_candidate import TraceCandidate
from outrider.schemas.trace_decision import TraceDecision
from outrider.schemas.trace_fetched_file import TraceFetchedFile
from outrider.schemas.triage_result import (
    ReviewTier,
    RiskLevel,
    TriageResult,
)

__all__ = [
    "AnalysisRound",
    "ChangedFile",
    "GitHubReviewCreated",
    "HITLDecision",
    "HITLRequest",
    "InlineComment",
    "PerFindingDecision",
    "PerFindingOutcome",
    "PRContext",
    "PublishDestination",
    "PublishResult",
    "ReviewDimension",
    "ReviewFinding",
    "ReviewState",
    "ReviewTier",
    "RiskLevel",
    "TraceCandidate",
    "TraceDecision",
    "TraceFetchedFile",
    "TriageResult",
]
