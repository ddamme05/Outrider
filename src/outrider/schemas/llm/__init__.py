"""Models for proposed model output and its validated form.

Providers decode responses into proposal models. Node parsers then apply
policy and coordinate checks before admitting results into review state.
"""

from outrider.schemas.llm.analyze import (
    AnalyzeFindingProposal,
    AnalyzeFindingProposalRaw,
    AnalyzeResponseRaw,
    TraceCandidateProposal,
    TraceCandidateProposalRaw,
)

__all__ = [
    "AnalyzeFindingProposal",
    "AnalyzeFindingProposalRaw",
    "AnalyzeResponseRaw",
    "TraceCandidateProposal",
    "TraceCandidateProposalRaw",
]
