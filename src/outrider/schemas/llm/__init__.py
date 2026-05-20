# See specs/2026-05-19-analyze-foundation.md §7.
"""LLM-provider-facing schemas for analyze.

Two-layer split per §7:
- Raw layer (`*Raw` suffix): bounded but non-canonical. What the LLM
  provider returns; the sister analyze-implementation spec's parser
  uses these for rejection-event hashing BEFORE admission.
- Admitted layer: post-admission shape with enum-constrained fields
  and post-`coordinates.validate_diff_path` candidate paths.
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
