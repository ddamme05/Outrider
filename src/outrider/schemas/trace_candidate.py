# See specs/2026-05-19-analyze-foundation.md §1.
"""`TraceCandidate` — analyze's deterministic request channel for trace.

The analyze node emits `TraceCandidate` instances when a finding's
description references code outside the current file set. The trace
node consumes the accumulated list and decides whether to fetch
candidate files. This module defines the schema only; the producer
(analyze) and consumer (trace) live in the sister analyze-implementation
spec.

`candidate_id` is a SHA-256 hex digest used as the LangGraph reducer's
merge key (see `outrider.agent.reducers.append_with_dedup_by` in §2).
Derived via `outrider.policy.canonical.compute_identity_hash` so
re-emission of the same candidate (e.g., from a checkpoint replay)
collapses on merge.

`source_proposal_hash` matches `FindingProposalRejectedEvent.proposal_hash`
(sister spec §5) so candidates and rejection events join in the audit
stream — a JUDGED-tier rejected proposal might still surface a
legitimate cross-file-to-look-at signal.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from outrider.policy.canonical import SHA256_HEX_PATTERN


class TraceCandidate(BaseModel):
    """One trace-candidate request from analyze to trace.

    `candidate_path` is post-`coordinates.validate_diff_path` normalized
    (repo-relative POSIX, no `..` traversal, no shell metacharacters) per
    the sister analyze-implementation spec's parser admission. The raw
    model-proposed path lives on `TraceCandidateProposalRaw` (sister
    spec §7); this admitted form is what reaches the trace node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    source_proposal_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    reason: Annotated[str, Field(max_length=500)]
    candidate_path: Annotated[str, Field(max_length=1024)]
