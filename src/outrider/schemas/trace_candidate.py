# See specs/2026-05-19-analyze-foundation.md §1 and specs/2026-05-23-trace-node.md.
"""`TraceCandidate` — analyze's deterministic request channel for trace.

The analyze node emits `TraceCandidate` instances when a finding's
description references code outside the current file set. The trace
node consumes the accumulated list and decides whether to fetch
candidate files. This module defines the schema only; the producer
(analyze) and consumer (trace) live in the sister analyze-implementation
and trace-node specs.

`candidate_id` is a SHA-256 hex digest used as the LangGraph reducer's
merge key (see `outrider.agent.reducers.append_with_dedup_by` in §2).
Derived via `outrider.policy.canonical.compute_identity_hash` so
re-emission of the same candidate (e.g., from a checkpoint replay)
collapses on merge.

`source_proposal_hash` matches `FindingProposalRejectedEvent.proposal_hash`
(sister spec §5) so candidates and rejection events join in the audit
stream — a JUDGED-tier rejected proposal might still surface a
legitimate cross-file-to-look-at signal. Per `DECISIONS.md#022`
(Accepted 2026-05-20) the underlying `compute_proposal_hash` recipe
is PR/file-scoped, so two analyze passes over different source files
emitting logically-identical proposals produce DISTINCT
`source_proposal_hash` values — preserving the per-source-file causal
edge on the candidate provenance trail. Per `DECISIONS.md#024`
(Accepted 2026-05-24, Amended 2026-05-24 for M8) trace candidates are
dotted Python import strings (V1; no file-path fallback); the trace
node's V1 resolver per M8 uses
`_candidate_paths_for` + `coordinates.validate_diff_path` +
`github.fetch.fetch_file_content_at` two-phase probes (the filesystem-
aware `coordinates.resolve_candidate_paths` is the V1.5+ future
shape). Dedup of actual file fetches is handled by the
`state.trace_fetched_files` reducer's `append_with_dedup_by(path)`.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import is_valid_import_string
from outrider.policy.canonical import SHA256_HEX_PATTERN, compute_candidate_id


class TraceCandidate(BaseModel):
    """One trace-candidate request from analyze to trace.

    `import_string` is a dotted Python import string (e.g. `foo.bar`)
    per `DECISIONS.md#024`. The schema-side field validator runs
    `coordinates.is_valid_import_string` — NFC normalization +
    identifier-validity + part-validation + shell-metachar / separator
    rejection — so the value reaching the trace node is canonical.
    The raw model-proposed form lives on `TraceCandidateProposalRaw`
    (sister analyze-impl spec §7); this admitted form is what reaches
    the trace node.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    source_proposal_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    reason: Annotated[str, Field(max_length=500)]
    import_string: Annotated[str, Field(max_length=1024)]

    @field_validator("import_string")
    @classmethod
    def _enforce_canonical_import_string(cls, value: str) -> str:
        """Reject values that aren't `coordinates.is_valid_import_string` output.

        `candidate_id` is content-derived from the candidate's payload.
        Non-canonical import strings produce non-deterministic IDs
        across producers, defeating replay idempotency of the dedup-by-
        `candidate_id` reducer. Pushing the rule down to the schema
        floor catches drift at every construction site; the shared
        predicate guarantees the schema validator and
        `resolve_candidate_paths` agree on the admit set.
        """
        return is_valid_import_string(value)

    @model_validator(mode="after")
    def _enforce_candidate_id_matches_payload(self) -> Self:
        """Assert `candidate_id == compute_candidate_id(...)` over this
        candidate's payload.

        Mirror of `AnalysisRound._enforce_round_id_matches_payload`.
        Without this, `candidate_id` was only pattern-checked — a
        caller could supply ANY 64-char hex string and Pydantic
        accepted it. The dedup-by-key reducer would then admit two
        logically-equivalent candidates under different `candidate_id`s
        and double-accumulate trace requests on replay.

        Routes through the existing `compute_candidate_id` typed wrapper
        rather than reinventing the recipe — single chokepoint property
        per `outrider.policy.canonical`. Field-validator already
        normalized `import_string`, so the value passed here is the
        post-`is_valid_import_string` form.
        """
        expected = compute_candidate_id(
            source_proposal_hash=self.source_proposal_hash,
            import_string=self.import_string,
            reason=self.reason,
        )
        if self.candidate_id != expected:
            raise ValueError(
                f"TraceCandidate.candidate_id={self.candidate_id!r} does not "
                f"match the canonical id computed from this candidate's "
                f"payload ({expected!r}). Construct via "
                f"`compute_candidate_id(...)` rather than passing an "
                f"arbitrary hex string."
            )
        return self
