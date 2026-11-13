"""Describe a request to inspect a file related to a finding.

Each candidate links an import string to its source finding. The trace node
validates and resolves the repository path before fetching it. Stable IDs
support deduplication and replay.

See DECISIONS.md#022, #024 and FUP-209.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from outrider.coordinates import is_valid_trace_import_string
from outrider.policy.canonical import SHA256_HEX_PATTERN, compute_candidate_id


class TraceCandidate(BaseModel):
    """One trace-candidate request from analyze to trace.

    `import_string` carries an absolute or package-relative Python module
    (e.g. `app.auth` or `..auth`) or a JS/TS relative path (e.g. `../auth`).
    Shared coordinate validation canonicalizes identifiers and Unicode and
    rejects unsafe syntax. Admission checks the import form against the source
    language. Probes check repository containment; Python-relative probes also
    require a `.py` source anchor. Raw model bytes stay in the retained exchange.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    source_proposal_hash: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    reason: Annotated[str, Field(max_length=500)]
    import_string: Annotated[str, Field(max_length=1024)]

    @field_validator("import_string")
    @classmethod
    def _enforce_canonical_import_string(cls, value: str) -> str:
        """Reject values that aren't `coordinates.is_valid_trace_import_string` output.

        Canonical bytes must agree across candidate, state and audit validators.
        Context-aware admission checks language. Probe construction checks repository
        containment, and Python-relative probes require a `.py` source anchor.
        Accepted syntax alone does not prove runtime resolution.
        """
        return is_valid_trace_import_string(value)

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
        post-`is_valid_trace_import_string` form.
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
