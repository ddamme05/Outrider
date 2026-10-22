# See specs/2026-05-23-trace-node.md (Q3 resolution).
"""Head-side content fetched after a trace candidate resolves.

Trace emits this when the target is outside pr_context.changed_files so
post-trace analysis can examine it. V1 uses API probes then a content fetch;
filesystem-aware resolution remains a separate future path per
`DECISIONS.md#024` point 4.

The state reducer deduplicates by path, keeping the first source_finding_id.
TraceDecisionEvent and state.trace_decisions retain every cause; query them
by target_file for complete provenance. This carrier has no changed-file
status or diff because the fetched target may be unchanged."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID  # noqa: TC003 — Pydantic field type, needs runtime import

from pydantic import BaseModel, ConfigDict, Field, field_validator

from outrider.coordinates import validate_diff_path


class TraceFetchedFile(BaseModel):
    """Head-side file content fetched by the trace node.

    Frozen + `extra="forbid"` per the cross-boundary schema discipline
    in `docs/conventions.md`. `path` is the dedup key on
    `ReviewState.trace_fetched_files`; the reducer is first-write-wins
    under `append_with_dedup_by(lambda f: f.path)`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Annotated[str, Field(max_length=1024)]
    """Post-`validate_diff_path` repo-relative POSIX form. Field validator
    re-runs `validate_diff_path` so the schema layer enforces the
    canonical-record discipline (same pattern as
    `ReviewFinding._enforce_canonical_file_path`)."""

    content_head: Annotated[str, Field(max_length=1_000_000)]
    """Head-side file content from
    `github.fetch.fetch_file_content_at(repo, path, head_sha)`.
    Stable across retries because head_sha is invariant per review.

    Capped at 1,000,000 chars to mirror the producer's byte cap
    (`github.fetch._PER_FILE_CONTENT_CAP_BYTES = 1_000_000`).
    char-count ≤ byte-count for UTF-8, so this is the conservative
    upper bound; the producer already skips files larger than the byte
    cap. The schema-level constraint defends against a buggy callsite
    that bypasses `fetch_file_content_at` and constructs a
    `TraceFetchedFile` from unbounded content, which would otherwise
    inflate checkpoints and the post-trace analyze prompt."""

    source_finding_id: UUID
    """First finding whose trace decision produced this fetch (per the
    `append_with_dedup_by(path)` reducer's first-write-wins semantics).
    For the complete provenance set when multiple findings resolve to
    the same target, query `state.trace_decisions` by `target_file`."""

    @field_validator("path")
    @classmethod
    def _enforce_canonical_path(cls, value: str) -> str:
        """Re-run `validate_diff_path` so the schema layer enforces the
        repo-relative POSIX invariant. Same shape as
        `ReviewFinding._enforce_canonical_file_path` — propagates the
        canonical-record discipline to every cross-boundary model
        that carries a diff-side path."""
        return validate_diff_path(value)
