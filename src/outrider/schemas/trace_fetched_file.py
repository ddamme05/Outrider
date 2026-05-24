# See specs/2026-05-23-trace-node.md (Q3 resolution).
"""`TraceFetchedFile` — head-side file content fetched by the trace node.

After the trace node resolves a candidate's import_string through
`coordinates.resolve_candidate_paths`, it fetches the resolved file's
head content from GitHub (when the path is NOT already in
`pr_context.changed_files`) and emits a `TraceFetchedFile` so analyze's
round-2 pass can examine the content.

Reducer key on `ReviewState.trace_fetched_files` is `path` alone (per
Q3 resolution + M2 audit-fold). First-write-wins on key collision —
when two findings resolve to the same target file, only the first
emission's TraceFetchedFile lands. Per M2: this is intentional, not
a bug. `TraceDecisionEvent` rows preserve full multi-cause provenance
(one event per source finding regardless of resolved-path collision);
recovering "which findings caused this fetch" is
`query state.trace_decisions where target_file == self.path`.

V1 field set per spec Q3 (Codex round-7 revision — no
source_import_string / source_proposal_hash fields):

- `path: str` — post-`validate_diff_path` repo-relative POSIX.
- `content_head: str` — head-side content from
  `github.fetch.fetch_file_content_at(repo, path, head_sha)`.
  Content is invariant per review's head_sha so stable across retries.
- `source_finding_id: UUID` — first finding whose trace decision produced
  this fetch (first-write-wins under the reducer's dedup-by-path).
  For the complete provenance set across all findings citing this path,
  query `state.trace_decisions` by `target_file`.

Why not reuse `ChangedFile`: ChangedFile is intake's post-PR-file shape
(status, additions/deletions, patch, base/head, rename fields). A trace-
fetched file is not necessarily a changed PR file and has no patch
semantics. Reusing ChangedFile would lie to analyze about the file's nature.

No `source_import_string` / `source_proposal_hash` fields per the
Codex round-7 audit-fold: those values vary across retries (LLM
proposes different rankings; multiple candidates may map to the same
target_file via different import strings) and would cause state-vs-
audit divergence. Provenance recovery via `state.trace_decisions`
cross-reference avoids the divergence and keeps the schema minimal.
"""

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

    content_head: str
    """Head-side file content from
    `github.fetch.fetch_file_content_at(repo, path, head_sha)`.
    Stable across retries because head_sha is invariant per review."""

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
