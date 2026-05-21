# =============================================================================
# coordinates: SINGLE TRANSLATION POINT — per docs/trust-boundaries.md §3
#
# Translation among diff coordinates, tree-sitter byte spans, GitHub comment
# locations, and source line numbers lives in this module and nowhere else.
# Agent nodes, audit handlers, and github/publisher.py consume the public
# surface below; coordinate math outside coordinates/ is a boundary violation.
# Backs invariant `coordinates-module-is-sole-translator` (docs/spec.md §5.6).
# =============================================================================
"""Coordinates module — translation surface per docs/spec.md §5.6.

V1 public surface:
- `tree_sitter_to_github(...)` — byte-span → GitHub comment location (§5.6).
- `diff_line_to_scope(...)` — diff line → owning ScopeUnit or None (§5.6).
- `resolve_candidate_paths(...)` — ImportPathResolver Protocol implementation
  per `src/outrider/ast_facts/base.py`; trust-boundary #5 places the
  implementation here (docs/trust-boundaries.md §5.3).
- `validate_diff_path(...)` — diff-side path validator (publisher-facing,
  before any reach the GitHub comment API per docs/spec.md §10.1).
- `file_in_patch(...)` — file-membership helper consumed by the publisher
  to distinguish unchanged-region from non-diffed-file routing
  (publish-routes-through-coordinates, docs/spec.md §4.1.7).
- `span_within_scope_unit(...)` — span containment in a ScopeUnit's byte
  range (analyze-foundation §4).
- `span_within_file(...)` — safety-floor file-bounds check (§4). NOT
  sufficient for degraded-mode admission; pair with `span_within_degraded_context`.
- `span_within_degraded_context(...)` — degraded-mode admission gate;
  intersection with addable diff hunks' byte ranges (§4).
- `span_to_line_range(...)` — byte Span → 1-indexed `(line_start, line_end)`
  over source text (§4).
- `scope_unit_diff_hunks(...)` — clip a unified-diff PatchedFile to hunks
  inside a ScopeUnit (§4).

V1 boundary types:
- `GitHubCommentLocation` — Pydantic model per docs/spec.md §7.2.
- `CoordinateError` — single failure-mode exception (§5.6).
"""

from outrider.coordinates.diff_parser import (
    COORDINATES_IMPORT_PATH_RESOLVER,
    diff_line_to_scope,
    file_in_patch,
    lookup_patched_file,
    resolve_candidate_paths,
    validate_diff_path,
)
from outrider.coordinates.errors import CoordinateError
from outrider.coordinates.spans import (
    bound_diff_hunks_text,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
    span_to_line_range,
    span_within_degraded_context,
    span_within_file,
    span_within_scope_unit,
)
from outrider.coordinates.translator import (
    GitHubCommentLocation,
    tree_sitter_to_github,
)

__all__ = [
    "COORDINATES_IMPORT_PATH_RESOLVER",
    "CoordinateError",
    "GitHubCommentLocation",
    "bound_diff_hunks_text",
    "diff_line_to_scope",
    "file_in_patch",
    "lookup_patched_file",
    "resolve_candidate_paths",
    "scope_unit_diff_hunks",
    "scope_unit_has_added_lines",
    "span_to_line_range",
    "span_within_degraded_context",
    "span_within_file",
    "span_within_scope_unit",
    "tree_sitter_to_github",
    "validate_diff_path",
]
