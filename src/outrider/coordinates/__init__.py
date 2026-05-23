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

V1 §5.6 public surface — three translation functions + three supporting
surfaces, all inside `coordinates/` per `coordinates-module-is-sole-translator`:

Three translation functions:

- `tree_sitter_to_github(...)` — byte-span → GitHub comment location (§5.6).
  Canonical analyze→publish path; consumers with byte spans use this directly.
- `diff_line_to_scope(...)` — diff line → owning ScopeUnit or None (§5.6).
- `source_line_to_github(...)` — source-line → GitHub comment location.
  Added Wave-3 publish-node fix 2026-05-22 as a line-coord publisher entry
  that bridges `ReviewFinding.line_start` / `line_end` to the byte-based
  canonical translator without inlining line→byte math at the publish-node
  call site (which would violate `coordinates-module-is-sole-translator`).

Three supporting surfaces:

- `validate_diff_path(...)` — diff-side path validator (publisher-facing,
  string-level surface for paths heading to the GitHub comment API per
  docs/spec.md §10.1 / trust-boundary §5 sub-rule 3a).
- `resolve_candidate_paths(...)` — ImportPathResolver Protocol implementation
  per `src/outrider/ast_facts/base.py`; root-aware surface for paths heading
  to filesystem stats per trust-boundary §5 sub-rule 3b.
- `file_in_patch(...)` — coordinates-owned patch-membership helper. **NOT
  called by V1 publish** (publish uses the in-memory `ChangedFile` registry
  short-circuit per `specs/2026-05-21-publish-node.md` FUP-057 resolution);
  remains canonical for non-registry consumers. Also listed under "V1
  supporting helpers" below.

V1 supporting helpers (analyze-foundation §4 + analyze-node spec §7):

- `file_in_patch(...)` — see "Three supporting surfaces" above.
- `lookup_patched_file(...)` — locate a `PatchedFile` by path inside a
  raw unified-diff string; returns None on absence (analyze §7 step 3a).
- `span_within_scope_unit(...)` / `span_within_file(...)` /
  `span_within_degraded_context(...)` — three span-containment checks
  the parser composes for the clean / degraded / file-bound admission
  paths (§4).
- `span_is_nonempty(...)` — predicate enforcing the prompt's stricter
  `byte_start < byte_end` rule (§4; the `Span` carrier admits
  zero-width by design for non-finding consumers).
- `span_to_line_range(...)` — byte Span → 1-indexed `(line_start,
  line_end)` over source text (§4).
- `scope_unit_diff_hunks(...)` — clip a unified-diff PatchedFile to
  hunks inside a ScopeUnit (§4).
- `scope_unit_has_added_lines(...)` / `patched_file_has_added_lines(...)`
  — addable-line predicates that own `unidiff.Line` attribute reads
  (§4 + analyze-node post-fold).
- `extract_scope_unit_body(...)` — UTF-8 byte slice of a ScopeUnit's
  byte range, returned as decoded `str`; owns the slice + decode +
  `errors="replace"` policy (analyze-node post-fold).
- `bound_diff_hunks_text(...)` — concatenate a `PatchedFile`'s lines
  under joint `max_lines` + `max_chars` caps, with a truncation
  sentinel that always fits inside `max_chars` (analyze-node §7).

V1 boundary types + constants:

- `GitHubCommentLocation` — Pydantic model per docs/spec.md §7.2.
- `CoordinateError` — single failure-mode exception (§5.6).
- `COORDINATES_IMPORT_PATH_RESOLVER` — module-level singleton
  `ImportPathResolver` implementation; wired into `build_graph` so
  nodes consume the singleton via closure injection rather than each
  constructing one.
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
    extract_scope_unit_body,
    patched_file_has_added_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
    span_is_nonempty,
    span_to_line_range,
    span_within_degraded_context,
    span_within_file,
    span_within_scope_unit,
)
from outrider.coordinates.translator import (
    GitHubCommentLocation,
    source_line_to_github,
    tree_sitter_to_github,
)

__all__ = [
    "COORDINATES_IMPORT_PATH_RESOLVER",
    "CoordinateError",
    "GitHubCommentLocation",
    "bound_diff_hunks_text",
    "diff_line_to_scope",
    "extract_scope_unit_body",
    "file_in_patch",
    "lookup_patched_file",
    "patched_file_has_added_lines",
    "resolve_candidate_paths",
    "scope_unit_diff_hunks",
    "scope_unit_has_added_lines",
    "source_line_to_github",
    "span_is_nonempty",
    "span_to_line_range",
    "span_within_degraded_context",
    "span_within_file",
    "span_within_scope_unit",
    "tree_sitter_to_github",
    "validate_diff_path",
]
