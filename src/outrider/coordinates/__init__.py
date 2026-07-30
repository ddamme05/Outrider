# =============================================================================
# coordinates: SINGLE TRANSLATION POINT — per docs/trust-boundaries.md §3
#
# Translation among diff coordinates, tree-sitter byte spans, GitHub comment
# locations, and source line numbers lives in this module and nowhere else.
# Agent nodes, audit handlers, and github/publisher.py consume the public
# surface below; coordinate math outside coordinates/ is a boundary violation.
# Backs invariant `coordinates-module-is-sole-translator` (docs/spec.md §5.6).
# =============================================================================
"""Translation among the four coordinate systems, per docs/spec.md §5.6.

The four are: unified-diff patch coordinates, tree-sitter byte spans and
(row, column) points, GitHub review-comment locations (line + side, 1-indexed),
and source line numbers — separately for base and head. They agree here or
nowhere, which is why `coordinates-module-is-sole-translator` confines the math
to this package.

The public surface, all re-exported below and each documented at its definition:

  - Translators, byte-span or line-range in, GitHub location or scope out.
  - Path validators — a string-level surface for paths bound for the GitHub API,
    and root-aware surfaces for paths bound for filesystem stats. The split is
    trust-boundary §5 sub-rule 3; the string-level one deliberately omits
    `.resolve()` because there is no meaningful root to resolve against.
  - Diff and span predicates the analyze admission gates compose.
  - Boundary types: `GitHubCommentLocation`, `ChangedLineSpan`,
    `ScopeChangedLineSpans`, and `CoordinateError` as the single failure mode.
    Raw `unidiff.Line` objects never cross out of this package; raw tree-sitter
    nodes stay behind the `ast_facts` firewall.
  - `COORDINATES_IMPORT_PATH_RESOLVER`, the singleton `build_graph` injects so
    nodes close over one resolver instead of each constructing their own.

Extending this surface means adding a function here with tests, never inlining a
parallel translator at a call site.
"""

from outrider.coordinates.diff_parser import (
    COORDINATES_IMPORT_PATH_RESOLVER,
    diff_line_to_scope,
    file_in_patch,
    is_relative_specifier_form,
    is_valid_import_string,
    is_valid_relative_specifier,
    is_valid_trace_import_string,
    lookup_patched_file,
    relative_specifier_candidate_paths,
    resolve_candidate_paths,
    resolve_specifier_candidate_paths,
    validate_diff_path,
)
from outrider.coordinates.errors import CoordinateError
from outrider.coordinates.spans import (
    ChangedLineSpan,
    ScopeChangedLineSpans,
    added_line_byte_ranges,
    added_line_numbers,
    bound_diff_hunks_text,
    changed_line_spans,
    extract_scope_unit_body,
    line_range_to_span,
    line_range_vetoed_by_parameterized_call,
    line_range_within_scope_unit,
    patched_file_has_added_lines,
    patched_file_has_removed_lines,
    scope_unit_diff_hunks,
    scope_unit_has_added_lines,
    span_is_nonempty,
    span_within_degraded_context,
    span_within_file,
)
from outrider.coordinates.translator import (
    GitHubCommentLocation,
    query_span_to_source_lines,
    source_line_to_github,
    tree_sitter_to_github,
)

__all__ = [
    "COORDINATES_IMPORT_PATH_RESOLVER",
    "ChangedLineSpan",
    "CoordinateError",
    "GitHubCommentLocation",
    "ScopeChangedLineSpans",
    "added_line_byte_ranges",
    "added_line_numbers",
    "bound_diff_hunks_text",
    "changed_line_spans",
    "diff_line_to_scope",
    "extract_scope_unit_body",
    "file_in_patch",
    "is_relative_specifier_form",
    "is_valid_import_string",
    "is_valid_relative_specifier",
    "is_valid_trace_import_string",
    "line_range_to_span",
    "line_range_vetoed_by_parameterized_call",
    "line_range_within_scope_unit",
    "lookup_patched_file",
    "patched_file_has_added_lines",
    "patched_file_has_removed_lines",
    "query_span_to_source_lines",
    "relative_specifier_candidate_paths",
    "resolve_candidate_paths",
    "resolve_specifier_candidate_paths",
    "scope_unit_diff_hunks",
    "scope_unit_has_added_lines",
    "source_line_to_github",
    "span_is_nonempty",
    "span_within_degraded_context",
    "span_within_file",
    "tree_sitter_to_github",
    "validate_diff_path",
]
