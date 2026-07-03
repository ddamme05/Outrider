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

V1 §5.6 public surface — five translation functions + five supporting
surfaces, all inside `coordinates/` per `coordinates-module-is-sole-translator`:

Five translation functions:

- `tree_sitter_to_github(...)` — byte-span → GitHub comment location (§5.6).
  Canonical analyze→publish path; consumers with byte spans use this directly.
- `query_span_to_source_lines(...)` — `QueryMatchSpan` byte envelope →
  1-indexed inclusive source line range `(line_start, line_end)`. The
  OBSERVED-tier producer's bridge to `ReviewFinding.line_start`/`line_end`,
  built on the private `_byte_offset_to_line` so byte→line math stays in
  `coordinates/`. Requires a non-empty span (`byte_start < byte_end`);
  `byte_end` is exclusive. See `DECISIONS.md#047` (Cost Lever 3).
- `diff_line_to_scope(...)` — diff line → owning ScopeUnit or None (§5.6).
- `source_line_to_github(...)` — source-line → GitHub comment location.
  Line-coord publisher entry that bridges `ReviewFinding.line_start` /
  `line_end` to the byte-based canonical translator without inlining
  line→byte math at the publish-node call site (which would violate
  `coordinates-module-is-sole-translator`).
- `changed_line_spans(...)` — diff coordinates → per-side changed-line
  numbers + whole-line byte spans (`ScopeChangedLineSpans`), the
  trivial-scope veto's input. Base side is the prompt's kept-removed
  clipping frame via the private `_clip_hunk_lines` core, so the veto
  sees exactly the removed lines the prompt's clipped hunks show.
  See DECISIONS.md#044.

Five supporting surfaces:

- `validate_diff_path(...)` — diff-side path validator (publisher-facing,
  string-level surface for paths heading to the GitHub comment API per
  docs/spec.md §10.1 / trust-boundary §5 sub-rule 3a).
- `resolve_candidate_paths(...)` — ImportPathResolver Protocol implementation
  per `src/outrider/ast_facts/base.py`; root-aware surface for paths heading
  to filesystem stats per trust-boundary §5 sub-rule 3b. Module-form
  (dotted Python import string) only.
- `relative_specifier_candidate_paths(...)` — string-level construction
  surface for JS/TS relative-specifier trace candidates per
  `DECISIONS.md#024` (Amended 2026-07-03): specifier + importing file path
  → contained, `validate_diff_path`-gated pragmatic-six probe fan-out.
  Repo-root escapes return () before anything can be probed.
- `resolve_specifier_candidate_paths(...)` — the relative-specifier
  filesystem twin (root-aware sibling of `resolve_candidate_paths`,
  trust-boundary §5 sub-rule 3b): same construction, then the shared
  symlink-safe walk against `import_root`. The JS/TS adapters'
  `resolve_simple_direct_import` surface.
- `file_in_patch(...)` — coordinates-owned patch-membership helper. **NOT
  called by V1 publish** (publish uses the in-memory `ChangedFile` registry
  short-circuit per the publish-node spec's FUP-057 resolution); remains
  canonical for non-registry consumers. Also listed under "V1 supporting
  helpers" below.

V1 supporting helpers (analyze-foundation §4 + analyze-node spec §7):

- `file_in_patch(...)` — see "Three supporting surfaces" above.
- `lookup_patched_file(...)` — locate a `PatchedFile` by path inside a
  raw unified-diff string; returns None on absence (analyze §7 step 3a).
- `is_valid_import_string(...)` — validate + NFC-normalize a dotted Python
  import string. Shared predicate per `DECISIONS.md#024` point 1: the
  `TraceCandidate.import_string` field validator calls it (raises on invalid);
  `resolve_candidate_paths` calls it (catches + returns []). Single source of
  truth ensures producer-side schema validation and resolver-side input
  validation accept the same set of strings. (A helper, not one of the
  five canonical §5.6 supporting surfaces.)
- `is_valid_relative_specifier(...)` — validate + NFC-normalize a JS/TS
  relative import specifier (shape-only, context-free; containment is
  the construction surfaces' job). The relative-specifier sibling of
  `is_valid_import_string` per `DECISIONS.md#024` (Amended 2026-07-03).
- `is_valid_trace_import_string(...)` — the shared two-form dispatcher
  for `TraceCandidate.import_string`: leading `.` → relative-specifier
  form, otherwise module form. The single dispatch point the schema and
  audit shape-validation sites call so state ↔ audit canonical-bytes
  lockstep holds for both forms.
- `span_within_file(...)` — byte file-bounds check the analyze parser's
  degraded path composes (§4).
- `span_within_degraded_context(...)` — byte-space intersection check: a
  degraded JUDGED finding's span must land within an addable diff hunk. Wired
  into the analyze parser's degraded admission gate (FUP-138) alongside
  `span_within_file` (§4).
- `added_line_byte_ranges(...)` / `added_line_numbers(...)` — deterministic
  producers of a `PatchedFile`'s addable-line byte ranges (+ source) / 1-indexed
  line numbers; the byte ranges feed `span_within_degraded_context` (FUP-138).
- `span_is_nonempty(...)` — degraded-path non-empty floor (`byte_start <
  byte_end`) applied to the byte `Span` from `line_range_to_span` before the
  `span_within_file` check; the `Span` carrier admits zero-width by design for
  non-finding consumers (§4).
- `line_range_within_scope_unit(...)` — line-space containment of a
  1-indexed range in a ScopeUnit; the analyze parser's admission gate for
  line-based proposals (see `specs/2026-06-01-analyze-span-frame-mismatch.md`).
- `line_range_vetoed_by_parameterized_call(...)` — line-space veto decision
  for JUDGED sql_injection proposals landing on a provably-parameterized
  execute call (FUP-162; specs/2026-06-12-sqli-parameterized-call-veto.md).
- `line_range_to_span(...)` — 1-indexed line range → whole-line byte Span;
  raises past EOF.
- `scope_unit_diff_hunks(...)` — clip a unified-diff PatchedFile to
  hunks inside a ScopeUnit (§4).
- `scope_unit_has_added_lines(...)` / `patched_file_has_added_lines(...)` /
  `patched_file_has_removed_lines(...)` — changed-line predicates that own
  `unidiff.Line` attribute reads (§4 + analyze-node post-fold; the removed-line
  sibling is the trivial-scope filter's fail-closed missing-base pre-check,
  DECISIONS.md#044 arc).
- `extract_scope_unit_body(...)` — UTF-8 byte slice of a ScopeUnit's
  byte range, returned as decoded `str`; owns the slice + decode +
  `errors="replace"` policy (analyze-node post-fold).
- `bound_diff_hunks_text(...)` — concatenate a `PatchedFile`'s lines
  under joint `max_lines` + `max_chars` caps, with a truncation
  sentinel that always fits inside `max_chars` (analyze-node §7).

V1 boundary types + constants:

- `GitHubCommentLocation` — Pydantic model per docs/spec.md §7.2.
- `ChangedLineSpan` / `ScopeChangedLineSpans` — coordinates-owned domain
  shape returned by `changed_line_spans` (per-side 1-indexed line numbers +
  whole-line byte `Span`s); raw `unidiff.Line` objects never cross the
  module boundary (DECISIONS.md#044).
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
