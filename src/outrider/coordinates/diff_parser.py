"""diff_line_to_scope + path validators + file-in-patch helper.

Per docs/spec.md §5.6 (diff_line_to_scope), §10.1 (path validation),
and §4.1.7 (publish routing — the file-in-patch helper backs the
publish-routes-through-coordinates invariant). The ImportPathResolver
Protocol implemented here is canonical at src/outrider/ast_facts/base.py.
"""

from __future__ import annotations

import keyword
import re
import unicodedata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Final

from unidiff import PatchSet
from unidiff.errors import UnidiffParseError

from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from outrider.ast_facts.models import ScopeUnit


# Shell metacharacters per `docs/spec.md` §10.1 + `docs/trust-boundaries.md` §5.3.
# Conservative reject set: any character that has special meaning in POSIX shells,
# could be used for command injection if the path ever flows to a subprocess, or
# breaks GitHub API path semantics. Newline / NUL prevent header-injection and
# null-byte attacks; glob characters prevent unintended pattern expansion at
# downstream consumers.
_SHELL_METACHARS_RE: Final = re.compile(r"[;&|`$()<>\n\r\x00*?~\[\]{}'\"]")

# Unicode bidi-override and invisible-format characters per CVE-2021-42574
# ("Trojan Source"). A path containing U+202E (RLO) renders left-to-right
# in editors and audit logs differently from what the bytes say it is —
# e.g., `report‮xls.py` displays as `reportyp.slx`. Operators reading
# audit logs or PR dashboards would see a different filename than the one
# actually fetched from GitHub.
#
# Narrow reject set — only chars that enable the trojan-source disguise
# AND have no legitimate filename use:
#   - U+200B Zero Width Space (invisible, no legitimate filename use)
#   - U+200E, U+200F LTR/RTL Mark (bidi format)
#   - U+202A-U+202E LRE/RLE/PDF/LRO/RLO (bidi format, CVE-2021-42574 core)
#   - U+2066-U+2069 LRI/RLI/FSI/PDI (bidi isolate, CVE-2021-42574 core)
#   - U+FEFF Byte Order Mark / Zero Width No-Break Space (invisible)
#
# Deliberately EXCLUDED from reject (legitimate in real scripts):
#   - U+200C Zero Width Non-Joiner (Persian, Arabic word-joining)
#   - U+200D Zero Width Joiner (Hindi/Devanagari conjuncts, emoji ZWJ
#     sequences, ligature control)
# Including these would block legitimate non-Latin-script filename
# contributors.
_TROJAN_SOURCE_CHARS_RE: Final = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")

# Reject paths whose FIRST path component is `.git` (case-insensitive).
# `.git/config`, `.git/HEAD`, etc. are not legitimate PR-modifiable files
# but the validator otherwise admits them (they're relative, have no `..`,
# no shell metachars). Workflow files under `.github/workflows/` ARE
# legitimate (Outrider doesn't audit `.github/` content but the API call
# is permitted). Narrow the reject to `.git` exactly (not `.gitignore`,
# `.github`, etc.) via component-equality, not prefix-match.
_GIT_INTERNAL_FIRST_COMPONENT: Final = ".git"

# Windows drive-letter prefix (e.g., `C:/`, `D:\\`, even `C:foo` for
# drive-relative). `PurePosixPath("C:/Users/file.py").is_absolute()` returns
# False (POSIX considers absolute = leading `/`), so a drive-prefixed path
# slips through the standard `pp.is_absolute()` check. Reject it explicitly
# so absolute Windows paths can't reach the GitHub comment API surface.
_WINDOWS_DRIVE_PREFIX_RE: Final = re.compile(r"^[A-Za-z]:")


def diff_line_to_scope(
    file_path: str,
    diff_line: int,
    scope_units: list[ScopeUnit],
) -> ScopeUnit | None:
    """Find the scope unit containing a given diff line.

    Returns None for top-level changes (module-level code outside any function),
    for `diff_line` outside any scope's line range, and for scopes belonging to
    a different file. Innermost-scope rule: when nested scopes contain the same
    line, return the deepest match (smallest line span wins).

    Per docs/spec.md §5.6. The six edge cases enumerated in the Month 0 spike
    `spikes/tree_sitter/demos/demo_q6_diff_line_to_scope.py` form the unit-test
    surface; see DECISIONS.md#006-two-month-0-spikes-not-five for the
    "exhaustive unit tests" discipline this implementation honors.

    Raises `CoordinateError` for `diff_line < 1`. Source lines are 1-indexed
    per git-diff convention; a 0 or negative value signals caller
    kind-confusion (e.g., a tree-sitter `Node.start_point[0]` row, which is
    0-indexed, accidentally passed instead of being converted). Surfacing
    the error explicitly keeps the silent-`None` failure mode reserved for
    the legitimate "no enclosing scope" case.

    Multi-file safety: `scope_units` may contain scopes for multiple files;
    only scopes where `ScopeUnit.file_path == file_path` are eligible. A
    `diff_line` that would otherwise map to a scope in a different file
    returns None. Both `file_path` and each `ScopeUnit.file_path` are
    expected in canonical POSIX form (the form `validate_diff_path` returns);
    divergent surface forms compare unequal and silently miss matches.
    """
    if diff_line < 1:
        raise CoordinateError(
            f"diff_line {diff_line} is not a valid 1-indexed source line",
            kind=CoordinateErrorKind.INVALID_DIFF_LINE,
        )
    candidates = [
        unit
        for unit in scope_units
        if unit.file_path == file_path and unit.line_start <= diff_line <= unit.line_end
    ]
    if not candidates:
        return None
    # Innermost = smallest enclosing line span. Matches ast_facts'
    # `_innermost_scope_containing` pattern (byte-span there; line-span here
    # because the input is a diff line, not a byte offset).
    return min(candidates, key=lambda unit: unit.line_end - unit.line_start)


def is_valid_import_string(value: str) -> str:
    """Validate and NFC-normalize a dotted Python import string.

    Raises ValueError on invalid input; returns the NFC-normalized value on
    valid input. Single source of truth shared by `TraceCandidate.import_string`
    field validator (raises directly) AND `resolve_candidate_paths` (catches +
    returns []) per `DECISIONS.md#024` point 1 and `specs/2026-05-23-trace-node.md` M3.

    NFC normalization runs first so the schema-time validator + the resolver +
    the audit-shadow `validate_diff_path` all see the same canonical form.
    Homoglyph identifiers (e.g., Cyrillic `а` U+0430 vs Latin `a` U+0061) pass
    through unchanged — NFC is composition normalization, not transliteration —
    but they pass through CONSISTENTLY, preventing hash divergence between the
    raw input fed to `compute_candidate_id` and the NFC'd path that lands in
    the audit log via `validate_diff_path`.

    Rejection cases (each raises ValueError with a discriminating message):

    - empty input
    - any backslash or forward-slash (Python imports use `.` as separator)
    - any shell metacharacter from the existing `_SHELL_METACHARS_RE` set
    - leading dot, trailing dot, or empty interior part (e.g., `foo..bar`)
    - any part that is not a valid Python identifier (e.g., `123abc`)
    - any part that is a reserved Python keyword (e.g., `class`, `for`)
    """
    if not value:
        raise ValueError("import_string must not be empty")
    normalized = unicodedata.normalize("NFC", value)
    if "\\" in normalized or "/" in normalized:
        raise ValueError("import_string must not contain path separators (use `.`)")
    if _SHELL_METACHARS_RE.search(normalized):
        raise ValueError("import_string contains shell metacharacters")
    parts = normalized.split(".")
    if not all(parts):
        raise ValueError(
            "import_string has empty leading/trailing/interior part "
            "(e.g., '.foo', 'foo.', 'foo..bar')"
        )
    bad_parts = [p for p in parts if not p.isidentifier() or keyword.iskeyword(p)]
    if bad_parts:
        raise ValueError(
            f"import_string parts not valid Python identifiers "
            f"(or are reserved keywords): {bad_parts!r}"
        )
    return normalized


def resolve_candidate_paths(
    import_string: str,
    import_root: Path,
) -> list[Path]:
    """ImportPathResolver Protocol implementation per `src/outrider/ast_facts/base.py`.

    The **root-aware** surface of the two-surface path-validation rule in
    `docs/spec.md` §10.1 / `docs/trust-boundaries.md` §5.3 (the other being
    `validate_diff_path` for the GitHub API path).

    Translates a dotted Python import string (e.g., `"foo.bar"`) into the
    two repo-relative candidate paths Python's import machinery would
    consider: the module file (`foo/bar.py`) and the package init
    (`foo/bar/__init__.py`). Each candidate is validated against the full
    Protocol contract before being returned:

    - relative-only, no `..` traversal in any component
    - no shell metacharacters or backslashes / forward-slashes in the
      import string itself
    - prefix-validated against `import_root` (after resolving symlinks,
      the candidate must still lie under `import_root`)
    - no path component is a symlink — final, any ancestor, AND
      `import_root` itself. Per the Protocol's "any ancestor up to
      `import_root`" rule (read as inclusive), a symlinked root is
      rejected up-front because `python_adapter.resolve_simple_direct_import`
      calls `is_file(follow_symlinks=False)` which only guards the FINAL
      component — ancestor symlinks (including root) would otherwise be
      followed at stat time.

    Candidates that cannot be guaranteed symlink-free, fail prefix-validation,
    or hit any filesystem error during the safety walk are omitted from the
    returned list per the ast_facts spec contract — `ast_facts/` treats
    omitted paths as "did not exist." Returns an empty list for malformed
    import strings (empty, leading/trailing dot, empty interior part, any
    rejected character, or any part that is not a valid Python identifier —
    e.g., numeric prefix `123abc`, a Python keyword like `class`).

    Input-string validation delegates to `is_valid_import_string` (shared with
    `TraceCandidate.import_string` field validator); validation failures map
    to the empty-list return per this function's existing "treats as does-
    not-exist" contract.
    """
    try:
        normalized = is_valid_import_string(import_string)
    except ValueError:
        return []

    parts = normalized.split(".")

    # Two candidates: foo/bar.py and foo/bar/__init__.py
    base = PurePosixPath(*parts)
    module_relative = Path(base.with_suffix(".py"))
    package_relative = Path(base / "__init__.py")

    try:
        if import_root.is_symlink():
            return []
        root_resolved = import_root.resolve(strict=False)
    except (OSError, RuntimeError):
        return []

    safe_candidates: list[Path] = []
    for candidate in (module_relative, package_relative):
        # Defensive — construction guarantees these, but check anyway.
        if candidate.is_absolute() or ".." in candidate.parts:
            continue

        absolute = import_root / candidate
        try:
            resolved = absolute.resolve(strict=False)
        except (OSError, RuntimeError):
            continue

        # Prefix-validation: resolved path must lie under import_root.
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            continue

        # Symlink-component check: walk from the candidate's full path up to
        # (but not including) `import_root`, checking is_symlink() at each
        # level. Per the ast_facts spec contract, ANY symlink component
        # disqualifies the candidate.
        if _has_symlink_component(absolute, import_root):
            continue

        safe_candidates.append(candidate)

    return safe_candidates


def _has_symlink_component(absolute: Path, root: Path) -> bool:
    """True if `absolute` or any ancestor up to (but not including) `root` is
    a symlink. Returns True on any filesystem error (treats unstat-able
    components as unsafe). Returns True if the walk reaches the filesystem
    root without ever hitting `root` (i.e., `absolute` is not actually under
    `root`).
    """
    current = absolute
    while current != root:
        try:
            if current.is_symlink():
                return True
        except OSError:
            return True
        if current.parent == current:
            # Walked off the top of the filesystem without hitting `root`.
            return True
        current = current.parent
    return False


def validate_diff_path(file_path: str) -> str:
    """Diff-side path validation surface — publisher-facing.

    The **string-level** surface of the two-surface path-validation rule
    in docs/spec.md §10.1 / docs/trust-boundaries.md §5.3 (the other being
    `resolve_candidate_paths` for filesystem use). Backs the
    `paths-validated-before-use` invariant (security-critical).

    Rejects, with `CoordinateError`:
    - empty strings
    - absolute paths (`is_absolute()` on a `PurePosixPath`)
    - Windows drive-letter prefixes (`C:/`, `C:\\`, `C:foo`) — `PurePosixPath`
      treats these as relative, so they need a separate rejection
    - `..` traversal in any path component
    - backslash characters (Windows separators; GitHub paths are POSIX)
    - shell metacharacters (`;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `<`, `>`,
      `\\n`, `\\r`, NUL, `*`, `?`, `~`, `[`, `]`, `{`, `}`, `'`, `"`)

    Returns the validated path in repo-relative POSIX form, normalized
    to Unicode NFC. The NFC normalization is the load-bearing step for
    identity-hash stability per spec §1: two filesystems / two clients
    that submit the same logical path under different normalization
    forms (NFC `café` vs NFD `cafe + combining-acute`) would otherwise
    produce different byte sequences and different content-derived
    hashes (round_id, candidate_id), defeating the dedup-by-key
    reducer's idempotency promise on replay.

    No `.resolve()` and no prefix-validation here — those apply to the
    root-aware surface (`resolve_candidate_paths`), per the amended
    canonical's two-surface split. The GitHub comment API consumes string
    paths, and there is no host filesystem to resolve against in this
    surface.
    """
    if not file_path:
        raise CoordinateError(
            "file_path is empty",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    # NFC normalization FIRST, before any other check, so all downstream
    # validators (Trojan-Source, metachars, traversal) see the same byte
    # sequence the hash recipes will see. spec §1 promised NFC; implementation initially
    # omitted it. Pure-ASCII paths are NFC-idempotent so the change
    # affects only multibyte paths — which is exactly the surface where
    # the hash drift would manifest.
    file_path = unicodedata.normalize("NFC", file_path)
    if "\\" in file_path:
        raise CoordinateError(
            f"file_path {file_path!r} contains a backslash (POSIX separators only)",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _SHELL_METACHARS_RE.search(file_path):
        raise CoordinateError(
            f"file_path {file_path!r} contains shell metacharacters",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _TROJAN_SOURCE_CHARS_RE.search(file_path):
        # Per CVE-2021-42574. Path bytes that render as a different
        # filename in audit logs / dashboards break the audit story
        # (operators see a different path than the one fetched).
        raise CoordinateError(
            f"file_path {file_path!r} contains Unicode bidi-override or "
            "zero-width characters (CVE-2021-42574 / trojan source)",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if _WINDOWS_DRIVE_PREFIX_RE.match(file_path):
        raise CoordinateError(
            f"file_path {file_path!r} has a Windows drive-letter prefix; "
            "must be repo-relative POSIX",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    pp = PurePosixPath(file_path)
    if pp.is_absolute():
        raise CoordinateError(
            f"file_path {file_path!r} is absolute; must be repo-relative",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if ".." in pp.parts:
        raise CoordinateError(
            f"file_path {file_path!r} contains '..' traversal",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    if pp.parts and pp.parts[0].lower() == _GIT_INTERNAL_FIRST_COMPONENT:
        # `.git/HEAD`, `.git/config`, etc. — not legitimate PR targets.
        # Component-equality (not prefix-match) so `.github/`, `.gitignore`,
        # and `.gitkeep` are unaffected.
        raise CoordinateError(
            f"file_path {file_path!r} targets the `.git` internal directory; "
            "not a legitimate PR-modifiable path",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    return pp.as_posix()


def _wrap_github_hunks_with_headers(patch: str, file_path: str) -> str:
    """Synthesize ``--- a/X`` / ``+++ b/X`` file headers around a hunks-only
    patch so `unidiff.PatchSet` can parse it.

    Wire-format reality: GitHub's ``/pulls/{number}/files`` API returns
    each file's ``patch`` field as hunks only — no preceding ``--- a/...``
    / ``+++ b/...`` headers, no ``diff --git`` line. `unidiff` requires
    those headers to attach hunks to a file; without them, the first
    ``@@`` line raises ``UnidiffParseError("Unexpected hunk found")``.
    This helper detects the hunks-only shape and synthesizes a minimal
    header pair using ``file_path`` for both source and target sides.

    **Detector policy.** Strictly narrow: the input is treated as
    hunks-only ONLY when its first non-blank line begins with ``@@``.
    Anything else (``---`` / ``+++`` already present, ``diff --git ...``
    prefix, ``Index: ...`` extended header, leading metadata lines) is
    passed through unchanged — `unidiff` knows how to parse those.
    A too-narrow detector (e.g. only checking ``startswith("--- ")``)
    can wrap an already-unified diff and create a malformed hybrid.

    **What the wrap loses.** Using the same ``file_path`` on both header
    lines is enough for the three current consumers (membership query in
    ``file_in_patch``, path-keyed lookup in ``lookup_patched_file``, and
    publisher-facing translation in ``_find_patched_file`` — the
    wire-format normalization extension that lets ``tree_sitter_to_github``
    accept GitHub's ``/pulls/{n}/files`` hunks-only shape) but deliberately
    discards:

    - The rename source path. ``PatchedFile.is_rename`` will report
      ``False`` even for actually-renamed files.
    - The ``is_added_file`` / ``is_removed_file`` semantics derived
      from ``/dev/null`` source/target sides.

    The three callers here don't consult those flags. If a future caller
    needs rename-aware parsing, this helper must NOT be used — that
    caller should accept ``(patch, current_path, previous_path)`` and
    construct the headers accordingly, OR the producer side (intake +
    ``ChangedFile`` schema) should normalize to full unified-diff form.
    Both options are deferred to the rules-layer pass; for now this
    helper stays module-private to discourage misuse.
    """
    # First non-blank line: strip leading whitespace/newlines AND the
    # UTF-8 BOM (U+FEFF) — `str.lstrip()` does not strip BOM, so a
    # patch beginning `﻿@@ ...` would otherwise survive the
    # detector unchanged, get passed unwrapped to `unidiff.PatchSet`,
    # and silently produce an empty PatchSet (no parse error). That
    # collapses to `lookup_patched_file → None → NO_REVIEWABLE_CONTEXT`
    # in analyze, i.e. a silent review-tier downgrade for any file
    # whose patch text echoes a BOM (a file authored with BOM that
    # GitHub forwards in the diff payload). Strip the BOM ONCE (the
    # spec allows at most one) so it never reaches unidiff.
    leading_stripped = patch.lstrip().removeprefix("﻿").lstrip()
    if leading_stripped.startswith("@@"):
        # Synthesize headers around the BOM-stripped body so `unidiff`
        # sees neither BOM nor leading whitespace.
        return f"--- a/{file_path}\n+++ b/{file_path}\n{leading_stripped}"
    return patch


def file_in_patch(file_path: str, patch: str) -> bool:
    """True if `file_path` matches any normalized `unidiff.PatchedFile.path` in `patch`.

    `PatchedFile.path` is the operation-dependent canonical path:
    additions, modifications, and renames return the **target** (head-side)
    path; deletions return the **source** path because the target is
    `/dev/null`. Both sides of the comparison are normalized: `file_path`
    runs through `validate_diff_path` (canonicalizing `./foo.py` → `foo.py`,
    `a//b.py` → `a/b.py`), and `unidiff.PatchedFile.path` runs through
    `PurePosixPath(...).as_posix()`. The rename commitment ("match
    `to_file` only, not `from_file`") follows from the same rule because
    `PatchedFile.path` returns `to_file` for renames.

    Boolean-helper policy: returns `False` for empty patches (`patch == ""`),
    for paths absent from the diff, AND for paths that fail
    `validate_diff_path` (caller passed `..`, an absolute path, shell
    metachars, etc.). The publisher's "in-patch vs not-in-patch" distinction
    routes a malformed caller path the same way it routes an absent file —
    to the `non_diffed_file` / `DASHBOARD_ONLY` tier — without coupling
    `file_in_patch` to a security gate it isn't responsible for.
    `tree_sitter_to_github` is the path-shape-validation gate (raises);
    `file_in_patch` is the membership query (returns bool).

    Raises `CoordinateError` on malformed patch input (any underlying
    `unidiff` parse exception is wrapped, never leaked) and on patches
    containing duplicate file entries (webhook-attacker input per trust
    boundary #5; a duplicate is ambiguous routing input that deterministic
    systems reject).

    Backs the `publish-routes-through-coordinates` invariant: the
    publisher uses this to distinguish `unchanged_region` (in-patch) from
    `non_diffed_file` (absent) routing reasons WITHOUT inlining patch-
    membership math, which would violate trust boundary #3.
    """
    if not patch:
        return False
    try:
        normalized_file_path = validate_diff_path(file_path)
    except CoordinateError:
        # Boolean-helper policy: malformed caller path → not-in-patch.
        # `tree_sitter_to_github` is the surface that raises on bad shape;
        # routing-membership queries return False.
        return False
    try:
        patchset = PatchSet(_wrap_github_hunks_with_headers(patch, normalized_file_path))
    except UnidiffParseError as e:
        raise CoordinateError(
            f"malformed patch input: {e}",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        ) from e

    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == normalized_file_path]
    if len(matches) > 1:
        raise CoordinateError(
            f"patch contains {len(matches)} duplicate entries for {normalized_file_path!r}",
            kind=CoordinateErrorKind.DUPLICATE_FILE_ENTRY,
        )
    return bool(matches)


class _CoordinatesImportPathResolver:
    """`ImportPathResolver` Protocol implementation wrapping the
    module-level `resolve_candidate_paths` function.

    The standalone function is the actual implementation (and the
    public surface readers reach for); this class is the
    Protocol-satisfying bridge so `build_graph`'s `isinstance(...)`
    gate accepts a stateless singleton resolver. Stateless by design —
    one instance per process is sufficient; constructed at module
    import time as `COORDINATES_IMPORT_PATH_RESOLVER`.

    Per `docs/trust-boundaries.md §5.3`: `ast_facts/` consumes
    already-validated paths via this Protocol; the validation
    (relative-only, no `..` traversal, prefix-validation, no symlinks)
    lives in the function this class wraps. Two-surface path-validation
    rule preserved.
    """

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return resolve_candidate_paths(import_string, import_root)


COORDINATES_IMPORT_PATH_RESOLVER: Final = _CoordinatesImportPathResolver()
"""Process-wide singleton `ImportPathResolver` instance. Used by
`lifespan.py` to wire `build_graph(import_path_resolver=...)` against
the canonical coordinates implementation. Stateless; safe to share
across concurrent reviews."""


def lookup_patched_file(patch: str | None, file_path: str) -> PatchedFile | None:
    """Return the `PatchedFile` matching `file_path` in `patch`, or None if absent.

    Mirrors `file_in_patch`'s defensive shape but returns the matched
    `PatchedFile` (or None) rather than a bool. None covers three
    semantically distinct cases the caller may want to distinguish from
    "in-patch": empty/None patch input, path that fails
    `validate_diff_path`, and file absent from a well-formed patch.
    The boolean-helper policy mirrors `file_in_patch` so a malformed
    caller path doesn't surface as a routing exception — the consumer
    interprets None as "no addable-text view of this file."

    Distinguished from `_find_patched_file` (translator.py), which
    raises `CoordinateError` on missing. Used by the analyze node body
    for changed-region intersection, where the absent case is a
    legitimate `skipped+NO_CHANGED_SCOPE_UNITS` outcome rather than an
    error.

    Raises `CoordinateError` only on malformed patch input (any
    underlying `unidiff` parse exception is wrapped) and on patches
    containing duplicate entries for the same normalized path — same
    discipline as `file_in_patch`.
    """
    if not patch:
        return None
    try:
        normalized_file_path = validate_diff_path(file_path)
    except CoordinateError:
        return None
    try:
        patchset = PatchSet(_wrap_github_hunks_with_headers(patch, normalized_file_path))
    except UnidiffParseError as e:
        raise CoordinateError(
            f"malformed patch input: {e}",
            kind=CoordinateErrorKind.MALFORMED_PATCH,
        ) from e
    matches = [pf for pf in patchset if PurePosixPath(pf.path).as_posix() == normalized_file_path]
    if len(matches) > 1:
        raise CoordinateError(
            f"patch contains {len(matches)} duplicate entries for {normalized_file_path!r}",
            kind=CoordinateErrorKind.DUPLICATE_FILE_ENTRY,
        )
    return matches[0] if matches else None
