"""diff_line_to_scope + path validators + file-in-patch helper.

Per docs/spec.md §5.6 (diff_line_to_scope), §10.1 (path validation),
and §4.1.7 (publish routing — the file-in-patch helper backs the
publish-routes-through-coordinates invariant). The ImportPathResolver
Protocol implemented here is canonical at src/outrider/ast_facts/base.py.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from outrider.coordinates.errors import CoordinateError

if TYPE_CHECKING:
    from outrider.ast_facts.models import ScopeUnit


# Shell metacharacters per `docs/spec.md` §10.1 + `docs/trust-boundaries.md` §5.3.
# Conservative reject set: any character that has special meaning in POSIX shells,
# could be used for command injection if the path ever flows to a subprocess, or
# breaks GitHub API path semantics. Newline / NUL prevent header-injection and
# null-byte attacks; glob characters prevent unintended pattern expansion at
# downstream consumers.
_SHELL_METACHARS_RE = re.compile(r"[;&|`$()<>\n\r\x00*?~\[\]{}'\"]")


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

    Multi-file safety: `scope_units` may contain scopes for multiple files;
    only scopes where `ScopeUnit.file_path == file_path` are eligible. A
    `diff_line` that would otherwise map to a scope in a different file
    returns None.
    """
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
    - no path component is a symlink — final or any ancestor up to
      `import_root` (exclusive)

    Candidates that cannot be guaranteed symlink-free, fail prefix-validation,
    or hit any filesystem error during the safety walk are omitted from the
    returned list per the ast_facts spec contract — `ast_facts/` treats
    omitted paths as "did not exist." Returns an empty list for malformed
    import strings (empty, leading/trailing dot, empty interior part,
    explicit `..` part, or any rejected character).
    """
    if not import_string:
        return []
    if "\\" in import_string or "/" in import_string:
        return []
    if _SHELL_METACHARS_RE.search(import_string):
        return []

    parts = import_string.split(".")
    if not all(parts):
        return []
    if any(p == ".." for p in parts):
        return []

    # Two candidates: foo/bar.py and foo/bar/__init__.py
    base = PurePosixPath(*parts)
    module_relative = Path(base.with_suffix(".py"))
    package_relative = Path(base / "__init__.py")

    try:
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
    - `..` traversal in any path component
    - backslash characters (Windows separators; GitHub paths are POSIX)
    - shell metacharacters (`;`, `&`, `|`, `` ` ``, `$`, `(`, `)`, `<`, `>`,
      `\\n`, `\\r`, NUL, `*`, `?`, `~`, `[`, `]`, `{`, `}`, `'`, `"`)

    Returns the validated path in repo-relative POSIX form (str). No
    `.resolve()` and no prefix-validation here — those apply to the
    root-aware surface (`resolve_candidate_paths`), per the amended
    canonical's two-surface split. The GitHub comment API consumes string
    paths, and there is no host filesystem to resolve against in this
    surface.
    """
    if not file_path:
        raise CoordinateError("file_path is empty")
    if "\\" in file_path:
        raise CoordinateError(
            f"file_path {file_path!r} contains a backslash (POSIX separators only)"
        )
    if _SHELL_METACHARS_RE.search(file_path):
        raise CoordinateError(f"file_path {file_path!r} contains shell metacharacters")
    pp = PurePosixPath(file_path)
    if pp.is_absolute():
        raise CoordinateError(f"file_path {file_path!r} is absolute; must be repo-relative")
    if ".." in pp.parts:
        raise CoordinateError(f"file_path {file_path!r} contains '..' traversal")
    return pp.as_posix()


def file_in_patch(file_path: str, patch: str) -> bool:
    """True if `file_path` matches any hunk's normalized target path in `patch`.

    Comparison uses `unidiff.PatchedFile.path` (target path with `a/`/`b/`
    prefix stripped); not raw `+++` header text. For rename hunks
    (`from_file != to_file`), matches the target (head-side) path only.

    Returns False for empty patches (`patch == ""`) and for paths absent
    from the diff. Raises `CoordinateError` on malformed patch input.
    """
    raise NotImplementedError(
        "file_in_patch lands in a later commit per the implementation sequence"
    )
