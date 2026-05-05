"""diff_line_to_scope + path validators + file-in-patch helper.

Per docs/spec.md §5.6 (diff_line_to_scope), §10.1 (path validation),
and §4.1.7 (publish routing — the file-in-patch helper backs the
publish-routes-through-coordinates invariant). The ImportPathResolver
Protocol implemented here is canonical at src/outrider/ast_facts/base.py.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from outrider.coordinates.errors import CoordinateError

if TYPE_CHECKING:
    from pathlib import Path

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
    """ImportPathResolver Protocol implementation per ast_facts/base.py.

    Returns repo-relative `Path` candidates validated as: relative-only, no
    `..` traversal, no shell metacharacters, prefix-validated against
    `import_root`, and free of symlink components — final or any ancestor up
    to `import_root`. Candidates that cannot be guaranteed symlink-free are
    omitted from the returned list per the ast_facts spec contract.
    """
    raise NotImplementedError(
        "resolve_candidate_paths lands in a later commit per the implementation sequence"
    )


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
