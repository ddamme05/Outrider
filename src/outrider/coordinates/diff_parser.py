"""diff_line_to_scope + path validators + file-in-patch helper.

Per docs/spec.md §5.6 (diff_line_to_scope), §10.1 (path validation),
and §4.1.7 (publish routing — the file-in-patch helper backs the
publish-routes-through-coordinates invariant). The ImportPathResolver
Protocol implemented here is canonical at src/outrider/ast_facts/base.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.ast_facts.models import ScopeUnit


def diff_line_to_scope(
    file_path: str,
    diff_line: int,
    scope_units: list[ScopeUnit],
) -> ScopeUnit | None:
    """Find the scope unit containing a given diff line.

    Returns None for top-level changes (module-level code outside any function),
    for diff_line outside the file's line range, and for scopes belonging to a
    different file. Innermost-scope rule: when nested scopes contain the same
    line, return the deepest match.

    Per docs/spec.md §5.6.
    """
    raise NotImplementedError(
        "diff_line_to_scope lands in a later commit per the implementation sequence"
    )


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

    Validates a repository-relative file path before it reaches the GitHub
    comment API. Canonical construction is `pathlib.Path` with `.resolve()`
    and prefix validation per docs/spec.md §10.1 / docs/trust-boundaries.md
    §5.3. Returns the validated repo-relative POSIX form (str).
    """
    raise NotImplementedError(
        "validate_diff_path lands in a later commit per the implementation sequence"
    )


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
