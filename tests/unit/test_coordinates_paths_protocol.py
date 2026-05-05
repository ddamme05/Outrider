"""Unit tests for `coordinates.resolve_candidate_paths` — ImportPathResolver
Protocol implementation per `src/outrider/ast_facts/base.py`.

Per docs/spec.md §10.1 / docs/trust-boundaries.md §5.3 (root-aware surface)
and the ast_facts spec's Protocol contract: relative-only, no `..` traversal,
no shell metacharacters, prefix-validated against `import_root`, and no
symlink components anywhere up to `import_root` (final or ancestor).

See DECISIONS.md#006-two-month-0-spikes-not-five for the test discipline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outrider.coordinates import resolve_candidate_paths

# ----------------------------------------------------------------------------
# Happy path — valid import strings produce both candidates as relative Paths
# ----------------------------------------------------------------------------


def test_simple_import_returns_module_and_package_candidates(tmp_path: Path) -> None:
    """`foo.bar` returns `foo/bar.py` AND `foo/bar/__init__.py` as candidates."""
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert result == [Path("foo/bar.py"), Path("foo/bar/__init__.py")]


def test_single_part_import_returns_top_level_candidates(tmp_path: Path) -> None:
    """`foo` returns `foo.py` AND `foo/__init__.py`."""
    result = resolve_candidate_paths("foo", tmp_path)
    assert result == [Path("foo.py"), Path("foo/__init__.py")]


def test_deeply_nested_import_returns_nested_candidates(tmp_path: Path) -> None:
    """`a.b.c.d` returns `a/b/c/d.py` AND `a/b/c/d/__init__.py`."""
    result = resolve_candidate_paths("a.b.c.d", tmp_path)
    assert result == [Path("a/b/c/d.py"), Path("a/b/c/d/__init__.py")]


def test_returned_paths_are_relative(tmp_path: Path) -> None:
    """Per the Protocol contract, returned Path objects are repo-relative."""
    result = resolve_candidate_paths("foo.bar", tmp_path)
    for candidate in result:
        assert not candidate.is_absolute()


# ----------------------------------------------------------------------------
# Rejections — empty / malformed import strings
# ----------------------------------------------------------------------------


def test_empty_import_returns_empty_list(tmp_path: Path) -> None:
    """`""` → empty list."""
    assert resolve_candidate_paths("", tmp_path) == []


def test_leading_dot_returns_empty_list(tmp_path: Path) -> None:
    """`".foo"` (split has empty first part) → empty list. Catches relative-import
    leakage that should have been filtered upstream by ast_facts."""
    assert resolve_candidate_paths(".foo", tmp_path) == []


def test_trailing_dot_returns_empty_list(tmp_path: Path) -> None:
    """`"foo."` → empty list."""
    assert resolve_candidate_paths("foo.", tmp_path) == []


def test_consecutive_dots_returns_empty_list(tmp_path: Path) -> None:
    """`"foo..bar"` (empty interior part) → empty list."""
    assert resolve_candidate_paths("foo..bar", tmp_path) == []


def test_triple_dots_return_empty_list(tmp_path: Path) -> None:
    """`"foo...bar"` → empty list (multiple consecutive dots produce empty
    interior parts; rejected by the not-all-parts check, since `.split(".")`
    can never produce a literal `".."` part)."""
    assert resolve_candidate_paths("foo...bar", tmp_path) == []


# ----------------------------------------------------------------------------
# Rejections — forbidden characters in the import string
# ----------------------------------------------------------------------------


def test_forward_slash_in_import_returns_empty(tmp_path: Path) -> None:
    """`"foo/bar"` (path-separator-as-import) → empty list. Python imports
    use `.` as the separator; `/` is malformed."""
    assert resolve_candidate_paths("foo/bar", tmp_path) == []


def test_backslash_in_import_returns_empty(tmp_path: Path) -> None:
    """`"foo\\bar"` → empty list (Windows separator forbidden)."""
    assert resolve_candidate_paths("foo\\bar", tmp_path) == []


@pytest.mark.parametrize(
    "metachar",
    [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?", "~", "[", "]", "{", "}", "'", '"'],
)
def test_shell_metacharacter_in_import_returns_empty(metachar: str, tmp_path: Path) -> None:
    """Every shell metacharacter in the import string → empty list."""
    assert resolve_candidate_paths(f"foo{metachar}bar", tmp_path) == []


def test_newline_in_import_returns_empty(tmp_path: Path) -> None:
    """Newline in import string → empty list (header-injection prevention)."""
    assert resolve_candidate_paths("foo\nbar", tmp_path) == []


def test_nul_byte_in_import_returns_empty(tmp_path: Path) -> None:
    """NUL byte in import string → empty list."""
    assert resolve_candidate_paths("foo\x00bar", tmp_path) == []


# ----------------------------------------------------------------------------
# Identifier validation per part — defensive narrowing of LLM-influenced surface
# ----------------------------------------------------------------------------


def test_numeric_prefix_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.123abc"` → empty list. `123abc` is not a valid Python identifier."""
    assert resolve_candidate_paths("foo.123abc", tmp_path) == []


def test_pure_numeric_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.42"` → empty list."""
    assert resolve_candidate_paths("foo.42", tmp_path) == []


def test_keyword_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.class"` → empty list. `class` is a Python keyword, not a module name."""
    assert resolve_candidate_paths("foo.class", tmp_path) == []


def test_dunder_parts_accepted(tmp_path: Path) -> None:
    """`"foo.__init__"` → both candidates returned. Dunders are valid identifiers
    and are common in Python imports (e.g., explicit package init imports)."""
    result = resolve_candidate_paths("foo.__init__", tmp_path)
    assert Path("foo/__init__.py") in result


def test_pycache_part_accepted_but_resolves_normally(tmp_path: Path) -> None:
    """`"foo.__pycache__"` → both candidates returned. The string is a valid
    identifier sequence (`__pycache__` is just an identifier); ast_facts'
    existence check is the gate that prevents `__pycache__` directories from
    being treated as Python modules in practice. The resolver narrows the
    string-validity bar but doesn't impose a Python-import-policy bar."""
    result = resolve_candidate_paths("foo.__pycache__", tmp_path)
    assert Path("foo/__pycache__.py") in result


def test_hyphen_in_part_returns_empty(tmp_path: Path) -> None:
    """`"foo.bar-baz"` → empty list. Hyphens aren't allowed in identifiers."""
    assert resolve_candidate_paths("foo.bar-baz", tmp_path) == []


# ----------------------------------------------------------------------------
# Symlink-component rejection — final component
# ----------------------------------------------------------------------------


def test_final_component_symlink_omits_candidate(tmp_path: Path) -> None:
    """If `foo/bar.py` is a symlink, that candidate is omitted; the
    `__init__.py` candidate (different final component) may still pass.
    """
    foo_dir = tmp_path / "foo"
    foo_dir.mkdir()
    real_file = tmp_path / "real.py"
    real_file.write_text("# real content")
    symlink_target = foo_dir / "bar.py"
    symlink_target.symlink_to(real_file)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    # foo/bar.py is a symlink → omitted
    assert Path("foo/bar.py") not in result
    # foo/bar/__init__.py would require foo/bar/ as a dir; it doesn't exist
    # but isn't a symlink, so it passes the safety check (existence is
    # ast_facts' job).
    assert Path("foo/bar/__init__.py") in result


def test_symlink_pointing_outside_import_root_omitted(tmp_path: Path) -> None:
    """A symlink pointing outside import_root → omitted via the prefix
    validation step (resolved path escapes root).
    """
    # Create a real file outside the import_root.
    outside_root = tmp_path.parent / "outside_repo"
    outside_root.mkdir(exist_ok=True)
    target = outside_root / "secret.py"
    target.write_text("# outside")
    # Inside import_root, place a symlink to that outside file.
    inside = tmp_path / "foo"
    inside.mkdir()
    symlink = inside / "bar.py"
    symlink.symlink_to(target)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    assert Path("foo/bar.py") not in result


# ----------------------------------------------------------------------------
# Symlink-component rejection — ancestor component
# ----------------------------------------------------------------------------


def test_symlinked_import_root_returns_empty(tmp_path: Path) -> None:
    """If `import_root` itself is a symlink, return [] regardless of the
    import string.

    The Protocol contract reads "no path component (final or any ancestor
    up to `import_root`) is a symlink" as INCLUSIVE of root. Without this
    guard, the per-candidate ancestor walk stops one level before root,
    and `python_adapter.resolve_simple_direct_import`'s
    `is_file(follow_symlinks=False)` only protects the FINAL component —
    so a symlinked root would silently be followed at stat time.
    """
    real_root = tmp_path / "real_root"
    real_root.mkdir()
    (real_root / "foo.py").write_text("# real")

    symlinked_root = tmp_path / "symlinked_root"
    symlinked_root.symlink_to(real_root)

    assert resolve_candidate_paths("foo", symlinked_root) == []
    assert resolve_candidate_paths("foo.bar", symlinked_root) == []


def test_ancestor_directory_symlink_omits_candidate(tmp_path: Path) -> None:
    """If `foo/` (a parent component) is a symlink, the `foo/bar.py`
    candidate is omitted — parent-directory-symlink-dodge prevention per
    ast_facts spec line 127.
    """
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    (real_dir / "bar.py").write_text("# inside real_dir")
    # Make foo/ a symlink to real_dir/
    foo_symlink = tmp_path / "foo"
    foo_symlink.symlink_to(real_dir)

    result = resolve_candidate_paths("foo.bar", tmp_path)

    # foo/bar.py would be reachable via the symlink — must be omitted
    # because `foo/` is a symlink even though `foo/bar.py` (its target) is not.
    assert Path("foo/bar.py") not in result
    assert Path("foo/bar/__init__.py") not in result


# ----------------------------------------------------------------------------
# All-clean filesystem — both candidates returned
# ----------------------------------------------------------------------------


def test_no_symlinks_returns_both_candidates(tmp_path: Path) -> None:
    """Plain directory tree with no symlinks → both candidates returned
    (existence-on-disk is not the resolver's concern; ast_facts checks).
    """
    foo = tmp_path / "foo"
    foo.mkdir()
    (foo / "bar.py").write_text("# real")

    result = resolve_candidate_paths("foo.bar", tmp_path)

    assert Path("foo/bar.py") in result
    assert Path("foo/bar/__init__.py") in result


def test_nonexistent_paths_pass_safety_check(tmp_path: Path) -> None:
    """If the candidate path does not exist on disk, the safety check still
    passes (non-existent components are not symlinks). Existence-on-disk
    is ast_facts' job, not the resolver's.
    """
    # tmp_path is empty; no foo/, no bar.py
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert Path("foo/bar.py") in result
    assert Path("foo/bar/__init__.py") in result


# ----------------------------------------------------------------------------
# Conformance: shape matches the ast_facts Protocol-mock harness
# ----------------------------------------------------------------------------


def test_call_signature_matches_protocol(tmp_path: Path) -> None:
    """The function accepts (import_string: str, import_root: Path) — same
    positional shape ast_facts' Protocol mock asserts via
    `assert_called_once_with("foo.bar", tmp_path)` at
    `tests/unit/test_ast_facts_python.py:339-447`.
    """
    # Calling positionally with the exact shape the Protocol mock uses.
    result = resolve_candidate_paths("foo.bar", tmp_path)
    assert isinstance(result, list)
    assert all(isinstance(p, Path) for p in result)


def test_returned_list_can_be_consumed_by_iteration(tmp_path: Path) -> None:
    """Returned list is a real list, iterable in deterministic order, with
    Path elements — matches ast_facts' downstream `for candidate in
    candidates: ...` consumption pattern.
    """
    result = resolve_candidate_paths("foo.bar", tmp_path)
    seen: list[Path] = []
    for candidate in result:
        seen.append(candidate)
    assert seen == result


# ----------------------------------------------------------------------------
# Idempotence
# ----------------------------------------------------------------------------


def test_repeated_calls_are_idempotent(tmp_path: Path) -> None:
    """Same inputs → equal output across repeated calls. No module-level
    state, no caching with mutable backing."""
    a = resolve_candidate_paths("foo.bar", tmp_path)
    b = resolve_candidate_paths("foo.bar", tmp_path)
    assert a == b


# ----------------------------------------------------------------------------
# import_root edge cases
# ----------------------------------------------------------------------------


def test_import_root_with_trailing_separator_works(tmp_path: Path) -> None:
    """`import_root` passed with or without trailing separator yields the
    same result (Path normalizes)."""
    result_a = resolve_candidate_paths("foo", tmp_path)
    result_b = resolve_candidate_paths("foo", Path(str(tmp_path) + "/"))
    assert result_a == result_b
