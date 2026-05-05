"""Unit tests for `coordinates.diff_line_to_scope`.

Per docs/spec.md §5.6 — covers the six edge cases enumerated in the Month 0
spike `spikes/tree_sitter/demos/demo_q6_diff_line_to_scope.py` (decorator
line, first/last line of nested function, class-body line between methods,
module-level comment, line outside file range), plus the multi-file
scope-list disambiguation case and the repeated-call idempotence property.

See DECISIONS.md#006-two-month-0-spikes-not-five for the test discipline.
"""

from __future__ import annotations

from typing import Literal

from outrider.ast_facts.models import ScopeUnit
from outrider.coordinates import diff_line_to_scope

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _make_scope(
    *,
    unit_id: str,
    kind: Literal["function", "method", "class"],
    name: str,
    file_path: str,
    line_start: int,
    line_end: int,
    parent_scope_id: str | None = None,
) -> ScopeUnit:
    """Construct a ScopeUnit for line-range testing.

    `byte_start`/`byte_end` are not consulted by `diff_line_to_scope`; set to
    placeholder values that satisfy the model's constraints.
    """
    return ScopeUnit(
        unit_id=unit_id,
        kind=kind,
        name=name,
        qualified_name=name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        byte_start=0,
        byte_end=1,
        parent_scope_id=parent_scope_id,
    )


# ----------------------------------------------------------------------------
# Spike edge case 1: decorator line above a method → method's enclosing scope
# ----------------------------------------------------------------------------


def test_decorator_line_above_method_resolves_to_method() -> None:
    """Per Q2 spike: tree-sitter's `decorated_definition` node spans from the
    decorator through the function body. A diff line on the decorator should
    resolve to the method (the innermost decorated scope), not the class.

    Fixture:
        class MyClass:        # line 1
            @decorator        # line 2 ← diff line
            def my_method(self):  # line 3
                return 1      # line 4
    """
    scope_units = [
        _make_scope(
            unit_id="cls",
            kind="class",
            name="MyClass",
            file_path="x.py",
            line_start=1,
            line_end=4,
        ),
        _make_scope(
            unit_id="m",
            kind="method",
            name="my_method",
            file_path="x.py",
            line_start=2,
            line_end=4,
            parent_scope_id="cls",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "m"


# ----------------------------------------------------------------------------
# Spike edge case 2: first line of function body → that function
# ----------------------------------------------------------------------------


def test_first_line_of_function_body_resolves_to_function() -> None:
    """Diff line on the first body line of a function → that function.

    Fixture:
        def func():        # line 1
            x = 42         # line 2 ← diff line (first body line)
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=2,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "f"


# ----------------------------------------------------------------------------
# Spike edge case 3: last line of nested function → nested function (innermost)
# ----------------------------------------------------------------------------


def test_last_line_of_nested_function_resolves_to_nested_not_outer() -> None:
    """Innermost-scope rule: a line shared by outer + nested scopes resolves
    to the nested (smallest line span wins).

    Fixture:
        def outer_function():            # line 1
            def nested_helper():         # line 2
                return 42                # line 3 ← diff line (last line of nested)
            return nested_helper()       # line 4
    """
    scope_units = [
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer_function",
            file_path="x.py",
            line_start=1,
            line_end=4,
        ),
        _make_scope(
            unit_id="nested",
            kind="function",
            name="nested_helper",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "nested"


# ----------------------------------------------------------------------------
# Spike edge case 4: class-body line between methods → the class
# ----------------------------------------------------------------------------


def test_class_body_line_between_methods_resolves_to_class() -> None:
    """A diff line in class-level code between two method bodies → the class
    scope (only the class contains it; methods don't).

    Fixture:
        class MyClass:           # line 1
            def method_a(self):  # line 2
                return 1         # line 3
                                 # line 4 ← diff line (between methods)
            def method_b(self):  # line 5
                return 2         # line 6
    """
    scope_units = [
        _make_scope(
            unit_id="cls",
            kind="class",
            name="MyClass",
            file_path="x.py",
            line_start=1,
            line_end=6,
        ),
        _make_scope(
            unit_id="ma",
            kind="method",
            name="method_a",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="cls",
        ),
        _make_scope(
            unit_id="mb",
            kind="method",
            name="method_b",
            file_path="x.py",
            line_start=5,
            line_end=6,
            parent_scope_id="cls",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=4, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "cls"


# ----------------------------------------------------------------------------
# Spike edge case 5: module-level comment line → None
# ----------------------------------------------------------------------------


def test_module_level_comment_line_returns_none() -> None:
    """A diff line outside any function/method/class scope → None.

    Fixture:
        # A module-level comment        # line 1 ← diff line
        def func():                     # line 2
            return 1                    # line 3
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=2,
            line_end=3,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=1, scope_units=scope_units)
    assert result is None


# ----------------------------------------------------------------------------
# Spike edge case 6: line outside the file's line range → None (deterministic)
# ----------------------------------------------------------------------------


def test_line_far_past_file_end_returns_none_deterministically() -> None:
    """A diff line outside the file's line range → None (no scope matches);
    must be deterministic, not raise.
    """
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=999, scope_units=scope_units)
    assert result is None


def test_line_zero_returns_none() -> None:
    """diff_line=0 (degenerate, before any 1-indexed scope) → None."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=0, scope_units=scope_units)
    assert result is None


# ----------------------------------------------------------------------------
# Multi-file scope list disambiguation (per spec line 84-85)
# ----------------------------------------------------------------------------


def test_multifile_scope_list_filters_by_file_path() -> None:
    """A scope_units list spanning multiple files: only scopes where
    ScopeUnit.file_path matches the file_path argument are eligible.

    Fixture: two scopes in two files, both covering line 5. file_path="a.py"
    should match a.py's scope, NOT b.py's, even though both contain line 5.
    """
    scope_units = [
        _make_scope(
            unit_id="a_func",
            kind="function",
            name="a_func",
            file_path="a.py",
            line_start=1,
            line_end=10,
        ),
        _make_scope(
            unit_id="b_func",
            kind="function",
            name="b_func",
            file_path="b.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="a.py", diff_line=5, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "a_func"
    assert result.file_path == "a.py"


def test_multifile_scope_list_returns_none_when_only_other_file_has_match() -> None:
    """If the file_path argument has no matching scopes but another file does,
    return None — never accidentally fall through to the other file's scope.
    """
    scope_units = [
        _make_scope(
            unit_id="b_func",
            kind="function",
            name="b_func",
            file_path="b.py",
            line_start=1,
            line_end=10,
        ),
    ]
    result = diff_line_to_scope(file_path="a.py", diff_line=5, scope_units=scope_units)
    assert result is None


# ----------------------------------------------------------------------------
# Idempotence (per spec test scenarios)
# ----------------------------------------------------------------------------


def test_repeated_calls_are_idempotent() -> None:
    """Same inputs → same unit_id returned across calls (or both None)."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
    ]
    a = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    b = diff_line_to_scope(file_path="x.py", diff_line=3, scope_units=scope_units)
    assert a is not None and b is not None
    assert a.unit_id == b.unit_id


def test_repeated_calls_idempotent_for_none_case() -> None:
    """Same inputs that miss → both calls return None."""
    scope_units = [
        _make_scope(
            unit_id="f",
            kind="function",
            name="func",
            file_path="x.py",
            line_start=10,
            line_end=20,
        ),
    ]
    a = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=scope_units)
    b = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=scope_units)
    assert a is None
    assert b is None


# ----------------------------------------------------------------------------
# Empty inputs
# ----------------------------------------------------------------------------


def test_empty_scope_units_returns_none() -> None:
    """Empty `scope_units` list → None deterministically."""
    result = diff_line_to_scope(file_path="x.py", diff_line=5, scope_units=[])
    assert result is None


# ----------------------------------------------------------------------------
# Innermost-scope tiebreaker stress: deeply nested scopes
# ----------------------------------------------------------------------------


def test_three_level_nesting_returns_innermost() -> None:
    """class > method > nested_inner_function: innermost wins.

    Fixture:
        class C:                          # line 1
            def m(self):                  # line 2
                def inner():              # line 3
                    return 1              # line 4 ← diff line
                return inner()            # line 5
    """
    scope_units = [
        _make_scope(
            unit_id="c",
            kind="class",
            name="C",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
        _make_scope(
            unit_id="m",
            kind="method",
            name="m",
            file_path="x.py",
            line_start=2,
            line_end=5,
            parent_scope_id="c",
        ),
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=3,
            line_end=4,
            parent_scope_id="m",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=4, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"


def test_innermost_when_scope_order_in_list_is_outer_first() -> None:
    """Result independent of scope_units list order: outer-first input still
    returns innermost.
    """
    scope_units = [
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"


def test_innermost_when_scope_order_in_list_is_inner_first() -> None:
    """Result independent of scope_units list order: inner-first input also
    returns innermost. Catches an implementation that returns the first match.
    """
    scope_units = [
        _make_scope(
            unit_id="inner",
            kind="function",
            name="inner",
            file_path="x.py",
            line_start=2,
            line_end=3,
            parent_scope_id="outer",
        ),
        _make_scope(
            unit_id="outer",
            kind="function",
            name="outer",
            file_path="x.py",
            line_start=1,
            line_end=5,
        ),
    ]
    result = diff_line_to_scope(file_path="x.py", diff_line=2, scope_units=scope_units)
    assert result is not None
    assert result.unit_id == "inner"
