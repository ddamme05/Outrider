# Per specs/2026-06-12-sqli-parameterized-call-veto.md (FUP-162).
"""The parameterized-call detection shape matrix.

Exercises the public scan API only (no raw tree-sitter in tests — the
AST firewall's test rule). Safe means: execute-like method, first
argument a PURE literal string, at least one further argument. The
placeholder STYLE inside the literal is irrelevant by design — the
shape check is what generalizes across `%s`, `%(k)s`, `:name`, `?`,
and `$1` alike.
"""

from __future__ import annotations

from outrider.ast_facts.parameterized_calls import scan_parameterized_calls


def _ranges(sites: tuple) -> list[tuple[int, int]]:  # type: ignore[type-arg]
    return sorted((s.line_start, s.line_end) for s in sites)


def _scan(source: str):  # type: ignore[no-untyped-def]
    return scan_parameterized_calls(source.encode("utf-8"))


# ---------------------------------------------------------------------------
# Safe shapes — literal SQL + separate args.
# ---------------------------------------------------------------------------


def test_indented_percent_s_with_tuple_is_safe() -> None:
    """The spec-pinned shape: single-line, indented, %s + tuple."""
    scan = _scan(
        "class Repo:\n"
        "    def find(self, cursor, q):\n"
        '        cursor.execute("SELECT * FROM t WHERE x = %s", (q,))\n'
    )
    assert _ranges(scan.safe_parameterized_calls) == [(3, 3)]
    assert _ranges(scan.all_execute_like_calls) == [(3, 3)]


def test_named_placeholder_with_dict_is_safe() -> None:
    scan = _scan('cursor.execute("SELECT 1 WHERE k = %(k)s", {"k": v})\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1)]


def test_django_raw_with_list_is_safe() -> None:
    scan = _scan('qs = Model.objects.raw("SELECT * FROM t WHERE id = %s", [pk])\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1)]


def test_holdout_placeholder_styles_are_safe_by_shape() -> None:
    """The styles the analyze prompt never exemplifies (DECISIONS.md#041's
    held-out set) — the shape check covers them without naming them."""
    scan = _scan(
        'conn.execute("SELECT 1 WHERE a = :name", {"name": v})\n'
        'cur.execute("SELECT 1 WHERE b = ?", (v,))\n'
        'conn.execute("SELECT 1 WHERE c = $1", [v])\n'
    )
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1), (2, 2), (3, 3)]


def test_multiline_executemany_is_safe_with_full_range() -> None:
    scan = _scan('cursor.executemany(\n    "INSERT INTO t VALUES (%s, %s)",\n    rows,\n)\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 4)]


def test_implicit_string_concatenation_of_literals_is_safe() -> None:
    scan = _scan('cursor.execute("SELECT * FROM t " "WHERE x = %s", (q,))\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1)]


def test_keyword_params_argument_counts_as_separate_args() -> None:
    scan = _scan('cursor.execute("SELECT 1 WHERE x = %s", params=(q,))\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1)]


def test_bare_function_call_shape_is_recognized() -> None:
    scan = _scan('execute("SELECT 1 WHERE x = %s", (q,))\n')
    assert _ranges(scan.safe_parameterized_calls) == [(1, 1)]


# ---------------------------------------------------------------------------
# Unsafe / excluded shapes — in `all_execute_like_calls` but never safe.
# ---------------------------------------------------------------------------


def test_fstring_sql_is_execute_like_but_never_safe() -> None:
    scan = _scan('cursor.execute(f"SELECT * FROM {table}", (q,))\n')
    assert scan.safe_parameterized_calls == ()
    assert _ranges(scan.all_execute_like_calls) == [(1, 1)]


def test_variable_sql_is_execute_like_but_never_safe() -> None:
    """Construction invisible — the scan cannot prove safety, so the
    model's proposal flows through to HITL."""
    scan = _scan("cursor.execute(query, (q,))\n")
    assert scan.safe_parameterized_calls == ()
    assert _ranges(scan.all_execute_like_calls) == [(1, 1)]


def test_format_and_percent_built_sql_never_safe() -> None:
    scan = _scan(
        'cursor.execute("SELECT {}".format(q), (x,))\n'
        'cursor.execute("SELECT %s" % q, (x,))\n'
        'cursor.execute("SELECT " + q, (x,))\n'
    )
    assert scan.safe_parameterized_calls == ()
    assert len(scan.all_execute_like_calls) == 3


def test_single_argument_execute_is_not_safe() -> None:
    """No separate params argument — nothing proves parameterization."""
    scan = _scan('cursor.execute("SELECT 1")\n')
    assert scan.safe_parameterized_calls == ()
    assert _ranges(scan.all_execute_like_calls) == [(1, 1)]


def test_concatenation_with_variable_part_is_not_safe() -> None:
    scan = _scan('cursor.execute("SELECT " f"{q}", (x,))\n')
    assert scan.safe_parameterized_calls == ()


def test_non_execute_methods_are_ignored_entirely() -> None:
    scan = _scan('log.info("SELECT %s", q)\nrun("cmd", arg)\n')
    assert scan.safe_parameterized_calls == ()
    assert scan.all_execute_like_calls == ()


# ---------------------------------------------------------------------------
# Parse-health guard.
# ---------------------------------------------------------------------------


def test_error_bearing_tree_returns_empty_scan() -> None:
    """ANY syntax error disables the veto: error recovery could misshape
    a call node, and a veto must never rest on an untrustworthy parse."""
    scan = _scan(
        "def broken(:\n"  # header syntax error
        '    cursor.execute("SELECT 1 WHERE x = %s", (q,))\n'
    )
    assert scan.safe_parameterized_calls == ()
    assert scan.all_execute_like_calls == ()


def test_empty_source_returns_empty_scan() -> None:
    scan = _scan("")
    assert scan.safe_parameterized_calls == ()
    assert scan.all_execute_like_calls == ()
