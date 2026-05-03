"""Unit tests for `outrider.ast_facts` — Python adapter.

Coverage tracks the spec's unit test list per
specs/2026-04-30-ast-facts-module.md.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from outrider.ast_facts import parse_python
from outrider.ast_facts.models import (
    AssignmentSite,
    CallSite,
    ImportRef,
    ImportResolution,
    ParseResult,
    QueryCaptureSpan,
    QueryMatchSpan,
    ScopeUnit,
    SkipReason,
    compute_unit_id,
)
from outrider.ast_facts.parser_outcome import (
    EXCLUSION_RULES,
    MAX_PARSE_BYTES,
    should_skip,
)
from outrider.ast_facts.python_adapter import PythonAdapter

# ---------------------------------------------------------------------------
# Domain model construction
# ---------------------------------------------------------------------------


def test_scope_unit_construction_admits_canonical_inputs() -> None:
    su = ScopeUnit(
        unit_id=compute_unit_id("f.py", "function", "process"),
        kind="function",
        name="process",
        qualified_name="process",
        file_path="f.py",
        line_start=1,
        line_end=10,
        byte_start=0,
        byte_end=100,
    )
    assert su.unit_id


def test_scope_unit_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        ScopeUnit(
            unit_id="x",
            kind="function",
            name="p",
            qualified_name="p",
            file_path="f.py",
            line_start=1,
            line_end=1,
            byte_start=0,
            byte_end=1,
            unknown_field="oops",  # type: ignore[call-arg]
        )


def test_scope_unit_no_field_validators_beyond_canonical() -> None:
    """No `byte_end < byte_start` or empty-`qualified_name` validators
    are added — these would be spec-fidelity drift per the V1 spec.
    """
    # Empty qualified_name admits (canonical doesn't reject it)
    su = ScopeUnit(
        unit_id="x",
        kind="function",
        name="",
        qualified_name="",
        file_path="f.py",
        line_start=1,
        line_end=1,
        byte_start=0,
        byte_end=0,
    )
    assert su.qualified_name == ""


# ---------------------------------------------------------------------------
# extract_scopes + qualified_name derivation
# ---------------------------------------------------------------------------


def test_extract_scopes_finds_canonical_constructs(
    canonical_python_source: bytes, canonical_python_path: str
) -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    scopes = adapter.extract_scopes(canonical_python_source, canonical_python_path)
    qualified_names = {s.qualified_name for s in scopes}
    assert "hello" in qualified_names
    assert "Greeter" in qualified_names
    assert "Greeter.greet" in qualified_names
    assert "Greeter.greet_async" in qualified_names
    assert "outer" in qualified_names
    assert "outer.inner" in qualified_names  # nested function, no <locals>
    assert "Outer.Inner" in qualified_names
    assert "Outer.Inner.method" in qualified_names


def test_qualified_name_derivation_specific_cases() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"""
def process():
    pass

class Cls:
    def method(self):
        pass

def outer():
    def inner():
        pass

class Outer:
    class Inner:
        def m(self):
            pass
"""
    scopes = adapter.extract_scopes(src, "f.py")
    by_qname = {s.qualified_name: s for s in scopes}
    assert by_qname["process"].kind == "function"
    assert by_qname["Cls"].kind == "class"
    assert by_qname["Cls.method"].kind == "method"
    assert by_qname["outer"].kind == "function"
    assert by_qname["outer.inner"].kind == "function"  # no <locals>
    assert by_qname["Outer"].kind == "class"
    assert by_qname["Outer.Inner"].kind == "class"
    assert by_qname["Outer.Inner.m"].kind == "method"


def test_lambdas_are_not_extracted_as_scopes() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"f = lambda x: x + 1\ndef wrapper():\n    g = lambda y: y * 2\n    return g\n"
    scopes = adapter.extract_scopes(src, "f.py")
    qnames = {s.qualified_name for s in scopes}
    assert qnames == {"wrapper"}


def test_unit_id_is_byte_stable_across_invocations() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"def process():\n    return 1\n"
    a1 = adapter.extract_scopes(src, "f.py")
    a2 = adapter.extract_scopes(src, "f.py")
    assert tuple(s.unit_id for s in a1) == tuple(s.unit_id for s in a2)


def test_decorated_function_byte_start_includes_decorators() -> None:
    """Decorated function's ScopeUnit byte_start equals decorated_definition's start,
    which precedes the inner function_definition.start_byte (Month 0 spike finding)."""
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"@decorator\ndef func():\n    pass\n"
    scopes = adapter.extract_scopes(src, "f.py")
    [scope] = scopes
    # byte_start should be 0 (start of `@decorator`), not 11 (start of `def`)
    assert scope.byte_start == 0
    # Per scaffold convention, decorators stored without the `@` prefix.
    assert scope.decorators == ("decorator",)


# ---------------------------------------------------------------------------
# extract_imports
# ---------------------------------------------------------------------------


def test_import_parsing_classifies_four_shapes() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"""
import os
from pathlib import Path
from .helpers import thing
from collections import *
"""
    imports = adapter.extract_imports(src, "f.py")
    by_kind = {i.import_kind: i for i in imports}
    assert by_kind["direct"].module == "os"
    assert by_kind["direct"].is_simple_direct is False
    assert by_kind["from"].module == "pathlib"
    assert by_kind["from"].names == ("Path",)
    assert by_kind["from"].is_simple_direct is True
    assert by_kind["relative"].module.startswith(".")
    assert by_kind["relative"].is_simple_direct is False
    assert by_kind["star"].module == "collections"
    assert by_kind["star"].is_simple_direct is False


# ---------------------------------------------------------------------------
# extract_call_sites
# ---------------------------------------------------------------------------


def test_call_sites_extracted_only_inside_scopes() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"""
import os

os.path.join("a", "b")  # module-level: should NOT produce a CallSite

def hello():
    return os.path.join("c", "d")  # inside scope: should produce
"""
    scopes = adapter.extract_scopes(src, "f.py")
    calls = adapter.extract_call_sites(src, "f.py", scopes)
    # Module-level call at line 4 must be skipped.
    assert all(c.line != 4 for c in calls)
    # The in-scope call should be present. `callee_name` is RAW SOURCE TEXT
    # per canonical spec.md §5.4 ("raw text; resolution is a separate
    # concern") — for `os.path.join(...)` the field is `"os.path.join"`,
    # not the final segment. Trace-node normalization is the consumer's
    # job, not this adapter's.
    assert any(c.callee_name == "os.path.join" and c.line == 7 for c in calls)


def test_call_form_decorator_produces_call_site() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b'@route("/api/foo")\ndef func():\n    pass\n'
    scopes = adapter.extract_scopes(src, "f.py")
    calls = adapter.extract_call_sites(src, "f.py", scopes)
    callees = {c.callee_name for c in calls}
    assert "route" in callees


def test_bare_name_decorator_produces_no_call_site() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"@property\ndef func(self):\n    return 1\n"
    scopes = adapter.extract_scopes(src, "f.py")
    calls = adapter.extract_call_sites(src, "f.py", scopes)
    assert calls == ()


def test_attribute_call_callee_name_is_raw_text_per_canonical() -> None:
    """`obj.method()` produces `callee_name="obj.method"` per canonical
    spec.md §5.4 ("raw text; resolution is a separate concern"). Trace-node
    normalization is the consumer's job. Pinned here so a future contributor
    "improving" the field by extracting the final segment knows to amend
    the spec via DECISIONS first.
    """
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"def f():\n    obj.method()\n"
    scopes = adapter.extract_scopes(src, "f.py")
    calls = adapter.extract_call_sites(src, "f.py", scopes)
    callees = {c.callee_name for c in calls}
    assert "obj.method" in callees


def test_innermost_scope_picks_inner_when_byte_starts_tie() -> None:
    """Regression: when an outer and inner scope share `byte_start` (which
    can happen after `_scope_byte_range` adjusts a decorated function's
    span to its parent's), `_innermost_scope_containing` must pick the
    inner (smaller-span) scope. A strict `byte_start >` tiebreaker would
    let the outer win, mis-attributing the call site.
    """
    from outrider.ast_facts.models import ScopeUnit, compute_unit_id
    from outrider.ast_facts.python_adapter import _innermost_scope_containing

    # Both scopes share byte_start=0; outer ends at 100, inner ends at 50.
    # The inner is strictly smaller and should win.
    outer = ScopeUnit(
        unit_id=compute_unit_id("f.py", "function", "outer"),
        kind="function",
        name="outer",
        qualified_name="outer",
        file_path="f.py",
        line_start=1,
        line_end=10,
        byte_start=0,
        byte_end=100,
    )
    inner = ScopeUnit(
        unit_id=compute_unit_id("f.py", "function", "outer.inner"),
        kind="function",
        name="inner",
        qualified_name="outer.inner",
        file_path="f.py",
        line_start=1,
        line_end=5,
        byte_start=0,
        byte_end=50,
    )
    # Sort matches what extract_call_sites/extract_assignments do.
    sorted_scopes = sorted([outer, inner], key=lambda s: (s.byte_start, -s.byte_end))
    # Query a span well inside both.
    result = _innermost_scope_containing(sorted_scopes, byte_start=10, byte_end=20)
    assert result is not None
    assert result.qualified_name == "outer.inner", (
        "tiebreaker on byte_start ties must pick the smaller span (inner scope), "
        "not the first scope encountered in sort order"
    )


# ---------------------------------------------------------------------------
# extract_assignments
# ---------------------------------------------------------------------------


def test_extract_assignments_returns_canonical_shape() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"def f():\n    x = 1\n    y = 2\n    return x + y\n"
    scopes = adapter.extract_scopes(src, "f.py")
    sites = adapter.extract_assignments(src, "f.py", scopes)
    target_names = {s.target_name for s in sites}
    assert target_names == {"x", "y"}
    # All enclosing_scope_id values must reference real unit_ids
    valid_unit_ids = {s.unit_id for s in scopes}
    assert all(s.enclosing_scope_id in valid_unit_ids for s in sites)


# ---------------------------------------------------------------------------
# resolve_simple_direct_import
#
# These tests are FOCUSED RESOLVER UNIT TESTS, not extractor-to-resolver
# contract tests. The `ImportPathResolver` Protocol is mocked, decoupling
# the resolver's behavior from the extractor's actual output. The
# `ImportRef` instances are hand-built to represent plausible Python
# source shapes (e.g., `from foo.bar import bar`) — they are NOT
# necessarily what `PythonAdapter.extract_imports(...)` would emit for
# any given source string. The contract-level test that exercises
# extractor→resolver end-to-end lives in the integration suite.
# ---------------------------------------------------------------------------


def test_resolved_import_returns_target_path(tmp_path: Path) -> None:
    # Hand-built ImportRef represents `from foo.bar import bar`.
    # Place a real file the resolver will return.
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "bar.py").write_text("x = 1")
    resolver = MagicMock()
    resolver.resolve_candidate_paths.return_value = [Path("foo/bar.py")]
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="from",
        module="foo.bar",
        names=("bar",),
        is_simple_direct=True,
    )
    result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "resolved"
    assert result.target_path == "foo/bar.py"
    resolver.resolve_candidate_paths.assert_called_once_with("foo.bar", tmp_path)


def test_ambiguous_import_returns_none_target_path(tmp_path: Path) -> None:
    # Two real files that the Protocol returns
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x.py").write_text("")
    resolver = MagicMock()
    resolver.resolve_candidate_paths.return_value = [
        Path("a/x.py"),
        Path("b/x.py"),
    ]
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="from",
        module="a.x",
        names=("x",),
        is_simple_direct=True,
    )
    result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "ambiguous"
    assert result.target_path is None


def test_unresolved_import_when_no_files_exist(tmp_path: Path) -> None:
    resolver = MagicMock()
    resolver.resolve_candidate_paths.return_value = [Path("nonexistent.py")]
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="from",
        module="nonexistent",
        names=("x",),
        is_simple_direct=True,
    )
    result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "unresolved"


def test_non_simple_direct_returns_unresolved(tmp_path: Path) -> None:
    resolver = MagicMock()
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="star",
        module="x",
        names=(),
        is_simple_direct=False,
    )
    result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "unresolved"
    resolver.resolve_candidate_paths.assert_not_called()


# ---------------------------------------------------------------------------
# Symlink-safety (Internal contracts allowlist)
# ---------------------------------------------------------------------------


def test_symlink_candidate_returns_unresolved(tmp_path: Path) -> None:
    """A symlinked candidate file is rejected by the symlink-safe stat."""
    target_outside = tmp_path.parent / "outside_target.py"
    target_outside.write_text("x = 1")
    symlink = tmp_path / "linked.py"
    try:
        symlink.symlink_to(target_outside)
    except OSError:
        pytest.skip("symlink creation unsupported on this filesystem")

    resolver = MagicMock()
    resolver.resolve_candidate_paths.return_value = [Path("linked.py")]
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="from",
        module="linked",
        names=("linked",),
        is_simple_direct=True,
    )
    result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "unresolved"


def test_symlink_following_primitives_not_called(tmp_path: Path) -> None:
    """Resolver must not call always-symlink-following pathlib/os primitives."""
    real_file = tmp_path / "real.py"
    real_file.write_text("")
    resolver = MagicMock()
    resolver.resolve_candidate_paths.return_value = [Path("real.py")]
    adapter = PythonAdapter(resolver=resolver)
    import_ref = ImportRef(
        file_path="caller.py",
        line=1,
        import_kind="from",
        module="real",
        names=("real",),
        is_simple_direct=True,
    )
    # Patch each forbidden primitive to raise. `os.stat` and
    # `pathlib.Path.stat` are deliberately NOT patched per the V1
    # ast_facts/ spec (the safe primitive `Path.is_file(follow_symlinks=False)`
    # uses them transitively).
    with (
        patch("pathlib.Path.exists", side_effect=AssertionError("Path.exists forbidden")),
        patch("pathlib.Path.resolve", side_effect=AssertionError("Path.resolve forbidden")),
        patch("pathlib.Path.is_dir", side_effect=AssertionError("Path.is_dir forbidden")),
        patch("os.path.exists", side_effect=AssertionError("os.path.exists forbidden")),
        patch("os.path.isfile", side_effect=AssertionError("os.path.isfile forbidden")),
        patch("os.path.isdir", side_effect=AssertionError("os.path.isdir forbidden")),
        patch("os.path.realpath", side_effect=AssertionError("os.path.realpath forbidden")),
        patch("os.access", side_effect=AssertionError("os.access forbidden")),
    ):
        result = adapter.resolve_simple_direct_import(import_ref, tmp_path)
    assert result.status == "resolved"


# ---------------------------------------------------------------------------
# compute_parser_outcome + has_error map
# ---------------------------------------------------------------------------


def test_compute_parser_outcome_clean_with_has_error_map() -> None:
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"def good():\n    return 1\n"
    scopes = adapter.extract_scopes(src, "f.py")
    # Fail-loud guard: without this, both `all(...)` quantifiers below pass
    # vacuously if extract_scopes regresses to empty, and the test would
    # claim to pin per-scope has_error reliability while testing nothing.
    # Same anti-pattern as test_per_scope_has_error_isolated_to_offending_scope.
    assert scopes, "fixture must produce the `good` scope; extract_scopes regression suspected"
    outcome, has_error = adapter.compute_parser_outcome(src, "f.py", scopes)
    assert outcome == "clean"
    assert all(scope.unit_id in has_error for scope in scopes)
    assert all(has_error[scope.unit_id] is False for scope in scopes)


def test_decorator_region_parse_error_surfaces_in_has_error() -> None:
    """Regression: a syntax error inside a `@decorator(...)` line must
    propagate to the ScopeUnit's `has_error`, even when the inner
    `function_definition` itself is structurally clean. Pre-fix,
    `_find_node_by_span` only matched function_definition/class_definition
    and returned the (clean) inner function for a decorated scope —
    masking decorator-region errors from downstream `degraded` derivation.
    Fix: include `decorated_definition` in the target types so the
    outermost wrapper (which carries the decorator's has_error) wins.

    Fixture chosen empirically: `@route(*invalid==)` produces a
    `decorated_definition` whose `has_error` is True while the inner
    `function_definition` parses cleanly with `name="func"`. Tree-sitter
    extracts the function name reliably so this test runs (no
    conditional skip).
    """
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"@route(*invalid==)\ndef func():\n    return 1\n"
    scopes = adapter.extract_scopes(src, "f.py")
    assert scopes, "fixture must reliably produce a scope"
    [scope] = scopes
    assert scope.name == "func"
    _, has_error = adapter.compute_parser_outcome(src, "f.py", scopes)
    # The decorator-region error must surface on the ScopeUnit's has_error,
    # even though the inner function_definition itself is clean.
    assert has_error[scope.unit_id] is True


def test_outer_scope_has_error_not_attributed_from_nested_clean_scope() -> None:
    """Regression: `_find_node_by_span` must pick the OUTERMOST contained
    function/class, not the deepest. A clean nested function inside a
    syntactically-broken outer one must NOT mask the outer's has_error.

    Fixture: outer has an incomplete-assignment syntax error (`x =\\n`)
    in its body, but the def line itself is valid so tree-sitter
    extracts both `outer` and `outer.inner` as ScopeUnits.
    """
    adapter = PythonAdapter(resolver=MagicMock())
    src = b"def outer():\n    x =\n    def inner():\n        return 1\n"
    scopes = adapter.extract_scopes(src, "f.py")
    _, has_error = adapter.compute_parser_outcome(src, "f.py", scopes)
    by_qname = {s.qualified_name: s for s in scopes}
    # Both scopes are extracted...
    assert "outer" in by_qname
    assert "outer.inner" in by_qname
    # ...and the outer's has_error is True (because of the broken
    # `x =` inside its body), while the nested inner is clean.
    # Pre-fix, _find_node_by_span returned the inner (deepest match)
    # for the outer's lookup and reported has_error=False — masking
    # the outer's actual error.
    assert has_error[by_qname["outer"].unit_id] is True
    assert has_error[by_qname["outer.inner"].unit_id] is False


def test_per_scope_has_error_isolated_to_offending_scope() -> None:
    """Scope A clean, scope B has a syntax error inside its body;
    `has_error` flags only B per Month 0 spike's per-scope reliability finding.

    Fixture shape is load-bearing: `def scope_b()` must parse SUCCESSFULLY
    as a function_definition (so extract_scopes produces a ScopeUnit for
    it), but the body must contain an ERROR node so has_error fires.
    A malformed def header (e.g., missing colon on the `def` line) would
    skip extracting scope_b entirely — tree-sitter treats it as an
    unrelated ERROR node, and the test would pass vacuously.
    """
    adapter = PythonAdapter(resolver=MagicMock())
    # scope_a is clean; scope_b's def line parses but the body has an
    # invalid expression (triple-equals) producing an inner ERROR node.
    src = b"""
def scope_a():
    return 1

def scope_b():
    x ===
"""
    scopes = adapter.extract_scopes(src, "f.py")
    _, has_error = adapter.compute_parser_outcome(src, "f.py", scopes)
    by_qname = {s.qualified_name: s for s in scopes}
    # Fail loud if the fixture stops producing both scopes — without this
    # assertion the test could pass vacuously after a scope-extraction
    # regression because both has_error assertions would be skipped.
    missing = {"scope_a", "scope_b"} - by_qname.keys()
    assert not missing, f"fixture must produce scope_a + scope_b; missing: {sorted(missing)}"
    assert has_error[by_qname["scope_a"].unit_id] is False
    assert has_error[by_qname["scope_b"].unit_id] is True


# ---------------------------------------------------------------------------
# parser_outcome.py: should_skip + EXCLUSION_RULES
# ---------------------------------------------------------------------------


def test_exclusion_rules_exact_shape() -> None:
    """Pin the full 11-rule tuple in declared precedence order."""
    from outrider.ast_facts.models import ExclusionRule

    expected: tuple[ExclusionRule, ...] = (
        ExclusionRule(reason=SkipReason.OVERSIZED, kind="size", pattern=MAX_PARSE_BYTES),
        ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="vendor/"),
        ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="node_modules/"),
        ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="third_party/"),
        ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern=".venv/"),
        ExclusionRule(reason=SkipReason.VENDORED, kind="path_prefix", pattern="venv/"),
        ExclusionRule(
            reason=SkipReason.GENERATED_FILENAME,
            kind="filename_suffix",
            pattern="_pb2.py",
        ),
        ExclusionRule(
            reason=SkipReason.GENERATED_FILENAME,
            kind="filename_suffix",
            pattern="_pb2_grpc.py",
        ),
        ExclusionRule(
            reason=SkipReason.GENERATED_FILENAME,
            kind="filename_suffix",
            pattern=".pyi",
        ),
        ExclusionRule(
            reason=SkipReason.MINIFIED,
            kind="filename_suffix",
            pattern=".min.py",
        ),
        ExclusionRule(
            reason=SkipReason.GENERATED_BANNER,
            kind="banner",
            pattern=b"DO NOT EDIT",
        ),
    )
    assert expected == EXCLUSION_RULES


@pytest.mark.parametrize(
    ("file_path", "source", "expected"),
    [
        ("vendor/foo.py", b"pass", SkipReason.VENDORED),
        ("node_modules/foo.py", b"pass", SkipReason.VENDORED),
        ("third_party/foo.py", b"pass", SkipReason.VENDORED),
        (".venv/foo.py", b"pass", SkipReason.VENDORED),
        ("venv/foo.py", b"pass", SkipReason.VENDORED),
        ("foo_pb2.py", b"pass", SkipReason.GENERATED_FILENAME),
        ("foo_pb2_grpc.py", b"pass", SkipReason.GENERATED_FILENAME),
        ("foo.pyi", b"pass", SkipReason.GENERATED_FILENAME),
        ("foo.min.py", b"pass", SkipReason.MINIFIED),
        ("foo.py", b"# DO NOT EDIT\n", SkipReason.GENERATED_BANNER),
        ("normal.py", b"def x(): pass\n", None),
    ],
)
def test_should_skip_per_variant(
    file_path: str, source: bytes, expected: SkipReason | None
) -> None:
    assert should_skip(file_path, source) == expected


@pytest.mark.parametrize(
    "banner_source",
    [b"# DO NOT EDIT\n", b"# Do Not Edit\n", b"# do not edit\n"],
)
def test_banner_match_is_case_insensitive(banner_source: bytes) -> None:
    assert should_skip("foo.py", banner_source) == SkipReason.GENERATED_BANNER


def test_skip_precedence_size_beats_path() -> None:
    big = b"# pad\n" * (MAX_PARSE_BYTES // 6 + 1)
    assert should_skip("vendor/big.py", big) == SkipReason.OVERSIZED


def test_skip_precedence_path_beats_filename() -> None:
    assert should_skip("vendor/foo_pb2.py", b"pass") == SkipReason.VENDORED


# ---------------------------------------------------------------------------
# QueryMatchSpan / QueryCaptureSpan validators
# ---------------------------------------------------------------------------


def test_query_capture_span_rejects_empty_name() -> None:
    with pytest.raises(ValidationError):
        QueryCaptureSpan(name="", byte_start=0, byte_end=10)


def test_query_capture_span_rejects_negative_byte_start() -> None:
    with pytest.raises(ValidationError):
        QueryCaptureSpan(name="x", byte_start=-1, byte_end=10)


def test_query_capture_span_rejects_inverted_span() -> None:
    with pytest.raises(ValidationError):
        QueryCaptureSpan(name="x", byte_start=10, byte_end=5)


def test_query_match_span_envelope_must_match_captures() -> None:
    captures = (
        QueryCaptureSpan(name="a", byte_start=5, byte_end=10),
        QueryCaptureSpan(name="b", byte_start=20, byte_end=30),
    )
    # Wrong envelope: actual is (5, 30), not (0, 30)
    with pytest.raises(ValidationError):
        QueryMatchSpan(byte_start=0, byte_end=30, captures=captures)
    # Correct envelope admits
    qms = QueryMatchSpan(byte_start=5, byte_end=30, captures=captures)
    assert qms.byte_start == 5


def test_query_match_span_rejects_empty_captures() -> None:
    with pytest.raises(ValidationError):
        QueryMatchSpan(byte_start=0, byte_end=0, captures=())


# ---------------------------------------------------------------------------
# parse_python orchestrator
# ---------------------------------------------------------------------------


def test_parse_python_clean_path_canonical_fixture(
    canonical_python_source: bytes, canonical_python_path: str
) -> None:
    resolver = MagicMock()
    result = parse_python(canonical_python_source, canonical_python_path, resolver)
    assert result.parser_outcome == "clean"
    assert result.skip_reason is None
    assert len(result.scope_units) > 0
    assert len(result.imports) > 0
    assert len(result.call_sites) > 0
    # All call_sites' enclosing_scope_id values reference real unit_ids
    valid_unit_ids = {s.unit_id for s in result.scope_units}
    assert all(c.enclosing_scope_id in valid_unit_ids for c in result.call_sites)
    # has_error keyed by real unit_ids
    assert set(result.has_error.keys()) == valid_unit_ids
    # Resolver mock not called on clean path
    resolver.resolve_candidate_paths.assert_not_called()


def test_parse_python_failed_path_non_utf8() -> None:
    result = parse_python(b"\xff\xfe def x(): pass", "f.py", MagicMock())
    assert result.parser_outcome == "failed"
    assert result.skip_reason is None
    assert result.scope_units == ()
    assert result.imports == ()
    assert result.call_sites == ()
    assert result.assignment_sites == ()
    assert result.has_error == {}


def test_parse_python_failed_outcome_discards_extracted_tuples() -> None:
    """Pins the contract that `parser_outcome == "failed"` carries the
    empty-tuples shape regardless of which pipeline stage decided the
    file is unrecoverable.

    `compute_parser_outcome` always returns `"clean"` in V1 (per its
    docstring), so the discard branch in `parse_python` is dead code
    today. This test pins the contract so a future contributor tightening
    `compute_parser_outcome` to ever return `"failed"` (an obvious-looking
    refinement) doesn't silently activate the discard without
    acknowledging that extracted scope context is being thrown away.

    Strategy: monkeypatch `_compute_parser_outcome_from_tree` to return
    `("failed", {})` despite the upstream extraction having succeeded.
    Verify the orchestrator discards everything per the empty-tuples
    contract.
    """
    from unittest.mock import patch

    src = b"def good():\n    return 1\n"
    with patch.object(
        PythonAdapter,
        "_compute_parser_outcome_from_tree",
        return_value=("failed", {}),
    ):
        result = parse_python(src, "f.py", MagicMock())

    assert result.parser_outcome == "failed"
    assert result.scope_units == ()
    assert result.imports == ()
    assert result.call_sites == ()
    assert result.assignment_sites == ()
    assert result.has_error == {}
    assert result.skip_reason is None


def test_parse_python_skipped_path_oversized() -> None:
    big = b"# pad\n" * (MAX_PARSE_BYTES // 6 + 1)
    result = parse_python(big, "big.py", MagicMock())
    assert result.parser_outcome == "skipped"
    assert result.skip_reason == SkipReason.OVERSIZED
    assert result.scope_units == ()


def test_parse_python_pipeline_size_before_decode() -> None:
    """Oversized invalid-UTF-8 returns skipped, not failed."""
    big_invalid = b"\xff\xfe" * (MAX_PARSE_BYTES // 2 + 1)
    result = parse_python(big_invalid, "big.py", MagicMock())
    assert result.parser_outcome == "skipped"
    assert result.skip_reason == SkipReason.OVERSIZED


def test_parse_python_pipeline_pattern_before_decode() -> None:
    """Small invalid-UTF-8 with .min.py suffix returns skipped, not failed."""
    result = parse_python(b"\xff\xfe def x(): pass", "evil.min.py", MagicMock())
    assert result.parser_outcome == "skipped"
    assert result.skip_reason == SkipReason.MINIFIED


def test_parse_python_rejects_non_bytes_source() -> None:
    """Non-bytes source surfaces TypeError at the top of the pipeline,
    not deep in `should_skip` with a confusing traceback."""
    with pytest.raises(TypeError, match="source must be bytes"):
        parse_python("def x(): pass", "f.py", MagicMock())  # type: ignore[arg-type]


def test_should_skip_size_boundary_is_strict() -> None:
    """Per canonical §5.5 ("exceeding MAX_PARSE_BYTES") and the approved
    ast_facts/ spec (`len(source) > MAX_PARSE_BYTES`), a file of EXACTLY
    MAX_PARSE_BYTES bytes passes through. One byte over is OVERSIZED.
    A future canonical amendment to `>=` semantics requires a
    `DECISIONS.md` entry; until then code matches spec text.
    """
    boundary = b"a" * MAX_PARSE_BYTES
    # Exactly at threshold passes through (None for a normal-named .py file)
    assert should_skip("foo.py", boundary) is None
    # One byte over the threshold is OVERSIZED
    one_over = b"a" * (MAX_PARSE_BYTES + 1)
    assert should_skip("foo.py", one_over) == SkipReason.OVERSIZED


def test_should_skip_rejects_absolute_path() -> None:
    """Defensive contract check per trust-boundary #5: file_path MUST be
    POSIX repo-relative when it reaches ast_facts/. An absolute path
    would silently bypass `path_prefix` exclusions (`/abs/vendor/x.py`
    doesn't startswith `vendor/`), so we fail loudly instead.
    """
    with pytest.raises(ValueError, match="POSIX repo-relative"):
        should_skip("/abs/path/foo.py", b"x = 1")


def test_should_skip_rejects_backslash_path() -> None:
    """Defensive contract check per trust-boundary #5: backslash paths
    are not the contract surface. Outrider runs on Linux only in V1;
    cross-platform path normalization is the input boundary's job.
    """
    with pytest.raises(ValueError, match="POSIX repo-relative"):
        should_skip("vendor\\foo.py", b"x = 1")


def test_compute_parser_outcome_v1_always_returns_clean() -> None:
    """V1 policy: `compute_parser_outcome` always returns `("clean", has_error)`.

    A future change tightening this (e.g., "any has_error => failed")
    requires a DECISIONS.md entry per spec-fidelity discipline. This
    test is the gate that forces a contributor to acknowledge the
    policy change rather than silently broaden the failed-path shape.
    """
    adapter = PythonAdapter(resolver=MagicMock())
    # File with malformed scope (missing colon)
    src = b"def broken()\n    return 1\n\ndef ok():\n    return 2\n"
    scopes = adapter.extract_scopes(src, "f.py")
    outcome, has_error = adapter.compute_parser_outcome(src, "f.py", scopes)
    # Even with an error in the parse, V1 returns "clean".
    assert outcome == "clean"
    # has_error map is populated; tree-sitter recovered enough to find some scopes.
    assert isinstance(has_error, dict)


# ---------------------------------------------------------------------------
# ParseResult cross-field validator
# ---------------------------------------------------------------------------


def test_parse_result_skipped_without_reason_raises() -> None:
    with pytest.raises(ValidationError):
        ParseResult(parser_outcome="skipped", skip_reason=None)


def test_parse_result_clean_with_reason_raises() -> None:
    with pytest.raises(ValidationError):
        ParseResult(parser_outcome="clean", skip_reason=SkipReason.VENDORED)


def test_parse_result_failed_with_reason_raises() -> None:
    with pytest.raises(ValidationError):
        ParseResult(parser_outcome="failed", skip_reason=SkipReason.OVERSIZED)


# ---------------------------------------------------------------------------
# ImportResolution cross-field validator
# ---------------------------------------------------------------------------


def test_import_resolution_resolved_without_path_raises() -> None:
    with pytest.raises(ValidationError):
        ImportResolution(status="resolved", target_path=None)


def test_import_resolution_ambiguous_with_path_raises() -> None:
    with pytest.raises(ValidationError):
        ImportResolution(status="ambiguous", target_path="x.py")


def test_import_resolution_unresolved_with_path_raises() -> None:
    with pytest.raises(ValidationError):
        ImportResolution(status="unresolved", target_path="x.py")


# ---------------------------------------------------------------------------
# Models sanity (CallSite/AssignmentSite construct)
# ---------------------------------------------------------------------------


def test_call_site_construction() -> None:
    cs = CallSite(
        file_path="f.py",
        line=10,
        callee_name="foo",
        enclosing_scope_id="abc",
    )
    assert cs.line == 10


def test_assignment_site_construction() -> None:
    as_ = AssignmentSite(
        file_path="f.py",
        line=5,
        target_name="x",
        enclosing_scope_id="abc",
    )
    assert as_.target_name == "x"


# ---------------------------------------------------------------------------
# LanguageAdapter registry (case-insensitivity, missing-extension)
# ---------------------------------------------------------------------------


def test_get_adapter_factory_canonical_lowercase_extension() -> None:
    from outrider.ast_facts.registry import get_adapter_factory

    factory = get_adapter_factory(".py")
    assert factory is not None
    assert factory is PythonAdapter


def test_get_adapter_factory_uppercase_extension_resolves_to_same_adapter() -> None:
    """`.PY` (legal on case-insensitive filesystems, observed via `Path(file).suffix`)
    must resolve to the same adapter as `.py`. Pre-fix, `_LANGUAGE_ADAPTERS.get(".PY")`
    returned None — silently skipping analysis on case-variant filenames.
    """
    from outrider.ast_facts.registry import get_adapter_factory

    factory_lower = get_adapter_factory(".py")
    factory_upper = get_adapter_factory(".PY")
    factory_mixed = get_adapter_factory(".Py")
    assert factory_upper is factory_lower
    assert factory_mixed is factory_lower


def test_get_adapter_factory_unregistered_returns_none() -> None:
    from outrider.ast_facts.registry import get_adapter_factory

    assert get_adapter_factory(".rs") is None
    assert get_adapter_factory(".js") is None  # V1.5 surface, not yet registered


def test_get_adapter_factory_empty_extension_returns_none() -> None:
    from outrider.ast_facts.registry import get_adapter_factory

    assert get_adapter_factory("") is None
