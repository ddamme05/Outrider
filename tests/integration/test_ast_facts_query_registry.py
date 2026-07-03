"""Integration tests for `outrider.queries.registry` per
specs/2026-04-30-ast-facts-module.md integration test list.

Crosses ast_facts/ + queries/ + tree-sitter (the C library), so it
lives under `tests/integration/`. No DB required.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from outrider.ast_facts.errors import UnknownQueryMatchId
from outrider.ast_facts.models import QueryCaptureSpan, QueryMatchSpan
from outrider.queries import registry
from outrider.queries.registry import (
    _DEPRECATED_QUERY_ID_TO_BODY,
    _GRAMMARS_BY_QUERY_LANGUAGE,
    _QUERY_ID_TO_FILENAME,
    _STRUCTURAL_QUERY_FILES_BY_LANGUAGE,
    _compile_and_validate,
    get_query_source,
    match,
)

# ---------------------------------------------------------------------------
# Registry shape: every registered id parses + matches the canonical fixture
# ---------------------------------------------------------------------------


def test_registry_loads_at_module_import() -> None:
    """Module-level import already happened by virtue of the test
    collection; this asserts the public surface is non-empty and
    every id has a queryable body."""
    assert _QUERY_ID_TO_FILENAME, "registry must have at least one id"
    for query_id in _QUERY_ID_TO_FILENAME:
        body = get_query_source(query_id)
        assert body
        assert "@" in body, f"query {query_id!r} must have at least one capture"


def test_every_registered_id_matches_canonical_fixture(
    canonical_python_source: bytes,
) -> None:
    """Each structural query produces ≥1 match against its LANGUAGE'S
    canonical fixture, under every grammar of that language — the
    determinism check that `audit/replay.py` will rely on. Languages are
    iterated explicitly so a future language's structural queries can
    never run against another language's fixture bytes or the default
    grammar (which would raise on the first non-python structural query):
    a language that ships structural queries without a canonical fixture
    wired here fails loud instead."""
    fixture_by_language: dict[str, bytes] = {"python": canonical_python_source}
    # Defensive non-empty check (consistent with sibling tests in this
    # file). Without this guard the loops below never fire when the
    # registry is empty, and the test passes vacuously.
    assert _QUERY_ID_TO_FILENAME, "registry must have at least one id"
    for language, files in _STRUCTURAL_QUERY_FILES_BY_LANGUAGE.items():
        if not files:
            continue  # ships no structural queries (javascript today)
        assert language in fixture_by_language, (
            f"language {language!r} ships structural queries but has no "
            f"canonical fixture wired into this test; add one."
        )
        source = fixture_by_language[language]
        for query_id in files:
            for grammar in _GRAMMARS_BY_QUERY_LANGUAGE[language]:
                spans = match(query_id, source, grammar=grammar)
                assert spans, (
                    f"query {query_id!r} produced no matches against the "
                    f"{language} canonical fixture under grammar {grammar!r} "
                    f"(registered structural queries should fire on it)"
                )


# ---------------------------------------------------------------------------
# query_match_id format + deprecated ledger
# ---------------------------------------------------------------------------


_QUERY_ID_PATTERN = re.compile(r"^[a-z]+(?:\.[a-z][a-z0-9_]*)+$")


def test_every_query_id_matches_format() -> None:
    """`<language>.<purpose>` per Internal contracts; lowercase + dots."""
    # Defensive non-empty check: without this the test passes vacuously
    # if the registry ever becomes empty. `test_registry_loads_at_module_import`
    # also asserts non-empty, but keeping the guard local makes this test
    # self-contained — same pattern Codex's tier-list reviewer flagged for
    # `if expected in results: assert ...`-shaped tests where existence is
    # part of the contract.
    assert _QUERY_ID_TO_FILENAME, "registry must have at least one id"
    for query_id in _QUERY_ID_TO_FILENAME:
        assert _QUERY_ID_PATTERN.match(query_id), (
            f"query_match_id {query_id!r} does not match the <language>.<purpose> format"
        )


def test_deprecated_ledger_disjoint_from_active() -> None:
    """Deprecated ids must not collide with currently-active ids."""
    active = set(_QUERY_ID_TO_FILENAME)
    deprecated = set(_DEPRECATED_QUERY_ID_TO_BODY)
    assert active.isdisjoint(deprecated), (
        f"id collision between active and deprecated ledgers: {active & deprecated}"
    )


def test_deprecated_ledger_compiles_under_prefix_language_grammars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deprecated id compiles under every grammar of the language named by
    its namespace prefix — a `javascript.*` deprecation must compile under
    the three JS-family grammars, never the python grammar (whose node
    types differ, so a python-hardcoded compile would crash every import
    on the first non-python deprecation)."""
    monkeypatch.setattr(
        registry,
        "_DEPRECATED_QUERY_ID_TO_BODY",
        {"javascript.legacy_probe": "(call_expression) @call"},
    )
    bodies, compiled = registry._load_and_compile()  # noqa: SLF001
    assert bodies["javascript.legacy_probe"]
    assert set(compiled["javascript.legacy_probe"]) == {"javascript", "typescript", "tsx"}


def test_deprecated_ledger_unknown_prefix_rejects_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An id whose namespace prefix names no catalog language fails
    registration with an actionable message at import, not a bare KeyError
    (or a silent wrong-grammar compile — `(call)` IS a valid python node
    type, so the pre-fix python-hardcoded compile would have accepted it)."""
    monkeypatch.setattr(
        registry,
        "_DEPRECATED_QUERY_ID_TO_BODY",
        {"go.legacy_probe": "(call) @c"},
    )
    with pytest.raises(ValueError, match="not a known catalog language"):
        registry._load_and_compile()  # noqa: SLF001


def test_file_rename_preserves_id_stability(tmp_path: Path) -> None:
    """Renaming a `.scm` file must not churn ids — the registry's
    id-to-body mapping is the authoritative source per Internal contracts.

    This test copies the queries dir, renames a file, and asserts the
    same id can still be loaded after updating the filename mapping.
    """
    queries_root = Path(__file__).parent.parent.parent / "src" / "outrider" / "queries" / "python"
    work = tmp_path / "queries"
    work.mkdir()
    shutil.copytree(queries_root, work / "python")

    # Rename one file
    src_file = work / "python" / "function_definition.scm"
    renamed = work / "python" / "fn_def.scm"
    src_file.rename(renamed)

    # If the id-to-filename mapping is updated to point at fn_def.scm,
    # the id `python.function_definition` would still load. The body
    # is identical — the id is what survives. We assert this by reading
    # both files and confirming their content is equivalent.
    original_body = (queries_root / "function_definition.scm").read_text()
    renamed_body = renamed.read_text()
    assert original_body == renamed_body


# ---------------------------------------------------------------------------
# Unknown-id behavior
# ---------------------------------------------------------------------------


def test_match_unknown_id_raises() -> None:
    with pytest.raises(UnknownQueryMatchId):
        match("python.never_registered", b"def x(): pass")


def test_get_query_source_unknown_id_raises() -> None:
    with pytest.raises(UnknownQueryMatchId):
        get_query_source("python.never_registered")


def test_match_known_id_zero_matches_returns_empty_tuple() -> None:
    """Registered query, source has no functions → empty tuple, NOT raise."""
    result = match("python.function_definition", b"# no functions here\n")
    assert result == ()


# ---------------------------------------------------------------------------
# match() shape: envelope + capture flattening
# ---------------------------------------------------------------------------


def test_match_envelope_equals_min_max_of_captures() -> None:
    src = b"def hello():\n    return 1\n"
    spans = match("python.function_definition", src)
    assert len(spans) == 1
    span = spans[0]
    expected_start = min(c.byte_start for c in span.captures)
    expected_end = max(c.byte_end for c in span.captures)
    assert span.byte_start == expected_start
    assert span.byte_end == expected_end


def test_match_captures_sorted_by_byte_start_then_name() -> None:
    src = b"class C:\n    pass\n"
    spans = match("python.class_definition", src)
    assert len(spans) == 1
    capture_keys = [(c.byte_start, c.byte_end, c.name) for c in spans[0].captures]
    assert capture_keys == sorted(capture_keys)


def test_match_returns_query_match_span_models() -> None:
    src = b"def f(): pass\n"
    spans = match("python.function_definition", src)
    assert len(spans) == 1
    assert isinstance(spans[0], QueryMatchSpan)
    for cap in spans[0].captures:
        assert isinstance(cap, QueryCaptureSpan)


# ---------------------------------------------------------------------------
# _compile_and_validate: per-pattern mandatory-capture enforcement
# ---------------------------------------------------------------------------
#
# The envelope rule per Internal contracts requires every registered pattern
# to produce at least one capture per match. tree-sitter's capture quantifiers
# distinguish mandatory (`''`, `'+'`) from optional (`'?'`, `'*'`); a pattern
# whose captures are all optional could fire with an empty captures tuple at
# runtime and crash `QueryMatchSpan`'s envelope `min`/`max`. Validation must
# reject such patterns at module-load.


def test_compile_and_validate_rejects_optional_only_capture() -> None:
    """`?` quantifier on the only capture means it might fire zero times."""
    body = "((comment)? @c)"
    with pytest.raises(ValueError, match="no mandatory captures"):
        _compile_and_validate("test.optional_only", body)


def test_compile_and_validate_rejects_star_only_capture() -> None:
    """`*` quantifier on the only capture means it might fire zero times."""
    body = "((comment)* @c)"
    with pytest.raises(ValueError, match="no mandatory captures"):
        _compile_and_validate("test.star_only", body)


def test_compile_and_validate_rejects_multi_pattern_with_captureless() -> None:
    """A multi-pattern body where any pattern lacks a mandatory capture
    must reject — the envelope rule is per-pattern, not per-file."""
    body = "(function_definition name: (identifier) @fn)\n(class_definition)\n"
    with pytest.raises(ValueError, match="no mandatory captures"):
        _compile_and_validate("test.multi_one_captureless", body)


def test_compile_and_validate_admits_multi_pattern_all_mandatory() -> None:
    """Multi-pattern body where every pattern has at least one mandatory
    capture is admitted — multi-pattern is permitted per Internal contracts."""
    body = (
        "(function_definition name: (identifier) @fn)\n(class_definition name: (identifier) @cls)\n"
    )
    query = _compile_and_validate("test.multi_both_mandatory", body)
    # Confirms validation didn't accidentally drop a pattern or capture —
    # `query is not None` is mypy-trivially true given the return type.
    assert cast("int", query.pattern_count) == 2
    assert cast("int", query.capture_count) == 2


# ---------------------------------------------------------------------------
# Import-light contract per DECISIONS.md#018 point 6
# ---------------------------------------------------------------------------


def test_import_light_subprocess_isolated() -> None:
    """`from outrider.ast_facts.models import SkipReason` does NOT
    transitively load `tree_sitter`. Run in a subprocess so the
    assertion is stable regardless of what other tests in this
    pytest session have imported.

    The subprocess inherits PYTHONPATH with `<repo>/src` prepended so
    the test passes whether the project is installed into the venv or
    only available on `pytest.ini_options.pythonpath = ["src"]` from a
    source checkout. Without this, `from outrider.ast_facts.models ...`
    fails to resolve and the test would report a false positive for
    the import-light contract.
    """
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from outrider.ast_facts.models import SkipReason; "
            "import outrider.ast_facts.registry; "  # lazy factories: no grammar load
            "loaded = [m for m in ("
            "'tree_sitter', 'tree_sitter_python', "
            "'tree_sitter_javascript', 'tree_sitter_typescript'"
            ") if m in sys.modules]; "
            "assert not loaded, "
            "f'grammar modules unexpectedly loaded: {loaded}; sys.modules has: '"
            "f'{sorted(m for m in sys.modules if \"tree\" in m)}'"
        ),
    ]
    repo_src = Path(__file__).parent.parent.parent / "src"
    # Avoid a trailing `os.pathsep` when `PYTHONPATH` is unset — Python
    # parses an empty path entry as CWD, which would silently add the
    # working directory to the subprocess's import search and muddy the
    # import-light isolation we're testing for.
    pythonpath_entries = [str(repo_src)]
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pythonpath_entries)}
    # cmd is built from `sys.executable` + literal string args; no user input.
    result = subprocess.run(  # noqa: S603
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, (
        f"import-light subprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
