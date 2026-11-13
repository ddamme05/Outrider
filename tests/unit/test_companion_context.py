"""Captured Python siblings add complete reference data, never primary scopes."""

from __future__ import annotations

import hashlib
import json

import pytest

from outrider.agent.nodes.analyze_context import (
    MAX_COMPANION_CONTEXT_BYTES,
    MAX_MODULE_CONTEXT_BYTES,
    PreparedReviewContext,
    _rendered_context_bytes,
    add_companion_function_context,
)
from outrider.ast_facts.registry import parse_source
from outrider.ast_facts.review_context import select_companion_functions
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

_STUB = "def encode(value: bytes) -> bytes: ...\n"
_IMPLEMENTATION = "@decorate\ndef encode(payload):\n    return payload\n"


def _scopes(source: str, path: str):
    return parse_source(source.encode(), path, COORDINATES_IMPORT_PATH_RESOLVER).scope_units


@pytest.mark.parametrize(
    ("path", "source", "sibling_path", "sibling_source"),
    [
        ("pkg/codec.pyi", _STUB, "pkg/codec.py", _IMPLEMENTATION),
        ("pkg/codec.py", _IMPLEMENTATION, "pkg/codec.pyi", _STUB),
    ],
)
def test_complete_sibling_reference_preserves_source_identity(
    path: str, source: str, sibling_path: str, sibling_source: str
) -> None:
    scopes = _scopes(source, path)
    heads = {sibling_path: sibling_source}
    selection = select_companion_functions(
        file_path=path, source=source, selected_scope_units=scopes, captured_heads=heads
    )
    (reference,) = selection.references
    assert not selection.omissions
    assert reference.file_path == sibling_path
    assert reference.excerpt.text == sibling_source
    assert reference.excerpt.line_start == 1

    original = PreparedReviewContext("Existing source context", ())
    result = add_companion_function_context(
        original,
        file_path=path,
        head_sha="a" * 40,
        source=source,
        selected_scope_units=scopes,
        captured_heads=heads,
    )
    metadata = json.loads(next(line for line in result.text.splitlines() if line.startswith("{")))
    assert metadata == {
        "file_path": sibling_path,
        "function_name": "encode",
        "source_side": "head",
        "head_sha": "a" * 40,
        "line_start": 1,
        "line_end": len(sibling_source.splitlines()),
        "source_sha256": hashlib.sha256(sibling_source.encode()).hexdigest(),
        "excerpt_sha256": hashlib.sha256(sibling_source.encode()).hexdigest(),
    }
    assert result.text.startswith(original.text)
    assert "not runtime binding proof" in result.text
    (entry,) = result.manifest
    assert entry.file_path == sibling_path
    assert entry.inclusion_reason == "captured_reference"
    assert entry.source_side == "head"
    assert (entry.line_start, entry.line_end) == (1, len(sibling_source.splitlines()))
    assert (
        add_companion_function_context(
            result,
            file_path=path,
            head_sha="a" * 40,
            source=source,
            selected_scope_units=scopes,
            captured_heads=heads,
        )
        is result
    )


@pytest.mark.parametrize(
    ("source", "heads"),
    [
        (_STUB, {"elsewhere/codec.py": _IMPLEMENTATION}),
        (_STUB + "encode = other\n", {"pkg/codec.py": _IMPLEMENTATION}),
        (_STUB, {"pkg/codec.py": _IMPLEMENTATION + "encode = other\n"}),
        (_STUB, {"pkg/codec.py": _IMPLEMENTATION + "\ndef encode(other): pass\n"}),
        (_STUB, {"pkg/codec.py": "def encode(payload):\n    value = " + repr("x" * 5000) + "\n"}),
    ],
)
def test_ineligible_siblings_leave_existing_context_untouched(
    source: str, heads: dict[str, str]
) -> None:
    original = PreparedReviewContext("Existing source context", ())
    result = add_companion_function_context(
        original,
        file_path="pkg/codec.pyi",
        head_sha="a" * 40,
        source=source,
        selected_scope_units=_scopes(source, "pkg/codec.pyi"),
        captured_heads=heads,
    )
    assert result is original


def test_only_selected_top_level_functions_participate() -> None:
    source = _STUB + "\nclass Codec:\n    def decode(self): ...\n"
    heads = {"pkg/codec.py": _IMPLEMENTATION + "\nclass Codec:\n    def decode(self): pass\n"}
    scopes = _scopes(source, "pkg/codec.pyi")
    selection = select_companion_functions(
        file_path="pkg/codec.pyi",
        source=source,
        selected_scope_units=tuple(scope for scope in scopes if scope.qualified_name != "encode"),
        captured_heads=heads,
    )
    assert not selection.references


def test_count_and_rendered_limits_omit_whole_definitions() -> None:
    source = "".join(f"def function_{index}(): ...\n" for index in range(4))
    sibling = "\n".join(
        f"def function_{index}():\n    return {repr('x' * 2800)}\n" for index in range(4)
    )
    scopes = _scopes(source, "pkg/codec.pyi")
    heads = {"pkg/codec.py": sibling}
    selection = select_companion_functions(
        file_path="pkg/codec.pyi", source=source, selected_scope_units=scopes, captured_heads=heads
    )
    assert [reference.excerpt.name for reference in selection.references] == [
        "function_0",
        "function_1",
        "function_2",
    ]
    original = PreparedReviewContext("Existing source context", ())
    result = add_companion_function_context(
        original,
        file_path="pkg/codec.pyi",
        head_sha="a" * 40,
        source=source,
        selected_scope_units=scopes,
        captured_heads=heads,
    )
    assert 0 < len(result.manifest) < len(selection.references)
    assert len(result.text.encode()) - len(original.text.encode()) <= MAX_COMPANION_CONTEXT_BYTES
    assert _rendered_context_bytes(result.text) <= MAX_MODULE_CONTEXT_BYTES
    for entry in result.manifest:
        exact_source = "\n".join(sibling.splitlines()[entry.line_start - 1 : entry.line_end])
        assert exact_source.splitlines()[-1] in result.text

    full = PreparedReviewContext("x" * MAX_MODULE_CONTEXT_BYTES, ())
    assert (
        add_companion_function_context(
            full,
            file_path="pkg/codec.pyi",
            head_sha="a" * 40,
            source=source,
            selected_scope_units=scopes,
            captured_heads=heads,
        )
        is full
    )


def test_growth_limit_counts_the_source_fence() -> None:
    source = "def first(): ...\ndef second(): ...\n"
    sibling = "".join(
        f"def {name}():\n    return {repr('`' * 2000)}\n" for name in ("first", "second")
    )
    original = PreparedReviewContext("Existing source context", ())
    result = add_companion_function_context(
        original,
        file_path="pkg/codec.pyi",
        head_sha="a" * 40,
        source=source,
        selected_scope_units=_scopes(source, "pkg/codec.pyi"),
        captured_heads={"pkg/codec.py": sibling},
    )

    assert result.manifest
    assert (
        _rendered_context_bytes(result.text) - _rendered_context_bytes(original.text)
        <= MAX_COMPANION_CONTEXT_BYTES
    )
