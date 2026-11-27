"""A local import alias is not the original name of a called Python helper."""

from unittest.mock import MagicMock

import pytest

from outrider.ast_facts.review_context import build_called_function_index, called_function_context


@pytest.mark.parametrize(
    ("source", "changed_line", "expected"),
    [
        # Reduced from Packaging #1171. Both names exist in the captured parser.
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "def restore(value):\n    return _parse_marker(value)\n",
            3,
            [],
        ),
        (
            "from ..grammar import (\n    parse_marker as _parse_marker,\n)\n"
            "def restore(value):\n    return _parse_marker(value)\n",
            5,
            [],
        ),
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "def restore(_parse_marker, value):\n    return _parse_marker(value)\n",
            3,
            ["_parse_marker"],
        ),
        (
            "from ._parser import parse_marker\n"
            "def restore(value):\n    return parse_marker(value)\n",
            3,
            ["parse_marker"],
        ),
        (
            "from ._parser import _parse_marker\n"
            "def elsewhere():\n    from ._parser import parse_marker as _parse_marker\n"
            "def restore(value):\n    return _parse_marker(value)\n",
            5,
            ["_parse_marker"],
        ),
        (
            "from ._parser import _parse_marker\n"
            "class Parser:\n    from ._parser import parse_marker as _parse_marker\n"
            "    def restore(self, value):\n        return _parse_marker(value)\n",
            5,
            ["_parse_marker"],
        ),
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "from ._parser import _parse_marker\n"
            "def restore(value):\n    return _parse_marker(value)\n",
            4,
            ["_parse_marker"],
        ),
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "def restore(value):\n    from ._parser import _parse_marker\n"
            "    return _parse_marker(value)\n",
            4,
            ["_parse_marker"],
        ),
        (
            "from .grammar import parse_marker as _parse_marker\n"
            "def configure():\n    global _parse_marker\n    from ._parser import _parse_marker\n"
            "configure()\ndef restore(value):\n    return _parse_marker(value)\n",
            7,
            ["_parse_marker"],
        ),
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "def restore(callback, value):\n"
            "    return (lambda _parse_marker: _parse_marker(value))(callback), "
            "_parse_marker(value)\n",
            3,
            ["_parse_marker"],
        ),
        (
            "from ._parser import parse_marker as _parse_marker\n"
            "def restore(parser, value):\n    _parse_marker(value)\n"
            "    return parser._parse_marker(value)\n",
            3,
            ["_parse_marker"],
        ),
    ],
    ids=[
        "recorded-alias",
        "multiline-relative-alias",
        "shadowed-alias",
        "unaliased",
        "other-scope",
        "class-scope",
        "module-rebind",
        "local-rebind",
        "explicit-global-rebind",
        "same-line-scopes",
        "receiver",
    ],
)
def test_name_matching_respects_python_import_aliases(
    source: str, changed_line: int, expected: list[str]
) -> None:
    path = "pkg/sub/markers.py"
    index = build_called_function_index(
        {
            path: source,
            "pkg/sub/_parser.py": (
                "def parse_marker(value):\n    return _parse_marker(value)\n"
                "def _parse_marker(tokenizer):\n    return tokenizer.read()\n"
            ),
        },
        import_path_resolver=MagicMock(),
    )
    refs = called_function_context(index, file_path=path, changed_lines=frozenset({changed_line}))
    assert [ref.excerpt.name.split("@")[0] for ref in refs] == expected


@pytest.mark.parametrize(
    "imports",
    [
        "from .helpers import inspect_value",
        "from .helpers import unrelated as local",
    ],
)
def test_called_helpers_skip_python_alias_parse_for_unrelated_imports(
    monkeypatch: pytest.MonkeyPatch, imports: str
) -> None:
    path = "pkg/handler.py"
    index = build_called_function_index(
        {
            path: f"{imports}\ndef changed(value):\n    return inspect_value(value)\n"
            "def observe(ctx):\n    with ctx as value:\n        return value\n",
            "pkg/helpers.py": "def inspect_value(value):\n    return value\n",
        },
        import_path_resolver=MagicMock(),
    )
    monkeypatch.setattr(
        "outrider.ast_facts.review_context._parse_context_source",
        lambda *_args: pytest.fail("Unexpected Python alias parse"),
    )
    refs = called_function_context(index, file_path=path, changed_lines=frozenset({3}))
    assert [ref.file_path for ref in refs] == ["pkg/helpers.py"]
