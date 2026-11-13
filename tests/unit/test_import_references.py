"""Explicit relative-import references preserve exact source and bound omissions."""

from dataclasses import asdict

import pytest

from outrider.ast_facts.import_references import (
    python_import_reference_excerpts,
    python_import_reference_requests,
)


def test_changed_scope_requests_original_imported_name_and_preserves_import_order() -> None:
    source = (
        "from .decoder import decode as convert, unused\n"
        "from ..checks import validate\n"
        "from .other import unrelated\n"
        "def changed(value):\n"
        "    return validate(convert(value))\n"
        "def untouched(value):\n"
        "    return unrelated(value)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/api/handler.py", changed_lines=frozenset({5})
    )

    assert [asdict(request) for request in selected.requests] == [
        {
            "module": "decoder",
            "level": 1,
            "imported_names": ("decode",),
            "importer_path": "pkg/api/handler.py",
            "line": 1,
        },
        {
            "module": "checks",
            "level": 2,
            "imported_names": ("validate",),
            "importer_path": "pkg/api/handler.py",
            "line": 2,
        },
    ]
    assert not selected.omissions


def test_selected_function_local_import_is_relevant_but_other_function_import_is_not() -> None:
    source = (
        "def changed(value):\n"
        "    from .decoder import decode as convert\n"
        "    return convert(value)\n"
        "def untouched(value):\n"
        "    from .other import convert\n"
        "    return convert(value)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({3})
    )
    assert [(item.module, item.imported_names) for item in selected.requests] == [
        ("decoder", ("decode",))
    ]


def test_unsupported_import_forms_are_explicit_without_guessing_targets() -> None:
    source = (
        "import importlib as loader\n"
        "from external import absolute\n"
        "from . import possible_child\n"
        "from .unknown import *\n"
        "def changed(value):\n"
        "    loader.import_module(value)\n"
        "    return absolute(possible_child(value))\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({6})
    )
    assert not selected.requests
    assert any("absolute import omitted" in note for note in selected.omissions)
    assert any("child-module form omitted" in note for note in selected.omissions)
    assert any("star import omitted" in note for note in selected.omissions)
    assert any("dynamic import omitted" in note for note in selected.omissions)


@pytest.mark.parametrize(
    "body",
    [
        "def changed(convert):\n    return convert('value')\n",
        "def changed():\n    convert = other\n    return convert('value')\n",
        "convert = other\ndef changed():\n    return convert('value')\n",
        ("if enabled:\n    convert = other\ndef changed():\n    return convert('value')\n"),
        (
            "try:\n    run()\nexcept Exception as convert:\n    pass\n"
            "def changed():\n    return convert('value')\n"
        ),
        (
            "match value:\n    case {'decoder': convert}:\n        pass\n"
            "def changed():\n    return convert('value')\n"
        ),
        (
            "def changed():\n    try:\n        run()\n    except Exception as convert:\n"
            "        return convert('value')\n"
        ),
        (
            "def changed(value):\n    match value:\n        case {'decoder': convert}:\n"
            "            return convert('value')\n"
        ),
        (
            "def changed():\n    def convert(value):\n        return value\n"
            "    return convert('value')\n"
        ),
    ],
)
def test_obvious_shadowing_never_requests_the_imported_implementation(body: str) -> None:
    source = "from .decoder import decode as convert\n" + body
    selected = python_import_reference_requests(
        source,
        file_path="pkg/handler.py",
        changed_lines=frozenset({len(source.splitlines())}),
    )
    assert not selected.requests


def test_another_selected_function_parameter_does_not_hide_a_used_module_import() -> None:
    source = (
        "from .decoder import decode as convert\n"
        "def shadowed(convert):\n"
        "    return convert('value')\n"
        "def changed():\n"
        "    return convert('value')\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({3, 5})
    )
    assert [item.imported_names for item in selected.requests] == [("decode",)]


def test_exact_complete_function_keeps_decorators_docstring_guard_and_dependency_request() -> None:
    source = (
        "from .policy import allowed as permitted\n"
        "\n"
        "@decorate\n"
        "async def decode(value):\n"
        '    """Keep the full implementation, including this documentation."""\n'
        "    if not permitted(value):\n"
        "        raise ValueError('invalid')\n"
        "    return value\n"
        "\n"
        "def unused(value):\n"
        "    return value\n"
    )
    selected = python_import_reference_excerpts(
        source, file_path="pkg/decoder.py", requested_names=("decode",)
    )
    assert len(selected.excerpts) == 1
    excerpt = selected.excerpts[0]
    assert (excerpt.line_start, excerpt.line_end) == (3, 8)
    assert excerpt.text == "".join(source.splitlines(keepends=True)[2:8])
    assert [(item.module, item.imported_names) for item in selected.import_requests] == [
        ("policy", ("allowed",))
    ]
    assert not selected.omissions


def test_complete_classes_lowercase_bindings_and_reexports_have_exact_ranges() -> None:
    source = (
        "from .actual import decode as convert\n"
        "value: int = 2\n"
        "left, right = (1, 2)\n"
        "class Policy:\n"
        "    def check(self):\n"
        "        return value\n"
    )
    selected = python_import_reference_excerpts(
        source,
        file_path="pkg/decoder.py",
        requested_names=("convert", "value", "left", "right", "Policy"),
    )
    assert [item.text for item in selected.excerpts] == [
        "from .actual import decode as convert\n",
        "value: int = 2\n",
        "left, right = (1, 2)\n",
        "class Policy:\n    def check(self):\n        return value\n",
    ]
    assert [(item.module, item.imported_names) for item in selected.import_requests] == [
        ("actual", ("decode",))
    ]


def test_ambiguous_missing_nested_and_long_definitions_are_omitted_whole() -> None:
    source = (
        "value = 1\n"
        "value = 2\n"
        "def outer():\n"
        "    def inner():\n"
        "        return True\n"
        "    return inner()\n"
        "def long_helper():\n"
        '    """' + "documentation " * 40 + '"""\n'
        "    if not enabled:\n"
        "        raise ValueError('disabled')\n"
        "    return True\n"
    )
    selected = python_import_reference_excerpts(
        source,
        file_path="pkg/decoder.py",
        requested_names=("value", "absent", "inner", "long_helper"),
        max_excerpt_chars=200,
    )
    assert not selected.excerpts
    assert not selected.import_requests
    assert selected.omissions == (
        "name value: ambiguous top-level binding",
        "name absent: no top-level binding",
        "name inner: no top-level binding",
        "name long_helper: complete excerpt exceeds character bound",
    )


def test_excerpt_limit_is_explicit_and_does_not_request_dependencies_of_omitted_excerpts() -> None:
    source = "from .remote import external\n" + "".join(
        f"def helper{number}():\n    return external()\n" for number in range(5)
    )
    selected = python_import_reference_excerpts(
        source,
        file_path="pkg/decoder.py",
        requested_names=tuple(f"helper{number}" for number in range(5)),
    )
    assert len(selected.excerpts) == 4
    assert "name helper4: four-excerpt limit" in selected.omissions
    assert len(selected.import_requests) == 1


@pytest.mark.parametrize(
    ("source", "path", "reason"),
    [
        ("def broken(:\n", "pkg/helper.py", "not eligible parsed Python"),
        ("value = 1\n", "pkg/helper.ts", "unsupported language"),
        ("# DO NOT EDIT\nvalue = 1\n", "pkg/helper.py", "source excluded"),
        ("value = '" + "é" * 65536 + "'\n", "pkg/helper.py", "128-KiB"),
        ("value = 1\n", "vendor/helper.py", "source excluded"),
    ],
)
def test_source_boundaries_produce_omissions_in_both_selectors(
    source: str, path: str, reason: str
) -> None:
    requests = python_import_reference_requests(
        source, file_path=path, changed_lines=frozenset({1})
    )
    excerpts = python_import_reference_excerpts(source, file_path=path, requested_names=("value",))
    assert not requests.requests
    assert not excerpts.excerpts
    assert any(reason in item for item in requests.omissions)
    assert any(reason in item for item in excerpts.omissions)


def test_no_function_scope_and_invalid_excerpt_cap_do_not_select_arbitrary_source() -> None:
    requests = python_import_reference_requests(
        "from .helper import external\nvalue = external()\n",
        file_path="pkg/handler.py",
        changed_lines=frozenset({2}),
    )
    assert not requests.requests
    assert requests.omissions == ("no selected changed Python functions",)
    with pytest.raises(ValueError, match="must be positive"):
        python_import_reference_excerpts(
            "value = 1\n",
            file_path="pkg/helper.py",
            requested_names=("value",),
            max_excerpt_chars=0,
        )


def test_function_local_import_suppresses_the_same_global_alias() -> None:
    source = (
        "from .global_decoder import decode\n"
        "def changed(value):\n"
        "    from .local_decoder import decode\n"
        "    return decode(value)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({4})
    )
    assert [(item.module, item.imported_names) for item in selected.requests] == [
        ("local_decoder", ("decode",))
    ]
    assert not selected.omissions


def test_competing_function_local_imports_leave_the_alias_unresolved() -> None:
    source = (
        "def changed(value):\n"
        "    from .first_decoder import decode\n"
        "    from .second_decoder import decode\n"
        "    return decode(value)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({4})
    )
    assert not selected.requests
    assert any("another local import binding" in note for note in selected.omissions)


def test_local_import_aliases_remain_independent_between_selected_functions() -> None:
    source = (
        "from .global_decoder import decode\n"
        "def first(value):\n"
        "    from .first_decoder import decode\n"
        "    return decode(value)\n"
        "def second(value):\n"
        "    from .second_decoder import decode\n"
        "    return decode(value)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({4, 7})
    )
    assert [(item.module, item.imported_names) for item in selected.requests] == [
        ("first_decoder", ("decode",)),
        ("second_decoder", ("decode",)),
    ]
    assert not selected.omissions


@pytest.mark.parametrize("type_parameter", ["decode", "**decode", "*decode"])
def test_function_type_parameters_shadow_module_import_aliases(type_parameter: str) -> None:
    source = (
        f"from .decoder import decode\ndef changed[{type_parameter}](value):\n    return decode\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({3})
    )
    assert not selected.requests


def test_enclosing_class_type_parameter_is_not_a_module_import_reference() -> None:
    source = (
        "from .decoder import decode\nclass Consumer[decode]:\n"
        "    def changed(self, value):\n        return decode\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({4})
    )
    assert not selected.requests


def test_nested_function_does_not_reintroduce_a_shadowed_module_import() -> None:
    source = (
        "from .decoder import decode\ndef outer(decode):\n"
        "    def changed(value):\n        return decode(value)\n    return changed(1)\n"
    )
    selected = python_import_reference_requests(
        source, file_path="pkg/handler.py", changed_lines=frozenset({4})
    )
    assert not selected.requests
