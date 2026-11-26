"""Checks for the exact package lines sent to JS and TS test reviewers."""

from __future__ import annotations

import json

import pytest

from outrider.agent.nodes.analyze_context import PreparedReviewContext
from outrider.agent.nodes.test_launch_context import (
    MAX_TEST_LAUNCH_MANIFEST_BYTES,
    add_test_launch_context,
    has_test_launch_import,
    is_test_launch_file,
)
from outrider.ast_facts.models import ImportRef
from outrider.audit.events import ContextManifestEntry


def review_import(module: str, *, value: bool = True) -> ImportRef:
    return ImportRef(
        file_path="test.js",
        line=1,
        module=module,
        import_kind="direct",
        is_simple_direct=True,
        is_value_import=value,
    )


BASE_ENTRY = ContextManifestEntry(
    file_path="test.js",
    scope_unit_name="changed",
    line_start=2,
    line_end=3,
    inclusion_reason="changed_scope",
)
BASE = PreparedReviewContext("Existing context", (BASE_ENTRY,))
DEPS = {"tape": "4", "esm": "3", "tap-spec": "5", "ava": "6", "tsd": "1", "xo": "2", "vitest": "3"}


def package(command: str = "xo && ava && tsd", **extra: object) -> str:
    return json.dumps({"scripts": {"test": command}, "devDependencies": DEPS, **extra}, indent=2)


def select(
    source: str,
    *imports: ImportRef,
    test_path: str = "packages/ui/test.js",
    base: PreparedReviewContext = BASE,
) -> PreparedReviewContext | None:
    return add_test_launch_context(
        base,
        test_path=test_path,
        manifest_path="packages/ui/package.json",
        manifest_source=source,
        imports=imports or (review_import("ava"),),
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("test.js", True),
        ("web/tests/foo.cjs", True),
        ("foo.unit.spec.mts", True),
        ("web/__tests__/thing.tsx", True),
        ("src/latest.ts", False),
        ("tests/test_it.py", False),
    ],
)
def test_supported_test_path(path: str, expected: bool) -> None:
    assert is_test_launch_file(path) == expected


def test_runner_precheck_skips_relative_node_and_type_only_imports() -> None:
    assert not has_test_launch_import(
        (review_import("./ava"), review_import("node:test"), review_import("ava", value=False))
    )
    assert has_test_launch_import((review_import("vitest"),))


def test_command_and_selected_runner_options_use_exact_source_coordinates() -> None:
    source = package(
        ava={"serial": True, "workerThreads": False},
        xo={"envs": ["irrelevant"]},
        nested={"scripts": {"test": "another command"}},
    )
    ctx = select(source)
    assert ctx is not None
    assert ctx.manifest[0] == BASE_ENTRY
    command, options = ctx.manifest[1:]
    assert [
        (entry.scope_unit_name, entry.inclusion_reason, entry.source_side)
        for entry in (command, options)
    ] == [
        ("package.json scripts.test", "captured_reference", "head"),
        ("package.json ava", "captured_reference", "head"),
    ]
    assert command.file_path == options.file_path == "packages/ui/package.json"
    assert command.line_start == command.line_end == 3
    assert source.splitlines()[options.line_start - 1].strip() == '"ava": {'
    assert source.splitlines()[options.line_end - 1].strip() == "},"
    lines = [
        f"{idx} | {source.splitlines()[idx - 1]}"
        for entry in (command, options)
        for idx in range(entry.line_start, entry.line_end + 1)
    ]
    assert (
        ctx.text
        == "Existing context\n\nDeclared test launch at HEAD in packages/ui/package.json:\n"
        + "\n".join(lines)
    )
    edited = select(source.replace('"serial": true', '"serial": false'))
    assert edited is not None
    assert edited.text == ctx.text.replace('"serial": true', '"serial": false')
    assert select(source.replace("\n", "\r\n")) == ctx


@pytest.mark.parametrize(
    ("command", "test_path"),
    [
        ("tape -r esm test/*.js | tap-spec", "packages/ui/test/index.js"),
        ("tape test/*.js | tap-spec", "packages/ui/test/api.test.js"),
        ("tape --require esm tests/**/*.spec.ts", "packages/ui/tests/api/public.spec.ts"),
    ],
)
def test_direct_preload_and_test_glob(command: str, test_path: str) -> None:
    ctx = select(package(command), review_import("tape"), test_path=test_path)
    assert ctx is not None
    assert [e.line_start for e in ctx.manifest[1:]] == [3]
    assert ctx.text.endswith(f'3 |     "test": "{command}"')


@pytest.mark.parametrize(
    ("pattern", "test_path"),
    [
        ("test/*.js", "packages/ui/__tests__/main.spec.ts"),
        ("test/*.js", "packages/ui/test/main.ts"),
        ("test/*.js", "packages/ui/src/test/main.js"),
        ("test/*.js", "packages/other/test/main.js"),
        ("test/*.js", "packages/ui/test/.internal.js"),
        ("tests/**/*.spec.ts", "packages/ui/tests/main.spec.ts"),
        ("tests/**/*.spec.ts", "packages/ui/tests/sub/deeper/main.spec.ts"),
        ("tests/**/*.spec.ts", "packages/ui/tests/.cache/main.spec.ts"),
        ("tests/**/*.spec.ts", "packages/ui/tests/sub/main.ts"),
    ],
)
def test_explicit_glob_must_include_the_file_under_its_package(
    pattern: str, test_path: str
) -> None:
    command = f"tape -r esm {pattern} | tap-spec"
    assert select(package(command), review_import("tape"), test_path=test_path) is None


def test_piped_reporter_cannot_supply_the_test_runner_or_its_settings() -> None:
    source = package("xo && ava | tap-spec", ava={"serial": True}, **{"tap-spec": {"quiet": True}})
    assert select(source, review_import("tap-spec")) is None
    ctx = select(source, review_import("ava"), review_import("tap-spec"))
    assert ctx is not None
    assert '"test": "xo && ava | tap-spec"' in ctx.text
    assert '"serial": true' in ctx.text
    assert '"quiet"' not in ctx.text


@pytest.mark.parametrize(
    "command",
    [
        "ava --token SECRET",
        "ava secret-token",
        "ava -r undeclared",
        "ava -r esm key.js",
        "ava $TOKEN",
        "ava -r ./secrets/token.js",
        "ava && curl https://example.invalid/key",
        "ava && xo --token secret",
        "ava\nxo",
        "ava || xo",
        "ava | ava -r esm",
        "ava && ava -r esm",
        "ava; xo",
        "npm run test",
        "unlisted | ava",
        "xo && ava && vitest",
        "ava --config=/tmp/private",
        "ava --flag secret",
    ],
)
def test_unrecognized_and_ambiguous_commands_stay_out(command: str) -> None:
    assert select(package(command), review_import("ava"), review_import("vitest")) is None


def test_settings_must_be_a_bounded_literal_boolean_object() -> None:
    for settings in (
        {"token": "inline-secret"},
        [True],
        {"concurrency": 16},
        {},
        {"nested": {"ok": True}},
        {f"option{i}": True for i in range(5)},
    ):
        assert select(package(ava=settings)) is None
    assert select(package(), review_import("ava", value=False)) is None


def test_unrelated_properties_sharing_a_source_line_are_omitted() -> None:
    source = package(ava={"serial": True})
    assert (
        select(
            source.replace(
                '"test": "xo && ava && tsd"', '"test": "xo && ava && tsd", "secret": "do not send"'
            )
        )
        is None
    )
    assert select(source.replace('"ava": {\n', '"private": "do not send", "ava": {\n')) is None
    assert select(source.replace('"serial": true', '"serial": true, "serial": false')) is None
    assert select(source.replace('"serial": true', '"serial": NaN')) is None
    assert select(source[:-4]) is None


def test_source_and_rendered_size_bounds_leave_the_base_unchanged() -> None:
    assert select(package() + " " * MAX_TEST_LAUNCH_MANIFEST_BYTES) is None
    assert (
        select(
            package(ava={"serial": True}).replace('"serial": true', " " * 700 + '"serial": true')
        )
        is None
    )
    oversized = PreparedReviewContext("x" * 32768, (BASE_ENTRY,))
    assert select(package(), base=oversized) is None
    assert oversized.manifest == (BASE_ENTRY,)
