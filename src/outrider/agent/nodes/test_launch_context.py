"""Attach the owning package's declared test command to a JavaScript or TypeScript test."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from outrider.agent.nodes.analyze_context import (
    MAX_MODULE_CONTEXT_BYTES,
    PreparedReviewContext,
    _is_test_path,
    _rendered_context_bytes,
)
from outrider.ast_facts.registry import JAVASCRIPT_EXTENSIONS, TYPESCRIPT_DIALECT_BY_EXTENSION
from outrider.ast_facts.review_context import SourceExcerpt
from outrider.audit.events import ContextManifestEntry

if TYPE_CHECKING:
    from outrider.ast_facts.models import ImportRef

MAX_TEST_LAUNCH_MANIFEST_BYTES = 32 * 1024
_MAX_SECTION_BYTES = 640
_BIN = re.compile(r"[a-z][a-z0-9-]{0,63}")
_TEST_GLOB = re.compile(r"(?:tests?|__tests__)/(?:\*\*/)?\*(?:\.(?:test|spec))?\.[cm]?[jt]sx?")
_OPTION = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
_DECODER = json.JSONDecoder()


def is_test_launch_file(path: str) -> bool:
    file = PurePosixPath(path)
    return _is_test_path(file) and (
        file.suffix in JAVASCRIPT_EXTENSIONS or file.suffix in TYPESCRIPT_DIALECT_BY_EXTENSION
    )


def has_test_launch_import(imports: tuple[ImportRef, ...]) -> bool:
    """Only a value import of a bare executable name can match a test command."""
    return any(ref.is_value_import and _BIN.fullmatch(ref.module) for ref in imports)


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise ValueError("ambiguous package property")
    return obj


def _invalid_constant(value: str) -> object:
    raise ValueError(f"invalid JSON number {value}")


def _matches_test_glob(pattern: str, relative_test: PurePosixPath) -> bool:
    # A shell without globstar expands ** as one directory. Recursive glob
    # implementations also include that depth, but may include others.
    return (
        _TEST_GLOB.fullmatch(pattern) is not None
        and len(relative_test.parts) == len(pattern.split("/"))
        and relative_test.full_match(pattern)
    )


def _runner(
    command: object,
    imports: tuple[ImportRef, ...],
    dependencies: set[str],
    relative_test: PurePosixPath,
) -> str | None:
    if (
        not isinstance(command, str)
        or not 0 < len(command) <= 200
        or not command.isascii()
        or not command.isprintable()
    ):
        return None
    launchers = []
    for pipeline in command.split("&&"):
        stages = [part.split() for part in pipeline.split("|")]
        if any(
            not words or not _BIN.fullmatch(words[0]) or words[0] not in dependencies
            for words in stages
        ):
            return None
        # Piped consumers receive another program's output. Their arguments
        # and settings cannot establish how the test itself was launched.
        if any(len(words) != 1 for words in stages[1:]):
            return None
        launchers.append(stages[0])
    imported = {ref.module for ref in imports if ref.is_value_import}
    matches = [words for words in launchers if words[0] in imported]
    if len(matches) != 1:
        return None
    runner, *args = matches[0]
    if any(words[0] != runner and len(words) != 1 for words in launchers):
        return None
    # Only show typed launch arguments. Shell expressions and arbitrary
    # literal CLI values may include private data even in a package manifest.
    if len(args) >= 2 and args[0] in {"-r", "--require"}:
        if not _BIN.fullmatch(args[1]) or args[1] not in dependencies:
            return None
        args = args[2:]
    if args and (len(args) != 1 or not _matches_test_glob(args[0], relative_test)):
        return None
    return runner


def _skip_space(source: str, offset: int) -> int:
    while offset < len(source) and source[offset] in " \t\n\r":
        offset += 1
    return offset


def _member(source: str, object_start: int, name: str) -> tuple[int, int, int]:
    """Find a known key's source span in an already validated JSON object."""
    offset = _skip_space(source, object_start + 1)
    while source[offset] != "}":
        key, end = _DECODER.raw_decode(source, offset)
        value_start = _skip_space(source, _skip_space(source, end) + 1)
        _, end = _DECODER.raw_decode(source, value_start)
        if key == name:
            return offset, value_start, end
        offset = _skip_space(source, end)
        if source[offset] == ",":
            offset = _skip_space(source, offset + 1)
    raise ValueError("package property was not found")


def _excerpt(source: str, span: tuple[int, int, int], label: str) -> SourceExcerpt | None:
    key, _, end = span
    first = source.rfind("\n", 0, key) + 1
    last = source.find("\n", end)
    if last == -1:
        last = len(source)
    # The reviewer must never receive unrelated properties that share a line.
    if source[first:key].strip() or source[end:last].strip() not in {"", ","}:
        return None
    text = source[first:last].replace("\r\n", "\n").removesuffix("\r")
    start_line = source.count("\n", 0, first) + 1
    return SourceExcerpt(label, start_line, start_line + text.count("\n"), text)


def _select(
    source: str, imports: tuple[ImportRef, ...], relative_test: PurePosixPath
) -> list[SourceExcerpt]:
    package = json.loads(source, object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    if not isinstance(package, dict) or not isinstance(scripts := package.get("scripts"), dict):
        return []
    dependencies: set[str] = set()
    for name in ("dependencies", "devDependencies"):
        declared = package.get(name)
        if isinstance(declared, dict):
            dependencies.update(declared)
    runner = _runner(scripts.get("test"), imports, dependencies, relative_test)
    if runner is None:
        return []
    root = _skip_space(source, 0)
    _, scripts_start, _ = _member(source, root, "scripts")
    command = _excerpt(source, _member(source, scripts_start, "test"), "scripts.test")
    if command is None:
        return []
    if runner not in package:
        return [command]
    options = package[runner]
    if not isinstance(options, dict) or not 1 <= len(options) <= 4:
        return []
    if not all(
        _OPTION.fullmatch(key) and isinstance(value, bool) for key, value in options.items()
    ):
        return []
    settings = _excerpt(source, _member(source, root, runner), runner)
    return [command, settings] if settings else []


def add_test_launch_context(
    base: PreparedReviewContext,
    *,
    test_path: str,
    manifest_path: str,
    manifest_source: str,
    imports: tuple[ImportRef, ...],
) -> PreparedReviewContext | None:
    """Render exact HEAD lines for a direct test launcher and its literal boolean settings."""
    if len(manifest_source) > MAX_TEST_LAUNCH_MANIFEST_BYTES:
        return None
    try:
        if len(manifest_source.encode()) > MAX_TEST_LAUNCH_MANIFEST_BYTES:
            return None
        relative_test = PurePosixPath(test_path).relative_to(PurePosixPath(manifest_path).parent)
        if not relative_test.parts or any(part.startswith(".") for part in relative_test.parts):
            return None
        selected = _select(manifest_source, imports, relative_test)
    except (ValueError, RecursionError):
        return None
    if not selected:
        return None
    lines = [f"Declared test launch at HEAD in {manifest_path}:"]
    entries = []
    for excerpt in selected:
        lines.extend(
            f"{n} | {line}" for n, line in enumerate(excerpt.text.split("\n"), excerpt.line_start)
        )
        entries.append(
            ContextManifestEntry(
                file_path=manifest_path,
                scope_unit_name=f"package.json {excerpt.name}",
                line_start=excerpt.line_start,
                line_end=excerpt.line_end,
                inclusion_reason="captured_reference",
            )
        )
    section = "\n".join(lines)
    text = base.text + "\n\n" + section
    if (
        len(section.encode()) > _MAX_SECTION_BYTES
        or _rendered_context_bytes(text) > MAX_MODULE_CONTEXT_BYTES
    ):
        return None
    return PreparedReviewContext(text, base.manifest + tuple(entries))
