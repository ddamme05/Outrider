"""Select bounded, captured HEAD implementations for a changed README."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from outrider.ast_facts.parser_outcome import should_skip
from outrider.ast_facts.registry import get_adapter_factory
from outrider.coordinates import CoordinateError, lookup_patched_file

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.pr_context import ChangedFile

# Include complete small documents and modules so qualifiers and branches stay visible.
MAX_DOCUMENT_BYTES = 16_384
MAX_IMPLEMENTATION_BYTES = 12_288
_WORD = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]{2,}")


def is_readme(path: str) -> bool:
    return PurePosixPath(path).name.casefold() == "readme.md"


def implementation_sources(document: ChangedFile, files: Sequence[ChangedFile]) -> dict[str, str]:
    """Prefer implementation files sharing identifiers with the changed documentation."""
    if not document.content_head or len(document.content_head.encode()) > MAX_DOCUMENT_BYTES:
        return {}
    try:
        patch = lookup_patched_file(document.patch, document.path)
    except CoordinateError:
        return {}
    if patch is None:
        return {}
    lines = document.content_head.split("\n")
    identifiers: set[str] = set()
    for hunk in patch:
        for line in hunk:
            if not line.is_added:
                continue
            number = line.target_line_no
            if number is None or not 1 <= number <= len(lines):
                return {}
            value = line.value.removesuffix("\n")
            if value.rstrip("\r") != lines[number - 1].rstrip("\r"):
                return {}
            identifiers.update(_WORD.findall(value))
    if not identifiers:
        return {}
    ranked = []
    for file in files:
        path = PurePosixPath(file.path.lower())
        if (
            not file.content_head
            or get_adapter_factory(path.suffix) is None
            or path.name.endswith((".pyi", ".d.ts", ".d.mts", ".d.cts"))
            or {"test", "tests", "__tests__", "examples", "fixtures", "docs"}.intersection(
                path.parts
            )
            or {"test", "tests", "spec", "bench"}.intersection(re.split(r"[._-]", path.stem))
        ):
            continue
        data = file.content_head.encode()
        if len(data) > MAX_IMPLEMENTATION_BYTES or should_skip(file.path, data) is not None:
            continue
        matches = identifiers.intersection(_WORD.findall(file.content_head))
        if matches:
            ranked.append((-len(matches), file.path, file.content_head, len(data)))
    remaining = MAX_IMPLEMENTATION_BYTES
    selected = {}
    for _, name, source, size in sorted(ranked):
        if size <= remaining:
            selected[name] = source
            remaining -= size
    return selected
