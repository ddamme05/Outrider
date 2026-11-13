"""Prepare bounded BASE evidence for a validated whole-file removal.

The source was removed; it is not current code and cannot produce structural
vulnerability findings. Surviving code is supplied separately as reference data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from outrider.coordinates import (
    bound_diff_hunks_text,
    extract_line_range_body,
    source_line_count,
    validate_removed_file,
)

if TYPE_CHECKING:
    from unidiff import PatchedFile

    from outrider.schemas.pr_context import ChangedFile


@dataclass(frozen=True, slots=True)
class RemovedFileContext:
    patched_file: PatchedFile
    line_ranges: tuple[tuple[int, int], ...]
    numbered_source: str
    diff_hunks: str
    omitted: bool


def prepare_removed_file_context(
    changed_file: ChangedFile, *, max_lines: int, max_source_bytes: int
) -> RemovedFileContext:
    """Validate the entire deletion before selecting complete BASE source lines."""
    patched_file = validate_removed_file(
        file_path=changed_file.path,
        status=changed_file.status,
        base_content=changed_file.content_base,
        head_content=changed_file.content_head,
        patch=changed_file.patch,
    )
    source = changed_file.content_base
    if source is None:
        raise RuntimeError("validated removal has no BASE source")
    total_lines = source_line_count(source)
    end = min(total_lines, max_lines)
    rendered = ""
    while end > 0:
        body = extract_line_range_body(1, end, source)
        numbered = "\n".join(
            f"{line} | {text}" for line, text in enumerate(body.removesuffix("\n").split("\n"), 1)
        )
        rendered = f"BASE lines 1-{end} (file removed; no HEAD exists)\n{numbered}"
        if len(rendered.encode("utf-8")) <= max_source_bytes:
            break
        end -= 1
    return RemovedFileContext(
        patched_file=patched_file,
        line_ranges=((1, end),) if end > 0 else (),
        numbered_source=rendered if end > 0 else "",
        diff_hunks=bound_diff_hunks_text(patched_file, max_lines=100, max_chars=8192),
        omitted=end < total_lines,
    )
