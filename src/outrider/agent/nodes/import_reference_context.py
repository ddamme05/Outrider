"""Acquire a few pinned-HEAD import references before initial discovery.

Reference reads use the existing installation-scoped GitHub client. They never
change the PR inventory or expand finding locations.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from typing import TYPE_CHECKING

from outrider.agent.nodes.analyze_context import (
    MAX_MODULE_CONTEXT_BYTES,
    PreparedReviewContext,
    _reference_change_lines,
    _rendered_context_bytes,
)
from outrider.agent.nodes.head_reference_sources import HeadReferenceSources
from outrider.ast_facts.import_references import (
    python_import_reference_excerpts,
    python_import_reference_requests,
)
from outrider.audit.events import ContextManifestEntry
from outrider.coordinates import (
    CoordinateError,
    lookup_patched_file,
    python_relative_import_candidate_paths,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from outrider.ast_facts.base import ImportPathResolver
    from outrider.github import InstallationGitHubClient
    from outrider.schemas.pr_context import ChangedFile
    from outrider.schemas.review_state import ReviewState

MAX_REFERENCE_TEXT_BYTES = 12 * 1024
MAX_REFERENCE_DEPTH = 2


def omit_import_references(context: PreparedReviewContext, *, reason: str) -> PreparedReviewContext:
    introduction, separator, excerpts = context.text.partition("\n\n")
    notice = (
        f"Import references omitted ({reason}). Unshown code and callers remain unknown. "
        "Supplemental excerpts do not expand allowed finding locations."
    )
    if len(notice.encode()) > len(introduction.encode()):
        raise ValueError("reference omission notice exceeds the base introduction")
    return PreparedReviewContext(notice + separator + excerpts, context.manifest)


async def prepare_import_reference_contexts(
    state: ReviewState,
    *,
    files: Sequence[ChangedFile],
    base_contexts: dict[str, PreparedReviewContext],
    github_factory: Callable[[int], Awaitable[InstallationGitHubClient]],
    import_path_resolver: ImportPathResolver,
    sources: HeadReferenceSources | None = None,
) -> dict[str, PreparedReviewContext]:
    """Select exact named references with shared read/file/byte and depth bounds.

    A 404 establishes absence. An unusable response or incomplete candidate pair
    does not establish unique resolution. Other GitHub failures propagate.
    """
    if sources is None:
        sources = HeadReferenceSources(state.pr_context, github_factory)

    prepared = {}
    for file in files:
        if not file.path.endswith(".py") or file.content_head is None or not file.patch:
            continue
        try:
            patch = lookup_patched_file(file.patch, file.path)
        except CoordinateError:
            patch = None
        if patch is None:
            continue
        changed = _reference_change_lines(
            file_path=file.path,
            content=file.content_head,
            content_base=file.content_base,
            patched_file=patch,
            import_path_resolver=import_path_resolver,
            function_edits_only=True,
        )
        initial = python_import_reference_requests(
            file.content_head, file_path=file.path, changed_lines=changed
        )
        pending = deque((request, 1) for request in initial.requests)
        seen_requests = set()
        seen_excerpts = set()
        omissions = list(initial.omissions)
        sections = []
        manifest = []
        rendered_bytes = 0
        while pending:
            request, depth = pending.popleft()
            if request in seen_requests:
                continue
            seen_requests.add(request)
            try:
                candidates = python_relative_import_candidate_paths(
                    importer_path=request.importer_path, module=request.module, level=request.level
                )
            except CoordinateError:
                omissions.append(
                    f"{request.importer_path}:{request.line}: unsupported relative path"
                )
                continue
            matches = [(path, await sources.source_at(path)) for path in candidates]
            available = [(path, result) for path, result in matches if result.status == "available"]
            if any(result.status == "unavailable" for _, result in matches) or len(available) != 1:
                omissions.append(
                    f"{request.importer_path}:{request.line}: ambiguous or unavailable module"
                )
                continue
            path, result = available[0]
            if result.text is None:
                omissions.append(f"{path}: {result.reason}")
                continue
            selection = python_import_reference_excerpts(
                result.text,
                file_path=path,
                requested_names=request.imported_names,
                max_excerpt_chars=4096,
            )
            omissions.extend(selection.omissions)
            retained_selection = False
            for excerpt in selection.excerpts:
                identity = (path, excerpt.line_start, excerpt.line_end)
                if identity in seen_excerpts:
                    continue
                section = (
                    json.dumps(
                        {
                            "path": path,
                            "head_sha": state.pr_context.head_sha,
                            "source_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
                            "lines": [excerpt.line_start, excerpt.line_end],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                    + "\n".join(
                        f"{number} | {line}"
                        for number, line in enumerate(excerpt.text.splitlines(), excerpt.line_start)
                    )
                )
                if rendered_bytes + len(section.encode()) > MAX_REFERENCE_TEXT_BYTES - 2500:
                    omissions.append(f"{path}:{excerpt.line_start}: reference text limit")
                    continue
                seen_excerpts.add(identity)
                rendered_bytes += len(section.encode())
                sections.append(section)
                retained_selection = True
                manifest.append(
                    ContextManifestEntry(
                        file_path=path,
                        scope_unit_name=f"import reference {excerpt.name}"[:1024],
                        line_start=excerpt.line_start,
                        line_end=excerpt.line_end,
                        inclusion_reason="captured_reference",
                    )
                )
            if retained_selection and depth < MAX_REFERENCE_DEPTH:
                pending.extend((item, depth + 1) for item in selection.import_requests)
            elif retained_selection and selection.import_requests:
                omissions.append(f"{path}: dependency depth limit")

        shown: list[str] = []
        for omission in omissions:
            if len(json.dumps([*shown, omission]).encode()) > 1800:
                break
            shown.append(omission)
        inventory = (
            "Bounded HEAD import references; syntactic matches, not runtime resolution proof. "
            "Unshown definitions and callers remain unknown. Finding locations stay unchanged.\n"
            + json.dumps(
                {
                    "recipe": "relative-import-references-v1",
                    "reads": sources.reads,
                    "omissions_total": len(omissions),
                    "omissions_shown": shown,
                }
            )
        )
        base = base_contexts[file.path]
        _, separator, original_excerpts = base.text.partition("\n\n")
        text = inventory + separator + original_excerpts
        if sections:
            text += "\n\n" + "\n\n".join(sections)
        reference_text = "\n\n".join((inventory, *sections))
        if (
            len(reference_text.encode()) > MAX_REFERENCE_TEXT_BYTES
            or _rendered_context_bytes(text) > MAX_MODULE_CONTEXT_BYTES
        ):
            prepared[file.path] = omit_import_references(base, reason="rendered size limit")
        else:
            prepared[file.path] = PreparedReviewContext(text, base.manifest + tuple(manifest))
    return prepared
