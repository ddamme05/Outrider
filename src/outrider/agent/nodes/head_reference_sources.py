"""Share pinned, bounded GitHub reads among a review's optional source references."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal

from outrider.coordinates import CoordinateError, validate_diff_path
from outrider.github.fetch import fetch_file_content_at

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from outrider.github import InstallationGitHubClient
    from outrider.schemas.pr_context import PRContext

MAX_REFERENCE_READS = 16
MAX_REFERENCE_FILES = 4
MAX_REFERENCE_FILE_BYTES = 128 * 1024
MAX_REFERENCE_TOTAL_BYTES = 400 * 1024


@dataclass(frozen=True)
class HeadSource:
    status: Literal["available", "absent", "unavailable"]
    text: str | None
    reason: str = ""


class HeadReferenceSources:
    """Read from intake first, then the installation at this PR's HEAD when enabled."""

    def __init__(
        self,
        pr: PRContext,
        github_factory: Callable[[int], Awaitable[InstallationGitHubClient]] | None,
    ) -> None:
        self._pr = pr
        self._factory = github_factory
        self._client: InstallationGitHubClient | None = None
        self._sources = {
            file.path: HeadSource("available", file.content_head)
            for file in pr.changed_files
            if file.content_head is not None
        }
        self._removed = {file.path for file in pr.changed_files if file.status == "removed"}
        self._removed.update(file.previous_path for file in pr.changed_files if file.previous_path)
        self.reads = 0
        self._retained_files = 0
        self._retained_bytes = 0

    async def source_at(self, path: str) -> HeadSource:
        if path in self._sources:
            return self._sources[path]
        if self._factory is None or self.reads >= MAX_REFERENCE_READS:
            return HeadSource("unavailable", None, "read limit")
        self.reads += 1
        if self._client is None:
            self._client = await self._factory(self._pr.installation_id)
        try:
            raw = await fetch_file_content_at(
                self._client,
                owner=self._pr.owner,
                repo=self._pr.repo,
                path=path,
                ref=self._pr.head_sha,
            )
        except Exception as error:
            if getattr(getattr(error, "response", None), "status_code", None) != 404:
                raise
            result = HeadSource("absent", None, "not present at HEAD")
        else:
            if raw is None:
                result = HeadSource("unavailable", None, "unusable repository response")
            elif len(raw) > MAX_REFERENCE_FILE_BYTES:
                result = HeadSource("available", None, "source size limit")
            elif (
                self._retained_files == MAX_REFERENCE_FILES
                or self._retained_bytes + len(raw) > MAX_REFERENCE_TOTAL_BYTES
            ):
                result = HeadSource("available", None, "retained source limit")
            else:
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    result = HeadSource("available", None, "non-UTF8 source")
                else:
                    self._retained_files += 1
                    self._retained_bytes += len(raw)
                    result = HeadSource("available", text)
        self._sources[path] = result
        return result

    async def nearest_package(self, file_path: str, *, deadline: float) -> tuple[str, str] | None:
        """An ancestor owns the file only after every nearer manifest is known absent at HEAD."""
        for directory in PurePosixPath(file_path).parents:
            try:
                path = validate_diff_path(str(directory / "package.json"))
            except CoordinateError:
                return None
            if path in self._removed and path not in self._sources:
                continue
            if path not in self._sources and asyncio.get_running_loop().time() >= deadline:
                return None
            try:
                # Captured and cached entries need no I/O, even after the lookup window closes.
                async with asyncio.timeout_at(deadline):
                    source = await self.source_at(path)
            except Exception:
                # An optional launcher lookup may fail independently of the ordinary file review.
                # Other tests in this package must also leave ownership unresolved.
                self._sources[path] = HeadSource("unavailable", None, "package lookup failed")
                return None
            if source.status == "absent":
                continue
            if source.text is None:
                return None
            return path, source.text
        return None
