"""Provide fixture GitHub reads and capture proposed reviews for eval runs.

eval_driver.py builds the fixture data and supplies these adapters to the graph.
The default import resolver returns no local paths; trace resolves through the
fixture GitHub client. These adapters make no network or database calls.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from outrider.schemas.publish import GitHubReviewCreated

if TYPE_CHECKING:
    from pathlib import Path

    from outrider.github import InstallationGitHubClient
    from outrider.schemas.publish import InlineComment


class _FixtureContentNotFoundError(Exception):
    """Mimics githubkit's 404 `RequestFailed` for a path the fixture's repo
    does not contain.

    Carries `.response.status_code == 404` so consumers that special-case 404
    treat an absent path as a real GitHub "file not found" — matching production
    wire behavior. This is required for the trace node's two-phase probe, which
    fetches BOTH candidate paths for a dotted import (e.g. `app/models.py` AND
    `app/models/__init__.py`) and relies on a 404 to learn which one exists
    (see `trace.py` — it reads `exc.response.status_code`). No `githubkit`
    import — only the `.response.status_code` shape those consumers read.
    """

    def __init__(self, path: str, ref: str) -> None:
        self.response = SimpleNamespace(status_code=404)
        super().__init__(f"fixture repo has no file at path={path!r} ref={ref!r} (GitHub 404)")


@dataclass
class _FixtureFileMeta:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    previous_filename: str | None


@dataclass
class _FixtureContentFile:
    encoding: str
    content: str


@dataclass
class _FixtureResponse:
    parsed_data: Any


class _FixtureReposAPI:
    """Serves `async_get_content` from fixture content, keyed by (path, ref).

    Content is base64 with newline wrapping (`base64.encodebytes`) to match
    GitHub's real contents-API wire shape — intake strips `\\n`/`\\r` then
    decodes with `validate=True` (see `github/fetch.py`), so an unwrapped stub
    would not exercise the strip path real GitHub triggers
    (`feedback_test_stubs_match_wire_format`).
    """

    def __init__(self, content_by_path_ref: dict[tuple[str, str], str]) -> None:
        self._content = content_by_path_ref

    async def async_get_content(
        self, owner: str, repo: str, path: str, *, ref: str
    ) -> _FixtureResponse:
        try:
            raw = self._content[(path, ref)]
        except KeyError as exc:
            # Absent path → mimic GitHub's 404 (real wire behavior), NOT a hard
            # error: trace's candidate probe fetches paths that legitimately
            # don't exist and reads the 404 to resolve. Intake only requests
            # paths the fixture supplied, so a 404 there would itself be a
            # genuine fetch failure (surfaced loudly upstream).
            raise _FixtureContentNotFoundError(path, ref) from exc
        wrapped_b64 = base64.encodebytes(raw.encode()).decode("ascii")
        return _FixtureResponse(
            parsed_data=_FixtureContentFile(encoding="base64", content=wrapped_b64)
        )


class _FixturePullsAPI:
    def __init__(self, metas: list[_FixtureFileMeta]) -> None:
        self._metas = metas

    async def async_list_files(
        self, owner: str, repo: str, pull_number: int, **kwargs: Any
    ) -> _FixtureResponse:
        return _FixtureResponse(parsed_data=list(self._metas))


class _FixtureRestAPI:
    def __init__(self, repos: _FixtureReposAPI, pulls: _FixturePullsAPI) -> None:
        self.repos = repos
        self.pulls = pulls


class _FixtureGitHubClient:
    def __init__(self, rest: _FixtureRestAPI) -> None:
        self.rest = rest


@dataclass
class _CapturingPublisher:
    """Records `create_review` instead of POSTing; reports no prior review."""

    create_review_calls: list[dict[str, Any]] = field(default_factory=list)

    async def create_review(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        review_status: str,
        body_marker: str,
        body: str | None = None,
        comments: tuple[InlineComment, ...],
    ) -> GitHubReviewCreated:
        self.create_review_calls.append(
            {"review_status": review_status, "comments": tuple(comments), "body": body}
        )
        return GitHubReviewCreated(github_review_id=999, comments_posted=len(comments))

    async def find_existing_review_on_head_sha(
        self,
        *,
        gh: InstallationGitHubClient,
        owner: str,
        repo: str,
        pull_number: int,
        head_sha: str,
        body_marker: str,
    ) -> int | None:
        return None


class _NoOpImportPathResolver:
    """Default resolver that supplies no local-filesystem candidate paths.

    Trace resolves through GitHub probes independently. Callers may inject a real
    resolver for analyze's scope/import context."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:
        return []

    def resolve_specifier_candidate_paths(
        self, specifier: str, importing_file_path: str, import_root: Path
    ) -> list[Path]:
        return []
