"""Tests for `outrider.github.fetch` — the per-file content + file-list helpers.

Uses a hand-rolled stub `GitHub` client to avoid HTTP; the focus is on:
  - `coordinates.validate_diff_path` runs BEFORE the githubkit call
    (path-traversal payloads are rejected without any network attempt).
  - `fetch_file_content_at` correctly handles the inline-base64,
    directory, symlink, submodule, and no-content response shapes.
  - The 1 MB per-file decoded-size cap is enforced.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import pytest

from outrider.coordinates.errors import CoordinateError
from outrider.github.fetch import fetch_file_content_at, list_pr_files

# ---- Stub githubkit response objects --------------------------------------


@dataclass
class _StubResponse:
    parsed_data: Any


@dataclass
class _StubContentFile:
    """ContentFile-like — has `encoding` and `content`."""

    encoding: str
    content: str


@dataclass
class _StubContentSymlink:
    """ContentSymlink-like — no content/encoding."""

    target: str


# ---- Stub GitHub client ---------------------------------------------------


class _StubReposAPI:
    def __init__(self, response_data: Any) -> None:
        self._response_data = response_data
        self.calls: list[tuple[str, str, str, dict[str, Any]]] = []

    async def async_get_content(
        self,
        owner: str,
        repo: str,
        path: str,
        **kwargs: Any,
    ) -> _StubResponse:
        self.calls.append((owner, repo, path, kwargs))
        return _StubResponse(parsed_data=self._response_data)


class _StubPullsAPI:
    def __init__(self, response_data: list[Any]) -> None:
        self._response_data = response_data
        self.calls: list[tuple[str, str, int, dict[str, Any]]] = []

    async def async_list_files(
        self,
        owner: str,
        repo: str,
        pull_number: int,
        **kwargs: Any,
    ) -> _StubResponse:
        self.calls.append((owner, repo, pull_number, kwargs))
        return _StubResponse(parsed_data=self._response_data)


class _StubRestAPI:
    def __init__(self, *, repos: _StubReposAPI, pulls: _StubPullsAPI) -> None:
        self.repos = repos
        self.pulls = pulls


class _StubGitHub:
    """Minimum surface for fetch.py — only the methods that get called."""

    def __init__(
        self,
        *,
        content_response: Any = None,
        files_response: list[Any] | None = None,
    ) -> None:
        self.rest = _StubRestAPI(
            repos=_StubReposAPI(content_response),
            pulls=_StubPullsAPI(files_response or []),
        )


# ---- list_pr_files --------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pr_files_returns_parsed_data() -> None:
    files = [{"filename": "a.py"}, {"filename": "b.py"}]
    gh = _StubGitHub(files_response=files)
    result = await list_pr_files(gh, owner="acme", repo="widgets", pull_number=42)  # type: ignore[arg-type]
    assert result == files
    assert gh.rest.pulls.calls == [("acme", "widgets", 42, {"per_page": 31})]


@pytest.mark.asyncio
async def test_list_pr_files_empty() -> None:
    gh = _StubGitHub(files_response=[])
    result = await list_pr_files(gh, owner="acme", repo="widgets", pull_number=99)  # type: ignore[arg-type]
    assert result == []


# ---- fetch_file_content_at: path validation -------------------------------


@pytest.mark.asyncio
async def test_fetch_rejects_traversal_path_before_api_call() -> None:
    """A path with `..` is rejected by `validate_diff_path`; the
    githubkit call is NEVER reached. Defends `paths-validated-before-use`."""
    gh = _StubGitHub(content_response=None)
    with pytest.raises(CoordinateError):
        await fetch_file_content_at(
            gh,  # type: ignore[arg-type]
            owner="acme",
            repo="widgets",
            path="../etc/passwd",
            ref="abc",
        )
    # No call was made.
    assert gh.rest.repos.calls == []


@pytest.mark.asyncio
async def test_fetch_rejects_absolute_path_before_api_call() -> None:
    """An absolute path is rejected; no API call."""
    gh = _StubGitHub(content_response=None)
    with pytest.raises(CoordinateError):
        await fetch_file_content_at(
            gh,  # type: ignore[arg-type]
            owner="acme",
            repo="widgets",
            path="/etc/passwd",
            ref="abc",
        )
    assert gh.rest.repos.calls == []


@pytest.mark.asyncio
async def test_fetch_rejects_shell_metachars_before_api_call() -> None:
    """A path with shell metacharacters is rejected; no API call."""
    gh = _StubGitHub(content_response=None)
    with pytest.raises(CoordinateError):
        await fetch_file_content_at(
            gh,  # type: ignore[arg-type]
            owner="acme",
            repo="widgets",
            path="src/$(cat /etc/passwd).py",
            ref="abc",
        )
    assert gh.rest.repos.calls == []


# ---- fetch_file_content_at: response shapes -------------------------------


@pytest.mark.asyncio
async def test_fetch_returns_decoded_bytes_for_inline_base64() -> None:
    """Happy path: ContentFile with `encoding="base64"` returns decoded bytes."""
    expected = b"def hello():\n    return 'world'\n"
    response = _StubContentFile(
        encoding="base64",
        content=base64.b64encode(expected).decode("ascii"),
    )
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="src/example.py",
        ref="abc123",
    )

    assert result == expected
    # Validated path is what reached the API (here, the input was already
    # validation-clean, so it's identical).
    assert gh.rest.repos.calls == [("acme", "widgets", "src/example.py", {"ref": "abc123"})]


@pytest.mark.asyncio
async def test_fetch_returns_none_for_directory_response() -> None:
    """A path that GitHub treats as a directory returns `list[...]` —
    fetch returns None (out-of-scope for intake)."""
    response = [{"filename": "child1.py"}, {"filename": "child2.py"}]
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="src/somedir",
        ref="abc123",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fetch_returns_none_for_symlink() -> None:
    """A symlink response (no `encoding`/`content`) returns None."""
    response = _StubContentSymlink(target="../somewhere")
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="link.py",
        ref="abc",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fetch_returns_none_when_encoding_is_none() -> None:
    """GitHub returns `encoding="none"` for files too large for inline-base64;
    fetch returns None (caller maps to SkipReason.OVERSIZED)."""
    response = _StubContentFile(encoding="none", content="")
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="huge.py",
        ref="abc",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fetch_returns_none_when_decoded_exceeds_cap() -> None:
    """Even when inline-base64 is returned for a borderline file, the
    1 MB decoded-size cap fires (belt-and-suspenders)."""
    oversize = b"x" * 2_000_000
    response = _StubContentFile(
        encoding="base64",
        content=base64.b64encode(oversize).decode("ascii"),
    )
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="huge.py",
        ref="abc",
    )
    assert result is None


@pytest.mark.asyncio
async def test_fetch_accepts_borderline_under_cap() -> None:
    """Files just under the cap are accepted."""
    under_cap = b"y" * 999_999
    response = _StubContentFile(
        encoding="base64",
        content=base64.b64encode(under_cap).decode("ascii"),
    )
    gh = _StubGitHub(content_response=response)

    result = await fetch_file_content_at(
        gh,  # type: ignore[arg-type]
        owner="acme",
        repo="widgets",
        path="big-but-ok.py",
        ref="abc",
    )
    assert result == under_cap
