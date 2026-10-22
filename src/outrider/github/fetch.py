# Per-file content fetch helper for intake — vendor wrapper.
"""Fetch PR metadata, then individual file contents for intake.

list_pr_files returns one page sized for intake's preflight limit. Once that gate
passes, intake fetches file content in parallel. fetch_file_content_at validates
paths before the SDK call (paths-validated-before-use) and bounds encoded and
decoded content allocations. Vendor access stays here, outside the agent nodes
(vendor-sdks-only-in-wrappers).
"""

from __future__ import annotations

import base64
import binascii
from typing import TYPE_CHECKING, Any

from outrider.coordinates.diff_parser import validate_diff_path

if TYPE_CHECKING:
    from outrider.github.auth import InstallationGitHubClient

__all__ = [
    "fetch_file_content_at",
    "list_pr_files",
]


# Per-file decoded content cap (1 MB) per the intake-and-webhook spec's
# pre-flight + per-file gate. GitHub's contents API itself caps at 1 MB
# decoded for the inline-base64 path; larger files return a different
# response shape (git_url for a Blob fetch) and we treat them as oversize.
_PER_FILE_CONTENT_CAP_BYTES = 1_000_000


async def list_pr_files(
    gh: InstallationGitHubClient,
    *,
    owner: str,
    repo: str,
    pull_number: int,
    per_page: int = 31,
) -> list[Any]:
    """Phase-1 of intake: return the per-file metadata list for a PR.

    Wraps `gh.rest.pulls.async_list_files(owner, repo, pull_number, per_page=...)`.
    Each returned entry has `filename`, `status`, `additions`, `deletions`,
    optionally `patch`, and (for renames) `previous_filename`. The intake
    node consumes this list to (a) run the size pre-flight, (b) dispatch
    the phase-2 content fan-out, (c) assemble the `ChangedFile` tuple.

    **Pagination strategy.** GitHub's `/pulls/{number}/files` is paginated
    (default `per_page=30`, max 100). Intake's size gate skips PRs with
    more than `_SIZE_GATE_MAX_FILES` files outright. The caller passes
    `per_page = _SIZE_GATE_MAX_FILES + 1`: if the response returns
    `_SIZE_GATE_MAX_FILES + 1` entries, the size gate fires without
    paginating further; if ≤ `_SIZE_GATE_MAX_FILES` are returned, the
    list is complete. Default of 31 here preserves the invariant for
    callers that don't pass an explicit value (defends against silent
    gate bypass if intake's constant drifts up without a corresponding
    `per_page` argument change).

    Owner and repo strings come from the validated webhook payload via
    `RepositoryRef.owner.login` and `RepositoryRef.name` directly
    (the canonical per-field source); `full_name` is informational
    only and is NOT used to derive these values. They are not
    path-validated here because they're URL-segments, not filesystem
    paths.
    """
    resp = await gh.rest.pulls.async_list_files(owner, repo, pull_number, per_page=per_page)
    return list(resp.parsed_data)


async def fetch_file_content_at(
    gh: InstallationGitHubClient,
    *,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> bytes | None:
    """Return validated inline-base64 file bytes at ref, or None when unusable.

    Validate the path before requesting content. Directories, non-base64 responses,
    malformed base64, and oversized payloads return None. CoordinateError and SDK
    request failures propagate; intake owns their skip or failure handling.
    """
    # Path validation is the security-critical pre-call gate. The
    # validator returns the normalized path; we use the normalized form
    # for the githubkit call so the API request can't carry traversal
    # or shell-metacharacter content even if a buggy caller constructed
    # the path from raw payload strings.
    safe_path = validate_diff_path(path)

    resp = await gh.rest.repos.async_get_content(
        owner,
        repo,
        safe_path,
        ref=ref,
    )
    data = resp.parsed_data

    # `async_get_content` returns a union: ContentFile (for a file),
    # list[ContentDirectoryItems] (for a directory), ContentSymlink,
    # ContentSubmodule. Only ContentFile carries inline content; the
    # others are out-of-scope for intake.
    if isinstance(data, list):
        # Directory listing — caller asked for a path that GitHub
        # considers a directory.
        return None

    # ContentFile / ContentSymlink / ContentSubmodule all have these
    # attributes but only ContentFile populates content + encoding.
    encoding = getattr(data, "encoding", None)
    content_b64 = getattr(data, "content", None)
    if encoding != "base64" or content_b64 is None:
        # ContentSymlink, ContentSubmodule, or the no-content shape that
        # GitHub returns for files larger than 1 MB (where `content` is
        # an empty string and `encoding` is "none").
        return None

    # Pre-decode size cap: base64 expands ~4/3, so a 1 MB decoded cap
    # bounds the encoded form to ~1.4 MB. A 2× safety margin
    # (2_000_000 bytes encoded) covers `\n`-padded GitHub responses while
    # rejecting pathological inputs BEFORE the decode allocation. This
    # is the load-bearing defense — a hostile or compromised upstream
    # returning a multi-MB `content` string would otherwise force a
    # multi-MB UTF-8 buffer to be allocated before the post-decode cap
    # at line below catches it.
    if len(content_b64) > _PER_FILE_CONTENT_CAP_BYTES * 2:
        return None

    # GitHub's contents API wraps inline base64 at ~60 chars with `\n`
    # separators. Strip them BEFORE strict validation — otherwise
    # `validate=True` raises `Only base64 data is allowed` on every real
    # response (the pre-decode size cap above explicitly accounts for
    # wrapping, so the two halves of the contract have to agree).
    # `\r` is stripped too for defense against CRLF intermediate proxies;
    # GitHub itself uses `\n` only.
    #
    # `validate=True` on the normalized form raises on non-base64 bytes
    # rather than silently stripping them. `validate=False` (the original
    # shape pre-1e59980) would have let an upstream returning
    # `<valid base64><garbage>=` succeed by discarding the garbage,
    # expanding our attack surface. Strip-newlines-then-strict-validate
    # preserves the security gate while accepting GitHub's wire format.
    # `binascii.Error → None` so caller treats as skip.
    normalized_b64 = content_b64.replace("\n", "").replace("\r", "")
    try:
        decoded = base64.b64decode(normalized_b64, validate=True)
    except binascii.Error:
        return None

    if len(decoded) > _PER_FILE_CONTENT_CAP_BYTES:
        # Belt-and-suspenders: even after the pre-decode gate, enforce
        # the cap on the decoded form. Caller maps to SkipReason.OVERSIZED.
        return None

    return decoded
