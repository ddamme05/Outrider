"""Read the GitHub payload fields Outrider uses.

These input models allow extra fields so GitHub can add payload data without
breaking delivery. Other state and audit models still reject unknown fields.
The router checks signatures, actions and access separately. GitHub strings
remain data, never commands or prompt templates
(webhook-strings-are-data-not-format-strings).

See specs/2026-05-17-intake-and-webhook.md and docs/trust-boundaries.md#5-input-boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "InstallationEventPayload",
    "InstallationRepositoriesEventPayload",
    "InstallationRepositoryRef",
    "PullRequestEventPayload",
    "PullRequestRef",
    "RepositoryRef",
    "WebhookAccount",
    "WebhookInstallation",
    "WebhookInstallationDetail",
    "WebhookPullRequest",
    "WebhookUser",
]


class WebhookUser(BaseModel):
    """A bounded GitHub user id and login for canonical PRContext fields.

    The login pattern permits ASCII letters, digits, and hyphens. It deliberately
    does not enforce GitHub's leading/consecutive-hyphen restrictions; it excludes
    slashes, control characters, and shell metacharacters at the input boundary.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    # rust-regex has no lookaheads for GitHub's hyphen-placement rules. This
    # bounded ASCII subset still excludes slashes, controls and shell metacharacters.
    login: str = Field(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")
    id: int = Field(ge=1)


class WebhookInstallation(BaseModel):
    """`installation` field on every App-delivered webhook.

    The numeric `id` is what the webhook handler uses to look up the
    `installations` row (active-membership check before review insert).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)


class RepositoryRef(BaseModel):
    """Repository identification.

    `id` is the canonical numeric repo id used in idempotency
    (`(repo_id, pr_number, head_sha)` triple). `owner.login` is the
    canonical source for the owner string when constructing
    `PRContext.owner` downstream; `name` is the canonical source for the
    repo-name string when constructing `PRContext.repo`.

    `full_name` is preserved as an informational field (it appears in
    logs and audit messages where the combined `"owner/repo"` form is
    operator-readable shorthand). It's validated to have exactly one
    slash and two non-empty halves — GitHub canonically issues
    `owner/repo`. Rejecting other shapes at the input boundary prevents
    downstream code from constructing requests with empty owner or
    empty repo segments. Note: `full_name` is NOT used to DERIVE
    `owner`/`repo` at PRContext construction; per the canonical-source
    discipline, those come from `owner.login` and `name` directly.
    `name` carries the same character-class restrictions as
    `owner.login` (no slashes; GitHub disallows slashes in repo names).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)
    # Bound both URL segments to repository-name characters. The 200-char cap
    # covers a 39-char login, slash and 100-char repo name with margin.
    full_name: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$",
    )
    # Exclude slashes, whitespace and shell metacharacters before URL/prompt use.
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._-]+$")
    owner: WebhookUser


class PullRequestRef(BaseModel):
    """`head` or `base` of the pull request — the SHA and ref.

    `sha` is exactly 40 hex (SHA-1, GitHub's default) OR exactly 64 hex
    (SHA-256, GitHub's object-format migration on some surfaces). The
    alternation pattern rejects impossible intermediate lengths
    (41-63 chars, 65+) that a range-bound `min_length=40 max_length=64`
    would admit. The trust-boundary identity guarantee is that
    `(repo_id, pr_number, head_sha)` uniquely identifies a review.

    `ref` validation has two layers because Pydantic v2's rust-regex
    doesn't support negative lookaheads / lookbehinds:
      1. **Field pattern** — character class + bounded length. Admits
         `git check-ref-format`'s allowed chars (alphanumeric + `.`
         `_` `/` `-` `+`) and rejects shell-meta, control, whitespace.
      2. **field_validator** — structural rules the pattern can't
         express: no leading/trailing `/`, no `//` empty segments, no
         `..` traversal segment, no segment ending in `.lock` or `.`.
         Mirrors `git check-ref-format`'s rule subset that's
         enforceable without lookaround.

    Both layers are load-bearing — pattern alone admits `../head`,
    `foo/../bar`, `a//b`, `/lead`, `trail/` (chars all valid; structure
    pathological). validator alone would admit shell-meta chars
    (structure clean; chars hostile).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    sha: str = Field(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")
    ref: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._/\-+]+$")

    @field_validator("ref")
    @classmethod
    def _validate_ref_structure(cls, value: str) -> str:
        """Structural rules from `git check-ref-format` that Pydantic
        v2's rust-regex can't express. The Field pattern handles
        character class + length; this handles segment shape.
        """
        if value.startswith("/"):
            msg = f"ref {value!r} cannot start with '/'"
            raise ValueError(msg)
        if value.endswith("/"):
            msg = f"ref {value!r} cannot end with '/'"
            raise ValueError(msg)
        segments = value.split("/")
        for seg in segments:
            if seg == "":
                msg = f"ref {value!r} contains empty segment ('//' or boundary slash)"
                raise ValueError(msg)
            # `..` is a special case of "starts with '.'"; check it
            # FIRST so the more specific error message wins for the
            # most common traversal attempt.
            if seg == "..":
                msg = f"ref {value!r} contains '..' traversal segment"
                raise ValueError(msg)
            if ".." in seg:
                # Embedded `..` (e.g., `feature..x`, `a..b`) — the
                # `seg == ".."` check above catches the bare form,
                # `startswith(".")` below catches leading-dot. Embedded
                # `..` between non-dot chars slips both. git-check-ref-format
                # rejects ANY `..` in a ref name regardless of position.
                msg = f"ref {value!r} segment {seg!r} contains '..'"
                raise ValueError(msg)
            if seg.startswith("."):
                # git-check-ref-format rejects components starting with
                # `.` — covers `.hidden`, `release/.tmp`. Without this
                # rule, dot-prefixed segments cross the trust boundary
                # into fetch / state / audit.
                msg = f"ref {value!r} contains segment starting with '.'"
                raise ValueError(msg)
            if seg.endswith(".lock"):
                msg = f"ref {value!r} segment ends with '.lock' (git-reserved)"
                raise ValueError(msg)
            if seg.endswith("."):
                msg = f"ref {value!r} segment ends with '.'"
                raise ValueError(msg)
        return value


class WebhookPullRequest(BaseModel):
    """The `pull_request` field on a `pull_request` event.

    `title` and `body` are bounded at the input boundary. Without caps, a
    pathological PR with a multi-MB title floods the audit-table `payload`
    JSONB, log lines (via `extra={"author": ...}`-style structured logs),
    AND the LLM prompt (`PRContext.pr_title` / `pr_body` flow into the
    triage and analyze prompts). GitHub's own server-side limits are
    looser than what's reasonable for a review-tool; defensively pin at
    the boundary. The values 4096 (title) and 65536 (body) are comfortable
    margins above typical PR shapes while bounding worst-case cost.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    number: int = Field(ge=1)
    title: str = Field(max_length=4096)
    body: str | None = Field(default=None, max_length=65536)
    user: WebhookUser
    head: PullRequestRef
    base: PullRequestRef
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    # Arc B2 autorun: draft PRs are skipped (2xx no-op); the review runs when the
    # `ready_for_review` action fires (draft → ready). REQUIRED (no default): GitHub always
    # sends `draft` on pull_request events, so a payload missing it is malformed — the router
    # rejects it (ValidationError → 400 "invalid payload shape", router.py) rather than
    # defaulting to ready and autorunning what may be a draft. Fail-closed.
    draft: bool


class PullRequestEventPayload(BaseModel):
    """Top-level shape of a `pull_request` event after signature verification.

    `extra="ignore"` (not `forbid`): GitHub's `pull_request` payload is
    very wide and includes many fields we don't consume; rejecting
    unknown top-level keys would couple us to GitHub's payload version
    pinning. `frozen=True` keeps the model immutable downstream of
    parsing.

    The `installation` field is required because Outrider's flow is
    GitHub-App-only — a PAT-driven delivery would arrive without
    `installation`; the absence is a fail-loud parse error, which is
    the desired shape (we don't process those).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Keep unknown actions parseable; the router's current allowlist decides which run.
    action: str
    pull_request: WebhookPullRequest
    repository: RepositoryRef
    installation: WebhookInstallation


# ---------------------------------------------------------------------------
# Installation-lifecycle events (DECISIONS.md#065)
#
# Under #065 the local install tables are a CACHE, not the authorization
# authority (GitHub is checked live at intake/publish). These events maintain
# cache / retention / display state via simple upserts — never a security
# decision. Keep action as str and allowlist it in the handler: unknown actions
# receive a 2xx no-op. extra="ignore" tolerates GitHub's full payload; only the
# cache's consumed fields are modeled here.
# ---------------------------------------------------------------------------


class WebhookAccount(BaseModel):
    """The `installation.account` object (the user or org the App is installed on).

    Carries the fields the `installations` cache upserts on `installation.created`:
    `account_id` (`id`), `account_login` (`login`), `account_type` (`type`). `login`
    reuses the GitHub-username character class (input-boundary defense, same as
    `WebhookUser.login`); `type` is a bounded string (`User` / `Organization` / `Bot`),
    persisted as display state only.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)
    login: str = Field(min_length=1, max_length=39, pattern=r"^[A-Za-z0-9-]+$")
    type: str = Field(min_length=1, max_length=20)


class InstallationRepositoryRef(BaseModel):
    """A repository entry in an install-event `repositories` / `repositories_added` /
    `repositories_removed` array.

    GitHub sends a MINIMAL repo shape in these arrays — `{id, node_id, name, full_name,
    private}`, with NO nested `owner` object — so this is a lighter model than
    `RepositoryRef` (which requires `owner`). Only `id` (-> `installation_repositories.repo_id`)
    and `full_name` (-> `repo_full_name`) are consumed by the membership-cache upsert; the
    `full_name` pattern is the same input-boundary defense as `RepositoryRef.full_name`.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)
    full_name: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$",
    )


class WebhookInstallationDetail(BaseModel):
    """The full `installation` object on install-lifecycle events (`installation`,
    `installation_repositories`).

    Distinct from `WebhookInstallation` (id-only, consumed by review-scoped events for the
    membership lookup): this carries the account / permission / selection fields the
    `installation.created` upsert writes into the `installations` cache. `suspended_at`
    mirrors GitHub's field (null = active); `repository_selection` is `all` | `selected`
    (bounded str, persisted as-is — the cache is not the authority under #065).
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int = Field(ge=1)
    account: WebhookAccount
    app_slug: str = Field(min_length=1, max_length=100)
    repository_selection: str = Field(min_length=1, max_length=20)
    permissions: dict[str, Any] = Field(default_factory=dict)
    suspended_at: AwareDatetime | None = None


class InstallationEventPayload(BaseModel):
    """`installation` webhook event — App lifecycle (`created` / `deleted` / `suspend` /
    `unsuspend`).

    The handler allowlists actions and ignores unknown ones with a 2xx response.
    repositories seeds membership for selected installs on created; it defaults
    to empty for other actions. installation is required for this App-only flow.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    action: str
    installation: WebhookInstallationDetail
    repositories: tuple[InstallationRepositoryRef, ...] = ()


class InstallationRepositoriesEventPayload(BaseModel):
    """`installation_repositories` webhook event — repo-set changes (`added` / `removed`).

    Every event re-persists `repository_selection` on the install. `repositories_added` /
    `repositories_removed` carry the deltas; the handler upserts membership with re-add
    semantics (`removed_at = NULL` on add, `now()` on remove). Under #065 this
    maintains cache hints; intake's live GitHub check decides repository access.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    action: str
    installation: WebhookInstallationDetail
    repository_selection: str = Field(min_length=1, max_length=20)
    repositories_added: tuple[InstallationRepositoryRef, ...] = ()
    repositories_removed: tuple[InstallationRepositoryRef, ...] = ()
