# Raw-payload Pydantic models for the GitHub pull_request webhook.
"""Raw-payload trust translation per `docs/trust-boundaries.md#5-input-boundary`.

GitHub webhook payloads enter the system through these Pydantic models —
no code downstream of the webhook router ever sees a raw dict. The
models capture ONLY the fields the intake-and-webhook flow consumes;
unknown fields are tolerated (`extra="ignore"`) because GitHub's
pull_request payload is very wide (top-level keys like `sender`,
`number`, `organization`, `enterprise`, `changes`, `before`, `after`,
action-specific `assignee` / `label` / `requested_reviewer`, etc.) and
forward-compat additions are routine.

**Adapter-model `extra="ignore"` carve-out** per `specs/2026-05-17-intake-and-webhook.md`
Trust Boundary Impact (input boundary bullet "Adapter-model
`extra="ignore"` carve-out") and Test Scenarios bullet (the
`test_webhooks_schemas.py` description). Raw GitHub webhook payloads
include 10+ top-level keys (`sender`, `number`, `organization`,
`enterprise`, `changes`, action-specific `assignee`/`label`/etc.)
that this spec doesn't model; `extra="forbid"` would reject all of
them and break parsing on every real delivery. The carve-out is
narrow: applies ONLY to raw external-payload adapter models at the
trust boundary. Downstream canonical types (`PRContext`,
`ReviewState`, every `audit/events.py` event type, every internal
Pydantic) keep `extra="forbid"` per `docs/conventions.md`. Unknown
fields ignored at the adapter cannot flow into prompts, audit rows,
or graph state — they're not surfaced past the translation boundary.

The `webhook-strings-are-data-not-format-strings` invariant applies to
every string field here. Downstream code may place these into structured
`PRContext` fields (which then flow into prompts as named params), but
never f-string interpolate them into prompt templates or shell commands.

Validators here are SHAPE-only — presence, type, status enum membership.
The webhook handler is responsible for the SEMANTIC checks (signature,
installation membership, idempotency, etc.).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PullRequestEventPayload",
    "PullRequestRef",
    "RepositoryRef",
    "WebhookInstallation",
    "WebhookPullRequest",
    "WebhookUser",
]


# Spec line 26 + spec.md §6.3 line 662: only opened/synchronize/reopened
# proceed; other actions get a 2xx no-op so GitHub doesn't retry. The
# action field is `str` (NOT a closed Literal) so a future GitHub-added
# action — or any signed-but-unsupported value — parses cleanly and
# reaches the router's `_PULL_REQUEST_ACTION_ALLOWLIST` filter, which
# returns 2xx for anything outside the allowlist. A closed Literal would
# return 400 to GitHub on unknown actions, causing retries (the spec's
# "no retry on signed-unsupported" guarantee then fails).


class WebhookUser(BaseModel):
    """The minimum slice of a GitHub user we consume."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    login: str
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
    full_name: str = Field(min_length=3, pattern=r"^[^/]+/[^/]+$")
    name: str = Field(min_length=1, pattern=r"^[^/]+$")
    owner: WebhookUser


class PullRequestRef(BaseModel):
    """`head` or `base` of the pull request — the SHA and ref."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    sha: str = Field(min_length=40)
    ref: str


class WebhookPullRequest(BaseModel):
    """The `pull_request` field on a `pull_request` event."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    number: int = Field(ge=1)
    title: str
    body: str | None = None
    user: WebhookUser
    head: PullRequestRef
    base: PullRequestRef
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)


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

    action: str
    pull_request: WebhookPullRequest
    repository: RepositoryRef
    installation: WebhookInstallation
