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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
    """The minimum slice of a GitHub user we consume.

    `login` is bounded with the same character-class GitHub itself enforces
    server-side: 1-39 chars, ASCII alphanumeric plus hyphen, no leading
    hyphen, no consecutive hyphens. Defending here AT the input boundary
    closes the gap where a forged-or-replayed payload could embed empty /
    slashed / control-character logins into `PRContext.author` (which
    reaches LLM prompts) and into the URL segments of the
    `repos/{owner}/{repo}/...` API calls intake builds.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Pattern is ASCII alphanumeric + hyphen, length 1-39 (GitHub's own
    # cap). The fully-correct GitHub-username rule also forbids leading
    # hyphens and consecutive hyphens, but Pydantic's regex engine
    # (rust-regex) doesn't support lookaheads. The looser pattern still
    # closes the input-boundary attack surfaces this commit targets:
    # empty, slashed, control-char, shell-metachar, and unbounded-length
    # logins. Leading-hyphen / consecutive-hyphen would 404 at the GitHub
    # API anyway; not a security-relevant gap.
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
    # full_name = "<owner>/<name>". Both halves use the GitHub repo /
    # owner character class (alphanumeric + `.` `_` `-`) — the prior
    # `[^/]+` was too permissive, admitting whitespace, control chars,
    # and shell metachars that flow into URL segments and audit
    # rendering. Max length 200 covers the worst case (39-char login
    # + "/" + 100-char repo-name + small margin).
    full_name: str = Field(
        min_length=3,
        max_length=200,
        pattern=r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$",
    )
    # GitHub caps repo names at 100 chars server-side. Character class
    # matches what GitHub permits: alphanumeric + `.` `_` `-`. The
    # prior `[^/]+` admitted whitespace / control / shell-meta which
    # would flow into URL segments (`repos/{owner}/{name}/...`) and
    # prompts.
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
