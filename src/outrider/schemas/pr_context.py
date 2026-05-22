# PRContext + ChangedFile cross-boundary models per docs/spec.md §7.2
# See DECISIONS.md#020 — Webhook receiver constructs seed PRContext; intake enriches
"""PRContext + ChangedFile: structured carriers of GitHub PR data.

PRContext is the typed envelope that crosses every subsystem boundary in the
review pipeline. The webhook receiver constructs it from raw GitHub payload
data (separate spec); intake fetches/populates ChangedFile content using a
short-lived API token minted from `installation_id`; triage classifies its
files into ReviewTiers; analyze consumes the file content; publish references
the original PR coordinates. The structured-field shape is the implementation
of `docs/trust-boundaries.md` #5 sub-rule 2 — webhook payload strings (branch
names, PR titles, commit messages) cross into LLM prompts as PRContext fields,
never as f-string interpolation.

The `installation_id` field was added 2026-05-08 to close a canonical-vs-impl
drift: spec §15.2 build_graph snippet calls `state.pr_context.installation_id`
(line 1440) but the canonical §7.2 PRContext shape did not include the field.
Resolved by canonical amendment + implementation in this commit.

`installation_id` is plain `int` (no `Field(ge=1)`) per the eval-isolation
convention in `docs/schema.md` — eval factories use synthetic non-colliding
IDs that may include negative values. Production webhook validation enforces
real GitHub installation-ID semantics at the input boundary (webhook-receiver
spec); this shared schema supports both production and eval contexts.

Both models use frozen=True: PRContext round-trips through LangGraph state
JSON on every checkpoint; immutability prevents mid-graph mutation by any
node and guarantees the value at HITL-resume matches the value at intake.
ChangedFile follows the same rationale — its patch and content fields are
the audit-trail anchor for every finding's coordinate translation, and an
in-place mutation would break replay equivalence.

ChangedFile.status uses Literal["added", "modified", "removed", "renamed"]
per spec §7.2 verbatim. content_base / content_head are Optional because the
side-that-doesn't-exist is structurally None for `added` (no base) and
`removed` (no head). `patch` is Optional because GitHub's
`/pulls/{number}/files` API omits the `patch` property for binary diffs
and for diffs too large for the API to include — applies to any status.
Downstream code that consumes `patch` must treat `None` as "no textual
diff available from GitHub" (not as an empty unified diff).

The wire format when present is GitHub's hunks-only shape — the value
begins with one or more ``@@ ...`` hunk headers and contains NO leading
``--- a/...`` / ``+++ b/...`` file headers and NO ``diff --git`` line.
Consumers that pass this value to a unified-diff parser (e.g. `unidiff`'s
`PatchSet`) must synthesize file headers first; see
`coordinates.diff_parser._wrap_github_hunks_with_headers` for the
canonical wrap used by `file_in_patch` and `lookup_patched_file`. The carrier's
Optional typing alone admits malformed shapes (e.g., `added` with both
content sides set, `modified` with both None); the post-intake
`enforce_status_invariants` model_validator rejects all
GitHub-API-impossible shapes. Four invariant classes are enforced:

- Status↔content : the side-that-doesn't-exist is None;
  the side-that-does-exist is non-None.
- Status↔count : `added` requires deletions=0; `removed`
  requires additions=0 (GitHub-impossible otherwise).
- Rename path-shape : `status="renamed"` requires
  `previous_path` set AND `previous_path != path` (a same-path
  rename is GitHub-impossible).
- Cross-status (Rounds 14, 16): non-`renamed` statuses must NOT
  carry `previous_path`.
"""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChangedFile(BaseModel):
    """One file in a PR's changed-files list.

    Frozen: see module docstring. status is a Literal-typed string, not an
    enum, per spec §7.2 verbatim — GitHub's API uses the string values
    directly and Literal tracks the API contract more transparently than
    a parallel enum would.

    Per `DECISIONS.md#020`, `ChangedFile` instances are constructed by
    intake AFTER fetching base/head content. `enforce_status_invariants`
    rejects every GitHub-API-impossible shape so a buggy intake or
    fixture can't silently produce a malformed instance:

    Status↔content :
    - `added`     → `content_head` set, `content_base` None
    - `removed`   → `content_base` set, `content_head` None
    - `modified`  → both `content_base` and `content_head` set
    - `renamed`   → both set, plus `previous_path` set to the old path

    Status↔count :
    - `added`     → `deletions == 0` (added file has no pre-existing
                    content to delete from)
    - `removed`   → `additions == 0` (removed file has nothing being
                    added)

    Rename path-shape :
    - `renamed`   → `previous_path != path` (a same-path rename is
                    GitHub-impossible — that would be modified or
                    no change)

    Cross-status (Rounds 14, 16):
    - non-`renamed` → must NOT carry `previous_path`

    `previous_path` is None for non-rename statuses; for `renamed`,
    intake reads the value from GitHub's `/pulls/{number}/files`
    response field `previous_filename`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    status: Literal["added", "modified", "removed", "renamed"]
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str | None = None
    content_base: str | None = None
    content_head: str | None = None
    language: str | None = None
    previous_path: str | None = None

    @model_validator(mode="after")
    def enforce_status_invariants(self) -> Self:
        """Status-conditional invariants per `DECISIONS.md#020` post-intake
        contract — covers content-side completeness (R14), count-side
        impossibility (R15), and rename path-shape (R16). GitHub's API
        cannot produce any of the rejected shapes; admitting them at the
        schema level would let a buggy fixture or upstream silently slip
        a malformed instance past the post-intake contract.

        Renamed from `enforce_status_content` (R18) to reflect the wider
        scope after R15/R16 added count and rename-path-shape checks."""
        if self.status == "added":
            if self.content_head is None:
                raise ValueError("status='added' requires content_head to be non-None")
            if self.content_base is not None:
                raise ValueError("status='added' requires content_base to be None")
            if self.deletions != 0:
                raise ValueError(
                    f"status='added' requires deletions=0 (got {self.deletions}); "
                    "an added file has no pre-existing content to delete from"
                )
        elif self.status == "removed":
            if self.content_base is None:
                raise ValueError("status='removed' requires content_base to be non-None")
            if self.content_head is not None:
                raise ValueError("status='removed' requires content_head to be None")
            if self.additions != 0:
                raise ValueError(
                    f"status='removed' requires additions=0 (got {self.additions}); "
                    "a removed file has nothing being added"
                )
        elif self.status == "modified":
            if self.content_base is None or self.content_head is None:
                raise ValueError(
                    "status='modified' requires both content_base and content_head to be non-None"
                )
        elif self.status == "renamed":
            if self.content_base is None or self.content_head is None:
                raise ValueError(
                    "status='renamed' requires both content_base and content_head to be non-None"
                )
            if self.previous_path is None:
                raise ValueError(
                    "status='renamed' requires previous_path (the pre-rename path); "
                    "GitHub's /pulls/{number}/files API returns this as `previous_filename`"
                )
            if self.previous_path == self.path:
                raise ValueError(
                    f"status='renamed' requires previous_path != path "
                    f"(got {self.path!r} for both); a same-path rename "
                    "is GitHub-impossible — that would be modified or no change"
                )
        # Non-rename statuses must NOT carry previous_path
        if self.status != "renamed" and self.previous_path is not None:
            raise ValueError(
                f"status={self.status!r} must not carry previous_path "
                "(previous_path is renamed-status-specific)"
            )
        return self


class PRContext(BaseModel):
    """Everything the agent needs to review a PR.

    Frozen: see module docstring. `changed_files` is `tuple[ChangedFile, ...]`
    not `list[ChangedFile]` because `frozen=True` is faux-immutable over
    `.append()` / `.clear()` on a list field — same convention as
    HITLDecision.decisions (spec §7.4 line 290 + hitl.py module docstring) and
    ReviewFinding.trace_path (review_finding.py module docstring). Tuple
    delivers what frozen=True is meant to deliver. Spec.md §7.2 was amended
    same-day (2026-05-08) to match this commitment.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    installation_id: int
    owner: str
    repo: str
    pr_number: int = Field(ge=1)
    pr_title: str
    pr_body: str | None = None
    base_sha: str
    head_sha: str
    author: str
    changed_files: tuple[ChangedFile, ...]
    total_additions: int = Field(ge=0)
    total_deletions: int = Field(ge=0)


__all__ = [
    "ChangedFile",
    "PRContext",
]
