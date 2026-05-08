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
spec); this shared schema supports both production and eval contexts. The
constraint was tried-and-removed in Round 7 after an external reviewer
flagged the eval-factory conflict.

Both models use frozen=True: PRContext round-trips through LangGraph state
JSON on every checkpoint; immutability prevents mid-graph mutation by any
node and guarantees the value at HITL-resume matches the value at intake.
ChangedFile follows the same rationale — its patch and content fields are
the audit-trail anchor for every finding's coordinate translation, and an
in-place mutation would break replay equivalence.

ChangedFile.status uses Literal["added", "modified", "removed", "renamed"]
per spec §7.2 verbatim. content_base / content_head are Optional because
"added" files have no base content and "removed" files have no head content;
the consuming nodes (analyze, coordinates) handle the None case explicitly.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChangedFile(BaseModel):
    """One file in a PR's changed-files list.

    Frozen: see module docstring. status is a Literal-typed string, not an
    enum, per spec §7.2 verbatim — GitHub's API uses the string values
    directly and Literal tracks the API contract more transparently than
    a parallel enum would.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    status: Literal["added", "modified", "removed", "renamed"]
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    patch: str
    content_base: str | None = None
    content_head: str | None = None
    language: str | None = None


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
    pr_body: str
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
