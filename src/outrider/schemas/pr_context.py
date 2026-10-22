# PRContext + ChangedFile cross-boundary models per docs/spec.md §7.2
# See DECISIONS.md#020 — Webhook receiver constructs seed PRContext; intake enriches
"""PR metadata and enriched changed files shared by the review pipeline.

The webhook seeds PRContext; intake replaces it with fetched ChangedFile
content; triage classifies files; analyze reads source; publish uses PR
coordinates. Fixed templates render these fields as data, never as format
strings (docs/trust-boundaries.md #5 sub-rule 2). Triage sends title and
file metadata/patches; analyze sends selected source. pr_body and author
are retained but not prompted (#013 Amended 2026-07-16).

installation_id permits synthetic eval IDs; webhook validation owns live
GitHub ID semantics. Frozen models and tuple changed_files prevent
in-place mutation across checkpoints.

A missing patch means GitHub supplied no textual diff, not an empty diff.
Present GitHub patches are hunks-only; coordinates.diff_parser adds the
headers needed by unidiff. ChangedFile validates post-intake content sides,
counts, and rename paths per `DECISIONS.md#020`."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChangedFile(BaseModel):
    """One post-intake file, with status-specific source and rename fields.

    Per `DECISIONS.md#020`, added has head only and no deletions; removed
    has base only and no additions; modified/renamed have both sources.
    Only renamed carries previous_path, and it must differ from path.
    intake reads that path from GitHub's previous_filename. The validator
    checks these rules for every construction path."""

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
        """Validate source sides, counts, and rename paths after intake.

        These are the `DECISIONS.md#020` status-conditional contract."""
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
    """Immutable metadata and changed-file tuple for one PR review.

    Tuple containers prevent mutation that frozen=True alone cannot block."""

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
