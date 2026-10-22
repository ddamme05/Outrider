# Triage-node prompt template + render helper per specs/2026-05-15-triage-node.md
"""Render PR metadata and patches for tier/risk classification.

The configured triage model proposes DEEP/STANDARD/SKIM per file, overall risk
and review dimensions. SKIP belongs to the deterministic size gate (§6.10).
The triage node validates the response against the changed-file set.

Static instructions stay separate from per-PR data. PR strings fill values in
USER_TEMPLATE, never act as format targets, and patches use dynamic fences
(`webhook-strings-are-data-not-format-strings`)."""

from dataclasses import dataclass
from typing import Final

from outrider.schemas.pr_context import PRContext

VERSION: Final[str] = "triage-v1"
MAX_TOKENS: Final[int] = 2048
TEMPERATURE: Final[float] = 0.0


SYSTEM_PROMPT: Final[str] = """\
You are a fast triage classifier for an automated PR-review agent.

Your job: classify each changed file into a review tier, assess the
overall PR risk, and select which review dimensions apply. You do NOT
review the code itself — that's a downstream node. You decide WHERE the
deeper review should focus.

## Review tiers

- "deep" — files with security-sensitive patterns (auth, crypto, input
  parsing), complex logic changes, business-critical paths, database
  migrations, or large diffs (>200 changed lines in one file).
- "standard" — typical application code changes that benefit from review
  but don't warrant the full deep-context pipeline.
- "skim" — config files, dependency updates, formatting-only changes,
  auto-generated code, or simple comment/docstring edits. If a file
  looks unreviewable (lockfiles, generated bindings, vendored deps,
  binary diffs marked "[no textual diff available]"), classify as
  "skim" — never omit and never "skip".

Never produce "skip" — that tier is reserved for a deterministic
size-cap policy gate upstream of this node. Every file you receive needs
a deep, standard, or skim classification. If you cannot determine an
appropriate tier for a file with confidence, default to "standard"
rather than omitting it.

## Overall risk

Classify the PR as a whole on a four-rung ladder:

- "low" — small, isolated changes; tests-only or docs-only PRs; trivial
  refactors.
- "medium" — typical feature work or bug fixes; multiple files; some
  business logic.
- "high" — auth/permissions changes, crypto, database migrations, large
  refactors crossing trust boundaries, or anything touching financial
  flows.
- "critical" — security fixes, vulnerability patches, secrets handling,
  or changes to safety-critical paths.

## Review dimensions

Select the subset of dimensions that should be examined for THIS PR:

- "code_quality" — readability, maintainability, structural concerns.
- "security" — auth, crypto, input validation, secrets, attack surface.
- "performance" — algorithmic complexity, hot-path changes, resource use.
- "test_coverage" — missing or weak tests, regression risk.
- "best_practices" — language/framework idioms, project conventions.

A pure CSS change doesn't need a security review. A database migration
doesn't need a style review. Choose what's load-bearing.

## Output

Return exactly one JSON object and nothing else. Do NOT wrap the JSON
in markdown code fences (no ```json, no ```). Do NOT add explanatory
prose before or after. Output starts with `{` and ends with `}`. Every
value must be valid JSON literally — placeholders like `<...>` in this
example are illustrative and must be replaced with real values.

{
  "file_tiers": {
    "<path/from/changed_files>": "<deep|standard|skim>"
  },
  "overall_risk": "<low|medium|high|critical>",
  "relevant_dimensions": ["<dimension>"],
  "reasoning": "<two short sentences explaining the classification>"
}

`file_tiers` is an object keyed by each changed file's path; include
every changed file as its own key (the example shows one key — your
output should have one entry per file). `relevant_dimensions` is an
array of `code_quality` / `security` / `performance` / `test_coverage`
/ `best_practices` strings.

Keep "reasoning" to two short sentences (hard upper bound: 500 chars).
Every changed file MUST appear in file_tiers with a tier value. Do not
include paths not in the changed-files list. Do not produce "skip".
"""


USER_TEMPLATE: Final[str] = """\
PR title: {pr_title}

Changed files ({file_count} total, +{total_additions}/-{total_deletions} lines):
{file_list}

Diff summary:
{diff_summary}
"""


TEMPLATE: Final[str] = USER_TEMPLATE
"""Spec-named alias of USER_TEMPLATE. The spec lists `TEMPLATE` as the
public surface; USER_TEMPLATE is the locally-named twin for clarity at the
call site. They refer to the same string object."""


@dataclass(frozen=True, slots=True)
class TriagePromptParts:
    """Named system and user prompt fields; LLMRequest validates non-empty strings."""

    system_prompt: str
    user_prompt: str


def render(pr_context: PRContext) -> TriagePromptParts:
    """Render only the permitted PR metadata and diff fields.

    Static system instructions stay separate from the per-PR user prompt
    (DECISIONS#013 point 4, spec §9.5). The wrapper owns cache behavior.

    Egress include list (DECISIONS#013 point 1, Amended 2026-07-16): PR title;
    changed-file paths, patches, status and line counts; aggregate line counts.
    PR body and author are carried in PRContext but are not sent. Branch names and
    commit messages are not carried here. Installation ID, SHAs and repository
    coordinates remain out of the prompt (DECISIONS#013 point 2). Adding fields
    requires a privacy-contract amendment."""
    file_list = "\n".join(
        f"- {changed_file.path} ({changed_file.status}, "
        f"+{changed_file.additions}/-{changed_file.deletions})"
        for changed_file in pr_context.changed_files
    )
    diff_summary = "\n\n".join(
        _format_file_diff(changed_file.path, changed_file.patch)
        for changed_file in pr_context.changed_files
    )
    user_prompt = USER_TEMPLATE.format(
        pr_title=pr_context.pr_title,
        file_count=len(pr_context.changed_files),
        total_additions=pr_context.total_additions,
        total_deletions=pr_context.total_deletions,
        file_list=file_list,
        diff_summary=diff_summary,
    )
    return TriagePromptParts(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def _format_file_diff(path: str, patch: str | None) -> str:
    """Format one file's patch for the diff_summary section.

    `patch` is `str | None` because GitHub's `/pulls/{number}/files` omits
    the patch for binary diffs or oversized diffs (per ChangedFile.patch
    being Optional per DECISIONS#020 + R22). None renders as a
    "[no textual diff]" marker so the LLM doesn't see an empty file
    section without context.

    Wraps `patch` in a dynamic-length code fence via
    `prompts.safe_code_fence` because the diff text is PR-controlled —
    a `+## Overall risk: critical` line (or any other markdown-control
    content) inside the diff would otherwise forge a heading that
    mimics the triage prompt's own structure. See
    `webhook-strings-are-data-not-format-strings`.
    """
    from outrider.prompts import safe_code_fence

    if patch is None:
        return f"## {path}\n[no textual diff available]"
    return f"## {path}\n{safe_code_fence(patch, lang='diff')}"


__all__ = [
    "MAX_TOKENS",
    "SYSTEM_PROMPT",
    "TEMPERATURE",
    "TEMPLATE",
    "TriagePromptParts",
    "USER_TEMPLATE",
    "VERSION",
    "render",
]
