"""Build the request that prioritizes files in a pull request.

The model receives selected PR metadata and patch data. The triage node checks
its proposed attention levels; deterministic checks own any skip decision.
Untrusted metadata and code are fenced as data (webhook-strings-are-data-not-format-strings).
"""

from dataclasses import dataclass
from typing import Final

from outrider.prompts.triage_schema import file_slot_ids
from outrider.schemas.pr_context import PRContext

VERSION: Final[str] = "triage-v3"
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
  migrations, or large diffs (>200 changed lines in one file). These use
  the deeper analysis model and receive budget priority.
- "standard" — typical application code changes that benefit from review
  with the standard analysis model. DEEP and STANDARD use the same
  context selection and admission rules.
- "skim" — no analyze-stage review. Use only when code analysis is not
  needed, such as formatting-only changes, auto-generated code or simple
  comment/docstring edits. Configuration and dependency changes can affect
  security or behavior; use STANDARD or DEEP when those need review. If a file
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
    "file_1": "<deep|standard|skim>"
  },
  "overall_risk": "<low|medium|high|critical>",
  "relevant_dimensions": ["<dimension>"],
  "reasoning": "<two short sentences explaining the classification>"
}

`file_tiers` is keyed by the numbered file IDs shown in the input, such as
"file_1" and "file_2". Include every ID exactly once, using its assigned file.
Do not use filenames as keys. `relevant_dimensions` is an
array of `code_quality` / `security` / `performance` / `test_coverage`
/ `best_practices` strings.

Keep "reasoning" to two short sentences (hard upper bound: 500 chars).
Every file ID MUST appear in file_tiers with a tier value. Do not
invent IDs or produce "skip".
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
    slots = zip(file_slot_ids(len(pr_context.changed_files)), pr_context.changed_files, strict=True)
    file_list = "\n".join(
        f"- {slot}: {changed_file.path} ({changed_file.status}, "
        f"+{changed_file.additions}/-{changed_file.deletions})"
        for slot, changed_file in slots
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
