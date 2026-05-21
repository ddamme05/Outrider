# Triage-node prompt template + render helper per specs/2026-05-15-triage-node.md
"""Triage prompt template, version, knobs, and render helper.

The triage node uses a fast Haiku pass to classify each changed file into
a `ReviewTier` (DEEP / STANDARD / SKIM — never SKIP, that's the §6.10
policy-gate path), assess `overall_risk`, and select the applicable
`ReviewDimension`s. This module owns the prompt surface for that pass.

Surfaces (per the triage-node spec's Reference Reconciliation):

- `SYSTEM_PROMPT: Final[str]` — fully static instructions + JSON output
  schema. Goes into `LLMRequest.system_prompt` (cached per `cache_control:
  ephemeral` set by the wrapper); reused across the analyze ⇄ trace loop
  for cache hits.
- `USER_TEMPLATE: Final[str]` — per-PR `str.format`-style template with
  structural placeholders (`{pr_title}`, `{file_list}`, `{diff_summary}`).
  Values are filled at `render()` time; the placeholder names are template
  STRUCTURE — attacker-supplied content in PR fields cannot escape into
  the template (per `webhook-strings-are-data-not-format-strings`).
- `TEMPLATE` — alias for `USER_TEMPLATE`; this is the spec's named surface
  for the `str.format` target.
- `VERSION: Final[str] = "triage-v1"` — flows to `LLMRequest.prompt_template_version`.
- `MAX_TOKENS: Final[int] = 2048` — fits per-file tier classification +
  risk + dimensions + 500-char reasoning across the §6.10 size-cap upper
  bound (~30 files).
- `TEMPERATURE: Final[float] = 0.0` — deterministic-leaning; minimizes
  drift across replay.
- `TriagePromptParts` — frozen dataclass result of `render()`. Dataclass
  not NamedTuple because dataclasses don't subclass `tuple` → positional
  unpacking fails at runtime (the object isn't iterable). Swap-prone
  access patterns like `(system, user) = render(...)` parse and compile
  fine but raise `TypeError` on execution, so the swap can't ship
  silently.
- `render(pr_context: PRContext) -> TriagePromptParts` — pure function;
  builds the user prompt via `USER_TEMPLATE.format(**kwargs)` and pairs
  it with the constant `SYSTEM_PROMPT`.

Render is intentionally non-raising. All str-keyed placeholder names are
provided as keyword arguments at the format call, so `str.format` cannot
hit a missing key. None field values stringify to `"None"`; they do not
raise. The triage node's full post-start-emission failure matrix lives
in `agent.nodes.triage` (currently five sources: request construction,
provider call, schema validation, policy validation, and end-phase
emission); this module's contract is just that `render()` does not add
a sixth source.

V1.5+ extension hook: when multi-version prompt comparison surfaces,
`prompts/triage.py` becomes `prompts/triage/registry.py` (or similar) and
this module's flat constants graduate to registry entries. Until then,
flat constants keep the surface small.
"""

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
    """Render output: the (system, user) pair plus the cache-boundary contract.

    Dataclass (not NamedTuple) because dataclasses do NOT subclass tuple,
    so the swap-prone shape `(system, user) = render(...)` parses and
    compiles fine but raises `TypeError` at runtime when the iterator
    protocol fails. The swap cannot ship silently — attribute access
    (`parts.system_prompt`, `parts.user_prompt`) is the supported usage
    pattern; positional unpacking fails loud on the very first call.

    Both fields are str (validated downstream by `LLMRequest.system_prompt`
    and `.user_prompt` which carry `min_length=1`).
    """

    system_prompt: str
    user_prompt: str


def render(pr_context: PRContext) -> TriagePromptParts:
    """Build the (system, user) prompt pair for the triage LLM call.

    Pure function. Uses `USER_TEMPLATE.format(**kwargs)` with all
    placeholder names supplied as keyword arguments — missing-key
    `KeyError` cannot fire under correct programming. None field values
    stringify to "None"; they do not raise.

    Cache-boundary contract per DECISIONS#013 point 4 + spec §9.5: the
    returned `system_prompt` is fully static (a `Final` constant) and gets
    `cache_control: ephemeral` from the wrapper — reusing it across the
    analyze ⇄ trace loop produces cache hits. The `user_prompt` carries
    the volatile per-PR data (title, file list, diffs) and stays outside
    the cache boundary by design.

    Egress-eligible `PRContext` fields per DECISIONS#013 point 1
    ("Egress include list"):
      - `pr_title`              ✓ used here
      - `pr_body`                   not currently used; would be eligible
      - `changed_files[].path`  ✓ used (file list)
      - `changed_files[].patch` ✓ used (diff summary; = file content delta)
      - `changed_files[].status`/`.additions`/`.deletions` ✓ used (metadata)
      - `author`                    not currently used; eligible
      - branch names                not in PRContext today

    `PRContext` fields that are NOT egress-eligible per DECISIONS#013
    point 2 ("Egress exclude list") AND any future-refactor that
    enriches the prompt MUST keep these OUT of the LLM payload:
      - `installation_id`       — operational secret (auth scope)
      - `base_sha` / `head_sha` — commit identifiers; not explicitly
                                  excluded in #013 but not in the include
                                  list either; treat as metadata that
                                  belongs in audit rows, NOT in prompts
      - `owner` / `repo`        — repository coordinates; same reasoning
                                  as SHAs above

    If a future refactor needs additional fields, surface a DECISIONS#013
    supersession (per point 2's "If a future code path would send any
    excluded item, that's a bug *and* a #013 supersession").
    """
    file_list = "\n".join(
        f"- {cf.path} ({cf.status}, +{cf.additions}/-{cf.deletions})"
        for cf in pr_context.changed_files
    )
    diff_summary = "\n\n".join(
        _format_file_diff(cf.path, cf.patch) for cf in pr_context.changed_files
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
