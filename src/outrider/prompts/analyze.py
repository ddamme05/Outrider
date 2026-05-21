# See specs/2026-05-19-analyze-node.md §5
"""Analyze prompt template, version, knobs, and render helpers.

The analyze node runs one Sonnet call per eligible file. Prompts split
into:

- **System prompt** (cacheable): Outrider-wide invariants (output schema,
  `FindingType` enum, `EvidenceTier` proof rules, severity-set-by-policy
  and confidence-is-computed reminders) PLUS file-scoped context
  (changed scope units with bodies + same-file callers/callees + imports
  + decorators + pre-fired `query_match_id` set). Stable for one file
  across the analyze ⇄ trace loop, so the provider's
  `cache_control: ephemeral` produces cross-pass cache hits.
- **User prompt** (volatile): pass-specific instruction + scope-unit-
  clipped diff hunks. Outside the cache boundary.

For degraded calls (parse failure or `has_error` nodes intersecting
changed regions), the prompt swaps to a `judged`-only directive set;
the registry/walk context is empty by construction, so the system
prompt is shorter and the user prompt carries bounded changed hunks
instead of scope-unit-clipped ones.

Surfaces:

- `SYSTEM_PROMPT_INVARIANTS` — fully static head of the system prompt.
- `SYSTEM_FILE_CONTEXT_TEMPLATE` — file-scoped tail appended by `render`.
- `USER_TEMPLATE` — pass directives + diff hunks for clean calls.
- `DEGRADED_USER_TEMPLATE` — directives + bounded hunks for degraded calls
  (admits only `evidence_tier="judged"`).
- `TEMPLATE = USER_TEMPLATE` — spec-named alias.
- `VERSION = "analyze-v1"` — flows to `LLMRequest.prompt_template_version`.
  Bump on any template change.
- `MAX_TOKENS = 8192` — fits up to ~50 findings per response.
- `TEMPERATURE = 0.0` — deterministic-leaning; minimizes replay drift.
- `AnalyzePromptParts` — frozen dataclass result. NOT a NamedTuple, so
  positional unpacking `(sys, usr) = render(...)` fails loud rather
  than silently masking a field swap.
- `render` / `render_degraded` — build the (system, user) pair.

Per `webhook-strings-are-data-not-format-strings`: PR-sourced content
enters via `str.format(**kwargs)` against structural placeholders;
attacker-controlled content cannot escape the template structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

VERSION: Final[str] = "analyze-v1"
MAX_TOKENS: Final[int] = 8192
TEMPERATURE: Final[float] = 0.0


SYSTEM_PROMPT_INVARIANTS: Final[str] = """\
You are an automated code-review agent analyzing one file at a time
from a pull request. A deterministic pipeline takes your structured
output, applies a proof-boundary gate, looks up severity from a
policy table, and routes findings to the human reviewer.

## Your role

You IDENTIFY candidate findings. You do NOT:
- propose severity — that is set by a deterministic policy table keyed
  on `finding_type`. Any `severity` field in your output is rejected.
- propose confidence — that is computed from `evidence_tier`. Any
  `confidence` field in your output is rejected.
- propose dimension — that is looked up from `finding_type`. Any
  `dimension` field in your output is rejected.

The system rejects outputs that include those fields. Don't include them.

## FindingType enum

Pick exactly one value for `finding_type`. A value outside this enum
causes the proposal to be rejected with audit reason
"finding_type_not_in_enum".

- Security: `sql_injection`, `xss`, `hardcoded_secret`, `auth_bypass`,
  `path_traversal`, `missing_input_validation`
- Performance: `n_plus_one_query`, `blocking_call_in_async`
- Code quality: `unused_import`, `missing_error_handling`
- Test coverage: `missing_test`
- Best practices: `deprecated_api`

## Evidence tier (proof rules)

Pick exactly one value for `evidence_tier`. V1 admits two tiers:

- `observed` — a tree-sitter query in our registry matched a structural
  pattern. You MUST cite a real `query_match_id` from the pre-supplied
  registry set below; a fabricated id causes rejection with reason
  "query_match_id_not_in_registry".
- `judged` — your own interpretation; no structural artifact required.
  Use this when you cannot cite a registry query match.

Do NOT emit `evidence_tier="inferred"` in V1. The trace resolver lands
in a future spec; until then, every `inferred` proposal is rejected
with `trace_path_not_admissible`. Pick `judged` for cross-file or
walk-derived reasoning.

Failed admission DROPS the proposal — it does not downgrade to a lower
tier. Pick `judged` upfront if you cannot cite structural evidence.

## Output shape

Return exactly one JSON object, nothing else. No markdown fences, no
prose before or after. Output starts with `{` and ends with `}`. Every
value must be valid JSON literally (`null`, a string, a number, an
array, or another object) — placeholders like `<...>` in this example
are illustrative and must be replaced with real values.

{
  "findings": [
    {
      "finding_type": "<enum value>",
      "evidence_tier": "<observed|judged>",
      "query_match_id": "<id from registry, or null>",
      "trace_path": null,
      "title": "<short summary, ≤120 chars>",
      "description": "<explanation, ≤1000 chars>",
      "evidence": "<verbatim quote from the code, ≤2000 chars>",
      "span": {"byte_start": 0, "byte_end": 1},
      "trace_candidates": [
        {"candidate_path_raw": "<repo-relative path>", "reason": "<text>"}
      ]
    }
  ]
}

Field semantics:
- `query_match_id`: a string id from the registry above when
  `evidence_tier="observed"`; `null` otherwise.
- `trace_path`: always `null` in V1 (the `inferred` tier that consumes
  it lands with the trace-node spec).
- `span.byte_start` / `span.byte_end`: integer UTF-8 byte offsets into
  the file. `byte_start` must be less than `byte_end`.
- `trace_candidates`: an array (possibly empty) of `{candidate_path_raw,
  reason}` objects. The field name is `candidate_path_raw` — the
  parser canonicalizes it to `candidate_path` after admission.

Up to 50 findings per response (`AnalyzeResponseRaw.findings` is bounded
at max_length=50). Up to 20 trace_candidates per finding.
"""


SYSTEM_FILE_CONTEXT_TEMPLATE: Final[str] = """\

## File under review

File: {file_path}

## Scope-unit context

The file's changed scope units (functions, classes, methods) and their
same-file context (callers/callees, imports, decorators) are listed
below. Findings should land within the byte ranges of these units.

{scope_unit_context}

## Pre-fired query matches

Use these `query_match_id` values when claiming `evidence_tier="observed"`:

{query_match_id_list}
"""


USER_TEMPLATE: Final[str] = """\
Pass: analyze-pass-{pass_index}

## Changed diff (scope-unit-clipped)

The unified-diff hunks below are clipped to the included scope units.
The full file is NOT in this prompt; only changed regions reach you.

{diff_hunks}
"""


DEGRADED_USER_TEMPLATE: Final[str] = """\
File: {file_path}
Pass: analyze-pass-{pass_index}
Mode: DEGRADED ({degradation_reason})

This file could not be parsed structurally (or has tree-sitter errors
intersecting the changed regions). The pre-fired query-match registry
and import/call walks are unavailable for this call.

You MAY emit findings only with `evidence_tier="judged"`. Any `observed`
or `inferred` claims will be rejected.

## Bounded changed hunks

The diff hunks below are bounded (max 100 unidiff Line objects total,
max 8192 chars of text) to cap the degraded-path cost.

{bounded_hunks}
"""


TEMPLATE: Final[str] = USER_TEMPLATE
"""Spec-named alias of USER_TEMPLATE. Same string object."""


@dataclass(frozen=True, slots=True)
class AnalyzePromptParts:
    """Render output: the (system, user) pair for one analyze LLM call.

    Dataclass, not NamedTuple — positional unpacking
    `(system, user) = render(...)` raises `TypeError` at runtime rather
    than silently masking a field swap. Use attribute access:
    `parts.system_prompt`, `parts.user_prompt`.
    """

    system_prompt: str
    user_prompt: str


def render(
    *,
    file_path: str,
    scope_unit_context: str,
    query_match_id_list: str,
    diff_hunks: str,
    pass_index: int,
) -> AnalyzePromptParts:
    """Build the (system, user) prompt pair for a clean-outcome call.

    `system_prompt` carries stable-per-file content (invariants +
    file-scoped scope-unit/query context); the provider's
    `cache_control: ephemeral` produces cross-pass cache hits. The
    `user_prompt` carries pass-specific volatile content (pass index +
    scope-unit-clipped diff hunks).

    Wraps `diff_hunks` in a dynamic-length `diff`-fence via
    `safe_code_fence`, matching the `render_degraded` shape. The clean
    diff content is PR-controlled identically to the degraded case — a
    diff line containing `## Heading` or ` ``` ` markdown would forge
    sections that mimic the prompt's own structure. See
    `webhook-strings-are-data-not-format-strings`.
    """
    from outrider.prompts import safe_code_fence

    system_prompt = SYSTEM_PROMPT_INVARIANTS + SYSTEM_FILE_CONTEXT_TEMPLATE.format(
        file_path=file_path,
        scope_unit_context=scope_unit_context,
        query_match_id_list=query_match_id_list,
    )
    user_prompt = USER_TEMPLATE.format(
        pass_index=pass_index,
        diff_hunks=safe_code_fence(diff_hunks, lang="diff"),
    )
    return AnalyzePromptParts(system_prompt=system_prompt, user_prompt=user_prompt)


def render_degraded(
    *,
    file_path: str,
    bounded_hunks: str,
    pass_index: int,
    degradation_reason: str,
) -> AnalyzePromptParts:
    """Build the (system, user) prompt pair for a degraded-outcome call.

    `degradation_reason` is the typed `LLMRequest` field value
    (`parse_failed` or `tree_has_error_in_changed_regions`); it appears
    in the prompt so the model knows structural-tier claims will reject.

    `bounded_hunks` MUST already satisfy the per-file degraded budget
    cap (≤100 unidiff Line objects AND ≤8192 chars). The node body
    bounds before calling; this function does not re-enforce.

    Wraps `bounded_hunks` in a dynamic-length `diff`-fence via
    `safe_code_fence` because diff content is PR-controlled — a diff
    line containing `## Heading` or ` ``` ` markdown would otherwise
    forge sections that mimic the prompt's own structure. See
    `webhook-strings-are-data-not-format-strings`.
    """
    from outrider.prompts import safe_code_fence

    system_prompt = SYSTEM_PROMPT_INVARIANTS
    user_prompt = DEGRADED_USER_TEMPLATE.format(
        file_path=file_path,
        pass_index=pass_index,
        degradation_reason=degradation_reason,
        bounded_hunks=safe_code_fence(bounded_hunks, lang="diff"),
    )
    return AnalyzePromptParts(system_prompt=system_prompt, user_prompt=user_prompt)


__all__ = [
    "DEGRADED_USER_TEMPLATE",
    "MAX_TOKENS",
    "SYSTEM_FILE_CONTEXT_TEMPLATE",
    "SYSTEM_PROMPT_INVARIANTS",
    "TEMPERATURE",
    "TEMPLATE",
    "USER_TEMPLATE",
    "VERSION",
    "AnalyzePromptParts",
    "render",
    "render_degraded",
]
