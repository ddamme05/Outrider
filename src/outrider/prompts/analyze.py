# Analyze-node prompt template + render helpers per specs/2026-05-19-analyze-node.md §5
"""Analyze prompt template, version, knobs, and render helpers.

The analyze node runs one Sonnet call per eligible file. Per the
analyze-node spec §5, prompts decompose into:

- **System prompt** (cacheable): Outrider-wide invariants (output schema,
  `FindingType` enum, `EvidenceTier` proof rules, severity-set-by-policy
  reminder, confidence-is-computed reminder) PLUS file-scoped context
  (the file's changed scope units with bodies + same-file callers/callees
  + imports + decorators + pre-fired `query_match_id` set). The
  combined string is stable across the analyze ⇄ trace loop for one
  file, so the wrapper's `cache_control: ephemeral` produces a cache
  hit on the second pass.
- **User prompt** (volatile): pass-specific instruction + diff hunks
  clipped to changed scope-unit boundaries. Outside the cache boundary.

For files that hit the degraded path (parse failure or `has_error`
nodes intersecting changed regions per `parse-errors-degrade-to-judged`),
the prompt swaps to a `judged`-only directive set; the registry/walk
context is empty by construction, so the system prompt for degraded
calls is shorter and the user prompt carries the bounded changed
hunks instead of scope-unit-clipped ones.

Surfaces (per the analyze-node spec's Reference Reconciliation):

- `SYSTEM_PROMPT_INVARIANTS: Final[str]` — fully static head of the
  system prompt; the file-scoped tail is appended by `render(...)`.
- `USER_TEMPLATE: Final[str]` — pass-specific directives + diff hunks
  template for clean-outcome calls. `str.format`-style placeholders.
- `DEGRADED_USER_TEMPLATE: Final[str]` — pass-specific directives +
  bounded changed hunks for degraded-outcome calls. Admits only
  `evidence_tier="judged"` proposals.
- `TEMPLATE: Final[str] = USER_TEMPLATE` — spec-named alias.
- `VERSION: Final[str] = "analyze-v1"` — flows to
  `LLMRequest.prompt_template_version`. Bump on any template change.
- `MAX_TOKENS: Final[int] = 8192` — fits up to ~50 findings per
  response (the raw layer's `max_length=50` per `AnalyzeResponseRaw`).
- `TEMPERATURE: Final[float] = 0.0` — deterministic-leaning; minimizes
  drift across replay.
- `AnalyzePromptParts` — frozen dataclass result of `render(...)` /
  `render_degraded(...)`. Mirrors `TriagePromptParts`'s shape rationale:
  dataclass not NamedTuple so positional unpacking (`(sys, usr) =
  render(...)`) fails loud at runtime rather than silently masking a
  swap.
- `render(...)` — clean-outcome render; builds system prompt from
  static invariants + file-scoped context and user prompt from
  pass-specific directives + scope-unit-clipped diff hunks.
- `render_degraded(...)` — degraded-outcome render; builds the
  `judged`-only system + degraded user prompts.

Per `webhook-strings-are-data-not-format-strings`: PR-sourced content
(file paths, scope-unit names, diff hunks, query match IDs) enters the
prompt via `str.format(**kwargs)` against structural placeholders;
attacker-controlled content cannot escape the template structure.

Implementation note (spec divergence recorded for Actual Outcome):
the analyze-node spec §5 originally described a `PromptRegistry` class
+ a `make_analyze_node` factory. The shipped codebase convention
(`prompts/triage.py` + `async def triage(...)` + `functools.partial`
at graph wire time) is followed instead — strictly, no local
abstraction. Dependencies-via-closure is satisfied by `functools.partial`
in `build_graph(...)`; the spec invariant is preserved, only the
syntactic surface differs.
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

Pick exactly one value for `evidence_tier`. Each tier carries different
admission rules:

- `observed` — a tree-sitter query in our registry matched a structural
  pattern. You MUST cite a real `query_match_id` from the pre-supplied
  registry set below; a fabricated id causes rejection with reason
  "query_match_id_not_in_registry".
- `inferred` — a deterministic walk through our import/call/symbol
  registry resolves the steps in `trace_path`. The trace resolver
  validates each step; unwalkable steps cause rejection with reason
  "trace_path_not_admissible".
- `judged` — your own interpretation; no structural artifact required.
  Use this for findings without an available query match or trace walk.

Failed admission DROPS the proposal — it does not downgrade to a lower
tier. Pick `judged` upfront if you cannot cite structural evidence.

## Output shape

Return exactly this JSON, nothing else. No markdown fences, no prose
before or after. Output starts with `{` and ends with `}`.

{
  "findings": [
    {
      "finding_type": "<enum value>",
      "evidence_tier": "<observed|inferred|judged>",
      "query_match_id": "<id from registry, or null>",
      "trace_path": ["<step>", "..."] | null,
      "title": "<short summary, ≤120 chars>",
      "description": "<explanation, ≤1000 chars>",
      "evidence": "<verbatim quote from the code, ≤2000 chars>",
      "span": {"byte_start": <int>, "byte_end": <int>},
      "trace_candidates": [
        {"candidate_path": "<repo-relative path>", "reason": "<text>"},
        ...
      ]
    },
    ...
  ]
}

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
"""Spec-named alias of USER_TEMPLATE. The spec lists `TEMPLATE` as the
public surface; USER_TEMPLATE is the locally-named twin for clarity at
the call site. They refer to the same string object."""


@dataclass(frozen=True, slots=True)
class AnalyzePromptParts:
    """Render output: the (system, user) pair for one analyze LLM call.

    Dataclass (not NamedTuple) because dataclasses do NOT subclass tuple,
    so the swap-prone shape `(system, user) = render(...)` parses and
    compiles fine but raises `TypeError` at runtime when the iterator
    protocol fails. The swap cannot ship silently — attribute access
    (`parts.system_prompt`, `parts.user_prompt`) is the supported
    pattern; positional unpacking fails loud on the very first call.
    Mirrors `TriagePromptParts` exactly.

    Both fields are str (validated downstream by `LLMRequest.system_prompt`
    and `.user_prompt` which carry `min_length=1`).
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
    """Build the (system, user) prompt pair for a clean-outcome analyze call.

    `system_prompt` carries stable-per-file content (invariants +
    file-scoped scope-unit/query context) so the provider's
    `cache_control: ephemeral` produces cross-pass cache hits for the
    same file. `user_prompt` carries pass-specific volatile content
    (pass index + scope-unit-clipped diff hunks).

    PR-sourced strings enter via `.format(**kwargs)` against structural
    placeholders per `webhook-strings-are-data-not-format-strings`.
    """
    system_prompt = SYSTEM_PROMPT_INVARIANTS + SYSTEM_FILE_CONTEXT_TEMPLATE.format(
        file_path=file_path,
        scope_unit_context=scope_unit_context,
        query_match_id_list=query_match_id_list,
    )
    user_prompt = USER_TEMPLATE.format(
        pass_index=pass_index,
        diff_hunks=diff_hunks,
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

    `degradation_reason` is the typed LLMRequest field value
    ("parse_failed" or "tree_has_error_in_changed_regions"); it appears
    in the prompt so the model knows why it's in degraded mode and
    why structural-tier claims will be rejected.

    `bounded_hunks` MUST satisfy the per-file degraded budget cap
    described in spec §7 step 3c: ≤100 `unidiff.Line` objects total
    AND ≤8192 chars of text. The caller (node body) bounds before
    calling render_degraded; this function does not re-enforce the cap.
    """
    system_prompt = SYSTEM_PROMPT_INVARIANTS
    user_prompt = DEGRADED_USER_TEMPLATE.format(
        file_path=file_path,
        pass_index=pass_index,
        degradation_reason=degradation_reason,
        bounded_hunks=bounded_hunks,
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
