# See specs/2026-05-23-trace-node.md M6 + Q4.
"""Render the selected imports for one trace ranking call.

Analyze pass 0 supplies candidates; trace deduplicates and caps each finding's
group before calling this renderer. The model must rank every selected ID
exactly once. Ranking changes priority, while deterministic probes decide
resolution and file fetching.

Static instructions and request data remain separate. Candidate imports and
reasons use dynamic fences (`webhook-strings-are-data-not-format-strings`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from outrider.prompts import safe_code_fence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from outrider.schemas.trace_candidate import TraceCandidate

VERSION: Final[str] = "trace-v1"
# Output budget for the ranking JSON. Truncation is rejected by trace_parser;
# the trace node then retains the supplied order.
MAX_TOKENS: Final[int] = 8192
TEMPERATURE: Final[float] = 0.0


SYSTEM_PROMPT: Final[str] = """\
You are a fast candidate-ranking assistant for an automated PR-review
agent. The agent's analyze node identified findings in changed files
and proposed Python import strings whose resolution might improve the
findings' proof tier (e.g., turning a JUDGED finding into an INFERRED
one by walking the imported symbol's body).

Your job: rank the proposed import-string candidates in DESCENDING
order of likelihood that resolving them and fetching the resolved file
will materially improve at least one finding's evidence quality. The
deterministic pipeline downstream resolves the top-ranked candidates
via fetch-probes and admits the resolved ones into a follow-up analyze
pass.

## Your role

You ORDER candidates. You do NOT:
- decide whether a candidate "resolves" — that's a deterministic
  fetch-probe step downstream.
- fetch any files — the GitHub fetch-probes happen after ranking
  (Phase 1 per-candidate fanout, then Phase 2 content fetch only for
  candidates that resolve to exactly one path).
- propose new candidates — only the candidates supplied below may
  appear in your output.
- skip candidates you consider low-value — order them, but include
  every supplied `candidate_id`. The downstream pipeline applies its
  own gates (depth limit, dedup, already-traced); your ranking
  informs priority, not admission.

## Output

Return exactly one JSON object and nothing else. Do NOT wrap the JSON
in markdown code fences (no ```json, no ```). Do NOT add explanatory
prose before or after. Output starts with `{` and ends with `}`. Every
value must be valid JSON literally — placeholders like `<...>` in this
example are illustrative and must be replaced with real values.

{
  "ranked_candidate_ids": [
    "<candidate_id from the supplied list, in descending priority>"
  ]
}

`ranked_candidate_ids` is a JSON array of strings. Every string MUST
be a `candidate_id` from the candidate list below — fabricated ids
cause the response to be rejected. The array MUST include every
supplied candidate_id exactly once; omitting or duplicating any id
causes the response to be rejected.

Keep your output to just the JSON object. No reasoning text.
"""


USER_TEMPLATE: Final[str] = """\
Rank the following {n_candidates} trace candidate(s) in descending
order of likelihood that resolving them will materially improve at
least one finding's evidence quality.

Candidates:
{candidate_list}
"""


TEMPLATE: Final[str] = USER_TEMPLATE
"""Spec-named alias of USER_TEMPLATE."""


@dataclass(frozen=True, slots=True)
class TracePromptParts:
    """Named system and user prompt fields; positional unpacking is unsupported."""

    system_prompt: str
    user_prompt: str


def render(candidates: Sequence[TraceCandidate]) -> TracePromptParts:
    """Build the (system, user) prompt pair for the trace LLM call.

    Pure function. Uses `USER_TEMPLATE.format(**kwargs)` with all
    placeholder names supplied as keyword arguments.

    Each candidate becomes one list item showing `candidate_id`,
    `import_string`, and `reason`. The `import_string` and `reason`
    fields are PR-and-LLM-derived (candidate_id is content-derived),
    so they enter the prompt inside structural markers, not as
    `.format()` interpolation targets — defends against the
    `webhook-strings-are-data-not-format-strings` invariant.
    """
    candidate_list = "\n".join(_format_candidate(candidate) for candidate in candidates)
    user_prompt = USER_TEMPLATE.format(
        n_candidates=len(candidates),
        candidate_list=candidate_list,
    )
    return TracePromptParts(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def _format_candidate(candidate: TraceCandidate) -> str:
    """Render one candidate, fencing its import and reason as untrusted data.

    The content-derived candidate_id is a SHA-256 hex value."""
    fenced_import = safe_code_fence(candidate.import_string, lang="text")
    fenced_reason = safe_code_fence(candidate.reason, lang="text")
    return (
        f"- candidate_id: {candidate.candidate_id}\n"
        f"  import_string:\n{fenced_import}\n"
        f"  reason:\n{fenced_reason}"
    )


__all__ = [
    "MAX_TOKENS",
    "SYSTEM_PROMPT",
    "TEMPERATURE",
    "TEMPLATE",
    "TracePromptParts",
    "USER_TEMPLATE",
    "VERSION",
    "render",
]
