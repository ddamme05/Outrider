# Synthesize-node prompt template + render helper per specs/2026-05-28-synthesize-node.md
"""Synthesize prompt template, version, knobs, and render helper.

The synthesize node uses one LLM pass (config-routed; Haiku by default
per `DECISIONS.md#043`) to generate a free-form prose summary of the
review's deduplicated findings. The prompt asks for a concise,
reviewer-facing summary of overall risk + dominant themes + suggested
next steps. The model does NOT classify severity (the
`severity-set-by-policy` invariant pins that upstream); the prompt is
prose-only.

Surfaces (per the synthesize-node spec's Reference Reconciliation):

- `SYSTEM_PROMPT: Final[str]` — fully static instructions. Goes into
  `LLMRequest.system_prompt`. NB: prompt caching has a per-model
  minimum-tokens floor (`llm/pricing.py::MIN_CACHEABLE_TOKENS` —
  4096 for Haiku 4.5, the `DECISIONS.md#043` default); this prompt
  (~700 tokens) is far below any floor, so cache_control attempts
  fall through with a per-(model, hash) wrapper warning. That is
  ACCEPTED per #043 (closing FUP-163): synthesize fires ONCE per
  review, so a same-review cache hit is structurally impossible and
  growing the prompt solely to clear a floor would buy nothing —
  per-call latency is completion-dominated, not prompt hydration.
- `USER_TEMPLATE: Final[str]` — per-review `str.format` template with
  structural placeholders (`{overall_risk}`, `{findings_summary}`,
  `{metrics_summary}`). Values are filled at `render()` time; the
  placeholder names are template STRUCTURE — model-derived content
  (finding titles/evidence/descriptions) gets fence-wrapped via
  `safe_code_fence` per `webhook-strings-are-data-not-format-strings`.
- `TEMPLATE` — alias for USER_TEMPLATE.
- `VERSION: Final[str] = "synthesize-v1"` — flows to
  `LLMRequest.prompt_template_version`.
- `MAX_TOKENS: Final[int] = 1024` — bounds the summary output to the
  `Field(max_length=2000)` codepoint cap on `ReviewReport.summary`
  with headroom for the token-to-codepoint ratio.
- `TEMPERATURE: Final[float] = 0.3` — slightly higher than triage's
  0.0 because summary prose benefits from non-deterministic phrasing,
  but bounded enough that replay produces semantically-equivalent
  output (the content_hash check at the schema layer pins logical
  identity).
- `SynthesizePromptParts` — frozen dataclass result of `render()`.
- `render(overall_risk, findings, metrics) -> SynthesizePromptParts`
  — pure function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_report import ReviewMetrics
    from outrider.schemas.triage_result import RiskLevel


VERSION: Final[str] = "synthesize-v1"
MAX_TOKENS: Final[int] = 1024
TEMPERATURE: Final[float] = 0.3


SYSTEM_PROMPT: Final[str] = """\
You are the synthesis layer of an automated PR-review agent. Your job is
to write a concise summary of the review's findings for a human reviewer.

You receive:
  - The PR-level overall_risk classification (low / medium / high / critical)
    set upstream by the triage layer; you do NOT re-classify it.
  - A deduplicated list of findings already classified by severity and
    dimension (you do NOT re-classify these — `severity-set-by-policy`
    pins the severity from finding_type upstream).
  - Aggregate review metrics (files examined, LLM calls made, cost, etc.).

Your job: write a 2-5 sentence prose summary that helps the human
reviewer prioritize their attention. Lead with what matters most. Be
specific about themes when patterns emerge across findings (e.g., "three
SQL-injection findings in user-input handlers"); be direct about
auto-publish vs HITL-gated findings without over-explaining the workflow.

CRITICAL CONSTRAINTS:

- Plain prose only. No markdown headers, no bullet lists, no code fences.
  The output is composed into a GitHub review body by the publish layer;
  markdown structure is the publisher's responsibility, not yours.
- 2000 character maximum (the schema cap rejects anything longer).
- Do NOT classify severity. Do NOT recommend severity overrides. Do NOT
  invent findings that aren't in the input list.
- Do NOT include any markdown links, URLs, or @-mentions (these are
  composed by the publisher from finding metadata).
- Write in third person. "This PR introduces ..." not "I observed ...".
"""


USER_TEMPLATE: Final[str] = """\
Overall risk: {overall_risk}

Findings ({n_findings} total):
{findings_summary}

Metrics:
{metrics_summary}

Write the summary now.
"""


TEMPLATE: Final[str] = USER_TEMPLATE


@dataclass(frozen=True, slots=True)
class SynthesizePromptParts:
    """Render output: (system, user) pair. Dataclass (not NamedTuple)
    per the triage-precedent rationale (positional unpacking should
    raise loudly, not silently swap)."""

    system_prompt: str
    user_prompt: str


def render(
    *,
    overall_risk: RiskLevel,
    findings: tuple[ReviewFinding, ...],
    metrics: ReviewMetrics,
) -> SynthesizePromptParts:
    """Build the (system, user) prompt pair for the synthesize LLM call.

    Pure function. Renders deduplicated findings + metrics into the user
    prompt. Finding titles/descriptions are passed through
    `safe_code_fence` to defend against PR-author-crafted prompt-
    injection content embedded in finding text (the analyze layer's
    review of the diff can paraphrase attacker prose into finding
    `description`/`evidence` fields).
    """
    from outrider.prompts import safe_code_fence

    if not findings:
        findings_summary = "(no findings — clean review)"
    else:
        lines: list[str] = []
        for f in findings:
            # Each finding is fence-wrapped because title/description
            # paraphrase PR content. The fence width is dynamic per
            # safe_code_fence so a finding text containing N+1
            # backticks can't break out.
            lines.append(
                f"- [{f.severity.value} / {f.dimension.value}] "
                f"{safe_code_fence(f.title, lang='')} @ "
                f"{f.file_path}:{f.line_start}-{f.line_end}\n"
                f"  {safe_code_fence(f.description, lang='')}"
            )
        findings_summary = "\n".join(lines)

    # LLM-aggregate metrics are populated from the audit stream (FUP-093) but
    # stay Optional[X] (nullable for historical-row read-compat, #030). Render
    # "unknown" defensively for any None rather than crashing on `:.4f`
    # format-spec against NoneType. _render_metric_value / _render_cost_value
    # handle the None/numeric union safely.
    metrics_summary = (
        f"- Files examined: {metrics.files_examined}\n"
        f"- Files traced beyond diff: {metrics.files_traced_beyond_diff}\n"
        f"- LLM calls made: {_render_metric_value(metrics.llm_calls_made)}\n"
        f"- Tokens: {_render_metric_value(metrics.total_input_tokens)} in / "
        f"{_render_metric_value(metrics.total_output_tokens)} out\n"
        f"- Cost: {_render_cost_value(metrics.total_cost_usd)}\n"
        f"- Wall clock: {metrics.wall_clock_seconds:.1f}s"
    )

    user_prompt = USER_TEMPLATE.format(
        overall_risk=overall_risk.value,
        n_findings=len(findings),
        findings_summary=findings_summary,
        metrics_summary=metrics_summary,
    )
    return SynthesizePromptParts(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def _render_metric_value(value: int | None) -> str:
    """Render an Optional[int] metric for prose output.

    LLM-aggregate metrics are populated from the audit stream (FUP-093) but
    stay Optional (None for historical rows / nullable per #030); render
    "unknown" so the prompt remains valid prose when the value is absent.
    None values formatted via `:d` or
    `:.4f` raise TypeError — this helper is the safety adapter at
    the prompt-render boundary.
    """
    return "unknown" if value is None else str(value)


def _render_cost_value(value: float | None) -> str:
    """Render an Optional[float] cost metric for prose output.

    Same None-safety rationale as `_render_metric_value` but with a
    `$N.NNNN` shape when the value is present.
    """
    return "unknown" if value is None else f"${value:.4f}"
