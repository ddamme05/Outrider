# Synthesize-node prompt template + render helper per specs/2026-05-28-synthesize-node.md
"""Render admitted findings and metrics for the review summary.

The configured model writes prose about findings, risk and remediation themes.
It does not assign severity (`severity-set-by-policy`) or describe pipeline
actions. Summary policy follows DECISIONS.md#043; the node separately enforces
the ReviewReport summary-length cap.

Static instructions stay separate from per-review data. Finding text is fenced
(`webhook-strings-are-data-not-format-strings`); absent historical metrics render
as "unknown". Prompt/version constants below remain the replay contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.review_report import ReviewMetrics
    from outrider.schemas.triage_result import RiskLevel


# Bumped 2026-06-10 (was "synthesize-v2"): the side-by-side prose read
# caught an UNSATISFIABLE instruction — "be direct about auto-publish vs
# HITL-gated findings" with no gating data in the input, which made both
# models fabricate workflow claims (Haiku called a CRITICAL finding
# "auto-published"). The clause is removed and a CRITICAL CONSTRAINT now
# forbids pipeline-behavior claims outright. The v2 bump (same day)
# corrected the stale GitHub-review-body surface claim (V1's only summary
# surface is the dashboard, FUP-149). Each bump keeps replay attribution
# exact.
VERSION: Final[str] = "synthesize-v3"
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
SQL-injection findings in user-input handlers").

CRITICAL CONSTRAINTS:

- Plain prose only. No markdown headers, no bullet lists, no code fences.
  The output is rendered into the review report shown to the reviewer;
  presentation structure is the rendering layer's responsibility, not yours.
- 2000 character maximum (the schema cap rejects anything longer).
- Do NOT classify severity. Do NOT recommend severity overrides. Do NOT
  invent findings that aren't in the input list.
- Do NOT make claims about pipeline behavior, including gating, HITL,
  publishing, auto-publish, review submission, or what happens next.
  Describe only the findings, risk, and remediation themes present in
  the input.
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
        for finding in findings:
            # Each finding is fence-wrapped because title/description
            # paraphrase PR content. The fence width is dynamic per
            # safe_code_fence so a finding text containing N+1
            # backticks can't break out.
            lines.append(
                f"- [{finding.severity.value} / {finding.dimension.value}] "
                f"{safe_code_fence(finding.title, lang='')} @ "
                f"{finding.file_path}:{finding.line_start}-{finding.line_end}\n"
                f"  {safe_code_fence(finding.description, lang='')}"
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
    """Render an integer metric, or "unknown" for historical NULL values (FUP-093, #030)."""
    return "unknown" if value is None else str(value)


def _render_cost_value(value: float | None) -> str:
    """Render a known cost to four decimal places, or "unknown" for NULL."""
    return "unknown" if value is None else f"${value:.4f}"
