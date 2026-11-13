"""Build a summary request from accepted findings and triage risk.

The model summarizes the supplied facts; policy assigns severity and workflow
code owns review status. Deterministic report fields own review metrics.
All untrusted finding text is fenced as data.

See DECISIONS.md#043, webhook-strings-are-data-not-format-strings and
severity-set-by-policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from outrider.schemas.review_finding import ReviewFinding
    from outrider.schemas.triage_result import RiskLevel


# Version the prompt for replay attribution. The summary appears in the
# dashboard (FUP-149); its input has no gating data, so workflow claims are forbidden.
VERSION: Final[str] = "synthesize-v5"
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
- Do NOT include LLM call counts, token counts, cost, or elapsed time in the
  summary.
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
) -> SynthesizePromptParts:
    """Build the (system, user) prompt pair for the synthesize LLM call.

    Pure function. Renders deduplicated findings and upstream risk into the user
    prompt. Finding titles/descriptions are passed through
    `safe_code_fence` to defend against PR-author-crafted prompt-
    injection content embedded in finding text (the analyze layer's
    review of the diff can paraphrase attacker prose into finding
    `description`/`evidence` fields).
    """
    from outrider.prompts import safe_code_fence

    if not findings:
        findings_summary = "(no findings)"
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

    user_prompt = USER_TEMPLATE.format(
        overall_risk=overall_risk.value,
        n_findings=len(findings),
        findings_summary=findings_summary,
    )
    return SynthesizePromptParts(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
