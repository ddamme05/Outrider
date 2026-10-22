# See DECISIONS.md#040 — suggested-patch generation (the synthesize patch pass).
"""Generate optional single-line fixes during synthesis (DECISIONS.md#040).

Select HIGH/CRITICAL single-line findings, CRITICAL first and capped, then make
one call to the configured patch model. Accepted replacements become validated
ReviewFinding.suggested_fix values. Publish later decides whether an inline
GitHub suggestion can be rendered; this module produces no markdown.

Malformed batches yield no suggestions. Valid batches reject duplicate IDs,
unknown findings, mismatched target-line echoes and unsafe replacements per item.
Provider errors leave findings unpatched; other unexpected failures propagate."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING
from uuid import (
    UUID,  # noqa: TC003 — runtime use: Pydantic resolves `finding_id: UUID` at model build
)

from pydantic import BaseModel, ConfigDict, ValidationError

from outrider.llm.base import LLMProviderError, LLMRequest
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy.output_sanitizer import is_safe_suggestion_replacement
from outrider.policy.publish_eligibility import is_hitl_gated_severity
from outrider.policy.severity import FindingSeverity
from outrider.prompts import patch as patch_prompt
from outrider.schemas.review_finding import ReviewFinding

if TYPE_CHECKING:
    from outrider.llm.base import LLMProvider
    from outrider.schemas.review_state import ReviewState

logger = logging.getLogger(__name__)


class PatchSuggestion(BaseModel):
    """One proposed fix, keyed by finding_id, with an exact target-line echo.

    Missing/non-UUID IDs and wrong field types reject the batch. Optional fields
    allow apply_patch_batch to discard unusable items independently. A null
    replacement means no suggested fix (DECISIONS.md#040)."""

    model_config = ConfigDict(extra="forbid")

    finding_id: UUID
    original_line: str | None = None
    replacement_line: str | None = None
    reason: str | None = None


class PatchSuggestionBatch(BaseModel):
    """The strict envelope for one batched patch-gen response."""

    model_config = ConfigDict(extra="forbid")

    items: list[PatchSuggestion]


def _baseline_severity(finding: ReviewFinding) -> FindingSeverity:
    """The policy severity the HITL gate fires on (pre-override) — matches how
    `hitl-gated` is determined elsewhere (S1 markers, publish eligibility)."""
    if finding.original_severity is not None:
        return finding.original_severity
    return finding.severity


def select_eligible_findings(
    findings: tuple[ReviewFinding, ...], *, max_suggestions: int
) -> tuple[ReviewFinding, ...]:
    """HIGH/CRITICAL + single-line (`line_start == line_end`), CRITICAL-first,
    capped at `max_suggestions`. Routing is NOT a gate input — publish decides
    renderability downstream (#040); patches for findings that turn out
    REVIEW_BODY/DASHBOARD_ONLY are bounded, cap-limited waste."""
    eligible: list[ReviewFinding] = []
    for patch_finding in findings:
        if (
            is_hitl_gated_severity(_baseline_severity(patch_finding))
            and patch_finding.line_start == patch_finding.line_end
        ):
            eligible.append(patch_finding)
    # CRITICAL before HIGH (severity priority for the cap); stable within a tier.
    eligible.sort(
        key=lambda finding: 0 if _baseline_severity(finding) is FindingSeverity.CRITICAL else 1
    )
    return tuple(eligible[:max_suggestions])


def _extract_target_line(content: str, line_start: int) -> str | None:
    """The 1-indexed source line at `line_start`, measured against `content` split on
    `"\\n"` — the SAME line basis ast_facts/tree-sitter use to derive `line_start`
    (byte/newline-counted), so this index aligns with the finding's coordinate. A
    final trailing newline yields an empty element that is NOT a real source line and
    is dropped; a trailing `"\\r"` per line is stripped (CRLF files). Returns None
    (FAIL CLOSED) when `line_start` is out of range — the caller then skips that
    finding rather than patching against a line it could not read."""
    if line_start < 1:
        return None
    lines = content.split("\n")
    if lines and lines[-1] == "":  # trailing-newline artifact, not a real line
        lines = lines[:-1]
    if line_start > len(lines):
        return None
    return lines[line_start - 1].rstrip("\r")


def extract_target_lines(
    eligible: tuple[ReviewFinding, ...], file_content_by_path: dict[str, str | None]
) -> dict[UUID, str]:
    """`{finding_id: target_line}` for the eligible findings whose target line can be
    read. FAILS CLOSED per finding: a finding whose file content is unavailable
    (`None` / missing key) or whose `line_start` is out of range is OMITTED — the
    patch model is never asked to fix a line we could not anchor. The returned map is
    both the prompt's per-finding context and the parser's echo-check authority."""
    out: dict[UUID, str] = {}
    for finding in eligible:
        content = file_content_by_path.get(finding.file_path)
        if content is None:
            continue  # no head content for this path → no patch
        target = _extract_target_line(content, finding.line_start)
        if target is None:
            continue  # line_start out of range → no patch
        out[finding.finding_id] = target
    return out


def _is_valid_replacement(replacement: str, original_line: str) -> bool:
    """A `replacement_line` must pass the shared single-line GitHub-suggestion safety
    gate (`is_safe_suggestion_replacement` — one line, backtick-free, non-empty, no diff
    marker, no Trojan-Source codepoint, no HTML-comment marker-forgery; #040) AND be an
    actual CHANGE from the original (drops no-op rewrites). The renderer enforces the
    same shared gate independently (defense in depth)."""
    return is_safe_suggestion_replacement(replacement) and replacement != original_line


def apply_patch_batch(
    raw_response: str,
    eligible: tuple[ReviewFinding, ...],
    target_lines: dict[UUID, str],
) -> dict[UUID, str]:
    """Parse the batched response → `{finding_id: replacement_line}` for items
    passing EVERY rule. FAILS CLOSED: a malformed batch (parse / schema failure, or
    a missing / non-UUID `finding_id` — the key) → empty dict (zero suggestions); a
    bad ITEM is dropped, the rest kept. Per-item rules: `finding_id` is in the
    eligible set (drops unknown / ineligible) and appears exactly once (drops
    duplicates entirely); `original_line` EXACTLY echoes `target_lines[finding_id]`
    (drops drifted anchoring — the cheap anchor check); and `replacement_line` passes
    `_is_valid_replacement` against that original."""
    eligible_ids = {finding.finding_id for finding in eligible}
    try:
        batch = PatchSuggestionBatch.model_validate_json(strip_outer_json_fence(raw_response))
    except (ValidationError, ValueError) as exc:
        logger.warning(
            "patch batch failed to parse (%s); dropping all suggestions", exc.__class__.__name__
        )
        return {}  # fail closed — never half-trust a malformed batch
    id_counts = Counter(item.finding_id for item in batch.items)
    accepted_patches: dict[UUID, str] = {}
    for item in batch.items:
        finding_id = item.finding_id
        if id_counts[finding_id] > 1 or finding_id not in eligible_ids:
            continue  # duplicate / unknown / ineligible → drop
        target = target_lines.get(finding_id)
        if target is None or item.original_line != target:
            continue  # echo mismatch (or no target) → drop as untrusted anchoring
        if item.replacement_line is None or not _is_valid_replacement(
            item.replacement_line, target
        ):
            continue  # null / multi-line / markdown / diff-marker / no-op → drop
        accepted_patches[finding_id] = item.replacement_line
    return accepted_patches


def set_suggested_fix(
    findings: tuple[ReviewFinding, ...], patches_by_id: dict[UUID, str]
) -> tuple[ReviewFinding, ...]:
    """Return `findings` with `suggested_fix` set on those carrying a patch — via a
    validator-running `model_validate` rebuild (NEVER `model_copy(update=…)`, which
    skips validators, per #040). A patch that fails validation (e.g. > the 2000-char
    `suggested_fix` cap) drops THAT finding's patch, not the whole pass."""
    if not patches_by_id:
        return findings
    result: list[ReviewFinding] = []
    for finding in findings:
        replacement = patches_by_id.get(finding.finding_id)
        if replacement is None:
            result.append(finding)
            continue
        try:
            result.append(
                ReviewFinding.model_validate({**finding.model_dump(), "suggested_fix": replacement})
            )
        except ValidationError:
            result.append(finding)  # patch rejected (e.g. >2000 chars) → keep unpatched
    return tuple(result)


async def generate_patches(
    findings: tuple[ReviewFinding, ...],
    state: ReviewState,
    *,
    provider: LLMProvider,
    patch_model: str,
    patches_enabled: bool,
    max_suggestions: int,
) -> tuple[ReviewFinding, ...]:
    """Select eligible findings, read target lines, call the patch model and validate.

    Disabled/empty paths return the original tuple. LLMProviderError also returns
    unpatched findings; the summary call in synthesize has a separate fatal-error
    contract. The provider records this call under node_id="synthesize", so patch
    cost is included in synthesis metrics (DECISIONS.md#040)."""
    if not patches_enabled:
        return findings
    eligible = select_eligible_findings(findings, max_suggestions=max_suggestions)
    if not eligible:
        return findings
    content_by_path = {
        changed_file.path: changed_file.content_head
        for changed_file in state.pr_context.changed_files
    }
    target_lines = extract_target_lines(eligible, content_by_path)
    if not target_lines:
        return findings

    prompt_parts = patch_prompt.render(eligible, target_lines)
    request = LLMRequest(
        model=patch_model,
        system_prompt=prompt_parts.system_prompt,
        user_prompt=prompt_parts.user_prompt,
        max_tokens=patch_prompt.MAX_TOKENS,
        temperature=patch_prompt.TEMPERATURE,
        review_id=state.review_id,
        node_id="synthesize",
        is_eval=state.is_eval,
        prompt_template_version=patch_prompt.VERSION,
        degraded_mode=False,
    )
    try:
        response = await provider.complete(request)
    except LLMProviderError as exc:
        logger.warning(
            "patch generation provider call failed (%s); shipping findings unpatched",
            exc.__class__.__name__,
        )
        return findings  # best-effort: patch-call failure ships findings unpatched

    patches_by_id = apply_patch_batch(response.text, eligible, target_lines)
    logger.debug(
        "patch pass: %d patch(es) generated for %d eligible finding(s)",
        len(patches_by_id),
        len(eligible),
    )
    return set_suggested_fix(findings, patches_by_id)
