# See DECISIONS.md#040 — suggested-patch generation (the synthesize patch pass).
"""Patch generation for suggested fixes (DECISIONS.md#040).

The synthesize node calls `generate_patches(...)` over the deduped finding set:
it gates HIGH/CRITICAL single-line findings (CRITICAL-first, capped), makes ONE
batched Haiku call, and sets `ReviewFinding.suggested_fix` (the EXACT replacement
text — never markdown) on the findings the model returned a usable single-line
fix for. publish renders the GitHub ```suggestion (only for INLINE_COMMENT
routing); this module never renders markdown.

The prompt + parser are the TRUST SURFACE (model output → a comment a developer
one-click-applies), so the schema is boring and strict and parsing FAILS CLOSED:
a malformed batch yields ZERO suggestions (no half-trust); a bad item in an
otherwise-valid batch drops only that item. Routing is NOT an input here — it is
decided downstream in publish (graph order `synthesize → hitl → publish`), so a
patch generated for a finding that turns out un-renderable is bounded waste.
"""

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
    """One model-proposed fix. `original_line` is the model's ECHO of the exact
    target line we sent — an anchor sanity check that catches drift ("fixed the
    nearby line") without coordinate math (DECISIONS.md#040). `replacement_line=None`
    (+ a short `reason`) = "no good single-line fix." Only `finding_id` is required;
    the rest are optional so a malformed ITEM drops in `apply_patch_batch` without
    killing the whole batch (a non-UUID/missing `finding_id` IS batch-fatal — it is
    the key). Business rules (echo match, one-line, no-markdown, dedup, length) live
    in `apply_patch_batch` / `set_suggested_fix`, not raising validators here."""

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
    eligible = [
        f
        for f in findings
        if is_hitl_gated_severity(_baseline_severity(f)) and f.line_start == f.line_end
    ]
    # CRITICAL before HIGH (severity priority for the cap); stable within a tier.
    eligible.sort(key=lambda f: 0 if _baseline_severity(f) is FindingSeverity.CRITICAL else 1)
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
    eligible_ids = {f.finding_id for f in eligible}
    try:
        batch = PatchSuggestionBatch.model_validate_json(strip_outer_json_fence(raw_response))
    except (ValidationError, ValueError) as exc:
        logger.warning(
            "patch batch failed to parse (%s); dropping all suggestions", exc.__class__.__name__
        )
        return {}  # fail closed — never half-trust a malformed batch
    id_counts = Counter(item.finding_id for item in batch.items)
    out: dict[UUID, str] = {}
    for item in batch.items:
        fid = item.finding_id
        if id_counts[fid] > 1 or fid not in eligible_ids:
            continue  # duplicate / unknown / ineligible → drop
        target = target_lines.get(fid)
        if target is None or item.original_line != target:
            continue  # echo mismatch (or no target) → drop as untrusted anchoring
        if item.replacement_line is None or not _is_valid_replacement(
            item.replacement_line, target
        ):
            continue  # null / multi-line / markdown / diff-marker / no-op → drop
        out[fid] = item.replacement_line
    return out


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
    """The synthesize patch pass (DECISIONS.md#040): gate → extract target lines →
    ONE batched Haiku call → parse (echo-checked, fail-closed) → set `suggested_fix`.

    Returns `findings` UNCHANGED when patches are disabled, nothing is eligible, no
    target line is extractable, or no item survives parsing. The provider call emits
    an `LLMCallEvent` with `node_id="synthesize"` (the patch cost rolls into
    synthesize's aggregate — synthesize already makes a second call here). Patches are
    BEST-EFFORT: an `LLMProviderError` on the patch call fails CLOSED (findings ship
    unpatched) rather than failing the whole review — unlike synthesize's own summary
    call, whose failure is fatal. Routing is decided later in publish.
    """
    if not patches_enabled:
        return findings
    eligible = select_eligible_findings(findings, max_suggestions=max_suggestions)
    if not eligible:
        return findings
    content_by_path = {cf.path: cf.content_head for cf in state.pr_context.changed_files}
    target_lines = extract_target_lines(eligible, content_by_path)
    if not target_lines:
        return findings

    parts = patch_prompt.render(eligible, target_lines)
    request = LLMRequest(
        model=patch_model,
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
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
