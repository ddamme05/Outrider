"""prompts/synthesize.py contract pins (mechanics, not prose quality).

Minimal sibling of `test_prompts_analyze.py` — added with the DECISIONS#043
flip to pin the branch's mechanically-regressible surfaces: the provenance
VERSION, the corrected output-surface claim (V1's only summary surface is
the dashboard, not the GitHub review body — FUP-149 tracks that surface),
and the severity-discipline constraint the prompt must keep stating.
"""

from __future__ import annotations

from outrider.prompts.synthesize import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    USER_TEMPLATE,
    VERSION,
)


def test_version_is_named_synthesize_v3() -> None:
    """VERSION flows to LLMRequest.prompt_template_version. The v3 bump
    (2026-06-10) removed the unsatisfiable auto-publish/HITL instruction
    and forbade pipeline-behavior claims; v2 (same day) corrected the
    output-surface claim. Replay attributes a prompt row to the contract
    it was emitted under."""
    assert VERSION == "synthesize-v3"


def test_system_prompt_does_not_claim_a_github_surface() -> None:
    """The v1 prompt told the model its output 'is composed into a GitHub
    review body' — false for V1 (dashboard is the only summary surface).
    Pin the absence so the claim can't quietly return before the GitHub
    surface actually ships."""
    assert "GitHub" not in SYSTEM_PROMPT
    assert "GitHub" not in USER_TEMPLATE
    # The surface-neutral replacement wording is present.
    assert "rendered into the review report" in SYSTEM_PROMPT


def test_system_prompt_keeps_severity_discipline() -> None:
    """`severity-set-by-policy`: the prompt must keep telling the model not
    to classify severity, whatever model runs the call."""
    assert "Do NOT classify severity" in SYSTEM_PROMPT


def test_system_prompt_forbids_pipeline_behavior_claims() -> None:
    """The v3 fix: the prompt receives NO gating/publish data, so any
    workflow claim is fabrication — the side-by-side read caught Haiku
    calling a CRITICAL finding "auto-published" because v2 INSTRUCTED
    "be direct about auto-publish vs HITL-gated findings". Pin both the
    prohibition and the absence of the unsatisfiable instruction."""
    assert "Do NOT make claims about pipeline behavior" in SYSTEM_PROMPT
    assert "auto-publish vs HITL-gated" not in SYSTEM_PROMPT


def test_knobs_within_llm_request_bounds() -> None:
    """MAX_TOKENS/TEMPERATURE must satisfy LLMRequest field constraints
    (max_tokens le=8192, temperature 0..1)."""
    assert 0 < MAX_TOKENS <= 8192
    assert 0.0 <= TEMPERATURE <= 1.0
