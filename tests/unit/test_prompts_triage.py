"""prompts/triage.py contract tests.

Pins the triage prompt's public surface from the triage-node spec:
constants (VERSION, MAX_TOKENS, TEMPERATURE), templates (SYSTEM_PROMPT,
USER_TEMPLATE, TEMPLATE alias), TriagePromptParts swap-impossible
dataclass, and the pure render() helper.

Tests are organized as: surface contracts (what callers can rely on),
render-behavior (happy path + content sanity + edge cases), input-
boundary regression (PR text with format-string metacharacters cannot
escape the template structure).
"""

import dataclasses
import re
from typing import Literal

import pytest

from outrider.prompts.triage import (
    MAX_TOKENS,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TEMPLATE,
    USER_TEMPLATE,
    VERSION,
    TriagePromptParts,
    render,
)
from outrider.schemas.pr_context import ChangedFile, PRContext

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


_SENTINEL_DEFAULT_FILES: tuple[ChangedFile, ...] | None = None
"""Sentinel: distinguishes 'caller passed empty tuple' from 'caller didn't pass'."""


def _build_changed_file(
    *,
    path: str = "src/example.py",
    status: Literal["added", "modified", "removed", "renamed"] = "modified",
    additions: int = 5,
    deletions: int = 2,
    patch: str | None = "@@ -1,1 +1,1 @@\n-old\n+new\n",
) -> ChangedFile:
    """Single-file builder. Defaults to status='modified' with non-None
    content sides because the §7.2 invariants reject content=None on
    modified files."""
    return ChangedFile(
        path=path,
        status=status,
        additions=additions,
        deletions=deletions,
        patch=patch,
        content_base="old\n",
        content_head="new\n",
        previous_path=None,
        language=None,
    )


def _build_pr_context(
    *,
    pr_title: str = "Refactor auth check",
    pr_body: str | None = "Tightens the input validator on /login.",
    changed_files: tuple[ChangedFile, ...] | None = _SENTINEL_DEFAULT_FILES,
    total_additions: int = 5,
    total_deletions: int = 2,
) -> PRContext:
    """`changed_files=None` (the default sentinel) backfills one file;
    `changed_files=()` is respected as 'genuinely empty' for edge-case
    tests."""
    if changed_files is None:
        changed_files = (_build_changed_file(),)
    return PRContext(
        installation_id=12345,
        owner="acme",
        repo="widget",
        pr_number=42,
        base_sha="a" * 40,
        head_sha="b" * 40,
        pr_title=pr_title,
        pr_body=pr_body,
        author="someone",
        total_additions=total_additions,
        total_deletions=total_deletions,
        changed_files=changed_files,
    )


# ---------------------------------------------------------------------------
# Surface contracts: constants
# ---------------------------------------------------------------------------


def test_version_is_named_v1() -> None:
    """VERSION flows to LLMRequest.prompt_template_version. Pin the v1 name
    so future renames break the test and force a registry decision."""
    assert VERSION == "triage-v1"


def test_max_tokens_bounded_within_llm_request_limit() -> None:
    """MAX_TOKENS must respect LLMRequest.max_tokens Field constraint
    (gt=0, le=8192). Anything outside that range would fail at LLMRequest
    construction — pin the value here so a typo doesn't break the chain."""
    assert 0 < MAX_TOKENS <= 8192
    assert MAX_TOKENS == 2048  # current chosen value; intentional pin


def test_temperature_bounded_within_llm_request_limit() -> None:
    """TEMPERATURE must respect LLMRequest.temperature Field constraint
    (ge=0.0, le=1.0). Triage uses temperature=0.0 for determinism."""
    assert 0.0 <= TEMPERATURE <= 1.0
    assert TEMPERATURE == 0.0


def test_template_is_user_template_alias() -> None:
    """TEMPLATE is documented as an alias of USER_TEMPLATE. Different
    constants would let one drift from the other; the test pins they're
    the same object so renames stay coupled."""
    assert TEMPLATE is USER_TEMPLATE


def test_user_template_has_required_placeholders() -> None:
    """The USER_TEMPLATE must carry every placeholder render() supplies.
    If render() drifts (adds a new {placeholder} that's not in the
    template, or vice versa), str.format raises at first call. This test
    pins the placeholder set so additions go through a coordinated edit."""
    expected_placeholders = {
        "pr_title",
        "file_count",
        "total_additions",
        "total_deletions",
        "file_list",
        "diff_summary",
    }
    # str.format placeholders are {name} with simple-name extraction
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", USER_TEMPLATE))
    assert found == expected_placeholders, (
        f"USER_TEMPLATE placeholders ({found}) drift from render()'s "
        f"kwargs ({expected_placeholders}); either rename the template "
        f"or update render()."
    )


def test_system_prompt_has_no_placeholders() -> None:
    """SYSTEM_PROMPT is fully static (cacheable). Any {placeholder} in it
    would either be a template artifact left from a refactor OR an
    intentional change that needs to wire to render() somehow. Either
    way, the test surfaces it for review."""
    found = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", SYSTEM_PROMPT)
    assert found == [], f"SYSTEM_PROMPT must be fully static (no placeholders); found: {found}"


def test_system_prompt_documents_all_review_tiers() -> None:
    """The prompt must explain DEEP/STANDARD/SKIM to the LLM so it knows
    the vocabulary it's expected to produce. Missing any tier → LLM
    guesses → policy gate rejects → review halts. Pin the contract."""
    for tier in ("deep", "standard", "skim"):
        assert f'"{tier}"' in SYSTEM_PROMPT


def test_system_prompt_documents_no_skip_rule() -> None:
    """Non-goal #1: this node never produces SKIP. The system prompt must
    say so explicitly — if it doesn't, the LLM has license to produce
    SKIP and the policy gate fires more often than it should."""
    assert "skip" in SYSTEM_PROMPT.lower()
    # The prompt must specifically forbid producing it, not just mention it
    assert "never produce" in SYSTEM_PROMPT.lower() or "do not produce" in SYSTEM_PROMPT.lower()


def test_system_prompt_documents_all_risk_levels() -> None:
    """Same pinning for the RiskLevel enum the LLM must produce."""
    for level in ("low", "medium", "high", "critical"):
        assert f'"{level}"' in SYSTEM_PROMPT


def test_system_prompt_documents_all_review_dimensions() -> None:
    """And for ReviewDimension. Drift here = LLM produces wrong dimension
    names = policy/schema rejection downstream."""
    for dim in (
        "code_quality",
        "security",
        "performance",
        "test_coverage",
        "best_practices",
    ):
        assert f'"{dim}"' in SYSTEM_PROMPT


def test_system_prompt_describes_reasoning_length_cap() -> None:
    """TriageResult.reasoning has Field(max_length=500). The LLM should
    know — otherwise it produces 1000-char rationales that fail schema
    validation downstream. Reasoning cap is framed as natural-language
    guidance ('two short sentences') with the 500-char schema bound as
    a safety net (Haiku is unreliable at counting chars; sentence-count
    framing tracks more reliably). Pin both forms."""
    assert "500" in SYSTEM_PROMPT
    assert "two short sentences" in SYSTEM_PROMPT.lower(), (
        "reasoning cap must use sentence-count framing, not just a raw "
        "char count — see commit b61e7fb for the Haiku-drift rationale"
    )


def test_system_prompt_output_example_is_strict_json() -> None:
    """The prompt says 'Return exactly one JSON object' and shows an
    example. The example must itself be strict JSON, or a model that
    follows it literally produces unparseable output. Non-JSON literals
    like trailing `...` (a placeholder for "more entries follow") had
    previously slipped in.
    """
    import json
    import re

    text = SYSTEM_PROMPT
    # Anchor on the multi-line `{\n  "file_tiers"` pattern so inline
    # `{` characters in prose (e.g., "Output starts with `{`") don't
    # confuse the extractor. Walk brace depth from there, treating
    # string literals as opaque.
    anchor = re.search(r'\{\s*\n\s*"file_tiers"', text)
    assert anchor is not None, "no JSON example anchor found in prompt"
    start = anchor.start()
    depth = 0
    end = -1
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    assert end > start, "could not locate a balanced JSON object in the prompt"
    json_block = text[start:end]
    # json.loads must succeed; if it raises, the example is not JSON.
    parsed = json.loads(json_block)
    assert "file_tiers" in parsed
    assert "overall_risk" in parsed
    assert "relevant_dimensions" in parsed
    assert "reasoning" in parsed


def test_system_prompt_prohibits_markdown_fenced_json() -> None:
    """Haiku has a well-known habit of wrapping structured output in
    ```json ... ``` blocks 'to be helpful'. `TriageResult.model_validate_json`
    does NOT tolerate code-fence wrappers — it raises `JSONDecodeError`.
    The prompt must explicitly prohibit fences. Pin so a future prompt
    edit that softens the wording (e.g., drops the fence ban while
    keeping 'no surrounding text') doesn't silently re-open the failure
    mode. The prohibition's specific shape — naming ```json by name —
    matters because 'no surrounding text' alone is ambiguous about
    whether code fences count as text."""
    lowered = SYSTEM_PROMPT.lower()
    # Must explicitly name markdown / code fences as forbidden
    assert "markdown" in lowered or "code fence" in lowered or "```" in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT must explicitly prohibit markdown code fences around "
        "the JSON output — Haiku wraps structured output in ```json blocks "
        "by default; 'no surrounding text' alone is ambiguous"
    )
    # Must name the specific ```json fence the model defaults to
    assert "```json" in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT must name ```json explicitly — generic 'no markdown' "
        "wording lets Haiku rationalize that ```json isn't markdown"
    )


def test_system_prompt_guides_skim_for_unreviewable_files() -> None:
    """The deterministic floor (_enforce_triage_policy rule c) rejects
    missing paths, but the prompt should also tell the model what to do
    when a file looks unreviewable (lockfiles, generated bindings,
    binary diffs with '[no textual diff available]'). Without explicit
    guidance the model either omits (caught by rule c, expensive halt)
    or reaches for 'skip' (caught by rule a, also a halt). Pin the
    SKIM-for-unreviewable guidance so future prompt edits don't silently
    drop it and re-introduce the halt surface."""
    lowered = SYSTEM_PROMPT.lower()
    # The prompt names at least one unreviewable category and routes it to skim
    assert any(
        marker in lowered
        for marker in ("lockfile", "generated", "binary", "vendored", "unreviewable")
    ), "SYSTEM_PROMPT must name at least one unreviewable-file category"
    assert "no textual diff" in lowered, (
        "SYSTEM_PROMPT must reference the '[no textual diff available]' marker "
        "that _format_file_diff renders for None patches — otherwise the model "
        "won't connect the binary-diff case to the SKIM guidance"
    )


def test_system_prompt_guides_default_to_standard_when_uncertain() -> None:
    """The deterministic floor catches omitted paths but the prompt
    should steer the model into the policy before the floor has to
    fire. Pin the 'default to standard when uncertain' instruction so
    future edits don't drop it and re-open the silent-omission failure
    mode the floor catches post-hoc."""
    lowered = SYSTEM_PROMPT.lower()
    # The prompt must explicitly tell the model to default rather than omit
    assert "default" in lowered, (
        "SYSTEM_PROMPT must instruct the model to default to a specific tier "
        "when uncertain rather than omit — without this, missing paths route "
        "to _enforce_triage_policy rule c (halt + retry cost)"
    )
    assert '"standard"' in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT must name 'standard' as the uncertain-fallback tier — "
        "generic 'pick a tier' wording lets the model default to whatever it "
        "thinks is safest, which has varied across Haiku versions"
    )


# ---------------------------------------------------------------------------
# Swap-impossibility (the M1 fix)
# ---------------------------------------------------------------------------


def test_triage_prompt_parts_rejects_positional_unpacking() -> None:
    """The whole point of @dataclass(frozen=True) instead of NamedTuple:
    positional unpacking MUST fail. If this test ever passes (unpacks
    successfully), the swap-prone shape is back."""
    parts = TriagePromptParts(system_prompt="sys", user_prompt="usr")
    with pytest.raises(TypeError):
        _, _ = parts  # type: ignore[misc]


def test_triage_prompt_parts_is_frozen() -> None:
    """Frozen prevents post-construction mutation. A future code change
    that drops frozen=True would let consumers mutate the prompts between
    render() and the LLMRequest construction — defeats reproducibility."""
    parts = TriagePromptParts(system_prompt="sys", user_prompt="usr")
    with pytest.raises(dataclasses.FrozenInstanceError):
        parts.system_prompt = "tampered"  # type: ignore[misc]


def test_triage_prompt_parts_supports_attribute_access() -> None:
    """The expected access pattern works."""
    parts = TriagePromptParts(system_prompt="sys", user_prompt="usr")
    assert parts.system_prompt == "sys"
    assert parts.user_prompt == "usr"


# ---------------------------------------------------------------------------
# render() — happy path
# ---------------------------------------------------------------------------


def test_render_returns_static_system_prompt_unchanged() -> None:
    """system_prompt is fully static — must equal SYSTEM_PROMPT exactly.
    Per the cache-boundary contract (DECISIONS#013 point 4), the wrapper
    marks it `cache_control: ephemeral`; reusing the same string produces
    cache hits. If render() mutates SYSTEM_PROMPT for any PR, caching
    breaks."""
    pr_context = _build_pr_context()
    parts = render(pr_context)
    assert parts.system_prompt == SYSTEM_PROMPT
    assert parts.system_prompt is SYSTEM_PROMPT  # same string object


def test_render_user_prompt_contains_pr_title() -> None:
    """Volatile content in user_prompt: PR title should appear so the
    LLM sees what the PR is about."""
    pr_context = _build_pr_context(pr_title="Fix XSS in login form")
    parts = render(pr_context)
    assert "Fix XSS in login form" in parts.user_prompt


def test_render_user_prompt_contains_file_list() -> None:
    """Each changed file's path appears in user_prompt under the file_list
    section."""
    cf1 = _build_changed_file(path="src/a.py", additions=10, deletions=2)
    cf2 = _build_changed_file(path="src/b.py", additions=3, deletions=5)
    pr_context = _build_pr_context(changed_files=(cf1, cf2))
    parts = render(pr_context)
    assert "src/a.py" in parts.user_prompt
    assert "src/b.py" in parts.user_prompt
    assert "+10/-2" in parts.user_prompt
    assert "+3/-5" in parts.user_prompt


def test_render_user_prompt_contains_diff_for_each_file() -> None:
    """The diff_summary section includes per-file patches."""
    cf = _build_changed_file(
        path="src/auth.py", patch="@@ -10,3 +10,5 @@\n+    if not user:\n+        raise\n"
    )
    pr_context = _build_pr_context(changed_files=(cf,))
    parts = render(pr_context)
    assert "src/auth.py" in parts.user_prompt
    assert "if not user:" in parts.user_prompt


def test_render_handles_binary_patch_none() -> None:
    """ChangedFile.patch is Optional (GitHub omits for binary diffs / oversized).
    render() must not raise; should render a placeholder marker so the LLM
    sees the file exists without seeing an empty section."""
    cf = _build_changed_file(path="assets/logo.png", patch=None)
    pr_context = _build_pr_context(changed_files=(cf,))
    parts = render(pr_context)
    assert "assets/logo.png" in parts.user_prompt
    assert "no textual diff" in parts.user_prompt.lower()


def test_render_reports_total_lines_in_user_prompt() -> None:
    """Total additions/deletions appear in the file-list header so the
    LLM has the PR-size context."""
    pr_context = _build_pr_context(total_additions=42, total_deletions=17)
    parts = render(pr_context)
    assert "+42" in parts.user_prompt
    assert "-17" in parts.user_prompt


def test_render_pure_does_not_mutate_pr_context() -> None:
    """render() is pure — must not mutate the input. PRContext is frozen
    so mutation would raise anyway, but pin it: changed_files comparison
    after render(). If a future render() reaches into pr_context to
    'normalize' something, this test breaks loud."""
    original_files = tuple(_build_changed_file(path=f"f{i}.py") for i in range(3))
    pr_context = _build_pr_context(changed_files=original_files)
    snapshot = pr_context.model_copy(deep=True)
    _ = render(pr_context)
    assert pr_context.model_dump() == snapshot.model_dump()


# ---------------------------------------------------------------------------
# render() — edge cases
# ---------------------------------------------------------------------------


def test_render_handles_empty_changed_files() -> None:
    """Edge case: a PR with no changed files (degenerate but constructible
    per the seed PRContext shape). render() should not raise; file_list
    and diff_summary become empty strings; LLMRequest.user_prompt is still
    non-empty (template has static structure around the placeholders)."""
    pr_context = _build_pr_context(changed_files=(), total_additions=0, total_deletions=0)
    parts = render(pr_context)
    assert parts.user_prompt  # non-empty
    assert "0 total" in parts.user_prompt


def test_render_handles_pr_body_none() -> None:
    """ChangedFile.pr_body is Optional. render() doesn't currently use
    pr_body, but if a future render() does, None must stringify, not
    raise. Pin the None-admission contract."""
    pr_context = _build_pr_context(pr_body=None)
    parts = render(pr_context)
    # render shouldn't depend on pr_body; user_prompt still well-formed
    assert parts.user_prompt
    assert parts.system_prompt == SYSTEM_PROMPT


def test_render_handles_status_renamed_with_previous_path() -> None:
    """Renamed files have previous_path set and the schema enforces both
    content sides + non-equal old/new paths. render() should still render
    them sensibly."""
    cf = ChangedFile(
        path="src/new_name.py",
        status="renamed",
        additions=0,
        deletions=0,
        patch="@@ -1 +1 @@\n",
        content_base="hello\n",
        content_head="hello\n",
        previous_path="src/old_name.py",
        language=None,
    )
    pr_context = _build_pr_context(changed_files=(cf,))
    parts = render(pr_context)
    assert "src/new_name.py" in parts.user_prompt


# ---------------------------------------------------------------------------
# Input boundary regression (webhook-strings-are-data-not-format-strings)
# ---------------------------------------------------------------------------


def test_pr_title_with_format_metacharacters_does_not_escape_template() -> None:
    """Per `webhook-strings-are-data-not-format-strings`: PR-sourced
    strings entering the prompt must be DATA, not template control text.
    str.format treats VALUES as opaque strings — only the TEMPLATE STRING
    is parsed for {placeholder} markers.

    Test: an attacker-supplied PR title containing literal `{system_prompt}`
    or `{file_list}` characters must survive structurally into the rendered
    user_prompt as those characters, NOT be substituted by the format()
    machinery. This is the regression test for the input-boundary invariant."""
    hostile_title = "Refactor {system_prompt} and inject {file_list} via {diff_summary}"
    pr_context = _build_pr_context(pr_title=hostile_title)
    parts = render(pr_context)
    # The literal hostile substring appears in user_prompt as DATA
    assert "{system_prompt}" in parts.user_prompt
    assert "{file_list}" in parts.user_prompt
    assert "{diff_summary}" in parts.user_prompt


def test_pr_diff_content_with_format_metacharacters_does_not_escape() -> None:
    """Same regression at the patch-content level: a malicious diff
    containing literal `{...}` markers must not interpolate. This is the
    PR-content-injection variant; PR-title is the metadata-injection variant."""
    hostile_patch = "@@ -1 +1 @@\n-old\n+{pr_title}{file_count}\n"  # attacker-controlled
    cf = _build_changed_file(patch=hostile_patch)
    pr_context = _build_pr_context(changed_files=(cf,))
    parts = render(pr_context)
    assert "{pr_title}" in parts.user_prompt
    assert "{file_count}" in parts.user_prompt


def test_render_does_not_invoke_str_format_on_values() -> None:
    """Defense-in-depth: render() uses .format(**kwargs) on the
    TEMPLATE string only. Values from PRContext are passed AS values.
    A future refactor that inadvertently does
    `template.format(pr_context.pr_title)` (treating the title AS the
    format-string) is the input-boundary failure mode. This test pins
    by sending a value that would trip such a refactor."""
    pr_context = _build_pr_context(pr_title="{}{}{}{}{}")  # positional fmt markers
    # render must not raise (it would if {} were treated as positional
    # placeholders in a format string)
    parts = render(pr_context)
    assert "{}{}{}{}{}" in parts.user_prompt


# ---------------------------------------------------------------------------
# Smoke: render output is valid LLMRequest input
# ---------------------------------------------------------------------------


def test_render_outputs_satisfy_llm_request_field_constraints() -> None:
    """Both prompts must be ≥1 char (per LLMRequest.system_prompt and
    .user_prompt min_length=1). Render must not produce an empty either.
    Pins downstream compatibility."""
    pr_context = _build_pr_context()
    parts = render(pr_context)
    assert len(parts.system_prompt) >= 1
    assert len(parts.user_prompt) >= 1


# ---------------------------------------------------------------------------
# Module surfaces
# ---------------------------------------------------------------------------


def test_module_exports_all_documented_surfaces() -> None:
    """The spec's Reference Reconciliation lists these surfaces. The
    module's __all__ must include them so import * works correctly and
    the public surface stays explicit."""
    from outrider.prompts import triage

    expected = {
        "MAX_TOKENS",
        "SYSTEM_PROMPT",
        "TEMPERATURE",
        "TEMPLATE",
        "TriagePromptParts",
        "USER_TEMPLATE",
        "VERSION",
        "render",
    }
    assert set(triage.__all__) == expected
