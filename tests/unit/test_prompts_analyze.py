"""prompts/analyze.py contract tests.

Mirror of `tests/unit/test_prompts_triage.py` for the analyze prompt
module. Pins the surface from `specs/2026-05-19-analyze-node.md` §5:
constants (VERSION, MAX_TOKENS, TEMPERATURE), templates
(SYSTEM_PROMPT_INVARIANTS, USER_TEMPLATE + alias TEMPLATE,
DEGRADED_USER_TEMPLATE), AnalyzePromptParts swap-impossible dataclass,
the pure render() helper for clean outcomes, and render_degraded() for
parse-failed / has-error outcomes.

Tests organized as: surface contracts (what callers can rely on),
swap-impossibility (the dataclass-not-tuple discipline), render
happy paths (one per render fn), input-boundary regression (PR text
with format-string metacharacters cannot escape the template
structure).
"""

import dataclasses
import re

import pytest

from outrider.prompts.analyze import (
    DEGRADED_USER_TEMPLATE,
    MAX_TOKENS,
    SYSTEM_PROMPT_INVARIANTS,
    TEMPERATURE,
    TEMPLATE,
    USER_TEMPLATE,
    VERSION,
    AnalyzePromptParts,
    render,
    render_degraded,
)

# ---------------------------------------------------------------------------
# Surface contracts: constants
# ---------------------------------------------------------------------------


def test_version_is_named_v1() -> None:
    """VERSION flows to LLMRequest.prompt_template_version. Pin the v1
    name so future renames break the test and force a registry decision."""
    assert VERSION == "analyze-v1"


def test_max_tokens_bounded_within_llm_request_limit() -> None:
    """MAX_TOKENS must respect LLMRequest.max_tokens Field constraint
    (gt=0, le=8192). Anything outside that range would fail at LLMRequest
    construction. Pin the chosen value so a typo doesn't break the chain."""
    assert 0 < MAX_TOKENS <= 8192
    assert MAX_TOKENS == 8192  # current chosen value; intentional pin


def test_temperature_bounded_within_llm_request_limit() -> None:
    """TEMPERATURE must respect LLMRequest.temperature Field constraint
    (ge=0.0, le=1.0). Analyze uses temperature=0.0 for determinism."""
    assert 0.0 <= TEMPERATURE <= 1.0
    assert TEMPERATURE == 0.0


def test_template_is_user_template_alias() -> None:
    """TEMPLATE is documented as an alias of USER_TEMPLATE. Different
    constants would let one drift from the other; the test pins they're
    the same object so renames stay coupled."""
    assert TEMPLATE is USER_TEMPLATE


def test_user_template_has_required_placeholders() -> None:
    """The USER_TEMPLATE must carry every placeholder render() supplies.
    If render() drifts (adds a new {placeholder} not in the template, or
    vice versa), str.format raises at first call. Pin the placeholder
    set so additions go through a coordinated edit."""
    expected_placeholders = {
        "file_path",
        "pass_index",
        "scope_unit_context",
        "query_match_id_list",
        "diff_hunks",
    }
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", USER_TEMPLATE))
    assert found == expected_placeholders, (
        f"USER_TEMPLATE placeholders ({found}) drift from render()'s "
        f"kwargs ({expected_placeholders}); either rename the template "
        f"or update render()."
    )


def test_degraded_user_template_has_required_placeholders() -> None:
    """Same pinning for the degraded-outcome template + render_degraded()."""
    expected_placeholders = {
        "file_path",
        "pass_index",
        "degradation_reason",
        "bounded_hunks",
    }
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", DEGRADED_USER_TEMPLATE))
    assert found == expected_placeholders, (
        f"DEGRADED_USER_TEMPLATE placeholders ({found}) drift from "
        f"render_degraded()'s kwargs ({expected_placeholders})."
    )


def test_system_prompt_invariants_has_no_placeholders() -> None:
    """SYSTEM_PROMPT_INVARIANTS is fully static (cacheable). Any
    {placeholder} would either be a template artifact left from a
    refactor OR an intentional change that needs to wire to render().
    Either way, the test surfaces it for review."""
    found = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", SYSTEM_PROMPT_INVARIANTS)
    # The system prompt embeds the AnalyzeResponseRaw shape as an example;
    # `{byte_start}` and `{byte_end}` appear inside that shape example as
    # literal type markers, NOT as str.format placeholders. The render()
    # function never calls .format() on SYSTEM_PROMPT_INVARIANTS, so these
    # literals are safe — but we surface them so a future refactor that
    # tries to .format() the system prompt fails-loud.
    allowed_literals = {"byte_start", "byte_end"}
    unexpected = set(found) - allowed_literals
    assert unexpected == set(), (
        f"SYSTEM_PROMPT_INVARIANTS contains unexpected placeholders: {unexpected}. "
        f"Allowed literal markers (inside the JSON example): {allowed_literals}. "
        f"If a real placeholder needs adding, route it through render()'s kwargs "
        f"rather than mutating the cacheable static head."
    )


def test_system_prompt_documents_all_finding_types() -> None:
    """The prompt must enumerate every FindingType so the LLM knows the
    constrained vocabulary. Drift here = LLM produces unknown finding_type =
    parser rejects with `finding_type_not_in_enum`. Pin against drift."""
    expected = (
        "sql_injection",
        "xss",
        "hardcoded_secret",
        "auth_bypass",
        "path_traversal",
        "missing_input_validation",
        "n_plus_one_query",
        "blocking_call_in_async",
        "unused_import",
        "missing_error_handling",
        "missing_test",
        "deprecated_api",
    )
    for finding_type in expected:
        assert f"`{finding_type}`" in SYSTEM_PROMPT_INVARIANTS, (
            f"FindingType `{finding_type}` missing from SYSTEM_PROMPT_INVARIANTS; "
            f"the LLM won't know it as a valid value and parser rejection rates "
            f"climb until the prompt is fixed."
        )


def test_system_prompt_documents_all_evidence_tiers() -> None:
    """Same pinning for EvidenceTier — the three tiers must be named so
    the LLM picks from the constrained set."""
    for tier in ("OBSERVED", "INFERRED", "JUDGED"):
        assert f"`{tier}`" in SYSTEM_PROMPT_INVARIANTS, (
            f"EvidenceTier `{tier}` missing from SYSTEM_PROMPT_INVARIANTS"
        )


def test_system_prompt_prohibits_severity_proposal() -> None:
    """Per `severity-set-by-policy`, the model must NEVER propose
    severity — the deterministic table assigns it. The prompt must
    explicitly forbid the field so the model doesn't include it."""
    lowered = SYSTEM_PROMPT_INVARIANTS.lower()
    assert "severity" in lowered
    # Must explicitly forbid proposing it, not just mention the concept
    assert "do not" in lowered or "never" in lowered or "rejected" in lowered, (
        "SYSTEM_PROMPT must explicitly forbid model-proposed severity"
    )


def test_system_prompt_prohibits_confidence_proposal() -> None:
    """Per `confidence-is-computed-not-assigned`, confidence is computed
    deterministically from evidence_tier. Model must not propose it."""
    assert "confidence" in SYSTEM_PROMPT_INVARIANTS.lower()


def test_system_prompt_prohibits_dimension_proposal() -> None:
    """Per `evidence-tier-schema-enforced` + `FINDING_TYPE_TO_DIMENSION`,
    dimension is looked up deterministically from finding_type."""
    assert "dimension" in SYSTEM_PROMPT_INVARIANTS.lower()


def test_system_prompt_prohibits_markdown_fenced_json() -> None:
    """Sonnet (like Haiku) sometimes wraps structured output in
    ```json ... ``` blocks. `AnalyzeResponseRaw.model_validate_json`
    does NOT tolerate code-fence wrappers — it raises `JSONDecodeError`
    and the parser fires `AnalyzeResponseRejectedEvent`. The prompt
    must explicitly prohibit fences. Same pinning shape as triage's."""
    lowered = SYSTEM_PROMPT_INVARIANTS.lower()
    assert "markdown" in lowered or "fence" in lowered or "```" in SYSTEM_PROMPT_INVARIANTS
    assert "```json" in SYSTEM_PROMPT_INVARIANTS or "fence" in lowered, (
        "SYSTEM_PROMPT_INVARIANTS must name ```json or 'fence' explicitly — "
        "generic 'no markdown' wording lets the model rationalize that ```json "
        "isn't markdown"
    )


def test_system_prompt_names_the_response_schema_top_level_key() -> None:
    """The model needs to know the top-level shape — `{"findings": [...]}`
    not `[...]` or `{"results": [...]}`. Pin the key so renames in
    `AnalyzeResponseRaw` propagate through the prompt."""
    assert '"findings"' in SYSTEM_PROMPT_INVARIANTS


# ---------------------------------------------------------------------------
# Swap-impossibility (the M1 fix mirroring triage)
# ---------------------------------------------------------------------------


def test_analyze_prompt_parts_rejects_positional_unpacking() -> None:
    """The whole point of @dataclass(frozen=True, slots=True) instead of
    NamedTuple: positional unpacking MUST fail. If this test ever passes
    (unpacks successfully), the swap-prone shape is back."""
    parts = AnalyzePromptParts(system_prompt="sys", user_prompt="usr")
    with pytest.raises(TypeError):
        _, _ = parts  # type: ignore[misc]


def test_analyze_prompt_parts_is_frozen() -> None:
    """Frozen prevents post-construction mutation. A future code change
    that drops frozen=True would let consumers mutate the prompts between
    render() and LLMRequest construction — defeats reproducibility."""
    parts = AnalyzePromptParts(system_prompt="sys", user_prompt="usr")
    with pytest.raises(dataclasses.FrozenInstanceError):
        parts.system_prompt = "tampered"  # type: ignore[misc]


def test_analyze_prompt_parts_supports_attribute_access() -> None:
    """The expected access pattern works."""
    parts = AnalyzePromptParts(system_prompt="sys", user_prompt="usr")
    assert parts.system_prompt == "sys"
    assert parts.user_prompt == "usr"


# ---------------------------------------------------------------------------
# render() — happy path
# ---------------------------------------------------------------------------


def test_render_returns_static_system_prompt_unchanged() -> None:
    """system_prompt must equal SYSTEM_PROMPT_INVARIANTS exactly so the
    cache-boundary contract (DECISIONS#013 point 4) produces hits."""
    parts = render(
        file_path="src/example.py",
        scope_unit_context="<scope unit body>",
        query_match_id_list="(none)",
        diff_hunks="@@ -1,1 +1,1 @@",
        pass_index=0,
    )
    assert parts.system_prompt == SYSTEM_PROMPT_INVARIANTS
    assert parts.system_prompt is SYSTEM_PROMPT_INVARIANTS  # same string object


def test_render_user_prompt_contains_file_path() -> None:
    """File path must appear so the model knows which file it's reviewing."""
    parts = render(
        file_path="src/auth/login.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "src/auth/login.py" in parts.user_prompt


def test_render_user_prompt_contains_pass_index() -> None:
    """Pass index must appear so the model can distinguish first-analyze
    from trace-round-2 context. Spec §7 step 1 sets phase_id with the
    same suffix; the prompt's `analyze-pass-N` mirror keeps them aligned."""
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=3,
    )
    assert "analyze-pass-3" in parts.user_prompt


def test_render_user_prompt_contains_scope_unit_context() -> None:
    """The scope unit context (bodies + callers/callees + imports +
    decorators) must appear so the model has the structural information
    to reason about findings."""
    sentinel = "def login(user, password):\n    # SENTINEL"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=sentinel,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.user_prompt


def test_render_user_prompt_contains_query_match_id_list() -> None:
    """Pre-fired query matches must appear so the model can cite real
    IDs when claiming OBSERVED."""
    sentinel = "python.security.sql_injection:42"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list=sentinel,
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.user_prompt


def test_render_user_prompt_contains_diff_hunks() -> None:
    """Scope-unit-clipped diff hunks must appear so the model sees what
    changed."""
    sentinel = "@@ -10,3 +10,5 @@\n+    if not user:\n+        raise"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=sentinel,
        pass_index=0,
    )
    assert sentinel in parts.user_prompt


# ---------------------------------------------------------------------------
# render_degraded() — happy path
# ---------------------------------------------------------------------------


def test_render_degraded_returns_static_system_prompt_unchanged() -> None:
    """Degraded path uses the SAME system prompt invariants — the
    degraded directive lives in the user prompt, not the system prompt.
    Keeps the cache key shape consistent across clean/degraded calls."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="@@ -1,1 +1,1 @@",
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert parts.system_prompt == SYSTEM_PROMPT_INVARIANTS
    assert parts.system_prompt is SYSTEM_PROMPT_INVARIANTS


def test_render_degraded_user_prompt_signals_degraded_mode() -> None:
    """User prompt must mark itself as DEGRADED so the model knows
    OBSERVED/INFERRED claims will be rejected."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="",
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert "DEGRADED" in parts.user_prompt
    assert "parse_failed" in parts.user_prompt


def test_render_degraded_user_prompt_admits_only_judged() -> None:
    """The degraded path must instruct the model to use JUDGED only.
    Without this, the model emits OBSERVED/INFERRED and the parser
    rejects every proposal — a halt surface that's avoidable here."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="",
        pass_index=0,
        degradation_reason="tree_has_error_in_changed_regions",
    )
    assert "JUDGED" in parts.user_prompt


def test_render_degraded_user_prompt_contains_bounded_hunks() -> None:
    """The bounded changed hunks (≤100 lines, ≤8192 chars) must appear
    so the model sees what changed even without structural context."""
    sentinel = "@@ -1,3 +1,5 @@\n+raise"
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks=sentinel,
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert sentinel in parts.user_prompt


# ---------------------------------------------------------------------------
# Input boundary regression (webhook-strings-are-data-not-format-strings)
# ---------------------------------------------------------------------------


def test_render_hostile_scope_unit_context_does_not_escape_template() -> None:
    """Per `webhook-strings-are-data-not-format-strings`: source-controlled
    strings entering the prompt via .format(**kwargs) must be DATA, not
    template control text. `{file_path}` or `{diff_hunks}` literals
    inside scope_unit_context must survive AS those characters — they
    must not be substituted by the format() machinery."""
    hostile = "def f(): {file_path} {diff_hunks} {pass_index}"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=hostile,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    # The literal hostile content survives in user_prompt as DATA
    assert "{file_path}" in parts.user_prompt
    assert "{diff_hunks}" in parts.user_prompt
    assert "{pass_index}" in parts.user_prompt


def test_render_hostile_diff_hunks_does_not_escape_template() -> None:
    """Same regression at the diff-content level: a malicious patch
    containing literal `{...}` markers must not interpolate."""
    hostile = "@@ -1 +1 @@\n+evil = {file_path}\n+exec({scope_unit_context})\n"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=hostile,
        pass_index=0,
    )
    assert "{file_path}" in parts.user_prompt
    assert "{scope_unit_context}" in parts.user_prompt


def test_render_hostile_positional_metacharacters_do_not_trip_format() -> None:
    """Defense-in-depth: a value containing positional format markers
    (`{}{}{}`) must not be re-interpreted as a format-string. render()
    uses .format(**kwargs) — kwargs route by name only; positional `{}`
    markers would raise IndexError if treated as format string. Pin so
    a future refactor that inadvertently does `template.format(value)`
    (treating the value AS the format string) fails this test."""
    parts = render(
        file_path="src/x.py",
        scope_unit_context="{}{}{}",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "{}{}{}" in parts.user_prompt


def test_render_degraded_hostile_bounded_hunks_does_not_escape() -> None:
    """Same input-boundary regression for the degraded render path."""
    hostile = "@@ -1 +1 @@\n+{file_path}{degradation_reason}\n"
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks=hostile,
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert "{file_path}" in parts.user_prompt
    assert "{degradation_reason}" in parts.user_prompt


# ---------------------------------------------------------------------------
# Smoke: render outputs are valid LLMRequest input
# ---------------------------------------------------------------------------


def test_render_outputs_satisfy_llm_request_field_constraints() -> None:
    """Both prompts must be ≥1 char (per LLMRequest.system_prompt and
    .user_prompt min_length=1). Render must not produce an empty either."""
    parts = render(
        file_path="src/x.py",
        scope_unit_context="ctx",
        query_match_id_list="ids",
        diff_hunks="hunks",
        pass_index=0,
    )
    assert len(parts.system_prompt) >= 1
    assert len(parts.user_prompt) >= 1


def test_render_degraded_outputs_satisfy_llm_request_field_constraints() -> None:
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="hunks",
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert len(parts.system_prompt) >= 1
    assert len(parts.user_prompt) >= 1


# ---------------------------------------------------------------------------
# Module surfaces
# ---------------------------------------------------------------------------


def test_module_exports_all_documented_surfaces() -> None:
    """The spec's Reference Reconciliation lists these surfaces. The
    module's __all__ must include them so import * works correctly and
    the public surface stays explicit."""
    from outrider.prompts import analyze

    expected = {
        "DEGRADED_USER_TEMPLATE",
        "MAX_TOKENS",
        "SYSTEM_PROMPT_INVARIANTS",
        "TEMPERATURE",
        "TEMPLATE",
        "USER_TEMPLATE",
        "VERSION",
        "AnalyzePromptParts",
        "render",
        "render_degraded",
    }
    assert set(analyze.__all__) == expected
