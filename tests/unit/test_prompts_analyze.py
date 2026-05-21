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
    SYSTEM_FILE_CONTEXT_TEMPLATE,
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
    """Volatile (pass-specific) placeholders on USER_TEMPLATE. Stable
    file-scoped placeholders live on SYSTEM_FILE_CONTEXT_TEMPLATE for
    cross-pass caching."""
    expected_placeholders = {"pass_index", "diff_hunks"}
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", USER_TEMPLATE))
    assert found == expected_placeholders


def test_system_file_context_template_has_required_placeholders() -> None:
    """Stable-per-file placeholders. Combined with SYSTEM_PROMPT_INVARIANTS
    they form the cacheable system_prompt block; reuse across passes for
    the same file produces cache hits."""
    expected_placeholders = {"file_path", "scope_unit_context", "query_match_id_list"}
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", SYSTEM_FILE_CONTEXT_TEMPLATE))
    assert found == expected_placeholders


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
    """Same pinning for EvidenceTier — the three tier values must be
    named so the LLM picks from the constrained set. Values are
    lowercase per `docs/spec.md` §7.3 (matches `FindingType`,
    `FindingSeverity`, `ReviewTier`); a prior version of this prompt
    used uppercase and the parser would reject every proposal with
    `evidence_tier_not_in_enum`."""
    for tier in ("observed", "inferred", "judged"):
        assert f"`{tier}`" in SYSTEM_PROMPT_INVARIANTS, (
            f"EvidenceTier `{tier}` missing from SYSTEM_PROMPT_INVARIANTS"
        )


def test_safe_code_fence_escapes_triple_backticks_in_body() -> None:
    """`prompts.safe_code_fence` must produce a fence longer than any
    backtick run in the body, so PR-controlled content can't close the
    surrounding fence and break out into the prompt structure. Pins the
    `webhook-strings-are-data-not-format-strings` defense for prompt
    rendering.
    """
    from outrider.prompts import safe_code_fence

    # Body with no backticks: default 3-backtick fence.
    plain = safe_code_fence("def f(): pass", lang="python")
    assert plain.startswith("```python\n")
    assert plain.endswith("\n```")

    # Body containing a 3-backtick run: fence grows to 4.
    hostile = "def f():\n    '''\n    Example: ```python\n    print(1)\n    ```\n    '''\n"
    wrapped = safe_code_fence(hostile, lang="python")
    # The fence must NOT appear inside the body verbatim.
    fence = wrapped.split("python\n", 1)[0]
    assert len(fence) >= 4, "fence must grow past 3 backticks when body contains ```"
    # The body's `````` is preserved literally inside the wrapper.
    assert hostile in wrapped


def test_safe_code_fence_grows_past_arbitrary_backtick_runs() -> None:
    """Defensive: a body containing a 5-backtick run still gets a fence
    that's longer (≥6). The loop must grow the fence until it does not
    appear in the body.
    """
    from outrider.prompts import safe_code_fence

    hostile = "before " + ("`" * 5) + " after"
    wrapped = safe_code_fence(hostile, lang="diff")
    fence = wrapped.split("diff\n", 1)[0]
    assert fence.count("`") >= 6, f"expected ≥6 backticks, got {fence!r}"
    assert hostile in wrapped


def test_system_prompt_forbids_inferred_in_v1() -> None:
    """V1 admission stub auto-rejects every `inferred` proposal with
    `trace_path_not_admissible` (parser §6 step 4, deferred until the
    trace-node spec lands). Telling the model to emit `inferred` would
    burn per-file budget on guaranteed-reject calls. Pin the prohibition
    so a future prompt edit that silently re-permits inferred — before
    the trace resolver actually exists — fails this test rather than
    inflating rejection events in production.
    """
    # Three load-bearing signals in the prompt:
    # 1. The output-shape enum union DOES NOT list inferred as an option.
    # 2. The clean prompt has an explicit "Do NOT emit inferred" sentence.
    # 3. The degraded-mode reminder continues to say `inferred` is rejected.
    assert "<observed|judged>" in SYSTEM_PROMPT_INVARIANTS
    assert "<observed|inferred|judged>" not in SYSTEM_PROMPT_INVARIANTS
    assert 'Do NOT emit `evidence_tier="inferred"`' in SYSTEM_PROMPT_INVARIANTS


def test_system_prompt_prohibits_severity_proposal() -> None:
    """Per `severity-set-by-policy`, the model must NEVER propose
    severity — the deterministic table assigns it. The prompt must
    explicitly forbid the field at field-local resolution: a prohibition
    verb (`propose`, `do not`, `never`) must appear within 80 chars of
    the `severity` field name, AND the prompt must state that a
    `severity` field in the model's output is rejected."""
    import re

    text = SYSTEM_PROMPT_INVARIANTS.lower()
    assert re.search(r"(propose|do not|never)[^\n]{0,80}severity", text), (
        "SYSTEM_PROMPT must forbid `severity` within an 80-char window "
        "of a prohibition verb (propose / do not / never)"
    )
    assert re.search(r"`?severity`?[^\n]{0,80}rejected", text), (
        "SYSTEM_PROMPT must state that a model-supplied `severity` field is rejected"
    )


def test_system_prompt_prohibits_confidence_proposal() -> None:
    """Per `confidence-is-computed-not-assigned`, confidence is computed
    deterministically from evidence_tier. Model must not propose it.
    Field-local prohibition: a prohibition verb within 80 chars of
    `confidence` AND the rejection contract."""
    import re

    text = SYSTEM_PROMPT_INVARIANTS.lower()
    assert re.search(r"(propose|do not|never)[^\n]{0,80}confidence", text), (
        "SYSTEM_PROMPT must forbid `confidence` within an 80-char window "
        "of a prohibition verb (propose / do not / never)"
    )
    assert re.search(r"`?confidence`?[^\n]{0,80}rejected", text), (
        "SYSTEM_PROMPT must state that a model-supplied `confidence` field is rejected"
    )


def test_system_prompt_prohibits_dimension_proposal() -> None:
    """Per `evidence-tier-schema-enforced` + `FINDING_TYPE_TO_DIMENSION`,
    dimension is looked up deterministically from finding_type. Field-
    local prohibition: a prohibition verb within 80 chars of `dimension`
    AND the rejection contract."""
    import re

    text = SYSTEM_PROMPT_INVARIANTS.lower()
    assert re.search(r"(propose|do not|never)[^\n]{0,80}dimension", text), (
        "SYSTEM_PROMPT must forbid `dimension` within an 80-char window "
        "of a prohibition verb (propose / do not / never)"
    )
    assert re.search(r"`?dimension`?[^\n]{0,80}rejected", text), (
        "SYSTEM_PROMPT must state that a model-supplied `dimension` field is rejected"
    )


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


def test_system_prompt_trace_candidate_field_matches_raw_schema() -> None:
    """Pin: the prompt's trace_candidates example uses
    `candidate_path_raw`, not `candidate_path`.

    `TraceCandidateProposalRaw` has `extra="forbid"` and requires the
    `_raw` suffix; a model that follows the prompt literally and emits
    `candidate_path` causes `AnalyzeResponseRaw.model_validate_json` to
    reject the entire response.
    """
    assert "candidate_path_raw" in SYSTEM_PROMPT_INVARIANTS
    # The bare (admitted-layer) field name must not appear as an object
    # key in the example.
    assert '"candidate_path":' not in SYSTEM_PROMPT_INVARIANTS


def test_system_prompt_output_example_is_strict_json() -> None:
    """The prompt says 'Return exactly this JSON' and shows an example.
    The example must itself be strict JSON, or a model that follows it
    literally produces unparseable output. Non-JSON union syntax like
    `["step", "..."] | null` had previously slipped in.
    """
    # Anchor on the multi-line `{\n  "findings"` pattern so inline
    # `{` characters in prose (e.g., "Output starts with `{`") don't
    # confuse the extractor. Walk brace depth from there, treating
    # string literals as opaque (no nested-brace counting inside).
    import json
    import re

    text = SYSTEM_PROMPT_INVARIANTS
    anchor = re.search(r'\{\s*\n\s*"findings"', text)
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
    assert "findings" in parsed
    # Pin span ordering: the prompt's own rule says byte_start < byte_end,
    # so the example must not contradict it (a zero-width span teaches
    # the model an invalid shape).
    for finding in parsed["findings"]:
        span = finding["span"]
        assert span["byte_start"] < span["byte_end"]


def test_system_prompt_field_char_bounds_match_raw_schema() -> None:
    """The prompt advertises `title ≤120`, `description ≤1000`,
    `evidence ≤2000`. The raw schema enforces those exact bounds via
    `Field(max_length=...)`. If schema changes but prompt doesn't, the
    model produces output the schema rejects — pin the numbers as a
    cross-source agreement so future edits fail loudly.
    """
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

    fields = AnalyzeFindingProposalRaw.model_fields
    expected = {
        "title": 120,
        "description": 1000,
        "evidence": 2000,
    }
    for field_name, max_len in expected.items():
        # The prompt advertises the bound in the example placeholder
        # like `<short summary, ≤120 chars>` or `<explanation, ≤1000 chars>`.
        assert f"≤{max_len}" in SYSTEM_PROMPT_INVARIANTS, (
            f"prompt missing `≤{max_len}` for {field_name}"
        )
        # Schema's actual max_length matches the prompt claim.
        annotated_meta = fields[field_name].metadata
        max_length_values = [getattr(m, "max_length", None) for m in annotated_meta]
        assert max_len in max_length_values, (
            f"schema {field_name} max_length doesn't match prompt's {max_len}"
        )


def test_system_prompt_findings_cap_matches_raw_schema() -> None:
    """Prompt states 'Up to 50 findings per response' and 'Up to 20
    trace_candidates per finding'. Schema enforces both. Pin so a
    silent schema bump (e.g., 50 → 60) doesn't drift the prompt.
    """
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw, AnalyzeResponseRaw

    # findings cap (top-level)
    findings_field = AnalyzeResponseRaw.model_fields["findings"]
    findings_caps = [getattr(m, "max_length", None) for m in findings_field.metadata]
    assert 50 in findings_caps
    assert "50 findings" in SYSTEM_PROMPT_INVARIANTS

    # trace_candidates cap (per finding)
    tc_field = AnalyzeFindingProposalRaw.model_fields["trace_candidates"]
    tc_caps = [getattr(m, "max_length", None) for m in tc_field.metadata]
    assert 20 in tc_caps
    assert "20 trace_candidates" in SYSTEM_PROMPT_INVARIANTS


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


def test_render_system_prompt_starts_with_invariants() -> None:
    """system_prompt begins with the static SYSTEM_PROMPT_INVARIANTS so
    the cacheable prefix is byte-identical across calls; the per-file
    suffix follows it."""
    parts = render(
        file_path="src/example.py",
        scope_unit_context="<scope unit body>",
        query_match_id_list="(none)",
        diff_hunks="@@ -1,1 +1,1 @@",
        pass_index=0,
    )
    assert parts.system_prompt.startswith(SYSTEM_PROMPT_INVARIANTS)


def test_render_system_prompt_contains_file_path() -> None:
    """File path lives in system_prompt (per-file-stable) so the cacheable
    boundary covers it. Reviews of the same file across passes hit cache."""
    parts = render(
        file_path="src/auth/login.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "src/auth/login.py" in parts.system_prompt


def test_render_system_prompt_contains_scope_unit_context() -> None:
    """Scope-unit context (bodies + same-file callers/callees + imports +
    decorators) is per-file-stable — system_prompt block."""
    sentinel = "def login(user, password):\n    # SENTINEL"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=sentinel,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.system_prompt


def test_render_system_prompt_contains_query_match_id_list() -> None:
    """Pre-fired query matches are file-scoped and stable across passes —
    system_prompt block."""
    sentinel = "python.security.sql_injection:42"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list=sentinel,
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.system_prompt


def test_render_user_prompt_contains_pass_index() -> None:
    """Pass index is volatile per analyze-pass — user_prompt block."""
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=3,
    )
    assert "analyze-pass-3" in parts.user_prompt


def test_render_user_prompt_contains_diff_hunks() -> None:
    """Scope-unit-clipped diff hunks are pass-specific (the included
    units may change between trace-loop iterations) — user_prompt block."""
    sentinel = "@@ -10,3 +10,5 @@\n+    if not user:\n+        raise"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=sentinel,
        pass_index=0,
    )
    assert sentinel in parts.user_prompt


def test_render_user_prompt_diff_hunks_are_fenced() -> None:
    """Sibling to `render_degraded`: clean-mode diff_hunks are PR-controlled
    and must be wrapped in a `safe_code_fence(lang="diff")` block so a
    diff line containing `## Heading` or ` ``` ` markdown can't forge
    sections that mimic the prompt's own structure.

    Pin: the rendered user_prompt MUST contain a ```diff opening fence
    immediately before the diff content (or a dynamic-length variant if
    the content itself has triple-backtick runs).
    """
    sentinel = "@@ -1,1 +1,2 @@\n+    return 42"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=sentinel,
        pass_index=0,
    )
    # Default 3-backtick fence opens with `\n```diff\n` before content
    # and closes with `\n```\n`.
    assert "```diff\n" in parts.user_prompt
    assert sentinel in parts.user_prompt


def test_render_user_prompt_fence_escapes_hostile_diff_hunks() -> None:
    """If the diff hunk itself contains a triple-backtick run (e.g., a
    docstring with embedded markdown), `safe_code_fence` must grow the
    surrounding fence so the content can't close it early."""
    hostile = "@@ -1,1 +1,1 @@\n+    '''\n+    Example: ```\n+    '''"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks=hostile,
        pass_index=0,
    )
    # The fence must be at least 4 backticks long since the body has a
    # 3-backtick run.
    assert "````diff\n" in parts.user_prompt
    # Hostile content preserved verbatim, not stripped.
    assert hostile in parts.user_prompt


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
    `observed`/`inferred` claims will be rejected. The "DEGRADED"
    marker is a section header (uppercase by convention for prompt
    section labels); the enum-VALUE references in the prose stay
    lowercase to match the constrained vocabulary."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="",
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert "DEGRADED" in parts.user_prompt
    assert "parse_failed" in parts.user_prompt


def test_render_degraded_user_prompt_admits_only_judged() -> None:
    """The degraded path must instruct the model to use `judged` only.
    Without this, the model emits `observed`/`inferred` and the parser
    rejects every proposal — a halt surface that's avoidable here."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="",
        pass_index=0,
        degradation_reason="tree_has_error_in_changed_regions",
    )
    assert "judged" in parts.user_prompt


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
    """`webhook-strings-are-data-not-format-strings`: PR-sourced strings
    entering the prompt via .format(**kwargs) survive AS literal data;
    `{file_path}` / `{diff_hunks}` / `{pass_index}` markers in the input
    must not interpolate. scope_unit_context lives in system_prompt."""
    hostile = "def f(): {file_path} {diff_hunks} {pass_index}"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=hostile,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "{file_path}" in parts.system_prompt
    assert "{diff_hunks}" in parts.system_prompt
    assert "{pass_index}" in parts.system_prompt


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
    """Positional `{}` markers in input must not be re-interpreted as a
    format string. `template.format(**kwargs)` routes by name only;
    treating a value AS the format string would raise IndexError."""
    parts = render(
        file_path="src/x.py",
        scope_unit_context="{}{}{}",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "{}{}{}" in parts.system_prompt


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
        "SYSTEM_FILE_CONTEXT_TEMPLATE",
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
