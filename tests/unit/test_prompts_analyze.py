"""Analyze prompt contracts from specs/2026-05-19-analyze-node.md §5.

Cover constants, clean/degraded templates, the AnalyzePromptParts field contract,
rendering, and untrusted PR text remaining data inside the prompt structure."""

import dataclasses
import hashlib
import re

import pytest

from outrider.prompts.analyze import (
    DEGRADED_USER_TEMPLATE,
    FILE_CONTEXT_TEMPLATE,
    MAX_TOKENS,
    POST_TRACE_SYSTEM_PROMPT_SUFFIX,
    SYSTEM_PROMPT_CALIBRATION,
    SYSTEM_PROMPT_COMPATIBILITY,
    SYSTEM_PROMPT_EXEMPLARS,
    SYSTEM_PROMPT_INVARIANTS,
    SYSTEM_PROMPT_STABLE_PREFIX,
    TEMPERATURE,
    TEMPLATE,
    USER_TEMPLATE,
    VERSION,
    AnalyzePromptParts,
    render,
    render_degraded,
    render_post_trace,
)

# ---------------------------------------------------------------------------
# Surface contracts: constants
# ---------------------------------------------------------------------------


def test_version_is_named_analyze_v19() -> None:
    """Pin analyze-v19 as recorded provenance and cache identity.

    VERSION identifies the prompt contract in LLMRequest.prompt_template_version,
    the analyze_file_cache key and OpenAI prompt_cache_key. Replay does not re-render
    templates. Taxonomy and SQL-injection guidance follow DECISIONS.md#053 and
    DECISIONS.md#041; wording has separate checks below."""
    assert VERSION == "analyze-v19"


def test_system_prompt_ssrf_carveout_and_authority_exception() -> None:
    """Keep both sides of the SSRF destination-control rule.

    Appending unvalidated data to a fixed host's path or ordinary query parameter is
    the safe case. Data that can influence scheme, host or port remains SSRF by any
    route; the authority rule must not become an exhaustive token checklist."""
    # whitespace-normalized so phrases wrapped across lines match as substrings.
    text = " ".join(SYSTEM_PROMPT_INVARIANTS.lower().split())
    # (1) destination-control rule lead + the safe case (path/query of a fixed host)
    assert "scheme, host, or port" in text  # flag ONLY when the user controls these
    assert "appended as a path segment" in text  # the path-safe half
    assert "ordinary query parameter" in text  # the query-safe half
    assert "even when the value is unvalidated" in text  # v7: lack of validation != ssrf
    assert "do not flag" in text  # the worked DON'T example Haiku responds to
    assert "using the value as its target" in text  # but a proxy ?url= IS still ssrf
    # (2) principled, non-exhaustive authority rule — preserves real-SSRF recall
    assert "still ssrf whenever the value can reach" in text
    assert "by any means" in text
    assert "when genuinely unsure" in text
    assert "host/port/scheme" in text  # host, port, scheme all in scope
    assert "before the safe case" in text  # metadata escalation evaluated first


def test_system_prompt_warns_parameterized_queries_are_not_sqli() -> None:
    """Parameterized-query guidance preserves the distinctions in DECISIONS.md#041.

    Separate parameter binding is not SQL injection; building input into SQL text is.
    The exception applies only to injection, so other defects on the same query may
    still produce findings."""
    text = SYSTEM_PROMPT_INVARIANTS.lower()
    assert "parameterized queries are not injectable" in text  # (1) safe-side instruction
    # (2) still-injectable side — anchored to the injectable INSTRUCTION ("built INTO the SQL
    # string"), not just the bare words, so a refactor moving "f-string"/"concatenation" onto the
    # SAFE side (a blanket suppression) can't slip past this guard.
    assert "built into the sql string" in text  # the directive that string-built SQL IS injectable
    assert "f-string" in text and "concatenation" in text  # named as injectable forms
    # (3) scoped to INJECTION — must not generalize "safe" to "skip the query"; pins the
    # clause that recovers the n_plus_one recall the unscoped "SAFE" wording dropped.
    assert "only about injection" in text
    assert "n+1" in text


def test_rendered_xss_safe_rules_do_not_treat_markup_constructor_as_escaping() -> None:
    """Keep safe escaping guidance without treating a safe marker as an escaper."""
    parts = render(
        file_path="app.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    xss = parts.system_prompt.split("### xss\n", 1)[1].split("\n### ", 1)[0]
    safe_rules = [
        bullet for bullet in re.split(r"(?m)^- ", xss) if bullet.startswith("Do NOT flag")
    ]
    assert safe_rules, "The XSS instructions must retain their safe-case guidance."
    safe_text = " ".join(" ".join(safe_rules).split())
    assert "template" in safe_text.lower()
    assert re.search(r"\bescape\(", safe_text)
    assert not re.search(r"\b(?:markupsafe\.)?Markup\b(?!\.)", safe_text), (
        "The Markup constructor marks text safe without escaping it."
    )


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
    """Pass-specific placeholders on USER_TEMPLATE. Per-file placeholders
    live on FILE_CONTEXT_TEMPLATE; both render into the volatile
    user_prompt (the system_prompt is the cross-file stable prefix)."""
    expected_placeholders = {"pass_index", "diff_hunks"}
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", USER_TEMPLATE))
    assert found == expected_placeholders


def test_file_context_template_has_required_placeholders() -> None:
    """Per-file placeholders. Rendered into the USER prompt by render()
    (cache-packing repartition: per-file content stays out of the cached
    system prefix, which must be byte-identical across files)."""
    expected_placeholders = {"file_path", "scope_unit_context", "query_match_id_list"}
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", FILE_CONTEXT_TEMPLATE))
    assert found == expected_placeholders


def test_degraded_user_template_has_required_placeholders() -> None:
    """Same pinning for the degraded-outcome template + render_degraded()."""
    expected_placeholders = {
        "file_path",
        "pass_index",
        "degradation_reason",
        "degradation_context",
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
    # SYSTEM_PROMPT_INVARIANTS is fully static — zero `{placeholder}` markers.
    # (Pre-FUP-126 the JSON example surfaced `{byte_start}` / `{byte_end}`
    # literal markers that had to be whitelisted; the line-based example uses
    # literal integer values, so no brace-marker appears and the whitelist is
    # gone. Any brace-marker now fails-loud — a refactor that tries to .format()
    # the cacheable static head, or a stray placeholder, is surfaced.)
    assert found == [], (
        f"SYSTEM_PROMPT_INVARIANTS contains unexpected placeholders: {found}. "
        f"It must stay fully static (cacheable). Route any real placeholder "
        f"through render()'s kwargs rather than mutating the static head."
    )


def test_system_prompt_calibration_has_no_placeholders() -> None:
    """SYSTEM_PROMPT_CALIBRATION joins the never-`.format()`ed cached prefix, so
    (like INVARIANTS) it must carry zero `{placeholder}` markers — a stray brace
    would break a future .format() loudly or leak a template artifact into the
    cached block."""
    found = re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", SYSTEM_PROMPT_CALIBRATION)
    assert found == [], (
        f"SYSTEM_PROMPT_CALIBRATION contains unexpected placeholders: {found}. "
        f"It must stay fully static (cacheable, never .format()ed)."
    )


def test_system_prompt_calibration_pins_clean_is_common_rule() -> None:
    """The v8 calibration is the conservatism-probe-validated `clean-is-common` wording,
    with its one diff-scoped clause generalized to "the code under review" (the prefix is
    reused by render_post_trace in the pass-1/post-trace prompt for non-diff files). Pin the
    load-bearing phrases directly (a VERSION bump alone wouldn't catch a softened rule):
    dropping "empty findings list is ... valid" or "do not manufacture" reopens the broad
    over-eagerness the probe closed (28->5 FP). Asserted against STABLE_PREFIX so a refactor
    that drops CALIBRATION from the cached block also fails here."""
    text = " ".join(SYSTEM_PROMPT_STABLE_PREFIX.lower().split())
    assert "most code under review is fine" in text
    assert "an empty findings list is a valid, common, and correct result" in text
    assert "do not manufacture a finding" in text
    assert "return no findings" in text
    # Path-neutral: the calibration must NOT use diff-scoped wording, because the SAME
    # prefix is reused in the post-trace pass-1 prompt for whole files OUTSIDE the PR diff
    # (the prompt there states "NOT part of the PR diff"). A regression to "in the diff"
    # would contradict that path — guard it directly on the CALIBRATION constant.
    assert "if nothing in the code under review is concretely wrong" in text
    assert "diff" not in SYSTEM_PROMPT_CALIBRATION.lower()


def test_system_prompt_exemplars_brace_markers_are_the_known_examples() -> None:
    """SYSTEM_PROMPT_EXEMPLARS is fully static, but its FLAG examples
    legitimately show f-string interpolation (`f"...{owner}..."`) — the
    exact pattern the exemplar teaches the model to flag. Allowlist those
    example variables; any OTHER brace-marker is a stray placeholder or a
    refactor artifact that would break a future .format() loudly anyway.
    The constant is never .format()ed — module-level concat only."""
    found = set(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", SYSTEM_PROMPT_EXEMPLARS))
    assert found <= {"owner", "x"}, (
        f"unexpected brace-markers in SYSTEM_PROMPT_EXEMPLARS: "
        f"{found - {'owner', 'x'}}. Static exemplar f-string examples are "
        f"allowlisted by name; anything else needs review."
    )


def test_stable_prefix_includes_review_rules_in_the_validated_order() -> None:
    """Keep shared review rules inside the cached and versioned prefix."""
    assert SYSTEM_PROMPT_STABLE_PREFIX == (
        SYSTEM_PROMPT_INVARIANTS
        + SYSTEM_PROMPT_EXEMPLARS
        + SYSTEM_PROMPT_CALIBRATION
        + SYSTEM_PROMPT_COMPATIBILITY
    )
    assert SYSTEM_PROMPT_STABLE_PREFIX.endswith(SYSTEM_PROMPT_COMPATIBILITY)


def test_stable_prefix_clears_min_cacheable_floor_conservatively() -> None:
    """The cache-floor gate (analyze cache-packing spec §2): below the
    model's minimum cacheable prompt length, the API silently skips
    caching (no error). There is no exact local tokenizer (FUP-049), so
    this pins a deliberately UNDER-counting estimate — chars/5 lower-
    bounds tokens for any realistic mixed prose/code text — with a ~10%
    margin over the strictest configured-tier floor (Haiku 4.5: 4096).
    The provider-estimated count_tokens verification at closeout is the
    calibration; runtime cache_creation/cache_read activity is the
    definitive proof. If this fails after a prompt edit, the prefix
    SHRANK below the floor margin — grow it back or revisit the spec."""
    from outrider.llm.config import ModelConfig
    from outrider.llm.pricing import min_cacheable_tokens

    cfg = ModelConfig()
    # min_cacheable_tokens is host-qualified per DECISIONS.md#056: both analyze
    # tiers are Claude/Anthropic models, so the profile_id is "anthropic". The None
    # filter is defensive (the DECISIONS.md#056 unknown-floor sentinel — no current
    # Anthropic model returns it; Sonnet 5's floor is the documented 1024). Haiku
    # 4.5's 4096 is the strictest KNOWN floor and the binding one; clearing it with
    # margin also clears the DEEP-tier Sonnet 5 floor (1024).
    known_floors = [
        f
        for f in (
            min_cacheable_tokens("anthropic", cfg.analyze_model),
            min_cacheable_tokens("anthropic", cfg.standard_analyze_model),
        )
        if f is not None
    ]
    assert known_floors, "no known cacheable floor among the analyze tiers — gate is vacuous"
    strictest_floor = max(known_floors)
    conservative_tokens = len(SYSTEM_PROMPT_STABLE_PREFIX) // 5
    assert conservative_tokens >= int(strictest_floor * 1.1), (
        f"stable prefix conservatively estimates {conservative_tokens} tokens; "
        f"needs >= {int(strictest_floor * 1.1)} (strictest tier floor "
        f"{strictest_floor} + 10% margin) or the cache silently no-ops."
    )


def test_render_system_prompt_is_byte_identical_across_files() -> None:
    """THE cache-packing property: two different files (different paths,
    scope contexts, query registries, diffs, passes) produce the SAME
    system_prompt — the cross-file cache key. Identity (`is`) pins that
    render() returns the module constant, not a rebuilt equal string."""
    a = render(
        file_path="src/a.py",
        scope_unit_context="def a(): ...",
        query_match_id_list="qm-1",
        diff_hunks="@@ -1 +1 @@\n+a",
        pass_index=0,
    )
    b = render(
        file_path="lib/b.py",
        scope_unit_context="class B: ...",
        query_match_id_list="",
        diff_hunks="@@ -2 +2 @@\n+b",
        pass_index=1,
    )
    assert a.system_prompt == b.system_prompt
    assert a.system_prompt is SYSTEM_PROMPT_STABLE_PREFIX


def test_render_post_trace_system_prompt_is_byte_identical_across_files() -> None:
    """Pass-1 calls share a SECOND stable cache entry: stable prefix +
    the post-trace suffix, byte-identical across trace-fetched files."""
    from uuid import uuid4 as _uuid4

    common = {
        "scope_unit_context": "def t(): ...",
        "query_match_id_list": "",
        "source_finding_title": "t",
        "source_finding_description": "d",
        "source_finding_evidence": "e",
        "pass_index": 1,
    }
    a = render_post_trace(file_path="src/a.py", source_finding_id=_uuid4(), **common)
    b = render_post_trace(file_path="lib/b.py", source_finding_id=_uuid4(), **common)
    assert a.system_prompt == b.system_prompt
    assert a.system_prompt == SYSTEM_PROMPT_STABLE_PREFIX + POST_TRACE_SYSTEM_PROMPT_SUFFIX
    # Per-file + source-finding content lives in the user prompt.
    assert "src/a.py" in a.user_prompt
    assert "lib/b.py" in b.user_prompt


def test_system_prompt_documents_all_finding_types() -> None:
    """The prompt must enumerate every model-pickable FindingType so the
    LLM knows the constrained vocabulary it may emit.

    Under the dual-mode taxonomy (DECISIONS.md#053) every FindingType is
    model-pickable — a security type may be emitted JUDGED (contextual) or
    OBSERVED (a registry query fired) — so the prompt's `finding_type`
    vocabulary must cover the WHOLE enum. Drift = the LLM produces an
    unknown finding_type = parser rejects with `finding_type_not_in_enum`.
    Iterating `FindingType` (not a hardcoded list) auto-tracks future
    additions; the separate OBSERVED query vocabulary (which types carry a
    `.scm` producer) is pinned by the queries-registry tests, not here.
    """
    from outrider.policy.severity import FindingType

    for finding_type in FindingType:
        assert f"`{finding_type.value}`" in SYSTEM_PROMPT_INVARIANTS, (
            f"FindingType `{finding_type.value}` missing from SYSTEM_PROMPT_INVARIANTS; "
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


def test_system_prompt_pass_conditional_inferred_admission() -> None:
    """Per the trace-node arc (M8 loop): pass 0 prohibits `inferred`
    (no trace context yet); pass 1 (post-trace re-entry) admits it
    when trace_path is non-empty. The pass-0 prompt must keep the
    prohibition signal so the model doesn't burn budget on pass-0
    rejects; the pass-1 admission directive ships via
    `POST_TRACE_SYSTEM_PROMPT_SUFFIX`, appended at `render_post_trace`
    time.

    The shared `SYSTEM_PROMPT_INVARIANTS` is the pass-0 baseline;
    `render_post_trace` appends the pass-1 suffix. Pin both signals
    so a future prompt edit that silently relaxes pass-0 admission
    OR drops the pass-1 admission directive fails this test.
    """
    # Pass 0 (baseline) signals:
    # 1. The output-shape enum union DOES NOT list inferred as an option.
    # 2. The clean prompt explicitly rejects inferred on pass 0.
    assert '"evidence_tier": "judged"' in SYSTEM_PROMPT_INVARIANTS
    assert "do not emit it or supply a `query_match_id`" in SYSTEM_PROMPT_INVARIANTS
    assert "On pass 0" in SYSTEM_PROMPT_INVARIANTS
    assert 'do NOT emit\n`evidence_tier="inferred"`' in SYSTEM_PROMPT_INVARIANTS

    # Pass 1 (post-trace) admission signals — supplied via the
    # render_post_trace path. The suffix must
    # OVERRIDE the pass-0 output schema (not just append guidance);
    # the parser would otherwise reject INFERRED proposals citing
    # scope-unit-name trace_paths that came back null per the pass-0
    # schema.
    from outrider.prompts.analyze import POST_TRACE_SYSTEM_PROMPT_SUFFIX

    assert "Pass 1 (post-trace)" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    # The suffix MUST restate the output schema with `inferred`
    # admitted + non-null trace_path — otherwise the pass-0 schema
    # above (`judged` + `trace_path: null`) wins and the
    # model emits proposals the parser rejects.
    assert "<inferred|judged>" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    assert "model `observed` claims remain unsupported" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    assert "REPLACES the pass-0 schema" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    assert "non-empty array of scope-unit names" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    assert "deterministic-proof set" in POST_TRACE_SYSTEM_PROMPT_SUFFIX


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
    `import_string_raw`, not `import_string` (and never the old
    `candidate_path` / `candidate_path_raw` framing per DECISIONS.md#024).

    `TraceCandidateProposalRaw` has `extra="forbid"` and requires the
    `_raw` suffix; a model that follows the prompt literally and emits
    `import_string` (bare) causes `AnalyzeResponseRaw.model_validate_json`
    to reject the entire response.
    """
    assert "import_string_raw" in SYSTEM_PROMPT_INVARIANTS
    # The bare (admitted-layer) field name must not appear as an object key.
    assert '"import_string":' not in SYSTEM_PROMPT_INVARIANTS
    # The old field name must not appear at all (DECISIONS.md#024 rename).
    assert "candidate_path" not in SYSTEM_PROMPT_INVARIANTS


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
    # Pin line-range ordering: the prompt's own rule says both ≥ 1 and
    # line_start ≤ line_end (FUP-126 — proposals are line-based), so the
    # example must not contradict it.
    for finding in parsed["findings"]:
        assert finding["line_start"] >= 1
        assert finding["line_start"] <= finding["line_end"]


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
# Prompt fields cannot be swapped by positional unpacking
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
    """system_prompt begins with the static SYSTEM_PROMPT_INVARIANTS —
    the head of the cross-file stable prefix."""
    parts = render(
        file_path="src/example.py",
        scope_unit_context="<scope unit body>",
        query_match_id_list="(none)",
        diff_hunks="@@ -1,1 +1,1 @@",
        pass_index=0,
    )
    assert parts.system_prompt.startswith(SYSTEM_PROMPT_INVARIANTS)


def test_render_user_prompt_contains_file_path() -> None:
    """File path is per-file content — user_prompt (cache-packing
    repartition: the system prefix must be byte-identical across files,
    so nothing per-file may render into it)."""
    parts = render(
        file_path="src/auth/login.py",
        scope_unit_context="",
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert "src/auth/login.py" in parts.user_prompt
    assert "src/auth/login.py" not in parts.system_prompt


def test_render_user_prompt_contains_scope_unit_context_fenced() -> None:
    """Selected scope-unit source is per-file — user_prompt block, wrapped in a
    safe_code_fence(lang="text") because scope text is PR-controlled."""
    sentinel = "def login(user, password):\n    # SENTINEL"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=sentinel,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.user_prompt
    assert sentinel not in parts.system_prompt
    assert "```text\n" in parts.user_prompt


def test_render_user_prompt_contains_query_match_id_list() -> None:
    """Pre-fired query matches are file-scoped — user_prompt block."""
    sentinel = "python.security.sql_injection:42"
    parts = render(
        file_path="src/x.py",
        scope_unit_context="",
        query_match_id_list=sentinel,
        diff_hunks="",
        pass_index=0,
    )
    assert sentinel in parts.user_prompt
    assert sentinel not in parts.system_prompt


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


def test_render_degraded_returns_stable_prefix_unchanged() -> None:
    """Degraded path uses the SAME stable prefix as clean pass-0 calls —
    the degraded directive lives in the user prompt, not the system
    prompt. Degraded calls therefore SHARE the pass-0 cache entry
    (identity pins that render_degraded returns the module constant)."""
    parts = render_degraded(
        file_path="src/x.py",
        bounded_hunks="@@ -1,1 +1,1 @@",
        pass_index=0,
        degradation_reason="parse_failed",
    )
    assert parts.system_prompt == SYSTEM_PROMPT_STABLE_PREFIX
    assert parts.system_prompt is SYSTEM_PROMPT_STABLE_PREFIX


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


def test_degradation_context_mapping_is_total_over_reasons() -> None:
    """Every `DegradationReason` member has a provenance sentence, and the
    mapping carries no stale keys. `render_degraded` indexes the mapping
    FAIL-LOUD, so a future member added to the enum without a sentence
    here would KeyError on its first degraded call — this pin moves that
    failure to test time and rejects fail-open drift in either direction.
    (The vocabulary itself is single-sourced on the enum now; this mapping
    is its one companion that the type system cannot force.)"""
    from outrider.audit.events import DegradationReason
    from outrider.prompts.analyze import _DEGRADATION_CONTEXT  # noqa: PLC2701

    assert set(_DEGRADATION_CONTEXT) == set(DegradationReason)


def test_render_degraded_unknown_reason_fails_loud() -> None:
    """An unmapped reason raises rather than rendering wrong provenance."""
    with pytest.raises(KeyError):
        render_degraded(
            file_path="src/x.py",
            bounded_hunks="",
            pass_index=0,
            degradation_reason="future_unmapped_reason",
        )


def test_render_degraded_provenance_is_reason_aware() -> None:
    """The v9 provenance rule: parse-defect reasons keep the "could not be
    parsed" sentence; the module-scope routing reason
    (`module_level_observed_match`, a CLEAN parse —
    DECISIONS.md#062) must say the file parsed
    cleanly and must NOT claim a parse failure — false provenance biases the
    model's judged review of a perfectly parseable file."""
    for parse_reason in (
        "parse_failed",
        "tree_has_error_in_changed_regions",
        "tree_has_error_no_scope",
    ):
        parts = render_degraded(
            file_path="src/x.py",
            bounded_hunks="",
            pass_index=0,
            degradation_reason=parse_reason,
        )
        assert "could not be parsed" in parts.user_prompt, parse_reason

    module_parts = render_degraded(
        file_path="src/index.js",
        bounded_hunks="",
        pass_index=0,
        degradation_reason="module_level_observed_match",
    )
    assert "parsed cleanly" in module_parts.user_prompt
    assert "could not be parsed" not in module_parts.user_prompt


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
    must not interpolate. scope_unit_context lives in user_prompt
    (cache-packing repartition)."""
    hostile = "def f(): {file_path} {diff_hunks} {pass_index}"
    parts = render(
        file_path="src/x.py",
        scope_unit_context=hostile,
        query_match_id_list="",
        diff_hunks="",
        pass_index=0,
    )
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
        "FILE_CONTEXT_TEMPLATE",
        "MAX_TOKENS",
        # Post-trace prompt surfaces added 2026-05-24 for trace-node arc
        # pass-1 INFERRED admission. See trace.py and analyze.py for the
        # render_post_trace call site. `POST_TRACE_FILE_CONTEXT_TEMPLATE`
        # is the whole-file analogue of FILE_CONTEXT_TEMPLATE —
        # diff-scoped wording ("changed scope units") is wrong for
        # trace-fetched files outside the PR diff.
        "POST_TRACE_FILE_CONTEXT_TEMPLATE",
        "POST_TRACE_SYSTEM_PROMPT_SUFFIX",
        "POST_TRACE_USER_TEMPLATE",
        # Cache-packing surfaces (analyze-v4): the exemplars block + the
        # composed cross-file stable prefix (THE cached system block).
        # CALIBRATION (analyze-v8) is the third prefix component.
        "SYSTEM_PROMPT_CALIBRATION",
        "SYSTEM_PROMPT_EXEMPLARS",
        "SYSTEM_PROMPT_INVARIANTS",
        "SYSTEM_PROMPT_STABLE_PREFIX",
        "TEMPERATURE",
        "TEMPLATE",
        "USER_TEMPLATE",
        "VERSION",
        "AnalyzePromptParts",
        "render",
        "render_degraded",
        "render_deleted",
        "render_post_trace",
    }
    assert set(analyze.__all__) == expected


# --- protected-region pins (suite-v2 spec, second review 2026-07-15) ------------------------------
# The non-type EXEMPLARS regions have no eval metrics yet, so fixture-suite work must not touch
# them: an accidental edit fails here; a deliberate edit updates the pinned digest in the same
# diff, making the touch reviewable. The per-type blocks are NOT pinned — they are the
# sanctioned surface of future gated prompt work. The PREAMBLE (everything before the first
# `### ` heading) is pinned too: it carries the never-flag-exemplar-code and bare-JSON-output
# safety rules, which are exactly as unmeasured as the named sections.
_EXEMPLARS_PREAMBLE_KEY = "(preamble)"

_PROTECTED_EXEMPLAR_REGIONS = {
    _EXEMPLARS_PREAMBLE_KEY: ("f2eb5f3cabbe88dec398b1c032b757b932ffcbf5dd11f29c6898563f54b64221"),
    "Commonly-confused pairs (pick the root cause, not the symptom)": (
        "6d70d230a2da6a93697750cb5abae25dca4028c167c0528c2250ccf70041165c"
    ),
    "Finding quality (title / description / evidence)": (
        "4829a5de77b48cbe5095b4affdae7bbd605ebea6cb6483f22ef10f42059eb0af"
    ),
    "Line-number discipline": ("2f2673eb25f250d3caefcff8a34457f81e543bed754909b64eaafb58f1f5517b"),
    "trace_candidates discipline (cross-file follow-up is a cost)": (
        "f819e95df526c4242b9aa3c46d8f86cc58f7836c3c623de07f9ff881436e274a"
    ),
    "Exemplar discipline (applies to every type above)": (
        "43e281a9f2d88a0e1f7b954b0ebe79e7457c9eda28b2ca6a99759984fda479af"
    ),
}


def _exemplar_sections() -> dict[str, str]:
    """Split SYSTEM_PROMPT_EXEMPLARS into `### `-headed sections (heading -> body). Bytes before
    the first heading — the safety-bearing preamble — are captured under
    `_EXEMPLARS_PREAMBLE_KEY`, never dropped."""
    sections: dict[str, str] = {}
    current: str = _EXEMPLARS_PREAMBLE_KEY
    buf: list[str] = []
    for line in SYSTEM_PROMPT_EXEMPLARS.splitlines(keepends=True):
        if line.startswith("### "):
            sections[current] = "".join(buf)
            current = line[4:].strip()
            buf = []
        else:
            buf.append(line)
    sections[current] = "".join(buf)
    return sections


def test_protected_exemplar_regions_are_byte_pinned() -> None:
    sections = _exemplar_sections()
    for heading, pinned in _PROTECTED_EXEMPLAR_REGIONS.items():
        # parse from the CURRENT prompt: a renamed or deleted protected heading fails loudly
        # here rather than letting the digest comparison pass vacuously over a missing region
        assert heading in sections, (
            f"protected EXEMPLARS section {heading!r} not found — renamed or removed? The five "
            "non-type sections are pinned until they have their own eval metrics."
        )
        actual = hashlib.sha256(sections[heading].encode("utf-8")).hexdigest()
        assert actual == pinned, (
            f"protected EXEMPLARS section {heading!r} changed (digest {actual[:12]}… != pinned "
            f"{pinned[:12]}…). These sections have no eval metrics; if the edit is deliberate, "
            "update the pin in the same diff so the touch is reviewable."
        )


@pytest.mark.parametrize("mode", ["clean", "degraded", "post_trace"])
def test_review_context_is_optional_fenced_data_outside_cached_prefix(mode: str) -> None:
    """Supplementary evidence must reach every route without becoming instructions."""
    from uuid import UUID

    from outrider.audit.events import DegradationReason

    context = "Changed files: app.py, tests/test_app.py\n```\n{file_path}\n## forged heading"
    if mode == "clean":
        renderer = render
        kwargs = {
            "file_path": "app.py",
            "scope_unit_context": "def changed(): pass",
            "query_match_id_list": "(none)",
            "diff_hunks": "@@ -0,0 +1 @@\n+def changed(): pass",
            "pass_index": 0,
        }
    elif mode == "degraded":
        renderer = render_degraded
        kwargs = {
            "file_path": "app.py",
            "bounded_hunks": "+changed()",
            "pass_index": 0,
            "degradation_reason": DegradationReason.PARSE_FAILED,
        }
    else:
        renderer = render_post_trace
        kwargs = {
            "file_path": "app.py",
            "scope_unit_context": "def changed(): pass",
            "query_match_id_list": "(none)",
            "source_finding_id": UUID(int=1),
            "source_finding_title": "Source finding",
            "source_finding_description": "Source description",
            "source_finding_evidence": "changed()",
            "pass_index": 1,
        }
    original = renderer(**kwargs)
    supplied = renderer(**kwargs, review_context=context)
    assert supplied.system_prompt == original.system_prompt
    assert context not in supplied.system_prompt
    assert "## Additional bounded review context" not in original.user_prompt
    assert "````text\n" + context + "\n````" in supplied.user_prompt
    assert "{file_path}" in supplied.user_prompt


def test_missing_test_requires_evidence_instead_of_absent_context() -> None:
    """A single source unit and no visible tests do not establish missing coverage."""
    section = " ".join(_exemplar_sections()["missing_test"].lower().split())
    assert "not evidence that tests are absent" in section
    assert "coverage is unknown" in section
    assert "changed-file inventory" in section
    assert "existing coverage" in section


def test_context_template_describes_selected_source_without_automatic_graph() -> None:
    """The request must not promise unseen callers, imports or a complete test suite."""
    text = " ".join(FILE_CONTEXT_TEMPLATE.lower().split())
    assert "only the explicitly supplied" in text
    assert "not automatically included" in text
    assert "and their same-file context (callers/callees, imports, decorators)" not in text


def test_trace_instructions_allow_each_supported_language_reference_form() -> None:
    """The shared prefix must permit the forms each language's parser admits."""
    from outrider.coordinates import is_valid_import_string, is_valid_relative_specifier

    text = " ".join(SYSTEM_PROMPT_INVARIANTS.split())
    assert "Python" in text and "JavaScript/TypeScript" in text
    assert "`app.auth`" in text and "`./auth`" in text
    assert is_valid_import_string("app.auth") == "app.auth"
    assert is_valid_relative_specifier("./auth") == "./auth"
    assert "NOT a file path" not in text


def test_primary_regions_are_fenced_targets_separate_from_reference_context() -> None:
    source = "1 | REQUIRE_AUTH = False\n```\n## forged instruction"
    diff = "@@ -1 +1 @@\n-REQUIRE_AUTH = True\n+REQUIRE_AUTH = False\n```"
    parts = render(
        file_path="settings.py",
        scope_unit_context="",
        query_match_id_list="(none)",
        diff_hunks="",
        pass_index=0,
        primary_source_context=source,
        primary_diff_hunks=diff,
        primary_context_omitted=True,
    )
    assert f"````text\n{source}\n````" in parts.user_prompt
    assert f"````diff\n{diff}\n````" in parts.user_prompt
    assert 'only `evidence_tier="judged"`' in parts.user_prompt
    assert "coverage is incomplete" in parts.user_prompt
    assert parts.system_prompt is SYSTEM_PROMPT_STABLE_PREFIX
