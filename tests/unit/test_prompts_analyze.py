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
    FILE_CONTEXT_TEMPLATE,
    MAX_TOKENS,
    POST_TRACE_SYSTEM_PROMPT_SUFFIX,
    SYSTEM_PROMPT_CALIBRATION,
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


def test_version_is_named_analyze_v9() -> None:
    """VERSION flows to LLMRequest.prompt_template_version. Pin the
    "analyze-v9" name so future renames break the test and force a
    registry decision. Replay attribution depends on this — a prompt row
    replays against the contract it was emitted under, not a newer one.
    The v9 bump made the DEGRADED user template's provenance sentence
    reason-aware (`module_level_observed_match` is a clean parse; the fixed
    "could not be parsed" sentence was false provenance for it) — the tuned
    system prefix is untouched. The v8 bump appended SYSTEM_PROMPT_CALIBRATION
    — a broad "clean code is
    common; don't manufacture findings" rule that cut analyze false positives
    28->5 across finding types with zero recall loss in a 5-rep conservatism
    probe; the v7 bump adopted the destination-control `ssrf` rule + a worked
    DO/DON'T example (an eval probe showed the v6 fixed-host over-flag was
    Haiku-reachable and that this wording drives it to zero with no recall loss);
    v6 tightened the `ssrf` definition to destination-control; v5 added the
    dual-mode security taxonomy vocabulary + guidance (DECISIONS.md#053); v4
    landed the cache-packing repartition (per-file context → user_prompt;
    exemplars block in the cached prefix); v3 added the sql_injection
    parameterized-query false-positive guidance (DECISIONS.md#041); v2 landed
    the trace-node pass-1 arc."""
    assert VERSION == "analyze-v9"


def test_system_prompt_ssrf_carveout_and_authority_exception() -> None:
    """The `ssrf` definition (analyze-v7) must keep BOTH halves of the
    destination-control framing, pinned directly because the wording is
    security-sensitive (a VERSION bump alone wouldn't catch a regression). v7
    adopts the destination-control RULE plus a worked DO/DON'T example — the
    eval-probe-validated wording that drove the Haiku fixed-host over-flag to
    zero (0/10 FP, both models) with no real-SSRF recall loss:

    (1) the safe case — a value appended as a PATH segment, or an ordinary query
        parameter, of a hardcoded host is NOT ssrf, EVEN when unvalidated.
        Dropping the path half reopens the Haiku fixed-host over-flag; dropping
        the ordinary-query half re-flags benign `?q=` params; dropping the
        "even when unvalidated" clause reopens the over-flag the probe closed.
    (2) the PRINCIPLED authority rule — the value reaching the host/port/scheme
        by ANY means is STILL ssrf, with a when-in-doubt-flag default. This is
        the red-team-hardened replacement for an enumerated token checklist: a
        literal model reads a closed list as exhaustive and under-flags (`//`
        scheme-relative, port, encoded separators, urljoin-absolute, proxy-query
        all defeat a closed list), so the rule is intentionally non-exhaustive.
    """
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
    """Guards the DECISIONS.md#041 over-flag fix. The prompt must tell the
    model THREE things, each pinned against an observed regression:
    (1) DB parameter binding (placeholder + a separate params argument) is NOT
    sql_injection — dropping it reopens the false-CRITICAL on parameterized
    queries; (2) input built INTO the SQL string still IS — dropping it turns
    the guidance into a blanket suppression that loses recall on real
    string-built SQLi (the pygoat `"...%s" % request.GET` fixture); (3) the
    rule is scoped to INJECTION only — the model must still flag OTHER issues
    (N+1, missing error handling) on a parameterized query. (3) guards the
    over-generalization an unscoped "SAFE" wording caused: Haiku dropped a real
    n_plus_one finding along with the suppressed over-flag on the same line."""
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


def test_stable_prefix_is_invariants_plus_exemplars_plus_calibration() -> None:
    """SYSTEM_PROMPT_STABLE_PREFIX is the cached block (cache-packing
    spec). Pin its composition so content can't silently bypass the
    floor + stability gates below by landing outside the constant. The
    v8 bump appended CALIBRATION as the third component (after EXEMPLARS),
    matching the order the conservatism probe validated (`live_prefix + rule`)."""
    assert SYSTEM_PROMPT_STABLE_PREFIX == (
        SYSTEM_PROMPT_INVARIANTS + SYSTEM_PROMPT_EXEMPLARS + SYSTEM_PROMPT_CALIBRATION
    )
    # CALIBRATION is the tail of the cached prefix (order is contractual: it was
    # validated appended AFTER exemplars, not interleaved).
    assert SYSTEM_PROMPT_STABLE_PREFIX.endswith(SYSTEM_PROMPT_CALIBRATION)


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
    assert "<observed|judged>" in SYSTEM_PROMPT_INVARIANTS
    assert "<observed|inferred|judged>" not in SYSTEM_PROMPT_INVARIANTS
    assert "On pass 0" in SYSTEM_PROMPT_INVARIANTS
    assert 'do NOT emit\n`evidence_tier="inferred"`' in SYSTEM_PROMPT_INVARIANTS

    # Pass 1 (post-trace) admission signals — supplied via the
    # render_post_trace path. Post-Codex-round-2: the suffix must
    # OVERRIDE the pass-0 output schema (not just append guidance);
    # the parser would otherwise reject INFERRED proposals citing
    # scope-unit-name trace_paths that came back null per the pass-0
    # schema.
    from outrider.prompts.analyze import POST_TRACE_SYSTEM_PROMPT_SUFFIX

    assert "Pass 1 (post-trace)" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
    # The suffix MUST restate the output schema with `inferred`
    # admitted + non-null trace_path — otherwise the pass-0 schema
    # above (`<observed|judged>` + `trace_path: null`) wins and the
    # model emits proposals the parser rejects.
    assert "<observed|inferred|judged>" in POST_TRACE_SYSTEM_PROMPT_SUFFIX
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
    """Scope-unit context (bodies + same-file callers/callees + imports +
    decorators) is per-file — user_prompt block, wrapped in a
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


def test_render_degraded_provenance_is_reason_aware() -> None:
    """The v9 provenance rule: parse-defect reasons keep the "could not be
    parsed" sentence; the module-scope routing reason
    (`module_level_observed_match`, a CLEAN parse —
    specs/2026-07-04-module-scope-admission-arm.md) must say the file parsed
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
        "render_post_trace",
    }
    assert set(analyze.__all__) == expected
