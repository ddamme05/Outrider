"""Analyze parser contract + step-by-step behavior tests.

Pins the §6 parser module: the public surface (frozen dataclasses +
signatures) AND the admission flow as each step lands. Sections marked
by spec §6 step number.

Sections (in spec §6 step order): scaffolding (frozen+slots discipline,
signature pin, field sets, audit-context exclusion, module `__all__`),
step 0 (response parse + response-level rejection), commit-2 helper
unit tests (proposal_hash + rejection-payload helper), commit-3 +
commit-4 admission tests (enum + producer + span), commit-5 finding
construction + trace-candidate collection, commit-6 audit-fold
regression pins.
"""

import dataclasses
import inspect
from uuid import uuid4

import pytest

from outrider.agent.nodes.analyze_parser import (
    ParserCounters,
    ParserResult,
    ProposalRejection,
    ResponseRejection,
    parse_analyze_response,
)
from outrider.policy.findings import EvidenceTier
from outrider.policy.severity import FindingType

# ---------------------------------------------------------------------------
# Frozen + slots discipline
# ---------------------------------------------------------------------------


def test_proposal_rejection_rejects_positional_unpacking() -> None:
    """ProposalRejection is a dataclass, not a tuple — positional unpack
    raises `TypeError`. Same swap-impossibility discipline as
    `AnalyzePromptParts`."""
    rej = ProposalRejection(
        proposal_hash="a" * 64,
        file_path="src/x.py",
        claimed_finding_type_hash="b" * 16,
        claimed_finding_type_len=10,
        claimed_evidence_tier=EvidenceTier.OBSERVED,
        rejection_reason="query_match_id_not_in_registry",
        rejection_detail="",
    )
    with pytest.raises(TypeError):
        _, _ = rej  # type: ignore[misc]


def test_proposal_rejection_is_frozen() -> None:
    rej = ProposalRejection(
        proposal_hash="a" * 64,
        file_path="src/x.py",
        claimed_finding_type_hash="b" * 16,
        claimed_finding_type_len=10,
        claimed_evidence_tier=EvidenceTier.OBSERVED,
        rejection_reason="query_match_id_not_in_registry",
        rejection_detail="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rej.proposal_hash = "tampered"  # type: ignore[misc]


def test_response_rejection_rejects_positional_unpacking() -> None:
    rej = ResponseRejection(
        file_path="src/x.py",
        response_hash="a" * 64,
        rejection_reason="raw_response_unparseable",
        rejection_detail="",
    )
    with pytest.raises(TypeError):
        _, _ = rej  # type: ignore[misc]


def test_response_rejection_is_frozen() -> None:
    rej = ResponseRejection(
        file_path="src/x.py",
        response_hash="a" * 64,
        rejection_reason="raw_response_unparseable",
        rejection_detail="",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        rej.file_path = "tampered"  # type: ignore[misc]


def test_parser_counters_rejects_positional_unpacking() -> None:
    counters = ParserCounters(
        n_proposals_seen=0,
        n_findings_emitted=0,
        n_proposals_rejected=0,
        n_responses_rejected=0,
        n_trace_candidates_emitted=0,
    )
    with pytest.raises(TypeError):
        _, _ = counters  # type: ignore[misc]


def test_parser_counters_is_frozen() -> None:
    counters = ParserCounters(
        n_proposals_seen=0,
        n_findings_emitted=0,
        n_proposals_rejected=0,
        n_responses_rejected=0,
        n_trace_candidates_emitted=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        counters.n_proposals_seen = 999  # type: ignore[misc]


def test_parser_result_rejects_positional_unpacking() -> None:
    result = ParserResult(
        admitted_findings=(),
        trace_candidates=(),
        proposal_rejections=(),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=0,
            n_findings_emitted=0,
            n_proposals_rejected=0,
            n_responses_rejected=0,
            n_trace_candidates_emitted=0,
        ),
    )
    with pytest.raises(TypeError):
        _, _ = result  # type: ignore[misc]


def test_parser_result_is_frozen() -> None:
    result = ParserResult(
        admitted_findings=(),
        trace_candidates=(),
        proposal_rejections=(),
        response_rejection=None,
        counters=ParserCounters(
            n_proposals_seen=0,
            n_findings_emitted=0,
            n_proposals_rejected=0,
            n_responses_rejected=0,
            n_trace_candidates_emitted=0,
        ),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.admitted_findings = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dataclass field sets — pin what the parser produces, NOT what the
# audit events carry. Audit-context fields (review_id, event_id,
# timestamp, sequence_number, is_eval, node_id, event_type) MUST be
# absent from the parser payloads — they're added by the node body
# when lifting to event objects.
# ---------------------------------------------------------------------------


def test_proposal_rejection_field_set() -> None:
    """The parser-owned subset of FindingProposalRejectedEvent fields.
    Audit-context fields are deliberately omitted — the node body
    adds them at lift time."""
    fields = {f.name for f in dataclasses.fields(ProposalRejection)}
    expected = {
        "proposal_hash",
        "file_path",
        "claimed_finding_type_hash",
        "claimed_finding_type_len",
        "claimed_evidence_tier",
        "rejection_reason",
        "rejection_detail",
    }
    assert fields == expected


def test_proposal_rejection_omits_audit_context_fields() -> None:
    """Defense-in-depth: an audit-context field landing on the parser
    payload (e.g., the parser sets `event_id` itself) breaks the
    "parser is pure" boundary. Explicit exclusion list catches a
    future commit that adds them silently."""
    fields = {f.name for f in dataclasses.fields(ProposalRejection)}
    audit_context = {
        "event_id",
        "review_id",
        "timestamp",
        "sequence_number",
        "is_eval",
        "node_id",
        "event_type",
    }
    leakage = fields & audit_context
    assert leakage == set(), (
        f"ProposalRejection must not carry audit-context fields "
        f"(those are added at node-body lift time). Found: {leakage}"
    )


def test_response_rejection_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ResponseRejection)}
    expected = {
        "file_path",
        "response_hash",
        "rejection_reason",
        "rejection_detail",
    }
    assert fields == expected


def test_response_rejection_omits_audit_context_fields() -> None:
    fields = {f.name for f in dataclasses.fields(ResponseRejection)}
    audit_context = {
        "event_id",
        "review_id",
        "timestamp",
        "sequence_number",
        "is_eval",
        "node_id",
        "event_type",
    }
    leakage = fields & audit_context
    assert leakage == set()


def test_parser_counters_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ParserCounters)}
    expected = {
        "n_proposals_seen",
        "n_findings_emitted",
        "n_proposals_rejected",
        "n_responses_rejected",
        "n_trace_candidates_emitted",
        "n_trace_candidates_dropped_malformed",
        "n_trace_candidates_module_corrected",
        "n_findings_observed",
        "n_proposals_superseded_by_observed",
    }
    assert fields == expected


def test_parser_result_field_set() -> None:
    fields = {f.name for f in dataclasses.fields(ParserResult)}
    expected = {
        "admitted_findings",
        "trace_candidates",
        "proposal_rejections",
        "response_rejection",
        "counters",
    }
    assert fields == expected


# ---------------------------------------------------------------------------
# parse_analyze_response signature pin
# ---------------------------------------------------------------------------


def test_parse_analyze_response_signature() -> None:
    """The parser's public signature is locked at scaffolding time.
    Implementation commits land the body; a refactor that changes
    parameter names or kw-only-ness breaks call sites and is caught
    here at scaffolding-test time."""
    sig = inspect.signature(parse_analyze_response)
    params = sig.parameters

    # response_text is positional-or-keyword (the only positional input).
    assert "response_text" in params
    assert params["response_text"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

    # Every other parameter is keyword-only (the * separator after
    # response_text). Same misuse-resistance pattern as
    # compute_cost_usd, coordinates.tree_sitter_to_github — same-typed
    # parameters can't get swapped at the call site.
    expected_kwonly = {
        "review_id",
        "installation_id",
        "file_path",
        "file_content",
        "file_byte_length",
        "included_scope_units",
        "query_match_id_set",
        "degraded_mode",
        "active_policy_version",
        # `degraded_context_byte_ranges` added for FUP-138: the deterministic
        # addable-diff byte ranges a degraded JUDGED span must intersect
        # (`span_within_degraded_context`). Defaults to `()`; the node supplies it
        # from the patch via `coordinates.added_line_byte_ranges`.
        "degraded_context_byte_ranges",
        # `pass_index` added 2026-05-24 for trace-node arc's post-trace
        # INFERRED admission: pass 0 rejects every INFERRED (no trace
        # context yet); pass 1+ admits INFERRED with valid trace_path.
        "pass_index",
        # `valid_trace_path_elements` added 2026-05-24 (Codex round 2):
        # deterministic-proof set for pass-1 INFERRED admission per
        # `evidence-tier-schema-enforced` — every trace_path element
        # MUST appear in this set (scope-unit names from the trace-
        # fetched file's included_scope_units). Defaults to empty
        # frozenset for pass-0 calls.
        "valid_trace_path_elements",
        # `finish_reason` added 2026-06-06 (FUP-153): the provider's
        # LLMResponse.finish_reason, threaded so a max_tokens truncation
        # produces a rejection_detail that names truncation as the root
        # cause instead of the opaque downstream JSON ValidationError.
        # Defaults to "unknown"; degradation behaviour is unchanged.
        "finish_reason",
        # `parameterized_call_scan` added 2026-06-12 (FUP-162): the
        # structural facts for the sql_injection parameterized-call
        # veto. Defaults to None (veto disabled — the degraded-mode
        # value); the node supplies a scan for clean outcomes only.
        "parameterized_call_scan",
        # `import_refs` added 2026-07-02 (FUP-209 follow-up): the
        # analyzed file's ast_facts ImportRef tuple, ground truth for
        # from-import candidate correction — a trace candidate whose
        # trailing component the file visibly from-imports is rewritten
        # to the importing module at admission. Defaults to `()`
        # (correction disabled — the degraded/failed-parse value).
        "import_refs",
        # Dispatch spec: suppresses trace-candidate collection for files
        # whose language has no import resolver (JS/TS until the
        # resolver spec). Defaults True; the node passes is_python.
        "collect_trace_candidates",
    }
    kwonly = {name for name, p in params.items() if p.kind == inspect.Parameter.KEYWORD_ONLY}
    assert kwonly == expected_kwonly, (
        f"parse_analyze_response keyword-only parameter set drifted. "
        f"Expected: {expected_kwonly}. Got: {kwonly}."
    )


def test_parse_analyze_response_does_not_take_audit_persister() -> None:
    """The parser is a pure function — no IO. An `audit_persister` (or
    any sink/persister-like) parameter would re-introduce the IO that
    the boundary discipline keeps out of the parser. Explicit
    not-present check catches a future commit that adds it."""
    sig = inspect.signature(parse_analyze_response)
    forbidden = {"audit_persister", "persister", "audit_emitter", "sink"}
    found = set(sig.parameters.keys()) & forbidden
    assert found == set(), (
        f"parse_analyze_response must not take a persister/sink/emitter "
        f"parameter (parser is pure; node body owns persistence). "
        f"Found: {found}."
    )


def test_parse_analyze_response_returns_parser_result() -> None:
    """Return annotation pin — implementation commits must keep the
    return type stable. `from __future__ import annotations` makes
    the annotation a string; check the string identity directly.
    (Runtime resolution via `typing.get_type_hints` fails because
    the parser module keeps `ReviewFinding`/`TraceCandidate`/etc.
    inside a `TYPE_CHECKING` block to avoid runtime imports of pure-
    annotation deps.)"""
    sig = inspect.signature(parse_analyze_response)
    assert sig.return_annotation == "ParserResult"


# ---------------------------------------------------------------------------
# Spec §6 step 0 — response parse + response-level rejection
# ---------------------------------------------------------------------------


def _call_parser(response_text: str, **overrides: object) -> ParserResult:
    """Convenience: invoke `parse_analyze_response` with sane defaults
    for the per-file kwargs that don't matter for step-0 tests. Overrides
    supplied per-test."""
    defaults: dict[str, object] = {
        "review_id": uuid4(),
        "installation_id": 12345,
        "file_path": "src/x.py",
        "file_content": "",
        "file_byte_length": 0,
        "included_scope_units": (),
        "query_match_id_set": frozenset(),
        "degraded_mode": False,
        "active_policy_version": "1.0.0",
    }
    defaults.update(overrides)
    return parse_analyze_response(response_text, **defaults)  # type: ignore[arg-type]


def test_step0_malformed_json_returns_response_rejection() -> None:
    """Raw response that isn't valid JSON at all fails
    `AnalyzeResponseRaw.model_validate_json` with a JSON-decode error.
    Parser must record the response-level rejection without raising."""
    result = _call_parser("not even json {{{")
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"
    assert result.admitted_findings == ()
    assert result.trace_candidates == ()
    assert result.proposal_rejections == ()


def test_step0_max_tokens_truncation_names_root_cause() -> None:
    """A response truncated at the max_tokens limit still degrades to
    `raw_response_unparseable` (the review continues — degradation is
    unchanged), but `rejection_detail` names truncation as the root cause
    instead of the opaque downstream JSON ValidationError (FUP-153)."""
    result = _call_parser('{"findings": [', finish_reason="max_tokens")
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"
    detail = result.response_rejection.rejection_detail
    assert "max_tokens" in detail
    assert "truncated" in detail.lower()
    # Stays within the audit event's rejection_detail max_length=500.
    assert len(detail) <= 500


def test_step0_non_truncation_keeps_validation_error_detail() -> None:
    """The truncation message is max_tokens-specific: a normal finish_reason
    leaves the detail as the formatted ValidationError, so a genuinely
    malformed (not truncated) response is not mislabelled as truncated."""
    result = _call_parser('{"findings": [', finish_reason="end_turn")
    assert result.response_rejection is not None
    assert "truncated at the max_tokens" not in result.response_rejection.rejection_detail


def test_step0_valid_json_wrong_shape_returns_response_rejection() -> None:
    """Valid JSON but wrong shape (missing required `findings` key) also
    fails parsing and routes to the response-rejection path."""
    result = _call_parser('{"not_findings": []}')
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"


def test_step0_response_hash_is_full_text_not_truncated() -> None:
    """`compute_response_hash` hashes the FULL response text (no
    truncation cap). Two responses that differ only past 8 KiB still
    produce distinct hashes."""
    import hashlib

    response_text = "x" * 10000  # > 8 KiB; clearly malformed JSON too
    result = _call_parser(response_text)
    assert result.response_rejection is not None
    expected = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    assert result.response_rejection.response_hash == expected


def test_step0_response_rejection_file_path_matches_input() -> None:
    """The rejection payload's `file_path` is the one analyze passed in
    (already canonicalized at intake). Parser doesn't re-canonicalize."""
    result = _call_parser("malformed", file_path="src/path/to/file.py")
    assert result.response_rejection is not None
    assert result.response_rejection.file_path == "src/path/to/file.py"


def test_step0_rejection_detail_does_not_contain_response_text() -> None:
    """Per `DECISIONS.md#014` point 1: audit rows must not carry user
    code or prompt/completion content. The `rejection_detail` formatter
    must NEVER include the response text. Send a response with a
    distinctive sentinel and verify the sentinel does not appear."""
    sentinel = "SENSITIVE_CONTENT_DO_NOT_LEAK_42"
    result = _call_parser(f'{{"findings": [{{"evidence": "{sentinel}"}}]}}')
    assert result.response_rejection is not None
    detail = result.response_rejection.rejection_detail
    assert sentinel not in detail, f"rejection_detail leaked response text. Detail: {detail!r}"


def test_step0_rejection_detail_is_within_field_max_length() -> None:
    """`AnalyzeResponseRejectedEvent.rejection_detail` is
    `Field(max_length=500)`. The formatter must respect this even
    when many errors fire — pathological case is a response with many
    findings each carrying many invalid fields. Truncate with `"..."`
    marker."""
    # Construct a response with many findings, each missing many fields.
    bogus_finding = (
        '{"finding_type": null, "evidence_tier": null, "title": null, '
        '"description": null, "evidence": null, "line_start": null, "line_end": null}'
    )
    response = '{"findings": [' + ",".join([bogus_finding] * 50) + "]}"
    result = _call_parser(response)
    assert result.response_rejection is not None
    assert len(result.response_rejection.rejection_detail) <= 500


def test_step0_rejection_detail_uses_json_pointer_format() -> None:
    """Format pin per spec §3: `findings[0].finding_type x1, ...`. An
    index segment attaches to its parent with `[N]`; a field segment
    joins with `.`. Pin the exact shape so the dashboard renderer can
    parse it deterministically."""
    # A response where findings[0] has a missing required field — produces
    # an error with location ("findings", 0, "<field>").
    result = _call_parser('{"findings": [{}]}')
    assert result.response_rejection is not None
    detail = result.response_rejection.rejection_detail
    # At minimum, the format must show `findings[0].<something>` shape
    assert "findings[0]" in detail, (
        f"rejection_detail does not use JSON-pointer format. Got: {detail!r}"
    )


def test_step0_root_json_syntax_error_names_the_failure_class() -> None:
    """A root-level JSON-syntax failure (whole document invalid — e.g. an
    unescaped quote inside a string value, the 2026-06-12 live-run shape)
    has an EMPTY Pydantic error location; it must render as `json_syntax`,
    never the empty string. Regression: the persisted detail used to read
    `" x1"`, leaving the audit stream unable to name the failure class
    after `llm_call_content` purges (FUP-168)."""
    # Invalid JSON: raw unescaped quotes inside a string value — exactly the
    # live-run failure, NOT a fence-wrap (strip_outer_json_fence) case.
    raw = '{"findings": [{"evidence": "results = [{"id": i}]"}]}'
    result = _call_parser(raw)
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"
    detail = result.response_rejection.rejection_detail
    assert "json_syntax" in detail, f"empty-loc rendering regressed. Got: {detail!r}"
    assert not detail.startswith(" "), f"detail must not start with an empty path: {detail!r}"


def test_step0_valid_json_wrong_root_shape_names_root_schema() -> None:
    """Valid JSON whose ROOT is the wrong shape (`[]` / `null` where the
    response object is expected) is a schema failure of well-formed JSON,
    not a syntax failure — the detail must say `root_schema`, never
    `json_syntax` (Codex residual on the FUP-168 fold: conflating the two
    misdirects the diagnosis)."""
    for raw in ("[]", "null"):
        result = _call_parser(raw)
        assert result.response_rejection is not None, raw
        detail = result.response_rejection.rejection_detail
        assert "root_schema" in detail, f"{raw!r}: expected root_schema, got {detail!r}"
        assert "json_syntax" not in detail, f"{raw!r}: mislabeled as syntax: {detail!r}"


def test_step0_response_rejection_counters() -> None:
    """On response-level rejection: every counter is zero except
    `n_responses_rejected == 1`. The node body sums per-file counters
    into the per-pass `AnalyzeCompletedEvent`."""
    result = _call_parser("malformed")
    assert result.counters.n_proposals_seen == 0
    assert result.counters.n_findings_emitted == 0
    assert result.counters.n_proposals_rejected == 0
    assert result.counters.n_responses_rejected == 1
    assert result.counters.n_trace_candidates_emitted == 0
    assert result.counters.n_trace_candidates_dropped_malformed == 0


def test_step0_valid_empty_findings_returns_clean_zero_result() -> None:
    """Valid JSON with `findings: []` is the trivial happy path —
    parsing succeeded, no proposals to process, zero counters across
    the board, no rejections."""
    result = _call_parser('{"findings": []}')
    assert result.response_rejection is None
    assert result.admitted_findings == ()
    assert result.trace_candidates == ()
    assert result.proposal_rejections == ()
    assert result.counters.n_proposals_seen == 0
    assert result.counters.n_findings_emitted == 0
    assert result.counters.n_proposals_rejected == 0
    assert result.counters.n_responses_rejected == 0
    assert result.counters.n_trace_candidates_emitted == 0
    assert result.counters.n_trace_candidates_dropped_malformed == 0


# ---------------------------------------------------------------------------
# Fenced-JSON envelope tolerance (regression for the silent-coverage-hole
# bug: Sonnet wrapping its response in ```json...``` was producing a
# `raw_response_unparseable` ResponseRejection — analyze degraded
# gracefully but every fenced response lost ALL its findings silently)
# ---------------------------------------------------------------------------


def test_fenced_json_response_is_parsed_not_rejected() -> None:
    """Sonnet sometimes wraps the response in ```json...``` despite the
    "no markdown fences" prompt instruction. With `strip_outer_json_fence`
    applied before model_validate_json, the fenced response is parsed
    normally — `response_rejection` stays None, the file's findings
    (here: zero) flow through admission as usual. Previously this path
    produced a ResponseRejection and the file's true findings were lost."""
    inner = '{"findings": []}'
    fenced = f"```json\n{inner}\n```"
    result = _call_parser(fenced)
    assert result.response_rejection is None, (
        "fenced response was treated as raw_response_unparseable — the "
        "fence-stripping normalizer regressed; analyze silently loses "
        "every file whose response is fenced"
    )
    assert result.admitted_findings == ()
    assert result.counters.n_responses_rejected == 0


def test_bare_fenced_json_response_is_parsed_not_rejected() -> None:
    """The bare-fence shape (```\\n...\\n``` without `json` tag) is also
    tolerated."""
    inner = '{"findings": []}'
    fenced = f"```\n{inner}\n```"
    result = _call_parser(fenced)
    assert result.response_rejection is None
    assert result.counters.n_responses_rejected == 0


def test_malformed_fence_still_routes_to_response_rejection() -> None:
    """A wrapper missing its closing fence is malformed — the helper
    falls through unchanged, the existing try/except routes the failure
    to the response-rejection path. Pins the policy boundary so the
    fence helper stays narrow: it tolerates ONE well-formed wrapper,
    not arbitrary recovery."""
    inner = '{"findings": []}'
    malformed = f"```json\n{inner}"  # opener but no closer
    result = _call_parser(malformed)
    assert result.response_rejection is not None
    assert result.response_rejection.rejection_reason == "raw_response_unparseable"


# ---------------------------------------------------------------------------
# Spec §6 commit-2 scaffold — proposal iteration + rejection-payload helper
# (no admission decisions yet; commit 3 wires the admission checks)
# ---------------------------------------------------------------------------


def _build_raw_proposal(**overrides: object):  # type: ignore[no-untyped-def]
    """Build a minimal valid AnalyzeFindingProposalRaw for helper tests."""
    from outrider.schemas.llm.analyze import AnalyzeFindingProposalRaw

    defaults: dict[str, object] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "line_start": 1,
        "line_end": 1,
    }
    defaults.update(overrides)
    return AnalyzeFindingProposalRaw(**defaults)  # type: ignore[arg-type]


def test_build_proposal_rejection_preserves_caller_proposal_hash() -> None:
    """Per commit 5's refactor: `_build_proposal_rejection` no longer
    computes `proposal_hash` internally — the caller passes it (so the
    same hash feeds both the rejection payload AND
    `_collect_trace_candidates_for`'s `source_proposal_hash`). The
    helper preserves the supplied value verbatim."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal()
    sentinel_hash = "a" * 64
    rej = _build_proposal_rejection(
        raw,
        proposal_hash=sentinel_hash,
        file_path="src/x.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=EvidenceTier.JUDGED,
    )
    assert rej.proposal_hash == sentinel_hash
    assert len(rej.claimed_finding_type_hash) == 16  # short prefix
    assert rej.claimed_finding_type_len == len("sql_injection")
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_reason == "finding_type_not_in_enum"
    assert rej.rejection_detail == "no_near_enum_match"
    assert rej.file_path == "src/x.py"


def test_build_proposal_rejection_claimed_evidence_tier_can_be_none() -> None:
    """For `rejection_reason="evidence_tier_not_in_enum"`, no parsed
    enum exists, so `claimed_evidence_tier=None` is the valid shape.
    The lifted event's cross-field validator enforces this iff."""
    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal(evidence_tier="WRONG_TIER_VALUE")
    rej = _build_proposal_rejection(
        raw,
        proposal_hash="a" * 64,
        file_path="src/x.py",
        rejection_reason="evidence_tier_not_in_enum",
        rejection_detail="no_near_enum_match",
        claimed_evidence_tier=None,
    )
    assert rej.claimed_evidence_tier is None


def test_build_proposal_rejection_claimed_finding_type_hash_matches_recipe() -> None:
    """The hash recipe is pinned: sha256(raw.finding_type.encode())[:16].
    Drift here would mean event-side and parser-side claims disagree."""
    import hashlib

    from outrider.agent.nodes.analyze_parser import _build_proposal_rejection

    raw = _build_raw_proposal(finding_type="some_bogus_type")
    rej = _build_proposal_rejection(
        raw,
        proposal_hash="a" * 64,
        file_path="src/x.py",
        rejection_reason="finding_type_not_in_enum",
        rejection_detail="",
        claimed_evidence_tier=None,
    )
    expected = hashlib.sha256(b"some_bogus_type").hexdigest()[:16]
    assert rej.claimed_finding_type_hash == expected


# ---------------------------------------------------------------------------
# Spec §6 commit-3 — producer-admission checks (finding_type / evidence_tier /
# query_match_id / trace stub)
# ---------------------------------------------------------------------------


def _build_response_json(*proposals: dict[str, object]) -> str:
    """Wrap proposal dicts into a JSON `AnalyzeResponseRaw` shape."""
    import json

    return json.dumps({"findings": list(proposals)})


def _minimal_proposal(**overrides: object) -> dict[str, object]:
    """Minimum-fields raw proposal for parser tests."""
    base: dict[str, object] = {
        "finding_type": "sql_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "t",
        "description": "d",
        "evidence": "e",
        "line_start": 1,
        "line_end": 1,
    }
    base.update(overrides)
    return base


def _build_scope_unit(
    *, line_start: int = 1, line_end: int = 10, byte_start: int = 0, byte_end: int = 100
):  # type: ignore[no-untyped-def]
    """Minimal valid `ScopeUnit` for parser tests. Line range (default 1-10)
    drives the line-space admission gate (`line_range_within_scope_unit`); the
    default contains the `_minimal_proposal` default line range `(1, 1)`. Byte
    range is carried for completeness but no longer gates admission (FUP-126)."""
    from outrider.ast_facts.models import ScopeUnit, compute_unit_id

    return ScopeUnit(
        unit_id=compute_unit_id("src/x.py", kind="function", qualified_name="some_function"),
        kind="function",
        name="some_function",
        qualified_name="some_function",
        file_path="src/x.py",
        line_start=line_start,
        line_end=line_end,
        byte_start=byte_start,
        byte_end=byte_end,
    )


def test_step3_rejects_unknown_evidence_tier() -> None:
    """Raw evidence_tier outside `EvidenceTier` enum → rejection with
    `evidence_tier_not_in_enum` and `claimed_evidence_tier=None` (the
    bidirectional cross-field rule)."""
    response = _build_response_json(_minimal_proposal(evidence_tier="bogus_tier"))
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "evidence_tier_not_in_enum"
    assert rej.claimed_evidence_tier is None
    assert result.counters.n_proposals_seen == 1
    assert result.counters.n_proposals_rejected == 1
    assert result.admitted_findings == ()


def test_step2_rejects_unknown_finding_type_with_parsed_tier() -> None:
    """Raw finding_type outside `FindingType` enum → rejection with
    `finding_type_not_in_enum` and `claimed_evidence_tier` carrying
    the parsed enum value (the implementation-order swap ensures
    evidence_tier was admitted first)."""
    response = _build_response_json(
        _minimal_proposal(finding_type="unknown_type", evidence_tier="judged")
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "finding_type_not_in_enum"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED


def test_step4_observed_rejects_when_query_match_id_absent() -> None:
    """OBSERVED proposal without a `query_match_id` → producer-
    admission rejection with `query_match_id_not_in_registry`,
    `claimed_evidence_tier=OBSERVED`."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="observed", query_match_id=None)
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "query_match_id_not_in_registry"
    assert rej.claimed_evidence_tier == EvidenceTier.OBSERVED
    assert rej.rejection_detail == "<absent>"


def test_step4_observed_rejects_fabricated_query_match_id() -> None:
    """OBSERVED proposal claiming an id NOT in the pre-supplied
    registry set → rejection; the claimed id is the rejection_detail
    (safe because raw schema constrains the pattern + length)."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="fabricated_id",
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "query_match_id_not_in_registry"
    assert rej.rejection_detail == "fabricated_id"


def test_step4_observed_passes_when_query_match_id_in_registry() -> None:
    """Happy path: OBSERVED proposal with a real registry id passes
    producer admission AND span admission (containing scope unit
    supplied), reaches `ReviewFinding` construction, and is admitted."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="real_id",
        )
    )
    result = _call_parser(
        response,
        query_match_id_set=frozenset({"real_id"}),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,  # unused on the clean path (line-space gate); kept harmless
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    finding = result.admitted_findings[0]
    assert finding.evidence_tier == EvidenceTier.OBSERVED
    assert finding.query_match_id == "real_id"


def test_step4_inferred_rejects_on_pass_0_no_trace_context_yet() -> None:
    """Per trace-node arc: pass 0 (the first analyze pass over a PR
    diff) rejects every INFERRED — no trace context exists yet, so
    no trace_path can ground the claim. Pass 1+ admits INFERRED with
    a non-empty trace_path (covered by a separate test below).
    """
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="inferred",
            trace_path=("some.symbol", "step.two"),
        )
    )
    result = _call_parser(response)  # pass_index defaults to 0
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "trace_path_not_admissible"
    assert rej.claimed_evidence_tier == EvidenceTier.INFERRED
    assert "pass 0" in rej.rejection_detail


def test_step4_inferred_admits_on_pass_1_with_valid_trace_path() -> None:
    """Per trace-node arc + Codex round 2: pass 1 (post-trace re-entry)
    admits INFERRED when (a) trace_path is a non-empty array of non-
    empty strs AND (b) every element appears in
    `valid_trace_path_elements` (the deterministic-proof set of scope-
    unit names from the trace-fetched file). Without (b), the parser
    would trust model-claimed structural evidence — violating
    `evidence-tier-schema-enforced`.
    """
    valid_set = frozenset({"middleware.auth.authenticate", "middleware.auth.validate_token"})
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="inferred",
            trace_path=("middleware.auth.authenticate", "middleware.auth.validate_token"),
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        pass_index=1,
        valid_trace_path_elements=valid_set,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    finding = result.admitted_findings[0]
    assert finding.evidence_tier == EvidenceTier.INFERRED
    assert finding.trace_path == (
        "middleware.auth.authenticate",
        "middleware.auth.validate_token",
    )


def test_step4_inferred_rejects_on_pass_1_with_ungrounded_trace_path() -> None:
    """Codex round 2 regression: pass-1 INFERRED with trace_path elements
    NOT in the deterministic-proof set is rejected per
    `evidence-tier-schema-enforced`. The model cannot claim to have
    walked a scope unit that doesn't exist in the trace-fetched file.
    """
    valid_set = frozenset({"only.this.name"})
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="inferred",
            trace_path=("forged.symbol.one", "forged.symbol.two"),
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        pass_index=1,
        valid_trace_path_elements=valid_set,
    )
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "trace_path_not_admissible"
    # Detail describes the gap quantitatively — does NOT echo the
    # model-supplied ungrounded elements verbatim (would let a hostile
    # LLM ship arbitrary text into the audit row).
    assert "2 element(s) not in" in rej.rejection_detail
    assert "forged.symbol" not in rej.rejection_detail


def test_step4_inferred_rejects_on_pass_1_empty_trace_path() -> None:
    """Pass 1 admits INFERRED only when trace_path is non-empty.
    An empty trace_path is rejected with the same reason but a
    different rejection_detail naming the shape requirement."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="inferred",
            trace_path=(),
        )
    )
    result = _call_parser(response, pass_index=1)
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "trace_path_not_admissible"
    assert "non-empty trace_path" in rej.rejection_detail


def test_step4_judged_skips_producer_admission() -> None:
    """JUDGED is the only tier the model can claim unilaterally — no
    structural artifact required. Producer admission is skipped; with
    a containing scope unit supplied for span admission, the proposal
    reaches `ReviewFinding` construction and is admitted."""
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    assert result.admitted_findings[0].evidence_tier == EvidenceTier.JUDGED


def test_all_rejected_returns_aggregate_result() -> None:
    """Multiple proposals all rejected — each on a different reason —
    aggregate counters reflect the per-proposal outcomes; result is
    well-formed (no admitted findings, no trace candidates yet)."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="bogus_tier"),  # rejected at §3
        _minimal_proposal(finding_type="unknown_type"),  # rejected at §2
        _minimal_proposal(evidence_tier="inferred"),  # V1 stub rejection at §4
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 3
    reasons = {rej.rejection_reason for rej in result.proposal_rejections}
    assert reasons == {
        "evidence_tier_not_in_enum",
        "finding_type_not_in_enum",
        "trace_path_not_admissible",
    }
    assert result.counters.n_proposals_seen == 3
    assert result.counters.n_proposals_rejected == 3
    assert result.counters.n_findings_emitted == 0
    assert result.admitted_findings == ()


def test_proposal_rejection_uses_canonical_proposal_hash() -> None:
    """Every rejection's `proposal_hash` is the canonical hash from
    `compute_proposal_hash` — same recipe whether the proposal is
    admitted or rejected, so the join with `TraceCandidate.source_proposal_hash`
    works across both outcomes."""
    from outrider.policy.canonical import compute_proposal_hash

    response = _build_response_json(
        _minimal_proposal(finding_type="unknown_type", evidence_tier="judged")
    )
    result = _call_parser(response, file_path="src/example.py")
    rej = result.proposal_rejections[0]
    expected = compute_proposal_hash(
        source_file_path="src/example.py",
        finding_type="unknown_type",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        line_start=1,
        line_end=1,
    )
    assert rej.proposal_hash == expected


# ---------------------------------------------------------------------------
# Spec §6 commit-4 — span admission (per-outcome branch)
# ---------------------------------------------------------------------------


def test_step5_clean_line_range_inside_scope_unit_passes_admission() -> None:
    """Clean-outcome happy path: the proposal's line range lies inside one
    of the file's included scope units (line-space) → admission passes →
    the proposal is admitted (`line_start`/`line_end` from the model)."""
    response = _build_response_json(_minimal_proposal(line_start=2, line_end=3))
    scope = _build_scope_unit(line_start=1, line_end=10)  # contains lines 2-3
    result = _call_parser(response, included_scope_units=(scope,))
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


def test_step5_clean_line_range_inside_indented_scope_admits() -> None:
    """FUP-126 reproduction: a finding on the FIRST line of a non-file-head,
    indented scope unit (a method whose token `byte_start` is past column 0)
    is admitted under line-space containment. The pre-fix byte gate rejected
    this (`su.byte_start > whole-line span byte_start`) — the silent
    findings-loss bug."""
    # Scope unit at lines 5-20, byte_start=26 (indented — four lines precede
    # it). A finding on line 6 anchored by the model at the clip head used to
    # come in as byte 0, failing `26 <= 0`; now it's line 6 in [5, 20].
    response = _build_response_json(_minimal_proposal(line_start=6, line_end=6))
    scope = _build_scope_unit(line_start=5, line_end=20, byte_start=26, byte_end=400)
    result = _call_parser(response, included_scope_units=(scope,))
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1
    assert result.admitted_findings[0].line_start == 6


def test_step5_clean_line_range_outside_all_scope_units_rejects() -> None:
    """Clean outcome: line range doesn't land in any included scope unit →
    `span_outside_scope_unit` rejection with `claimed_evidence_tier`
    carrying the parsed enum value. (A past-EOF range is subsumed here — it
    is in no scope — preserving today's single-reason clean-path taxonomy.)"""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged", line_start=50, line_end=51)
    )
    scope = _build_scope_unit(line_start=1, line_end=10)  # excludes lines 50-51
    result = _call_parser(response, included_scope_units=(scope,))
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_scope_unit"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_detail == "(50,51)"


def test_step5_clean_multiple_scope_units_admits_if_any_contains() -> None:
    """Clean outcome: a line range landing in the second scope unit of a
    tuple still admits — the `any(...)` covers every included unit."""
    response = _build_response_json(_minimal_proposal(line_start=15, line_end=16))
    scope_a = _build_scope_unit(line_start=1, line_end=10)  # excludes
    scope_b = _build_scope_unit(line_start=12, line_end=20)  # contains 15-16
    result = _call_parser(response, included_scope_units=(scope_a, scope_b))
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


def test_step5_clean_empty_scope_units_rejects_every_proposal() -> None:
    """Edge case: a clean-outcome call with no included scope units
    rejects every proposal at `span_outside_scope_unit`. (The node
    body shouldn't make this call — outcome would be
    NO_CHANGED_SCOPE_UNITS in that case — but the parser handles it
    defensively.)"""
    response = _build_response_json(_minimal_proposal())
    result = _call_parser(response, included_scope_units=())
    assert len(result.proposal_rejections) == 1
    assert result.proposal_rejections[0].rejection_reason == "span_outside_scope_unit"


_DEGRADED_SRC = "a\nb\nc\nd"  # 4 lines; len 7 bytes


# Lines 2-3 of `_DEGRADED_SRC` ("a\nb\nc\nd") cover bytes [2,6) — the addable
# degraded context for a finding the patch added on those lines.
_DEGRADED_CTX_LINES_2_3 = ((2, 6),)


def test_step5_degraded_line_range_within_file_passes_admission() -> None:
    """Degraded-outcome happy path: the line range translates to a byte span
    within file bounds AND inside the addable degraded context → admission passes
    → finding admitted. No scope units consulted in degraded mode."""
    response = _build_response_json(_minimal_proposal(line_start=2, line_end=3))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_content=_DEGRADED_SRC,
        file_byte_length=len(_DEGRADED_SRC.encode()),
        degraded_context_byte_ranges=_DEGRADED_CTX_LINES_2_3,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


def test_step5_degraded_span_outside_addable_context_rejects() -> None:
    """FUP-138: a degraded JUDGED finding whose span is WITHIN FILE but OUTSIDE the
    addable diff context is rejected `span_outside_degraded_context` — it cannot
    anchor to in-file bytes the patch didn't add. Finding on lines 2-3 (span [2,6));
    context is line 4 only ((6,7)), which the span does not intersect."""
    response = _build_response_json(_minimal_proposal(line_start=2, line_end=3))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_content=_DEGRADED_SRC,
        file_byte_length=len(_DEGRADED_SRC.encode()),
        degraded_context_byte_ranges=((6, 7),),  # line 4 only — finding is on lines 2-3
    )
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_degraded_context"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert result.admitted_findings == ()


def test_step5_degraded_line_range_past_eof_rejects() -> None:
    """Degraded outcome: a line range past EOF (`line_range_to_span` raises)
    → `span_outside_file` rejection with `claimed_evidence_tier` carrying
    the parsed enum; detail shows the offending line range."""
    response = _build_response_json(_minimal_proposal(line_start=2, line_end=50))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_content="a\nb\nc",  # only 3 lines
        file_byte_length=5,
    )
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_file"
    assert rej.claimed_evidence_tier == EvidenceTier.JUDGED
    assert rej.rejection_detail == "(2,50)"


def test_step5_degraded_does_not_consult_scope_units() -> None:
    """Degraded outcome ignores `included_scope_units` — the deterministic bound is
    `span_within_file` + `span_within_degraded_context`, not line-space scope
    containment. Even an empty scope-unit tuple still admits when the line range is
    in file and inside the addable degraded context."""
    response = _build_response_json(_minimal_proposal(line_start=2, line_end=3))
    result = _call_parser(
        response,
        degraded_mode=True,
        file_content=_DEGRADED_SRC,
        file_byte_length=len(_DEGRADED_SRC.encode()),
        included_scope_units=(),  # ignored in degraded mode
        degraded_context_byte_ranges=_DEGRADED_CTX_LINES_2_3,
    )
    assert result.proposal_rejections == ()
    assert len(result.admitted_findings) == 1


def test_step5_degraded_trailing_empty_line_rejects_zero_width() -> None:
    """Degraded outcome: a schema-valid line range that translates to a
    ZERO-WIDTH span — the empty final line after a trailing newline — is
    rejected `span_outside_file` with the `zero_width:` detail prefix. (Line
    proposals are never zero-width in line space; this is the one byte-level
    edge the derived-span non-empty floor still guards.)"""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged", line_start=2, line_end=2)
    )
    # "a\n" → line 1 = "a", line 2 = the empty trailing line (starts at EOF).
    result = _call_parser(
        response,
        degraded_mode=True,
        file_content="a\n",
        file_byte_length=2,
    )
    assert len(result.proposal_rejections) == 1
    rej = result.proposal_rejections[0]
    assert rej.rejection_reason == "span_outside_file"
    assert rej.rejection_detail.startswith("zero_width:")


def test_step5_degraded_rejects_independent_of_query_match_id() -> None:
    """Degraded outcome: even if the model claimed `observed` with an
    id, the producer-admission step rejects FIRST (no registry-fired
    set in degraded mode). The span admission step never runs for an
    OBSERVED claim in degraded mode. This pin documents the rejection
    ordering."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id="some_id",
            line_start=2,
            line_end=3,
        )
    )
    # Empty query_match_id_set (typical for degraded mode) → producer
    # admission rejects with query_match_id_not_in_registry, before the
    # line-range step ever runs.
    result = _call_parser(
        response,
        degraded_mode=True,
        file_byte_length=100,
        query_match_id_set=frozenset(),
    )
    assert len(result.proposal_rejections) == 1
    assert result.proposal_rejections[0].rejection_reason == "query_match_id_not_in_registry"


# ---------------------------------------------------------------------------
# Spec §6 commit-5 — ReviewFinding + TraceCandidate construction + counters
# ---------------------------------------------------------------------------


def test_step8_admitted_finding_severity_from_policy_table() -> None:
    """`ReviewFinding.severity` is set from `SEVERITY_POLICY[finding_type]`
    per `severity-set-by-policy` — never proposed by the model."""
    from outrider.policy.severity import SEVERITY_POLICY, FindingType

    response = _build_response_json(
        _minimal_proposal(finding_type="sql_injection", evidence_tier="judged")
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    finding = result.admitted_findings[0]
    assert finding.severity == SEVERITY_POLICY[FindingType.SQL_INJECTION]


def test_step8_admitted_finding_dimension_from_mapping_table() -> None:
    """`ReviewFinding.dimension` is set from `FINDING_TYPE_TO_DIMENSION[finding_type]`
    per `evidence-tier-schema-enforced` — never proposed by the model."""
    from outrider.policy.dimensions import FINDING_TYPE_TO_DIMENSION
    from outrider.policy.severity import FindingType

    response = _build_response_json(
        _minimal_proposal(finding_type="sql_injection", evidence_tier="judged")
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    finding = result.admitted_findings[0]
    assert finding.dimension == FINDING_TYPE_TO_DIMENSION[FindingType.SQL_INJECTION]


def test_step8_admitted_finding_policy_version_from_closure() -> None:
    """`ReviewFinding.policy_version` is the value passed into the
    parser via the node-body closure (analyze passes
    `active_policy_version=ACTIVE_POLICY_VERSION` by default)."""
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        active_policy_version="1.2.3",
    )
    finding = result.admitted_findings[0]
    assert finding.policy_version == "1.2.3"


def test_step8_admitted_finding_review_and_installation_ids() -> None:
    """`review_id` + `installation_id` flow through from caller."""
    from uuid import uuid4

    review_id = uuid4()
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        review_id=review_id,
        installation_id=99999,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    finding = result.admitted_findings[0]
    assert finding.review_id == review_id
    assert finding.installation_id == 99999


def test_step8_admitted_finding_content_hash_matches_recipe() -> None:
    """`content_hash` is `compute_finding_content_hash(file_path,
    line_start, line_end, finding_type)`. Mirror of the
    `ReviewFinding._verify_content_hash` validator — the parser
    constructs this canonically so construction never fails on the
    content-hash mismatch."""
    from outrider.audit.events import compute_finding_content_hash
    from outrider.policy.severity import FindingType

    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection",
            evidence_tier="judged",
            line_start=1,
            line_end=1,
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_path="src/x.py",
    )
    finding = result.admitted_findings[0]
    expected = compute_finding_content_hash(
        file_path="src/x.py",
        line_start=finding.line_start,
        line_end=finding.line_end,
        finding_type=FindingType.SQL_INJECTION,
    )
    assert finding.content_hash == expected


def test_step8_line_range_from_proposal() -> None:
    """`line_start`/`line_end` on the admitted finding ARE the model's raw
    proposal lines (identity-preserved), not derived from a byte span — the
    proposal is line-based now (FUP-126). Pins that the parser carries the
    proposed line range straight onto the finding."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged", line_start=2, line_end=4)
    )
    result = _call_parser(response, included_scope_units=(_build_scope_unit(),))
    finding = result.admitted_findings[0]
    assert finding.line_start == 2
    assert finding.line_end == 4


def test_step10_trace_candidates_collected_from_admitted_proposal() -> None:
    """Step 10: trace_candidates are collected from admitted proposals
    AND stamped with the parent's `proposal_hash` as
    `source_proposal_hash`. `candidate_id` is content-derived via
    `compute_candidate_id`."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": "other", "reason": "calls helper"},
            ],
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    assert len(result.trace_candidates) == 1
    parent_hash = result.admitted_findings[0]
    cand = result.trace_candidates[0]
    assert cand.import_string == "other"
    assert cand.reason == "calls helper"
    # source_proposal_hash links back to the parent (same recipe)
    from outrider.policy.canonical import compute_proposal_hash

    expected_parent_hash = compute_proposal_hash(
        source_file_path="src/x.py",
        finding_type="sql_injection",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        line_start=1,
        line_end=1,
    )
    assert cand.source_proposal_hash == expected_parent_hash
    _ = parent_hash  # admitted_findings doesn't expose proposal_hash; verified via recipe


def test_step10_trace_candidates_collected_from_rejected_proposal() -> None:
    """Per spec §6 step 10: trace_candidates are collected from BOTH
    admitted and proposal-level-REJECTED raw proposals. A rejected
    proposal's child candidates still surface — rejected
    `JUDGED`-claim might still flag a legitimate cross-file signal."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="unknown_type",  # rejects at finding_type_not_in_enum
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": "related", "reason": "should still surface"},
            ],
        )
    )
    result = _call_parser(response)
    assert len(result.proposal_rejections) == 1
    assert len(result.admitted_findings) == 0
    # Trace candidate from the rejected proposal still appears
    assert len(result.trace_candidates) == 1
    assert result.trace_candidates[0].import_string == "related"


def test_step10_response_level_rejection_collects_no_trace_candidates() -> None:
    """Per spec §6 step 0: response-level rejections (parser step 0)
    have no proposals to collect from. Best-effort salvage from
    malformed JSON is forbidden per the clean-counter contract."""
    result = _call_parser("not valid json {{{")
    assert result.response_rejection is not None
    assert result.trace_candidates == ()


# ---------------------------------------------------------------------------
# Step 10, from-import candidate correction (FUP-209 follow-up): when the
# analyzed file's own imports contradict a candidate's module prefix, a
# corrected module-form SIBLING is emitted alongside the original — never
# instead of it (the trailing-name match is a heuristic; a coincidental
# collision must not clobber a valid candidate). Live instance:
# `svc.utils.normalize_owner` proposed while the file says
# `from svc.db import normalize_owner`.
# ---------------------------------------------------------------------------


def _from_import_ref(
    module: str,
    names: tuple[str, ...],
    *,
    kind: str = "from",
    line: int = 1,
) -> object:
    from outrider.ast_facts.models import ImportRef

    return ImportRef(
        file_path="src/x.py",
        line=line,
        import_kind=kind,  # type: ignore[arg-type]
        module=module,
        names=names,
        is_simple_direct=False,
    )


def _candidate_response(import_string_raw: str) -> str:
    return _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[{"import_string_raw": import_string_raw, "reason": "r"}],
        )
    )


def test_hallucinated_module_emits_corrected_sibling() -> None:
    """The live FUP-209 shape: the model proposes `svc.utils.normalize_owner`
    while the file from-imports `normalize_owner` from `svc.db`. A
    corrected module-form sibling is emitted ALONGSIDE the original
    (original first — insertion order feeds the bucket cap) and the
    correction is counted."""
    result = _call_parser(
        _candidate_response("svc.utils.normalize_owner"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("svc.db", ("normalize_owner", "run_query")),),
    )
    assert [c.import_string for c in result.trace_candidates] == [
        "svc.utils.normalize_owner",
        "svc.db",
    ]
    assert result.counters.n_trace_candidates_module_corrected == 1
    assert result.counters.n_trace_candidates_emitted == 2  # original + sibling


def test_coincidental_trailing_name_collision_preserves_original() -> None:
    """Review pin: `myproject.settings` proposed while the file has
    `from django.conf import settings` — a coincidental trailing-name
    collision. The valid original MUST survive (the probe ladder resolves
    whichever path exists); the django.conf sibling rides along and
    simply misses if not in-repo."""
    result = _call_parser(
        _candidate_response("myproject.settings"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("django.conf", ("settings",)),),
    )
    assert [c.import_string for c in result.trace_candidates] == [
        "myproject.settings",
        "django.conf",
    ]


def test_consistent_symbol_form_candidate_gets_no_sibling() -> None:
    """`svc.db.run_query` under `from svc.db import run_query` — the prefix
    already matches the importing module; the trace ladder handles the
    symbol form, so no sibling and no count."""
    result = _call_parser(
        _candidate_response("svc.db.run_query"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("svc.db", ("normalize_owner", "run_query")),),
    )
    assert [c.import_string for c in result.trace_candidates] == ["svc.db.run_query"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_module_form_submodule_import_gets_no_sibling() -> None:
    """`svc.db` under `from svc import db`: trailing 'db' maps to module
    'svc', but the candidate IS the imported submodule (prefix equals the
    mapped module) — a 'svc' sibling would probe the package instead of
    the module for no benefit. No sibling."""
    result = _call_parser(
        _candidate_response("svc.db"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("svc", ("db",)),),
    )
    assert [c.import_string for c in result.trace_candidates] == ["svc.db"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_bare_symbol_candidate_gets_corrected_sibling() -> None:
    """A bare-symbol candidate (`normalize_owner`) probes a repo-root
    module and dead-ends; the from-import map emits the importing module
    as a sibling."""
    result = _call_parser(
        _candidate_response("normalize_owner"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("svc.db", ("normalize_owner",)),),
    )
    assert [c.import_string for c in result.trace_candidates] == [
        "normalize_owner",
        "svc.db",
    ]
    assert result.counters.n_trace_candidates_module_corrected == 1


def test_unmatched_candidate_gets_no_sibling() -> None:
    """A candidate whose trailing component matches no from-imported name
    passes through alone (also the empty-import_refs default: failed
    or degraded parses disable correction)."""
    result = _call_parser(
        _candidate_response("pkg.ghost"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("svc.db", ("run_query",)),),
    )
    assert [c.import_string for c in result.trace_candidates] == ["pkg.ghost"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_relative_and_star_imports_do_not_correct() -> None:
    """Relative modules can't be rewritten into absolute dotted candidates
    without package context, and star imports name no symbols — neither
    participates in the correction map."""
    result = _call_parser(
        _candidate_response("svc.utils.run_query"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(
            _from_import_ref(".db", ("run_query",), kind="relative"),
            _from_import_ref("svc.legacy", (), kind="star"),
        ),
    )
    assert [c.import_string for c in result.trace_candidates] == ["svc.utils.run_query"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_later_from_import_shadows_earlier() -> None:
    """Two from-imports binding the same name: the later one wins,
    matching Python's runtime shadowing semantics — the sibling carries
    the later module."""
    result = _call_parser(
        _candidate_response("wrong.place.helper"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(
            _from_import_ref("first.mod", ("helper",), line=1),
            _from_import_ref("second.mod", ("helper",), line=2),
        ),
    )
    assert [c.import_string for c in result.trace_candidates] == [
        "wrong.place.helper",
        "second.mod",
    ]


def test_invalid_import_ref_module_emits_no_sibling() -> None:
    """Defensive: an ImportRef whose module isn't a valid dotted identifier
    (producer-bug shape) is dropped at map build — the model's canonical
    form passes through alone and nothing is counted as corrected."""
    result = _call_parser(
        _candidate_response("svc.utils.run_query"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("123bad", ("run_query",)),),
    )
    assert [c.import_string for c in result.trace_candidates] == ["svc.utils.run_query"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_invalid_module_does_not_resurrect_shadowed_valid_module() -> None:
    """Validation runs AFTER shadowing resolves: a later producer-bug ref
    shadowing a valid earlier one drops the name entirely rather than
    resurrecting the shadowed module — the file's runtime binding is the
    later (broken) one, so no sibling is trustworthy."""
    result = _call_parser(
        _candidate_response("wrong.place.helper"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(
            _from_import_ref("first.mod", ("helper",), line=1),
            _from_import_ref("123bad", ("helper",), line=2),
        ),
    )
    assert [c.import_string for c in result.trace_candidates] == ["wrong.place.helper"]
    assert result.counters.n_trace_candidates_module_corrected == 0


def test_nfd_imported_name_still_matches_nfc_candidate() -> None:
    """Map keys are NFC-normalized: a decomposed-Unicode imported name
    (NFD from the wire) still matches the canonicalized (NFC) candidate
    trailing component."""
    nfd_name = "cafe\u0301"  # e + combining acute (NFD)
    nfc_name = "caf\u00e9"  # composed (NFC)
    result = _call_parser(
        _candidate_response(f"wrong.mod.{nfc_name}"),
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
        import_refs=(_from_import_ref("real.mod", (nfd_name,)),),
    )
    assert [c.import_string for c in result.trace_candidates] == [
        f"wrong.mod.{nfc_name}",
        "real.mod",
    ]
    assert result.counters.n_trace_candidates_module_corrected == 1


def test_from_import_map_digest_properties() -> None:
    """The cache-key component over the from-import map: deterministic,
    ref-order-independent for the same resolved map, sensitive to a
    module change, and empty-map-distinct (no from-imports ≠ any
    populated map)."""
    from outrider.agent.nodes.analyze_parser import from_import_map_digest

    ref_a = _from_import_ref("svc.db", ("run_query",), line=1)
    ref_b = _from_import_ref("svc.utils", ("helper",), line=2)
    base = from_import_map_digest((ref_a, ref_b))  # type: ignore[arg-type]
    assert base == from_import_map_digest((ref_a, ref_b))  # type: ignore[arg-type]
    assert base == from_import_map_digest((ref_b, ref_a))  # type: ignore[arg-type]
    changed = from_import_map_digest(
        (_from_import_ref("svc.other", ("run_query",), line=1), ref_b)  # type: ignore[arg-type]
    )
    assert changed != base
    assert from_import_map_digest(()) != base


def test_counters_reflect_admitted_and_rejected_mix() -> None:
    """Mixed result: some proposals admitted, some rejected. Counters
    accurately reflect the per-proposal outcomes."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged"),  # admitted
        _minimal_proposal(evidence_tier="bogus_tier"),  # rejected at §3
        _minimal_proposal(evidence_tier="judged"),  # admitted
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.counters.n_proposals_seen == 3
    assert result.counters.n_findings_emitted == 2
    assert result.counters.n_proposals_rejected == 1
    assert result.counters.n_responses_rejected == 0
    assert result.counters.n_trace_candidates_emitted == 0


def test_counter_accounting_equation_holds() -> None:
    """`n_proposals_seen == n_findings_emitted + n_proposals_rejected`
    must hold at the parser layer (same equation the lifted
    `AnalyzeCompletedEvent._enforce_proposal_accounting` enforces at
    construction). Mix of admitted, rejected, and candidate-producing
    proposals."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged"),  # admitted
        _minimal_proposal(evidence_tier="inferred"),  # rejected (V1 stub)
        _minimal_proposal(finding_type="unknown_type"),  # rejected
        _minimal_proposal(evidence_tier="judged"),  # admitted
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert result.counters.n_proposals_seen == (
        result.counters.n_findings_emitted + result.counters.n_proposals_rejected
    )


# ---------------------------------------------------------------------------
# Spec §6 commit-6 — audit-fold regression pins
# ---------------------------------------------------------------------------


def test_hostile_import_string_does_not_crash_parser() -> None:
    """Sharp-edges H1 + general-purpose §4: a single hostile
    `import_string_raw` (e.g. path-shaped, shell metacharacter, Python
    keyword part) used to crash `parse_analyze_response` mid-loop
    because `TraceCandidate(...)` construction was outside any
    try/except. Fix: per-candidate try/except in
    `_collect_trace_candidates_for` drops the bad candidate; the
    parent proposal still produces its admission/rejection outcome.
    Post-DECISIONS.md#024 rename: hostile values are import-string-
    malformed instead of path-traversal-shaped, but the
    crash-resistance contract is identical."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                # Path separator — rejected by is_valid_import_string
                {"import_string_raw": "../../etc/passwd", "reason": "hostile"},
            ],
        )
    )
    # Must NOT raise — that was the bug
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1  # parent admitted
    assert len(result.trace_candidates) == 0  # hostile candidate dropped
    # Sharp-edges F1 audit-fold: the drop is counted, not silent.
    assert result.counters.n_trace_candidates_dropped_malformed == 1


def test_admitted_finding_carries_proposal_hash_from_compute_recipe() -> None:
    """Per DECISIONS.md#025: analyze's admission path threads proposal_hash
    from compute_proposal_hash through to ReviewFinding construction.
    The same value the rejected branch (FindingProposalRejectedEvent.proposal_hash)
    would emit appears on admitted findings — closes the trace-join contract."""
    from outrider.policy.canonical import compute_proposal_hash

    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    admitted = result.admitted_findings[0]

    # The expected proposal_hash matches the canonical recipe over the
    # admitted proposal's payload (same recipe the rejected branch uses).
    # Reading _minimal_proposal defaults: source_file_path=src/x.py,
    # finding_type=sql_injection, evidence_tier=judged, no query_match_id /
    # trace_path, title/description/evidence are "t"/"d"/"e", lines 1-1.
    expected_hash = compute_proposal_hash(
        source_file_path="src/x.py",
        finding_type="sql_injection",
        evidence_tier="judged",
        query_match_id=None,
        trace_path=None,
        title="t",
        description="d",
        evidence="e",
        line_start=1,
        line_end=1,
    )
    assert admitted.proposal_hash == expected_hash


def test_dropped_malformed_counter_increments_per_dropped_candidate() -> None:
    """Sharp-edges F1 audit-fold: the n_trace_candidates_dropped_malformed
    counter increments ONCE per dropped raw candidate, not per affected
    proposal. Operators reading AnalyzeCompletedEvent.n_trace_candidates_
    dropped_malformed should see aggregate drift across pass-level
    candidate emissions, not a binary "any proposal had a drop" signal."""
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                # Three different rejection cases — all drop, all count
                {"import_string_raw": "../../etc/passwd", "reason": "path"},
                {"import_string_raw": "foo;rm -rf", "reason": "shell"},
                {"import_string_raw": "foo.class", "reason": "keyword"},
                # One valid one — admits, doesn't count toward drops
                {"import_string_raw": "valid.module", "reason": "ok"},
            ],
        )
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.trace_candidates) == 1  # only the valid one
    assert result.counters.n_trace_candidates_emitted == 1
    assert result.counters.n_trace_candidates_dropped_malformed == 3


def test_admitted_finding_survives_hostile_sibling_candidate() -> None:
    """Multi-proposal regression: proposal[0] admits cleanly,
    proposal[1] has a hostile candidate. Pre-fix, the proposal[1]
    crash dropped proposal[0]'s admission too."""
    response = _build_response_json(
        _minimal_proposal(evidence_tier="judged", line_start=2, line_end=3),
        _minimal_proposal(
            evidence_tier="judged",
            line_start=4,
            line_end=5,
            trace_candidates=[
                # Python keyword part — rejected by is_valid_import_string
                {"import_string_raw": "foo.class", "reason": "hostile"},
            ],
        ),
    )
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),  # default line range 1-10
    )
    assert len(result.admitted_findings) == 2  # both parents survive
    assert len(result.trace_candidates) == 0  # only the hostile one was filtered


def test_decomposed_unicode_import_string_uses_canonical_nfc_form() -> None:
    """Post-DECISIONS.md#024 rename: the analog of the original "alias
    canonicalize" test (`./src/foo.py` → `src/foo.py`) for import strings
    is NFC normalization. The parser canonicalizes `import_string_raw`
    via `is_valid_import_string` before computing `candidate_id`;
    without that step, the raw NFD bytes would hash differently from
    the NFC bytes the schema validator stores, and
    `_enforce_candidate_id_matches_payload` would crash.
    Per M3 / adversarial-modeler #1."""
    # Construct via unicodedata.normalize — visually-identical string
    # literals compile to the SAME bytes in a UTF-8 source file
    # (whichever form the editor/save flow produced), defeating the
    # NFC-vs-NFD distinction the test claims to exercise.
    import unicodedata

    precomposed = "café.bar"
    decomposed = unicodedata.normalize("NFD", precomposed)
    # Sanity: NFD must differ byte-wise from NFC (otherwise the
    # canonicalization step is unobservable here).
    assert decomposed != precomposed
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="judged",
            trace_candidates=[
                {"import_string_raw": decomposed, "reason": "decomposed"},
            ],
        )
    )
    # Must NOT raise — that was the bug class
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1
    assert len(result.trace_candidates) == 1
    # Candidate's `import_string` is the canonical NFC form, not the raw NFD
    assert result.trace_candidates[0].import_string == precomposed


def test_query_match_id_rejection_detail_sanitizes_ansi_escape() -> None:
    """Adversarial HIGH-2: raw schema for `query_match_id` ships only
    `max_length=256` (no pattern), but spec §3 named
    `[A-Za-z0-9_./:-]+` as the safety class. Parser-side
    `_sanitize_query_match_id_for_detail` enforces the spec-promised
    class so attacker bytes can't land verbatim in `rejection_detail`."""
    hostile = "fake_id\x1b]8;;file:///etc/passwd\x07click\x1b]8;;\x07"
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id=hostile,
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert len(result.proposal_rejections) == 1
    detail = result.proposal_rejections[0].rejection_detail
    # ANSI escape sequences MUST be sanitized
    assert "\x1b" not in detail
    assert "]8" not in detail
    # Safe-class chars survive
    assert "fake_id" in detail
    # Out-of-class chars are replaced with `?`
    assert "?" in detail


def test_query_match_id_rejection_detail_preserves_safe_chars() -> None:
    """Benign IDs (matching the spec pattern) pass through unchanged so
    operators see the structural form."""
    benign = "python.security.sql_injection:42"
    response = _build_response_json(
        _minimal_proposal(
            evidence_tier="observed",
            query_match_id=benign,
        )
    )
    result = _call_parser(response, query_match_id_set=frozenset({"real_id"}))
    assert result.proposal_rejections[0].rejection_detail == benign


def test_narrow_exception_handler_lets_unexpected_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharp-edges M4 + adversarial M2: the `except ValidationError`
    block used to catch every exception class (MemoryError,
    RecursionError, etc.) and fold them into
    `schema_construction_failed`. Narrowed so genuinely-unexpected
    failures propagate as bugs rather than as misleading rejections.

    To prove narrowing actually narrows: monkeypatch the
    `compute_finding_content_hash` call (which fires inside the
    try-block during `ReviewFinding` construction) to raise a
    `RuntimeError`. The parser MUST propagate it instead of folding
    to `schema_construction_failed`. A re-broadening to
    `except Exception` would catch the RuntimeError and produce a
    rejection event — this test would fail with `RuntimeError` not
    raised, naming the regression precisely.

    Companion smoke assertion preserves the happy path.
    """
    from outrider.agent.nodes import analyze_parser

    # Smoke: clean admission still works (sanity that the test setup
    # is otherwise valid).
    response = _build_response_json(_minimal_proposal(evidence_tier="judged"))
    result = _call_parser(
        response,
        included_scope_units=(_build_scope_unit(),),
        file_content="x" * 200,
    )
    assert len(result.admitted_findings) == 1

    # Force a non-narrowed exception inside the try-block. If the
    # parser re-broadens its except, this RuntimeError gets folded
    # to a rejection and `pytest.raises` would fail.
    def _explode(**_kwargs: object) -> str:
        msg = "synthetic RuntimeError from monkeypatched compute_finding_content_hash"
        raise RuntimeError(msg)

    monkeypatch.setattr(analyze_parser, "compute_finding_content_hash", _explode)
    with pytest.raises(RuntimeError, match="synthetic RuntimeError"):
        _call_parser(
            response,
            included_scope_units=(_build_scope_unit(),),
            file_content="x" * 200,
        )


# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_exports_all_documented_surfaces() -> None:
    from outrider.agent.nodes import analyze_parser

    expected = {
        "ParserCounters",
        "ParserResult",
        "ProposalRejection",
        "ResponseRejection",
        "parse_analyze_response",
    }
    assert set(analyze_parser.__all__) == expected


# ---------------------------------------------------------------------------
# FUP-162 parameterized-call veto
# (specs/2026-06-12-sqli-parameterized-call-veto.md). The scan comes from
# the REAL detection module so these tests exercise the full deterministic
# chain: source bytes -> ast_facts scan -> coordinates veto -> rejection.
# ---------------------------------------------------------------------------

_PARAMETERIZED_SOURCE = (
    "def find(cursor, q):\n"  # 1
    "    x = 1\n"  # 2
    "    y = 2\n"  # 3
    '    cursor.execute("SELECT * FROM t WHERE x = %s", (q,))\n'  # 4
    '    cursor.execute(f"SELECT * FROM {q}")\n'  # 5
    "    return x\n"  # 6
)


def _parameterized_scan():  # type: ignore[no-untyped-def]
    from outrider.ast_facts.parameterized_calls import scan_parameterized_calls

    return scan_parameterized_calls(_PARAMETERIZED_SOURCE.encode("utf-8"))


def _veto_parser_kwargs() -> dict[str, object]:
    return {
        "file_content": _PARAMETERIZED_SOURCE,
        "file_byte_length": len(_PARAMETERIZED_SOURCE.encode("utf-8")),
        "included_scope_units": (_build_scope_unit(line_start=1, line_end=6),),
        "parameterized_call_scan": _parameterized_scan(),
    }


def test_veto_rejects_judged_sqli_on_indented_parameterized_call() -> None:
    """THE spec-pinned case: a single-line `cursor.execute("…%s…", (q,))`
    INDENTED inside a function body must veto. Byte containment would have
    silently missed it (whole-line span starts before the call node's
    token start) — line space is the load-bearing frame."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection", evidence_tier="judged", line_start=4, line_end=4
        )
    )
    result = _call_parser(response, **_veto_parser_kwargs())
    assert result.admitted_findings == ()
    [rejection] = result.proposal_rejections
    assert rejection.rejection_reason == "sql_injection_on_parameterized_call"
    assert rejection.rejection_detail == "(4,4)"
    assert rejection.claimed_evidence_tier == EvidenceTier.JUDGED
    assert result.counters.n_proposals_rejected == 1
    assert result.counters.n_findings_emitted == 0


def test_veto_disabled_when_scan_is_none() -> None:
    """scan=None (the degraded-mode value and the default) admits the same
    proposal — the veto never rests on an absent or untrustworthy parse."""
    kwargs = _veto_parser_kwargs()
    kwargs.pop("parameterized_call_scan")
    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection", evidence_tier="judged", line_start=4, line_end=4
        )
    )
    result = _call_parser(response, **kwargs)
    [finding] = result.admitted_findings
    assert finding.finding_type == FindingType.SQL_INJECTION


def test_veto_is_type_scoped_other_findings_on_same_call_admit() -> None:
    """Per DECISIONS.md#041 guidance: other finding types on the same
    parameterized query stay flaggable — the veto removes only the
    structurally-impossible sql_injection claim."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="missing_error_handling",
            evidence_tier="judged",
            line_start=4,
            line_end=4,
        )
    )
    result = _call_parser(response, **_veto_parser_kwargs())
    [finding] = result.admitted_findings
    assert finding.finding_type == FindingType.MISSING_ERROR_HANDLING


def test_veto_not_fired_when_range_touches_unsafe_execute() -> None:
    """Lines 4-5 span the safe call AND the f-string call — the proposal
    flows through to HITL (the spec's spanning rule)."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection", evidence_tier="judged", line_start=4, line_end=5
        )
    )
    result = _call_parser(response, **_veto_parser_kwargs())
    [finding] = result.admitted_findings
    assert finding.finding_type == FindingType.SQL_INJECTION


def test_veto_not_fired_on_the_unsafe_call_itself() -> None:
    """The f-string call at line 5 is exactly what sql_injection should
    flag — never vetoed."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection", evidence_tier="judged", line_start=5, line_end=5
        )
    )
    result = _call_parser(response, **_veto_parser_kwargs())
    [finding] = result.admitted_findings
    assert finding.finding_type == FindingType.SQL_INJECTION


def test_veto_requires_judged_tier_observed_sqli_admits() -> None:
    """An OBSERVED sql_injection with a registry-valid query_match_id is
    NOT vetoed — the veto is scoped to JUDGED claims; structural-evidence
    tiers carry their own deterministic proof requirements."""
    response = _build_response_json(
        _minimal_proposal(
            finding_type="sql_injection",
            evidence_tier="observed",
            query_match_id="python.function_definition",
            line_start=4,
            line_end=4,
        )
    )
    kwargs = _veto_parser_kwargs()
    kwargs["query_match_id_set"] = frozenset({"python.function_definition"})
    result = _call_parser(response, **kwargs)
    [finding] = result.admitted_findings
    assert finding.evidence_tier == EvidenceTier.OBSERVED
