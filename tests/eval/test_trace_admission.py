"""Trace admission instrument for the openai host (openai-native-host spec).

Frozen pass predicate (spec "Gates before any production-shaped use"),
covered across THREE surfaces because no single exchange carries it:

  - RANKING (the trace model's own call): the expected candidate ranked
    first, valid permutation, no fabricated candidate id, zero rejected
    responses — graded by the PRODUCTION `parse_trace_ranking`.
  - EMISSION (the analyze model proposing candidates — FUP-236's actual
    failure surface, which the finding-graded scorecard does not cover):
    graded through the PRODUCTION chain — real admission incl. the #024
    corrected sibling, join to an ADMITTED finding, deterministic ladder
    resolution. The honest bare-symbol form passes via its sibling; a
    fabricated module ("app.user_store" for a DI'd parameter) fails. An
    all-red emission result does NOT admit the host: change the production
    context and rerun, approve a spec amendment, or do not ship.
  - RESOLUTION: deterministic probe code, exercised end-to-end here by the
    real-node test (scripted ranking -> probe ladder -> resolved
    TraceDecision), not a model surface.

PRODUCTION-REACHABLE scenario: both ranking candidates share ONE finding's
provenance — production buckets candidates per finding and SKIPS the ranking
call when every bucket is a singleton, so cross-finding candidates would
capture a request the real node never sends. The real-node test below proves
the provider IS invoked for this exact scenario.

ONE canonical paid path: the paid rows live in the wire probe
(`<model>:trace` for node-capable models, `<model>:trace_emission` for all
full-matrix models). This file never spends: scripted negative twins prove
the graders can fail, and the fixture-graded halves resolve captures THROUGH
THE VERIFIED MANIFEST (`verified_capture_fixture` — versions, hashes, row
verdicts, adjudication all checked; a stale fixture left behind by a failed
rerun cannot grade). Scenario constants are duplicated in the probe
deliberately; drift fails loud through the parsers.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

from outrider.agent.nodes import trace as trace_module
from outrider.agent.nodes.analyze_parser import ParserResult, parse_analyze_response
from outrider.agent.nodes.trace import trace
from outrider.agent.nodes.trace_parser import (
    TraceRankingParsed,
    TraceRankingRejected,
    parse_trace_ranking,
)
from outrider.ast_facts.python_adapter import parse_python
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.canonical import compute_candidate_id, compute_round_id
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import trace as trace_prompt
from outrider.schemas import AnalysisRound, ReviewFinding, ReviewState
from outrider.schemas.pr_context import PRContext
from outrider.schemas.trace_candidate import TraceCandidate

from .model_comparison import _NoOpImportPathResolver
from .test_model_comparison import _ScriptedProvider
from .test_openai_scorecard import _PROBE_MANIFEST, verified_capture_fixture

if TYPE_CHECKING:
    from outrider.audit.events import ReviewPhaseEvent, TraceDecisionEvent

# --- The deterministic admission scenario (duplicated in the probe) ---------
# The FUP-236 shape: one REAL cross-file candidate (the finding's tainted
# call flows through run_query imported from app.db) and one peripheral
# distractor — BOTH from the same finding, so production's per-finding
# bucketing yields a 2-candidate bucket and the ranking call fires.
_SOURCE_HASH = "3" * 64
_REAL_IMPORT = "app.db"
_DISTRACTOR_IMPORT = "app.render_helpers"
_REAL_REASON = (
    "handlers.py builds the flagged query via run_query imported from app.db; "
    "the finding's tainted value flows directly into it"
)
_DISTRACTOR_REASON = (
    "render_error_page formats the error string shown when the request fails; "
    "cosmetic to the finding's data flow"
)
_REAL_ID = compute_candidate_id(
    source_proposal_hash=_SOURCE_HASH, import_string=_REAL_IMPORT, reason=_REAL_REASON
)
_DISTRACTOR_ID = compute_candidate_id(
    source_proposal_hash=_SOURCE_HASH,
    import_string=_DISTRACTOR_IMPORT,
    reason=_DISTRACTOR_REASON,
)

# Emission scenario (duplicated in the probe). PRODUCTION-EQUIVALENT
# VISIBILITY: the rendered prompt shows scope bodies + clipped hunks only —
# the import line and "app.db" are WITHHELD (the probe's dry run pins that),
# because production prompts don't carry module imports (cache/key.py,
# `_assemble_scope_unit_context`). The verdict GENUINELY hangs on the hidden
# imported `escape_owner` (does it escape quotes before the concatenated
# query?), so the analyze prompt's candidate discipline REQUIRES a candidate
# — a locally-proven defect would instead instruct an empty array (the
# probe-v4 scenario's defect). Grading runs the PRODUCTION chain
# end-to-end (`parse_analyze_response` admission incl. the #024 corrected
# sibling, join to an ADMITTED finding, deterministic resolution through the
# real probe ladder); a candidate that neither resolves nor is a visible
# bare from-import name is fabricated-from-hidden-information — the FUP-236
# failure — and fails GLOBALLY, because production retains and
# probe-resolves every admitted candidate.
_EMISSION_FROM_IMPORTS = {"run_query": "app.db", "escape_owner": "app.db"}
_EMISSION_EXPECTED_TYPE = "sql_injection"
_EMISSION_TAINT_LINE = 5
# The scenario's REAL file content (production holds the whole file even
# though the prompt renders scope bodies + clipped hunks only — the import at
# line 1 is file-only, never prompt-visible) and the scenario repository the
# deterministic ladder resolves against. app/db.py carries the UNSAFE
# escape_owner (strips whitespace, does NOT escape quotes) — what a real
# trace fetch would reveal.
_EMISSION_FILE = "app/handlers.py"
_EMISSION_FILE_CONTENT = (
    "from app.db import run_query, escape_owner\n"
    "\n"
    "def get_user_orders(request):\n"
    '    owner = escape_owner(request.GET["owner"])\n'
    '    return run_query("SELECT * FROM orders WHERE owner = \'" + owner + "\'")\n'
)
_SCENARIO_REPO: dict[str, bytes] = {
    "app/db.py": (
        b"def escape_owner(owner):\n    return owner.strip()\n\n\ndef run_query(sql):\n    ...\n"
    )
}


def _ladder_resolves(import_string: str) -> bool:
    """Deterministic resolution through the REAL probe ladder
    (`_probe_paths_for` + `_symbol_in_content` — production code, not a
    re-implementation) against the scenario repository. Level 0 reads the
    whole string as a module; level k strips k trailing components and
    requires the stripped symbol in the resolved content."""
    parts = import_string.split(".")
    for level in range(len(parts)):
        for path in trace_module._probe_paths_for(import_string, level, _EMISSION_FILE):
            content = _SCENARIO_REPO.get(path)
            if content is None:
                continue
            if level == 0 or trace_module._symbol_in_content(parts[-level], content):
                return True
    return False


# Models whose captures this instrument grades when present in the verified
# manifest (Terra appears only when a swap capture declares it).
_NODE_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")
_EMISSION_MODELS = ("gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra")


def _candidates() -> tuple[TraceCandidate, TraceCandidate]:
    real = TraceCandidate(
        candidate_id=_REAL_ID,
        source_proposal_hash=_SOURCE_HASH,
        reason=_REAL_REASON,
        import_string=_REAL_IMPORT,
    )
    distractor = TraceCandidate(
        candidate_id=_DISTRACTOR_ID,
        source_proposal_hash=_SOURCE_HASH,
        reason=_DISTRACTOR_REASON,
        import_string=_DISTRACTOR_IMPORT,
    )
    return (real, distractor)


def _ranking_json(*ids: str) -> str:
    return json.dumps({"ranked_candidate_ids": list(ids)})


def _grade_ranking(text: str) -> tuple[str, str]:
    """(verdict, detail) — verdict 'pass' | 'rejected' | 'misranked'. Built on
    the PRODUCTION parser so 'zero rejected responses' and 'no fabricated id'
    are graded by the exact code the trace node runs."""
    outcome = parse_trace_ranking(response_text=text, candidates=_candidates())
    if isinstance(outcome, TraceRankingRejected):
        return ("rejected", outcome.reason)
    assert isinstance(outcome, TraceRankingParsed)
    first = outcome.ordered_candidates[0]
    if first.candidate_id != _REAL_ID:
        return ("misranked", first.import_string)
    return ("pass", first.import_string)


def _emission_parse(text: str) -> ParserResult:
    """Run the captured response through the PRODUCTION admission chain:
    the REAL ast_facts parse of the scenario file (scope units + import
    refs — the from-import map production's #024 correction reads) and the
    REAL `parse_analyze_response` 10-step flow. NOTE: candidates from
    REJECTED proposals REMAIN in `ParserResult.trace_candidates` for
    forensic use (spec §6 step 10) — exactly as in production; the
    admitted-proposal-hash JOIN in the grader is what keeps a forensic-only
    candidate from certifying a pass."""
    parsed_file = parse_python(
        _EMISSION_FILE_CONTENT.encode("utf-8"), _EMISSION_FILE, _NoOpImportPathResolver()
    )
    units = tuple(u for u in parsed_file.scope_units if u.name == "get_user_orders")
    assert units, "scenario file must parse to the expected scope unit"
    return parse_analyze_response(
        text,
        review_id=UUID("00000000-0000-0000-0000-0000000000d3"),
        installation_id=42,
        file_path=_EMISSION_FILE,
        file_content=_EMISSION_FILE_CONTENT,
        file_byte_length=len(_EMISSION_FILE_CONTENT.encode("utf-8")),
        included_scope_units=units,
        query_match_id_set=frozenset(),
        degraded_mode=False,
        active_policy_version=ACTIVE_POLICY_VERSION,
        finish_reason="end_turn",
        import_refs=parsed_file.imports,
    )


def _grade_emission(text: str) -> tuple[str, str]:
    """(verdict, detail) for the analyze-emission surface, graded through the
    PRODUCTION chain end-to-end: `parse_analyze_response` admission (schema +
    proof boundary + candidate collection incl. the #024 corrected sibling),
    join to an ADMITTED expected finding, and deterministic resolution via
    the REAL probe ladder. Verdicts: 'rejected' (response-level rejection),
    'guessed' (an admitted candidate that neither resolves nor is a visible
    bare from-import name — fabricated from hidden information, the FUP-236
    failure), 'missing_finding' (the expected sql_injection finding was not
    ADMITTED — includes forensic-only shapes whose parent admission
    rejected), 'none_emitted' (admitted expected finding, no joined
    candidate), 'unresolved' (joined candidates but none lands on app/db.py),
    or 'pass'."""
    result = _emission_parse(text)
    if result.response_rejection is not None:
        return ("rejected", result.response_rejection.rejection_detail)
    fabricated = sorted(
        {
            c.import_string
            for c in result.trace_candidates
            if not _ladder_resolves(c.import_string)
            and c.import_string not in _EMISSION_FROM_IMPORTS
        }
    )
    if fabricated:
        return ("guessed", ", ".join(fabricated))
    admitted = [
        f
        for f in result.admitted_findings
        if f.finding_type is FindingType.SQL_INJECTION
        and f.line_start <= _EMISSION_TAINT_LINE <= f.line_end
    ]
    if not admitted:
        return (
            "missing_finding",
            f"no ADMITTED {_EMISSION_EXPECTED_TYPE} finding covering line "
            f"{_EMISSION_TAINT_LINE} (admission may have rejected the proposal)",
        )
    admitted_hashes = {f.proposal_hash for f in admitted}
    joined = [c for c in result.trace_candidates if c.source_proposal_hash in admitted_hashes]
    if not joined:
        return ("none_emitted", "expected finding carries no admitted trace candidate")
    if not any(_ladder_resolves(c.import_string) for c in joined):
        return ("unresolved", ", ".join(sorted({c.import_string for c in joined})))
    return ("pass", ", ".join(sorted({c.import_string for c in joined})))


class _NoOpPhaseSink:
    async def emit_phase(self, event: ReviewPhaseEvent) -> None:
        return None


class _EchoTraceSink:
    """Audit-first contract double: returns the incoming event as canonical;
    the read-side recovery surface reports no prior decisions."""

    async def emit_trace_decision(self, event: TraceDecisionEvent) -> TraceDecisionEvent:
        return event

    async def get_trace_decisions(self, *, review_id: UUID) -> tuple[TraceDecisionEvent, ...]:
        return ()


async def _stub_github_factory(_installation_id: int) -> object:
    return object()


def _finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id=UUID("00000000-0000-0000-0000-0000000000d1"),
        review_id=UUID("00000000-0000-0000-0000-0000000000d2"),
        installation_id=42,
        finding_type=FindingType.SQL_INJECTION,
        severity=lookup_severity(FindingType.SQL_INJECTION),
        file_path="app/handlers.py",
        line_start=5,
        line_end=5,
        title="SQL built by string concatenation from a request parameter",
        description="owner is concatenated into the orders query passed to run_query.",
        evidence='run_query("SELECT * FROM orders WHERE owner = \'" + owner + "\'")',
        dimension=lookup_dimension(FindingType.SQL_INJECTION),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path="app/handlers.py",
            line_start=5,
            line_end=5,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=_SOURCE_HASH,
    )


def _state() -> ReviewState:
    now = datetime.now(UTC)
    finding = _finding()
    analysis_round = AnalysisRound(
        round_id=compute_round_id(
            pass_index=0,
            files_examined=(finding.file_path,),
            files_skipped=(),
            finding_content_hashes=(finding.content_hash,),
        ),
        pass_index=0,
        findings=(finding,),
        files_examined=(finding.file_path,),
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )
    return ReviewState(
        review_id=finding.review_id,
        received_at=now,
        pr_context=PRContext(
            installation_id=42,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="t",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="dev",
            total_additions=5,
            total_deletions=0,
            changed_files=(),
        ),
        analysis_rounds=[analysis_round],
        trace_candidates=list(_candidates()),
        is_eval=True,
    )


@pytest.mark.asyncio
async def test_real_node_sends_ranking_and_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """PRODUCTION-REACHABILITY proof: driving the REAL trace node with this
    scenario (two candidates, one finding bucket) INVOKES the provider —
    singleton buckets would skip it — with both candidate ids on the wire,
    and the scripted ranking flows through the deterministic probe ladder to
    a RESOLVED TraceDecision for app/db.py (the predicate's resolution
    half)."""
    fetched: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        fetched.append(path)
        if path == "app/db.py":
            return b"def run_query(sql):\n    ...\n"
        return None

    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    provider = _ScriptedProvider(_ranking_json(_REAL_ID, _DISTRACTOR_ID))
    delta = await trace(
        _state(),
        provider=provider,  # type: ignore[arg-type]
        trace_model="gpt-5.6-luna",
        phase_event_sink=_NoOpPhaseSink(),  # type: ignore[arg-type]
        trace_sink=_EchoTraceSink(),  # type: ignore[arg-type]
        github_factory=_stub_github_factory,  # type: ignore[arg-type]
    )

    (request,) = provider.calls  # the ranking call FIRED — the scenario is reachable
    assert request.node_id == "trace"
    assert request.prompt_template_version == trace_prompt.VERSION
    assert _REAL_ID in request.user_prompt
    assert _DISTRACTOR_ID in request.user_prompt

    decisions = delta["trace_decisions"]
    resolved = [d for d in decisions if d.resolution_status == "resolved"]
    assert resolved, f"no resolved decision in {[d.resolution_status for d in decisions]}"
    assert resolved[0].target_file == "app/db.py"
    assert "app/db.py" in fetched


def test_grader_negative_twins() -> None:
    """The ranking grader can FAIL — one twin per parser rejection reason plus
    the misranking case, each differing from the passing control in exactly
    the graded property. (`ranking_id_missing` has no reachable twin here:
    with a matching count, no duplicates, and no fabricated ids, a
    two-candidate set is necessarily complete — that branch is defensive.)"""
    assert _grade_ranking(_ranking_json(_REAL_ID, _DISTRACTOR_ID)) == ("pass", _REAL_IMPORT)

    verdict, reason = _grade_ranking("I think the db module matters most.")
    assert (verdict, reason) == ("rejected", "raw_response_unparseable")

    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID))
    assert (verdict, reason) == ("rejected", "ranking_count_mismatch")

    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID, _REAL_ID))
    assert (verdict, reason) == ("rejected", "ranking_id_duplicated")

    # A fabricated candidate id — the ranking-layer analog of a guessed
    # module path — is rejected by the production parser.
    verdict, reason = _grade_ranking(_ranking_json(_REAL_ID, "f" * 64))
    assert (verdict, reason) == ("rejected", "ranking_id_fabricated")

    verdict, detail = _grade_ranking(_ranking_json(_DISTRACTOR_ID, _REAL_ID))
    assert (verdict, detail) == ("misranked", _DISTRACTOR_IMPORT)


def test_emission_grader_negative_twins() -> None:
    """The emission grader can FAIL — the FUP-236 guessed path
    (app.user_store for a DI'd parameter), the zero-candidate run, an
    UNTIED candidate (a stray finding carrying app.db while the expected
    finding has none), a missing expected finding, and a non-conforming
    response are each caught; the tied real-import emission passes."""

    def _finding_dict(
        candidates: list[dict[str, str]],
        *,
        finding_type: str = "sql_injection",
        line: int = _EMISSION_TAINT_LINE,
    ) -> dict[str, object]:
        return {
            "finding_type": finding_type,
            "evidence_tier": "judged",
            "title": "SQL concat from request parameter",
            "description": "owner concatenated into the orders query",
            "evidence": "run_query(...)",
            "line_start": line,
            "line_end": line,
            "trace_candidates": candidates,
        }

    def _analyze_response(findings: list[dict[str, object]]) -> str:
        return json.dumps({"findings": findings})

    real = [{"import_string_raw": _REAL_IMPORT, "reason": "taint sink"}]
    good = _analyze_response([_finding_dict(real)])
    assert _grade_emission(good) == ("pass", _REAL_IMPORT)

    # BARE-SYMBOL pass — the HONEST form under V1 visibility: the model names
    # the visible symbol; production admission adds the #024 corrected
    # sibling app.db, which resolves through the ladder. The joined set shows
    # both (original retained + sibling). Each mapped symbol is pinned as its
    # own variant — the load-bearing one (escape_owner, the sanitizer the
    # verdict hangs on) and the sink (run_query).
    for symbol in ("escape_owner", "run_query"):
        bare = [{"import_string_raw": symbol, "reason": "does it sanitize/escape quotes?"}]
        verdict, detail = _grade_emission(_analyze_response([_finding_dict(bare)]))
        assert verdict == "pass", f"{symbol}: {detail}"
        assert "app.db" in detail and symbol in detail

    # CONSISTENT SYMBOL-FORM pass: app.db.run_query names no foreign module —
    # the suffix-strip ladder resolves it (level 1 + symbol check) with
    # nothing guessed.
    consistent = [{"import_string_raw": "app.db.run_query", "reason": "taint sink"}]
    verdict, detail = _grade_emission(_analyze_response([_finding_dict(consistent)]))
    assert (verdict, detail) == ("pass", "app.db.run_query")

    # FORENSIC-ONLY shape: a pass-0 `inferred` proposal carries app.db, but
    # production ADMISSION rejects the parent (no valid trace_path elements at
    # pass 0) and the candidate never joins — the grader must not certify it.
    forensic = _analyze_response(
        [
            {
                "finding_type": "sql_injection",
                "evidence_tier": "inferred",
                "trace_path": ["app/db.py"],
                "title": "SQL concat from request parameter",
                "description": "owner concatenated into the orders query",
                "evidence": "run_query(...)",
                "line_start": _EMISSION_TAINT_LINE,
                "line_end": _EMISSION_TAINT_LINE,
                "trace_candidates": real,
            }
        ]
    )
    verdict, _ = _grade_emission(forensic)
    assert verdict == "missing_finding"

    # FOREIGN-MODULE form FAILS the strict predicate: the #024 correction
    # adds app.db ALONGSIDE, but production retains and probe-resolves the
    # guessed original — the guess survives, so it is still a guessed path.
    foreign = [{"import_string_raw": "app.database.run_query", "reason": "taint sink"}]
    verdict, detail = _grade_emission(_analyze_response([_finding_dict(foreign)]))
    assert (verdict, detail) == ("guessed", "app.database.run_query")

    guessed = _analyze_response(
        [
            _finding_dict(
                [
                    {
                        "import_string_raw": "app.user_store",
                        "reason": "guessed from a parameter name",
                    }
                ]
            )
        ]
    )
    verdict, detail = _grade_emission(guessed)
    assert (verdict, detail) == ("guessed", "app.user_store")

    # GLOBAL guess: the expected finding's own set is clean, but ANOTHER
    # finding carries an uncorrectable guess — production would retain and
    # probe-resolve it, so the row fails even with a clean expected set.
    stray_guess = _analyze_response(
        [
            _finding_dict(real),
            _finding_dict(
                [{"import_string_raw": "app.user_store", "reason": "guessed"}],
                finding_type="missing_error_handling",
                line=3,
            ),
        ]
    )
    verdict, detail = _grade_emission(stray_guess)
    assert (verdict, detail) == ("guessed", "app.user_store")

    verdict, _ = _grade_emission(_analyze_response([_finding_dict([])]))
    assert verdict == "none_emitted"

    # UNTIED: a stray, unrelated finding carries app.db while the expected
    # SQLi finding has no candidate — the candidate cannot feed the expected
    # finding's trace, so it must NOT rescue the row.
    untied = _analyze_response(
        [
            _finding_dict([], line=_EMISSION_TAINT_LINE),
            _finding_dict(real, finding_type="missing_error_handling", line=3),
        ]
    )
    verdict, _ = _grade_emission(untied)
    assert verdict == "none_emitted"

    verdict, _ = _grade_emission(
        _analyze_response([_finding_dict(real, finding_type="missing_error_handling")])
    )
    assert verdict == "missing_finding"

    verdict, _ = _grade_emission("prose, not JSON")
    assert verdict == "rejected"


def _skip_unless_capture() -> None:
    if not _PROBE_MANIFEST.exists():
        pytest.skip(
            "paid probe capture absent — run the wire probe first "
            "(op run --env-file=.env -- uv run python spikes/openai/probe.py)"
        )


def _fixture_message(tag: str) -> dict[str, object] | None:
    data = verified_capture_fixture(tag)
    if data is None:
        return None
    doc = json.loads(data.decode("utf-8"))
    message = doc["choices"][0]["message"]
    assert isinstance(message, dict)
    return message


def test_captured_ranking_passes_frozen_predicate() -> None:
    """Grade every node-capable model's captured ranking row through the
    VERIFIED manifest. A miss FAILS (Terra swap + rerun, never a softened
    gate); models absent from the declared matrix are skipped."""
    _skip_unless_capture()
    graded = []
    for model in _NODE_MODELS:
        message = _fixture_message(f"{model}:trace")
        if message is None:
            continue
        assert not message.get("refusal"), f"{model} trace row returned a refusal"
        verdict, detail = _grade_ranking(str(message.get("content") or ""))
        assert verdict != "rejected", f"{model} trace ranking REJECTED ({detail})"
        assert verdict == "pass", f"{model} ranked {detail!r} above the load-bearing candidate"
        graded.append(model)
        print(f"\n[trace admission: PASS — {model} ranked {detail!r} first]")  # noqa: T201
    assert graded, "verified manifest carried no node-model trace rows"


def test_captured_emission_passes_frozen_predicate() -> None:
    """Grade every full-matrix model's captured emission row through the
    VERIFIED manifest: real imported module strings only, no guessed paths,
    at least one candidate for the cross-file taint."""
    _skip_unless_capture()
    graded = []
    for model in _EMISSION_MODELS:
        message = _fixture_message(f"{model}:trace_emission")
        if message is None:
            continue
        assert not message.get("refusal"), f"{model} emission row returned a refusal"
        verdict, detail = _grade_emission(str(message.get("content") or ""))
        assert verdict == "pass", f"{model} emission {verdict.upper()}: {detail}"
        graded.append(model)
        print(f"\n[trace emission: PASS — {model} proposed {detail}]")  # noqa: T201
    assert graded, "verified manifest carried no emission rows"
