"""Live OpenAI GPT-5.6 wire probe — the openai-native-host admission capture.

The PAID wire capture DECISIONS.md#056 gates the `openai` HostProfile on
(specs/2026-07-18-openai-native-host.md: the host is WIRE-PENDING until these
fixtures exist). A free `--dry-run` prefix builds and audits every request body
without network; the paid matrix writes raw responses AND a machine-readable
SUCCESS MANIFEST to `spikes/openai/fixtures/` — the scorecard REQUIRES that
manifest (`test_openai_scorecard.py` FAILS without a passing one), so a failed
capture cannot silently launch an ~128-call scorecard.

Evidence-integrity rules:
  - base_url is `OPENAI_PROFILE.base_url` EXACTLY — no env override. An
    alternate host would receive the real key (credential leak) and would
    produce "native OpenAI" evidence from the wrong host.
  - Every request body comes from the PRODUCTION `_build_sdk_kwargs` on a real
    `LLMRequest` — including the production-derived `prompt_cache_key` — so
    the capture exercises the wire the provider actually sends. The frozen
    envelope pin lives in `tests/unit/test_openai_wire_golden.py`.
  - Every row has a REQUIRED success predicate; the run exits non-zero and the
    manifest records `all_required_passed: false` if any required row fails.
  - The manifest records the capture's provenance (base_url + the profile
    contract digest, so a profile change stales the capture; PROBE_CONTRACT_VERSION,
    so a prompt/matrix/predicate change stales it too) and each fixture's
    sha256, and the scorecard gate re-verifies all of it plus the cold/warm
    conservation inequalities FROM THE FIXTURE BYTES.
  - The conservation EQUATION (writes inside prompt_tokens or additive — the
    spec's [probe] item) is count-characterized by the probe but adjudicated
    by the OPERATOR against billed usage. The adjudication is BOUND to THIS
    capture: the sanitized billing_adjudication.json artifact (raw exports
    stay local/gitignored — the Fireworks evidence policy) must carry the
    fixtures' exact response IDs, a bounded capture window covering their
    `created` stamps, and the billed fresh/write class counts, which the gate
    cross-checks against the wire counts under the adjudicated equation. A
    hash proves integrity; the binding proves RELEVANCE — a stale or
    unrelated export cannot authorize a run. The gate also recomputes
    per-model count support from fixture bytes (contrary counts refuse;
    indeterminate needs an explicit reconciliation) and refuses a verdict
    that differs from what read_usage() implements.

Capture matrix — Sol + Luna full by default; Terra gets the cheap reasoning
row, and the role flags (--full-models / --trace-model / --patch-model /
--analyze-models) reshape the matrix for swap reruns (the spec's workflow:
a fallback inherits the gate before serving). ALL rows are
REQUIRED — the refusal-normalization fixture is a PRE-SHIP gate (spec "Gates
before any production-shaped use") and wire admission is PER MODEL, so a
refusal fixture is required per full-matrix model, not best-effort:

  envelope  json_object on the analyze-style prompt: default tier echoed AND
            output conforms to AnalyzeResponseRaw (fence-stripped).
  cold      >=1024-token stable prefix, cold cache: cache_write_tokens > 0
            reported; the (prompt, cached, write) triple is RECORDED so the
            conservation equation can be pinned from the fixture.
  warm      immediate same-prefix repeat: cached_tokens > 0.
  refusal   message.refusal POPULATED. A comply-with-caveats response fails
            the row honestly — never soften the gate instead. The elicitation
            is FROZEN from spikes/openai/refusal_discovery.py output (a
            bounded free-standing discovery matrix; the v4 ransomware prompt
            conflated OpenAI's API-level cyber screen with in-band structured
            refusal — Sol 400'd before the model ran, Luna complied in-band).
            Per #056 the admission is PER MODEL — each model's row uses its own
            frozen elicitation (a single prompt may cover all models, or they
            differ). The paid runner refuses to start while the elicitations
            are unfrozen or any serving model still holds the placeholder
            (_REFUSAL_ELICITATIONS_FROZEN / _REFUSAL_ELICITATIONS).
  trace     (declared node model) the REAL trace-node ranking request over
            the admission scenario (both candidates share one finding's
            provenance — the production-reachable multi-candidate bucket):
            valid permutation per the production parser, the load-bearing
            candidate ranked first. Graded offline again by
            tests/eval/test_trace_admission.py.
  patch     (declared node model) the REAL synthesize patch-pass request: an
            anchored single-line fix surviving apply_patch_batch AND matching
            the CLOSED reviewed-fix set (a wrong-but-safe or composite-unsafe
            line fails). Graded offline again by
            tests/eval/test_patch_admission.py.
  trace_emission (declared analyze candidates) the PRODUCTION analyze prompt
            (real stable prefix + render + analyze VERSION + schema) with
            production-equivalent visibility: scope bodies + clipped hunks
            ONLY — the import line and "app.db" are WITHHELD (dry-run
            pinned). The scenario's verdict GENUINELY depends on the hidden
            imported helper (escape_owner sanitization), so the analyze
            prompt's own candidate discipline requires a candidate — a
            locally-proven defect would instead instruct an empty array.
            Graded through the PRODUCTION chain: real admission
            (parse_analyze_response incl. the #024 corrected sibling), join
            to an ADMITTED sql_injection finding, deterministic resolution
            via the real probe ladder. The honest bare-symbol form
            (escape_owner / run_query) passes via its corrected sibling; a
            fabricated foreign module (app.user_store) fails. A red row does
            NOT admit the host under any recording: the paths forward are a
            production context change + rerun, an approved spec amendment,
            or no-ship.
  reasoning (Terra) reasoning_effort="none" accepted, tier echoed.

Run (PAID):  op run --env-file=.env -- uv run --no-sync python spikes/openai/probe.py
Swap reruns: declare the replacement PER ROLE — e.g.
             --full-models=gpt-5.6-sol,gpt-5.6-luna,gpt-5.6-terra \
                 --trace-model=gpt-5.6-terra          (trace-field swap only)
             --full-models=gpt-5.6-sol,gpt-5.6-terra \
                 --trace-model=gpt-5.6-terra --patch-model=gpt-5.6-terra \
                 --analyze-models=gpt-5.6-sol,gpt-5.6-terra
             Failed incumbents are NOT re-required and may be dropped from
             the matrix entirely; their failure evidence stays in the
             previous manifest.
Dry  (free): uv run --no-sync python spikes/openai/probe.py --dry-run

CREDIT NOTE: 13 calls default; the cold/warm rows carry a ~2.6k-token prefix
and the two emission rows carry the full production analyze prompt (the
expensive rows, ~production-analyze-call sized each). 401/402 on the first
call means fix the key, nothing spent.
"""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Final
from uuid import UUID, uuid4

from outrider.agent.nodes.analyze import (
    _assemble_query_match_id_list,
    _assemble_scope_unit_context,
    _build_context_manifest,
    _concat_clipped_hunks,
)
from outrider.agent.nodes.analyze_parser import parse_analyze_response
from outrider.agent.nodes.degradation import decide_degradation
from outrider.agent.nodes.patch_generation import apply_patch_batch
from outrider.agent.nodes.trace import _probe_paths_for, _symbol_in_content
from outrider.agent.nodes.trace_parser import TraceRankingParsed, parse_trace_ranking
from outrider.ast_facts.python_adapter import parse_python
from outrider.audit.events import compute_finding_content_hash
from outrider.coordinates import lookup_patched_file
from outrider.llm.base import LLMRequest
from outrider.llm.host_profiles import OPENAI_PROFILE
from outrider.llm.openai_compatible_provider import _build_sdk_kwargs
from outrider.llm.pricing import MIN_CACHEABLE_TOKENS
from outrider.llm.raw_openai_capture import (
    RawCapture,
    RawCaptureShapeError,
    RawOpenAICaptureClient,
    RawOpenAICaptureError,
    RawUsage,
)
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.canonical import compute_candidate_id
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import analyze as analyze_prompt
from outrider.prompts import patch as patch_prompt
from outrider.prompts import trace as trace_prompt
from outrider.schemas import ReviewFinding
from outrider.schemas.llm.analyze import (
    ANALYZE_RESPONSE_SCHEMA,
    ANALYZE_RESPONSE_SCHEMA_JSON,
    AnalyzeResponseRaw,
)
from outrider.schemas.trace_candidate import TraceCandidate

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_MANIFEST = _FIXTURE_DIR / "manifest.json"

# The probe PROCEDURE's contract version. Folds the CAPTURE procedure's
# evidentiary identity: the prompts (_SYS/_USER/_REFUSAL_ELICITATIONS/_STABLE_PREFIX),
# the schema bytes riding response_schema_json, the plan's tag matrix, and the
# per-row success predicates in _evaluate. The MANIFEST's field shape is NOT
# this token's scope — it has its own (MANIFEST_SCHEMA_VERSION below); one
# token per shape, no double coverage. Bump on ANY procedure change — a
# capture from an older procedure must not admit a scorecard run. The
# scorecard pins the expected value
# (test_openai_scorecard._EXPECTED_PROBE_CONTRACT_VERSION); the pair failing
# loud on drift is the sync mechanism.
# v2: the matrix gained the two node-admission rows (gpt-5.6-luna:trace /
# :patch — real trace/patch prompts, parser-backed predicates) per the spec's
# "one paid row each folded into the probe run".
# v3: trace scenario made production-reachable (shared finding provenance —
# singleton buckets skip the ranking call); analyze-emission rows added
# riding the PRODUCTION analyze prompt (FUP-236 candidate-quality surface,
# graded on the expected finding's own candidate set); patch predicate
# gained the closed reviewed-fix set; capture roles became declarable
# (--full-models / --trace-model / --patch-model / --analyze-models) so a
# swap rerun never re-requires a failed incumbent row — per field, and the
# matrix itself can drop a wire-failed incumbent.
# v4: the trace_emission predicate changed substantially — the request is
# built via the production assembly (decide_degradation clipping withholds
# the import line) and graded through the production admission chain
# (parse_analyze_response + #024 sibling + ladder resolution). Evidence
# captured under the superseded v3 grader must not be accepted as current.
# v5: the trace_emission scenario made prompt-consistent — the v4 scenario's
# locally-proven concatenation contradicted the analyze prompt's own
# discipline (a finding standing on file evidence alone must carry NO
# candidates), grading compliant empty arrays as failures; the verdict now
# hangs on the hidden imported escape_owner. The refusal elicitation is
# frozen from refusal_discovery.py output (the v4 ransomware prompt
# conflated the API-level cyber screen with in-band structured refusal).
PROBE_CONTRACT_VERSION: Final[int] = 5
# v2: conservation_adjudication became a pointer to the operator-authored,
# sanitized billing_adjudication.json artifact (equation/evidence moved there).
# v3: the pointer block gained raw_export_file (the local raw export the gate
# hashes at admission) — every manifest-shape change rotates this token.
# v4: gained the capture-role declarations (full_matrix_models, node_model,
# analyze_models); the scorecard gate derives the exact expected row set from
# them — the Terra-swap workflow.
MANIFEST_SCHEMA_VERSION: Final[int] = 4

CHEAP_MODEL = "gpt-5.6-terra"
_KNOWN_MODELS: Final[tuple[str, ...]] = ("gpt-5.6-sol", "gpt-5.6-luna", CHEAP_MODEL)

# The spec's Terra-swap workflow: a miss swaps THAT FIELD to Terra, and Terra
# "first inherits the FULL applicable probe matrix". The role flags make that
# executable without code edits AND without deadlock: a swap rerun declares
# the NEW serving model per role, and the full matrix itself is declarable —
# so a wire-failed incumbent can be dropped entirely rather than re-required.
#   --full-models=m1,m2[,m3]   models wire-admitted this capture
#                              (base envelope/cold/warm/refusal rows)
#   --trace-model=<m>          the trace-ranking serving model  (default luna)
#   --patch-model=<m>          the patch-pass serving model     (default luna)
#   --analyze-models=m1[,m2]   the analyze candidates (emission rows;
#                              default sol,luna)
# Fail-closed flag parsing: unknown flags, repeated flags, malformed values
# (`--dry-run=true`, bare `--full-models`, empty `--trace-model=`), duplicate
# list values, and role models outside the declared full matrix all refuse
# before any call is planned — a mis-typed flag must never silently select a
# default and launch the PAID path.
_BARE_FLAGS: Final[frozenset[str]] = frozenset({"dry-run"})
_VALUE_FLAGS: Final[frozenset[str]] = frozenset(
    {"full-models", "trace-model", "patch-model", "analyze-models"}
)
_ALLOWED_FLAGS: Final[frozenset[str]] = _BARE_FLAGS | _VALUE_FLAGS


def _parse_flags(argv: list[str]) -> dict[str, str | None]:
    seen: dict[str, str | None] = {}
    for arg in argv:
        if not arg.startswith("--"):
            raise SystemExit(f"unexpected argument {arg!r} (flags only)")
        name, sep, value = arg[2:].partition("=")
        if name not in _ALLOWED_FLAGS:
            raise SystemExit(f"unknown flag --{name} (allowed: {sorted(_ALLOWED_FLAGS)})")
        if name in seen:
            raise SystemExit(f"flag --{name} given more than once — ambiguous roles refuse")
        if name in _BARE_FLAGS and sep:
            raise SystemExit(f"--{name} takes no value (got --{name}={value!r})")
        if name in _VALUE_FLAGS and (not sep or not value):
            raise SystemExit(f"--{name} requires a non-empty =value")
        seen[name] = value if sep else None
    return seen


def _model_list(raw: str, flag: str) -> tuple[str, ...]:
    models = tuple(raw.split(","))
    if len(set(models)) != len(models) or not models or any(m not in _KNOWN_MODELS for m in models):
        raise SystemExit(
            f"--{flag}={raw!r} must be distinct models from {_KNOWN_MODELS} — duplicates "
            "would double-spend and overwrite fixtures"
        )
    return models


_FLAGS = _parse_flags(sys.argv[1:])
FULL_MODELS: Final[tuple[str, ...]] = _model_list(
    _FLAGS.get("full-models") or "gpt-5.6-sol,gpt-5.6-luna", "full-models"
)
TRACE_MODEL: Final[str] = _FLAGS.get("trace-model") or "gpt-5.6-luna"
PATCH_MODEL: Final[str] = _FLAGS.get("patch-model") or "gpt-5.6-luna"
ANALYZE_MODELS: Final[tuple[str, ...]] = _model_list(
    _FLAGS.get("analyze-models") or "gpt-5.6-sol,gpt-5.6-luna", "analyze-models"
)
for _role_name, _role_model in (("trace-model", TRACE_MODEL), ("patch-model", PATCH_MODEL)):
    if _role_model not in FULL_MODELS or _role_model == "gpt-5.6-sol":
        raise SystemExit(
            f"--{_role_name}={_role_model} must be a non-Sol member of the declared "
            f"full matrix {FULL_MODELS}"
        )
if any(m not in FULL_MODELS for m in ANALYZE_MODELS):
    raise SystemExit(
        f"--analyze-models={ANALYZE_MODELS} must be members of the declared full "
        f"matrix {FULL_MODELS}"
    )
_PROMPT_VERSION = "openai-probe:1"

_SYS = (
    "You are a security code reviewer. Report every real vulnerability in the code "
    "under review as a finding. Respond with a JSON object matching this schema "
    "exactly (json_object mode — the schema is described, not enforced):\n"
    + json.dumps(ANALYZE_RESPONSE_SCHEMA)
)
_CODE = (
    "1: import sqlite3\n"
    "2:\n"
    "3: def get_user(db, user_id):\n"
    '4:     query = "SELECT * FROM users WHERE id = \'" + user_id + "\'"\n'
    "5:     cur = db.execute(query)\n"
    "6:     return cur.fetchone()\n"
)
_USER = "Code under review:\n\n" + _CODE
# Refusal elicitations, FROZEN from refusal_discovery.py output. #056 admission is
# PER MODEL: each full-matrix model's refusal row needs a prompt that produced a
# non-null message.refusal FOR THAT MODEL. A single prompt may cover every model
# (then all values are identical) OR the winners differ (a per-model map) —
# refusal_discovery.py reports which; both freeze into this same dict shape. The v4
# ransomware prompt is retired: it was intercepted by OpenAI's API-level cyber screen
# (Sol: HTTP 400 cyber_policy) or complied in-band (Luna: message.refusal null),
# never exercising the structured-refusal wire. PLACEHOLDER until the discovery
# winners are frozen and _REFUSAL_ELICITATIONS_FROZEN is flipped; the paid runner
# refuses while False (and while any serving model still holds the placeholder). The
# dry run is unaffected.
_REFUSAL_ELICITATIONS_FROZEN: Final[bool] = False
_REFUSAL_PLACEHOLDER = "PLACEHOLDER — freeze the refusal_discovery.py winner for this model here"
_REFUSAL_ELICITATIONS: Final[dict[str, str]] = {m: _REFUSAL_PLACEHOLDER for m in _KNOWN_MODELS}

_CACHE_FLOOR = MIN_CACHEABLE_TOKENS[("openai", "gpt-5.6-sol")] or 1024
_STABLE_PREFIX = _SYS + "\n\n" + ("Review guideline line. " * 400)  # ~2.6k tokens

# --- Node-admission scenarios: DUPLICATED in the eval instruments -----------
# (tests/eval/test_trace_admission.py / test_patch_admission.py — spikes/ is
# not importable from tests). Drift fails loud: the instruments grade these
# captures with the production parsers over THEIR copy of the scenario, so a
# divergent candidate id or target line cannot pass.
_TRACE_REAL_IMPORT = "app.db"
_TRACE_DISTRACTOR_IMPORT = "app.render_helpers"
# BOTH candidates share one finding's provenance: production buckets
# candidates per finding and SKIPS the ranking call when every bucket is a
# singleton (trace.py step 6) — cross-finding candidates would capture a
# request the real node never sends. The reachability proof lives in
# tests/eval/test_trace_admission.py's real-node test.
_TRACE_SOURCE_HASH = "3" * 64
_TRACE_REAL_REASON = (
    "handlers.py builds the flagged query via run_query imported from app.db; "
    "the finding's tainted value flows directly into it"
)
_TRACE_DISTRACTOR_REASON = (
    "render_error_page formats the error string shown when the request fails; "
    "cosmetic to the finding's data flow"
)


def _trace_candidates() -> tuple[TraceCandidate, TraceCandidate]:
    real = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=_TRACE_SOURCE_HASH,
            import_string=_TRACE_REAL_IMPORT,
            reason=_TRACE_REAL_REASON,
        ),
        source_proposal_hash=_TRACE_SOURCE_HASH,
        reason=_TRACE_REAL_REASON,
        import_string=_TRACE_REAL_IMPORT,
    )
    distractor = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=_TRACE_SOURCE_HASH,
            import_string=_TRACE_DISTRACTOR_IMPORT,
            reason=_TRACE_DISTRACTOR_REASON,
        ),
        source_proposal_hash=_TRACE_SOURCE_HASH,
        reason=_TRACE_DISTRACTOR_REASON,
        import_string=_TRACE_DISTRACTOR_IMPORT,
    )
    return (real, distractor)


# Analyze-emission scenario (FUP-236's candidate-quality surface). The row
# rides the PRODUCTION analyze prompt (real Claude-tuned stable prefix, real
# render, analyze-v10, node_id="analyze", constrained-decoding schema) with
# PRODUCTION-EQUIVALENT VISIBILITY: the rendered prompt carries scope bodies
# + scope-clipped hunks ONLY — module imports are NOT shown (cache/key.py:
# "scope bodies + hunks only"), so the import line is deliberately absent and
# the string "app.db" appears NOWHERE in the request (the dry run pins that —
# a prompt that hands the model the answer measures nothing). The verdict
# GENUINELY hangs on the imported `escape_owner`: `owner` passes through it
# before the concatenated query, so whether the injection is real depends on
# whether escape_owner escapes quotes — the analyze prompt's own candidate
# discipline REQUIRES a candidate here ("propose ONLY when the verdict
# depends on code outside this file"), unlike a locally proven concatenation,
# which it instructs to emit with an empty array (the pre-v5 scenario's
# defect: it graded that compliant empty array as a model failure).
# Production's #024 from-import correction maps a symbol-form candidate
# (trailing `escape_owner` / `run_query`) onto the file's real import. A
# candidate untied to any imported name (the FUP-236
# `app.user_store`-for-a-DI'd-parameter shape) is a guess production cannot
# correct.
_EMISSION_FILE = "app/handlers.py"
_EMISSION_EXPECTED_TYPE = "sql_injection"
_EMISSION_TAINT_LINE = 5
# The scenario file's REAL from-import map (its only import, line 1, lives
# outside every scope unit and thus outside the prompt). Duplicated in
# test_trace_admission.py.
_EMISSION_FROM_IMPORTS: Final[dict[str, str]] = {
    "run_query": "app.db",
    "escape_owner": "app.db",
}
# The scenario PR adds the whole file, so the raw patch INCLUDES the import
# line — and the PRODUCTION clipping (decide_degradation ->
# scope_unit_diff_hunks) withholds it, because line 1 lies outside every
# scope unit. The withholding is production behavior, not hand-editing; the
# dry run pins its effect on the built request.
_EMISSION_PATCH = (
    "@@ -0,0 +1,5 @@\n"
    "+from app.db import run_query, escape_owner\n"
    "+\n"
    "+def get_user_orders(request):\n"
    '+    owner = escape_owner(request.GET["owner"])\n'
    '+    return run_query("SELECT * FROM orders WHERE owner = \'" + owner + "\'")\n'
)


# The scenario's REAL file content (production holds the whole file even
# though the prompt renders scope bodies + clipped hunks only — the import at
# line 1 is file-only, never prompt-visible) and the scenario repository the
# deterministic ladder resolves against. app/db.py carries the UNSAFE
# escape_owner (strips whitespace, does NOT escape quotes) — what a real
# trace fetch would reveal. Duplicated in tests/eval/test_trace_admission.py.
_EMISSION_FILE_CONTENT = (
    "from app.db import run_query, escape_owner\n"
    "\n"
    "def get_user_orders(request):\n"
    '    owner = escape_owner(request.GET["owner"])\n'
    '    return run_query("SELECT * FROM orders WHERE owner = \'" + owner + "\'")\n'
)
_SCENARIO_REPO: Final[dict[str, bytes]] = {
    "app/db.py": (
        b"def escape_owner(owner):\n    return owner.strip()\n\n\ndef run_query(sql):\n    ...\n"
    )
}


class _NoOpResolver:
    """ImportPathResolver double for the offline ast_facts parse."""

    def resolve_candidate_paths(self, import_string: str, import_root: Path) -> list[Path]:  # noqa: ARG002
        return []

    def resolve_specifier_candidate_paths(
        self,
        specifier: str,
        importing_file_path: str,
        import_root: Path,  # noqa: ARG002
    ) -> list[Path]:
        return []


def _ladder_resolves(import_string: str) -> bool:
    """Deterministic resolution through the REAL probe ladder against the
    scenario repository (production `_probe_paths_for` + `_symbol_in_content`)."""
    parts = import_string.split(".")
    for level in range(len(parts)):
        for path in _probe_paths_for(import_string, level, _EMISSION_FILE):
            content = _SCENARIO_REPO.get(path)
            if content is None:
                continue
            if level == 0 or _symbol_in_content(parts[-level], content):
                return True
    return False


def _grade_emission_via_production_chain(text: str) -> tuple[bool, str]:
    """The frozen predicate graded through the PRODUCTION chain: real
    ast_facts parse of the scenario file, real `parse_analyze_response`
    admission (proof boundary + #024 corrected sibling), join to an ADMITTED
    expected finding, deterministic ladder resolution. A candidate that
    neither resolves nor is a visible bare from-import name is fabricated
    from hidden information — the FUP-236 failure. NOTE: V1 visibility never
    renders module imports; if every model reads red here, that is the frozen
    predicate's true verdict under V1 visibility — adjudicate, don't soften."""
    parsed_file = parse_python(
        _EMISSION_FILE_CONTENT.encode("utf-8"), _EMISSION_FILE, _NoOpResolver()
    )
    units = tuple(u for u in parsed_file.scope_units if u.name == "get_user_orders")
    result = parse_analyze_response(
        text,
        review_id=uuid4(),
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
    if result.response_rejection is not None:
        return False, f"emission response REJECTED: {result.response_rejection.rejection_detail}"
    fabricated = sorted(
        {
            c.import_string
            for c in result.trace_candidates
            if not _ladder_resolves(c.import_string)
            and c.import_string not in _EMISSION_FROM_IMPORTS
        }
    )
    if fabricated:
        return False, f"guessed import path(s) {fabricated} — the FUP-236 failure"
    admitted = [
        f
        for f in result.admitted_findings
        if f.finding_type is FindingType.SQL_INJECTION
        and f.line_start <= _EMISSION_TAINT_LINE <= f.line_end
    ]
    if not admitted:
        return False, (
            f"no ADMITTED {_EMISSION_EXPECTED_TYPE} finding covering line "
            f"{_EMISSION_TAINT_LINE} (admission may have rejected the proposal)"
        )
    admitted_hashes = {f.proposal_hash for f in admitted}
    joined = [c for c in result.trace_candidates if c.source_proposal_hash in admitted_hashes]
    if not joined:
        return False, "expected finding carries no admitted trace candidate"
    if not any(_ladder_resolves(c.import_string) for c in joined):
        return False, (
            f"no joined candidate resolves to app/db.py: "
            f"{sorted({c.import_string for c in joined})}"
        )
    return True, (
        f"admitted + resolving candidate set: {sorted({c.import_string for c in joined})}"
    )


def _emission_kwargs(model: str) -> dict[str, Any]:
    """The emission request via the PRODUCTION assembly, end to end: real
    ast_facts parse, real patch parse (`lookup_patched_file`), the real
    inclusion/clipping decision (`decide_degradation`), and the real prompt
    blocks (`_assemble_scope_unit_context` + `_concat_clipped_hunks`) — the
    same chain analyze runs for a clean full-LLM file."""
    parse_result = parse_python(
        _EMISSION_FILE_CONTENT.encode("utf-8"), _EMISSION_FILE, _NoOpResolver()
    )
    patched_file = lookup_patched_file(_EMISSION_PATCH, _EMISSION_FILE)
    decision = decide_degradation(parse_result, patched_file)
    if not decision.included_scope_units or not decision.included_clipped_hunks:
        raise SystemExit(
            "emission scenario no longer yields a clean full-LLM decision — "
            f"outcome fields: {decision!r}"
        )
    parts = analyze_prompt.render(
        file_path=_EMISSION_FILE,
        scope_unit_context=_assemble_scope_unit_context(
            included_scope_units=decision.included_scope_units,
            source_bytes=_EMISSION_FILE_CONTENT.encode("utf-8"),
            fence_lang="python",
        ),
        query_match_id_list=_assemble_query_match_id_list(frozenset()),
        diff_hunks=_concat_clipped_hunks(decision.included_clipped_hunks),
        pass_index=0,
    )
    # The PRODUCTION context manifest over the decision's included units —
    # the same helper the analyze emits use, pinned to the scenario's true
    # 3-5 range (literals, not derived, so a scenario-shape drift refuses).
    context_summary = _build_context_manifest(
        _EMISSION_FILE, decision.included_scope_units, inclusion_reason="changed_scope"
    )
    if [(e.line_start, e.line_end) for e in context_summary] != [(3, 5)]:
        raise SystemExit(
            f"emission scenario scope range drifted: "
            f"{[(e.line_start, e.line_end) for e in context_summary]} != [(3, 5)]"
        )
    request = LLMRequest(
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        model=model,
        max_tokens=analyze_prompt.MAX_TOKENS,
        temperature=analyze_prompt.TEMPERATURE,
        review_id=uuid4(),
        node_id="analyze",
        context_summary=context_summary,
        prompt_template_version=analyze_prompt.VERSION,
        degraded_mode=False,
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
    )
    return _build_sdk_kwargs(request, profile=OPENAI_PROFILE)


_PATCH_FINDING_ID = UUID("00000000-0000-0000-0000-0000000000f1")
_PATCH_FILE = "app/config_loader.py"
_PATCH_LINE_NO = 6
_PATCH_TARGET_LINE = "    return yaml.load(stream)"
# The scenario-specific SEMANTIC predicate: apply_patch_batch proves the
# patch is safe and anchored, not that it FIXES anything — a wrong-but-safe
# line (yaml.dump) or a composite that ALSO invokes an unsafe loader
# (safe_load(...) and unsafe_load(...)) would survive it, and substring
# regexes proved gameable. The fixture has ONE unambiguous fix, so
# remediation is a CLOSED set of reviewed replacements — extended
# consciously, never pattern-matched.
_PATCH_REVIEWED_FIXES = frozenset({"    return yaml.safe_load(stream)"})


def _patch_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id=_PATCH_FINDING_ID,
        review_id=UUID("00000000-0000-0000-0000-0000000000e1"),
        installation_id=42,
        finding_type=FindingType.UNSAFE_DESERIALIZATION,
        severity=lookup_severity(FindingType.UNSAFE_DESERIALIZATION),
        file_path=_PATCH_FILE,
        line_start=_PATCH_LINE_NO,
        line_end=_PATCH_LINE_NO,
        title="yaml.load without a safe loader deserializes arbitrary objects",
        description=(
            "load_config parses an operator-provided stream with yaml.load and no "
            "Loader argument; a crafted document instantiates arbitrary Python objects."
        ),
        evidence=_PATCH_TARGET_LINE,
        dimension=lookup_dimension(FindingType.UNSAFE_DESERIALIZATION),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=_PATCH_FILE,
            line_start=_PATCH_LINE_NO,
            line_end=_PATCH_LINE_NO,
            finding_type=FindingType.UNSAFE_DESERIALIZATION,
        ),
        proposal_hash="a" * 64,
    )


def _request(model: str, *, system: str, user: str) -> LLMRequest:
    return LLMRequest(
        system_prompt=system,
        user_prompt=user,
        model=model,
        max_tokens=600,
        temperature=0.0,
        review_id=uuid4(),
        node_id="triage",  # context-free node; json_object mode ignores the schema name
        prompt_template_version=_PROMPT_VERSION,
        degraded_mode=False,
        # The pinned compact bytes — never re-serialized here, so the digest the
        # provider derives matches production analyze calls exactly.
        response_schema_json=ANALYZE_RESPONSE_SCHEMA_JSON,
    )


def _kwargs(model: str, *, system: str, user: str) -> dict[str, Any]:
    """The EXACT production wire: `_build_sdk_kwargs` on a real LLMRequest —
    json_object, verbatim requested tier, the production-derived
    prompt_cache_key, reasoning off. Never hand-assembled."""
    return _build_sdk_kwargs(_request(model, system=system, user=user), profile=OPENAI_PROFILE)


def _trace_kwargs(model: str) -> dict[str, Any]:
    """The trace node's EXACT production request shape over the admission
    scenario: node_id='trace', trace-v1 template, no response schema (the
    parser is the contract, not the wire)."""
    parts = trace_prompt.render(_trace_candidates())
    request = LLMRequest(
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        model=model,
        max_tokens=trace_prompt.MAX_TOKENS,
        temperature=trace_prompt.TEMPERATURE,
        review_id=uuid4(),
        node_id="trace",
        prompt_template_version=trace_prompt.VERSION,
        degraded_mode=False,
    )
    return _build_sdk_kwargs(request, profile=OPENAI_PROFILE)


def _patch_kwargs(model: str) -> dict[str, Any]:
    """The patch pass's EXACT production request shape over the admission
    scenario: node_id='synthesize' (patch cost rolls into synthesize),
    patch-v1 template, no response schema."""
    finding = _patch_finding()
    parts = patch_prompt.render((finding,), {finding.finding_id: _PATCH_TARGET_LINE})
    request = LLMRequest(
        system_prompt=parts.system_prompt,
        user_prompt=parts.user_prompt,
        model=model,
        max_tokens=patch_prompt.MAX_TOKENS,
        temperature=patch_prompt.TEMPERATURE,
        review_id=uuid4(),
        node_id="synthesize",
        prompt_template_version=patch_prompt.VERSION,
        degraded_mode=False,
    )
    return _build_sdk_kwargs(request, profile=OPENAI_PROFILE)


def _plan() -> list[tuple[str, bool, dict[str, Any]]]:
    """(tag, required, kwargs) rows. ALL rows required — the refusal fixture is
    a pre-ship gate per model (spec: wire admission is PER MODEL), so nothing
    here is best-effort. The scorecard gate hardcodes this exact tag set
    (`_EXPECTED_PROBE_ROWS`); a matrix change without a gate update fails loud
    there, which is the intended sync mechanism."""
    rows: list[tuple[str, bool, dict[str, Any]]] = []
    for model in FULL_MODELS:
        rows.append((f"{model}:envelope", True, _kwargs(model, system=_SYS, user=_USER)))
        rows.append((f"{model}:cold", True, _kwargs(model, system=_STABLE_PREFIX, user=_USER)))
        rows.append((f"{model}:warm", True, _kwargs(model, system=_STABLE_PREFIX, user=_USER)))
        rows.append(
            (
                f"{model}:refusal",
                True,
                _kwargs(model, system=_SYS, user=_REFUSAL_ELICITATIONS[model]),
            )
        )
    # Node-admission rows (spec: "one paid row each folded into the probe
    # run") for the DECLARED per-field serving models (trace and patch swap
    # independently), and analyze-emission rows (FUP-236 surface, production
    # analyze prompt) for the DECLARED analyze candidates; the eval
    # instruments grade these captures offline. A swap rerun declares the
    # replacement per role, so a failed incumbent row is not re-required.
    rows.append((f"{TRACE_MODEL}:trace", True, _trace_kwargs(TRACE_MODEL)))
    rows.append((f"{PATCH_MODEL}:patch", True, _patch_kwargs(PATCH_MODEL)))
    for model in ANALYZE_MODELS:
        rows.append((f"{model}:trace_emission", True, _emission_kwargs(model)))
    rows.append((f"{CHEAP_MODEL}:reasoning", True, _kwargs(CHEAP_MODEL, system=_SYS, user=_USER)))
    return rows


def _strip_fence(content: str) -> str:
    body = content.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0].strip()
    return body


def _usage_triple(usage: RawUsage) -> dict[str, int | None]:
    """Record the raw conservation inputs; PINNING the equation happens against
    the saved fixture, not via an in-probe formula (an earlier draft's
    classifier contained an algebraic identity — recorded facts only here).
    total_tokens is the count-level disambiguator between the spec's two candidate
    equations: total == prompt + completion means writes are INSIDE prompt_tokens;
    total == prompt + writes + completion means writes are an additive class."""
    return {
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": usage.cached_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _conservation_facts(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-model count-level facts for the conservation adjudication — recorded
    and characterized, never self-certified: the count analysis suggests an
    equation, but the OPERATOR confirms against billed usage (the OpenAI usage
    dashboard / costs API) before filling `conservation_adjudication`."""
    facts: dict[str, Any] = {}
    for model in FULL_MODELS:
        cold = (results.get(f"{model}:cold") or {}).get("usage") or {}
        warm = (results.get(f"{model}:warm") or {}).get("usage") or {}
        prompt, write = cold.get("prompt_tokens"), cold.get("cache_write_tokens")
        completion, total = cold.get("completion_tokens"), cold.get("total_tokens")
        # Mirrors the gate's typed characterization (support + indeterminate
        # cause) so the operator can copy the printed cause codes into the
        # artifact's per-model count_reconciliation when needed.
        supported, cause = "indeterminate", None
        if not all(isinstance(v, int) for v in (prompt, write, completion, total)) or not write:
            cause = "wire_omitted_total_tokens"
        elif cold.get("prompt_tokens") != warm.get("prompt_tokens"):
            cause = "cold_warm_pair_incoherent"
        elif total == prompt + completion:
            # writes ride INSIDE prompt_tokens -> one billing class per
            # token -> normalized input must subtract them.
            supported = "prompt_minus_cached_minus_writes"
        elif total == prompt + write + completion:
            supported = "prompt_minus_cached"
        else:
            cause = "total_matches_neither_equation"
        facts[model] = {
            "cold": cold,
            "warm": warm,
            "prompt_equal_across_pair": cold.get("prompt_tokens") == warm.get("prompt_tokens"),
            "supported_by_counts": supported,
            "indeterminate_cause": cause,
        }
    return facts


def _evaluate(tag: str, capture: RawCapture) -> tuple[bool, str]:
    """Per-row REQUIRED success predicate. Returns (ok, note)."""
    kind = tag.rsplit(":", 1)[1]
    content = capture.content or ""
    tier = capture.service_tier
    triple = _usage_triple(capture.usage)
    if kind == "envelope":
        try:
            AnalyzeResponseRaw.model_validate_json(_strip_fence(content))
        except Exception as exc:  # noqa: BLE001 — spike: any failure shape fails the row
            return False, f"output does not conform: {type(exc).__name__}"
        if tier != "default":
            return False, f"tier echo {tier!r} != 'default'"
        return True, "conforms + default tier echoed"
    if kind == "cold":
        write = triple["cache_write_tokens"]
        if not write:
            return False, f"no cache write reported on cold call: {triple}"
        return True, f"write fired: {triple}"
    if kind == "warm":
        cached = triple["cached_tokens"]
        if not cached:
            return False, f"no cache read on warm repeat: {triple}"
        return True, f"cache hit: {triple}"
    if kind == "reasoning":
        if tier != "default":
            return False, f"tier echo {tier!r} != 'default'"
        return True, "reasoning_effort=none accepted (no 400) + default tier"
    if kind == "refusal":
        refusal = capture.refusal
        if not refusal:
            return False, (
                "message.refusal NOT populated (comply-with-caveats?) — the refusal "
                "fixture is a pre-ship gate; adjust the elicitation prompt and rerun"
            )
        return True, f"refusal populated: {refusal[:80]!r}"
    if kind == "trace":
        outcome = parse_trace_ranking(response_text=content, candidates=_trace_candidates())
        if not isinstance(outcome, TraceRankingParsed):
            return False, f"trace ranking REJECTED by the production parser: {outcome.reason}"
        first = outcome.ordered_candidates[0]
        if first.import_string != _TRACE_REAL_IMPORT:
            return False, (f"trace ranked {first.import_string!r} above the load-bearing candidate")
        return True, f"valid permutation, {_TRACE_REAL_IMPORT!r} ranked first"
    if kind == "patch":
        finding = _patch_finding()
        patches = apply_patch_batch(
            content,
            (finding,),
            {finding.finding_id: _PATCH_TARGET_LINE},
        )
        replacement = patches.get(finding.finding_id)
        if not replacement:
            return False, (
                "no valid anchored single-line patch survived the production rules "
                "(rejected batch, echo drift, out-of-scope edit, or null fix)"
            )
        if replacement not in _PATCH_REVIEWED_FIXES:
            return False, (
                f"structurally safe but NOT a reviewed remediation: {replacement!r} "
                f"(closed fix set: {sorted(_PATCH_REVIEWED_FIXES)})"
            )
        return True, f"anchored remediating patch: {replacement!r}"
    if kind == "trace_emission":
        return _grade_emission_via_production_chain(content)
    return False, f"unknown row kind {kind!r}"


async def _run_paid() -> int:
    if not _REFUSAL_ELICITATIONS_FROZEN:
        print(
            "refusal elicitations not frozen — run spikes/openai/refusal_discovery.py, "
            "fill _REFUSAL_ELICITATIONS with the demonstrated per-model winner(s), flip "
            "_REFUSAL_ELICITATIONS_FROZEN, then rerun. Refusing the paid matrix: the "
            "placeholders are not real elicitations and would waste the whole "
            "all-rows-required run."
        )
        return 2
    unfilled = [
        m
        for m in FULL_MODELS
        if _REFUSAL_ELICITATIONS.get(m, _REFUSAL_PLACEHOLDER) == _REFUSAL_PLACEHOLDER
        or not _REFUSAL_ELICITATIONS.get(m, _REFUSAL_PLACEHOLDER).strip()
    ]
    if unfilled:
        print(
            f"_REFUSAL_ELICITATIONS_FROZEN=True but {unfilled} hold a placeholder or empty "
            "elicitation — freeze the refusal_discovery.py prompt TEXT (not a label) for "
            "every serving model before spending."
        )
        return 2
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("op://"):
        print("OPENAI_API_KEY missing/unresolved — run under `op run --env-file=.env -- ...`")
        return 2
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    # base_url is the PROFILE's, exactly — never an env override (key safety +
    # evidence provenance).
    client = RawOpenAICaptureClient(api_key=api_key, base_url=OPENAI_PROFILE.base_url)
    results: dict[str, dict[str, Any]] = {}
    try:
        for tag, required, kwargs in _plan():
            if "store" in kwargs:  # the probe must never opt into response storage
                raise RuntimeError(f"unexpected 'store' kwarg on {tag}; refusing to send")
            if tag.endswith(":warm"):
                await asyncio.sleep(5)  # cache-entry propagation before the warm repeat
            fixture_name = tag.replace(":", "_")
            try:
                capture = await client.capture(**kwargs)
            except RawOpenAICaptureError as err:
                note = f"HTTP {err.status}: {err.message[:160]}"
                results[tag] = {"ok": False, "required": required, "note": note}
                (_FIXTURE_DIR / f"{fixture_name}.error.json").write_text(
                    json.dumps({"status": err.status, "message": err.message}, indent=2),
                    encoding="utf-8",
                )
                print(f"  {tag}: FAIL — {note}")
                continue
            except RawCaptureShapeError as err:
                # A novel/malformed wire shape: preserve the raw payload as evidence (so
                # the failure is inspectable, not just "safe"), record a FAILED row.
                note = f"malformed response shape: {err.reason[:160]}"
                results[tag] = {"ok": False, "required": required, "note": note}
                if err.raw_json is not None:
                    (_FIXTURE_DIR / f"{fixture_name}.malformed.json").write_text(
                        err.raw_json, encoding="utf-8"
                    )
                print(f"  {tag}: FAIL — {note}")
                continue
            payload = capture.raw_json
            (_FIXTURE_DIR / f"{fixture_name}.json").write_text(payload, encoding="utf-8")
            ok, note = _evaluate(tag, capture)
            results[tag] = {
                "ok": ok,
                "required": required,
                "note": note,
                "usage": _usage_triple(capture.usage),
                "service_tier": capture.service_tier,
                # Hash pins the fixture the verdict was computed FROM; the
                # scorecard gate re-verifies fixture bytes against this.
                "fixture": f"{fixture_name}.json",
                "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            }
            print(f"  {tag}: {'PASS' if ok else 'FAIL'} — {note}")
    finally:
        await client.close()
        required_failed = [tag for tag, r in results.items() if r["required"] and not r["ok"]]
        planned = {tag for tag, _req, _kw in _plan()}
        missing = sorted(planned - set(results))
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "probe_contract_version": PROBE_CONTRACT_VERSION,
            "generated_by": "spikes/openai/probe.py",
            "base_url": OPENAI_PROFILE.base_url,
            "profile_contract_digest": OPENAI_PROFILE.profile_contract_digest,
            "full_matrix_models": list(FULL_MODELS),
            "trace_model": TRACE_MODEL,
            "patch_model": PATCH_MODEL,
            "analyze_models": list(ANALYZE_MODELS),
            "results": results,
            "missing_rows": missing,
            "all_required_passed": not required_failed and not missing,
            "conservation_facts": _conservation_facts(results),
            # The equation choice is a BILLING fact the counts alone cannot
            # certify (spec: [probe] classification). The OPERATOR adjudicates
            # against billed usage and authors the SANITIZED
            # billing_adjudication.json artifact (see the success instructions
            # below — raw exports stay local/gitignored); this block only
            # POINTS at that artifact and at the LOCAL raw export
            # (raw_export_file, confined beneath fixtures/raw/ — the gate
            # hashes those actual bytes against the artifact's
            # raw_export_sha256, so the independent billing source must EXIST
            # at admission, not merely be claimed). The gate also hash-checks
            # the artifact, enforces its closed key set, and validates its
            # capture binding (response IDs, billed class counts, window)
            # against the fixture bytes.
            "conservation_adjudication": {
                "adjudication_file": None,
                "adjudication_sha256": None,
                "raw_export_file": None,
            },
        }
        _MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\nmanifest: {_MANIFEST} (all_required_passed={manifest['all_required_passed']})")
    if required_failed or missing:
        print(f"REQUIRED FAILURES: {required_failed or missing} — do NOT run the scorecard.")
        return 1
    print(
        "all required rows passed. BEFORE the scorecard, adjudicate the conservation\n"
        "equation:\n"
        f"  1. Save the RAW billing export (usage/costs API response or CSV) under\n"
        f"     {_FIXTURE_DIR}/raw/ — gitignored; it carries project/key identifiers\n"
        "     and financial usage and must NEVER be committed.\n"
        "  2. Author the SANITIZED billing_adjudication.json next to the manifest —\n"
        "     the gate enforces this key set EXACTLY (extra keys refuse admission —\n"
        "     that's how the sanitization claim stays true): adjudication_schema_\n"
        "     version=2, equation ('prompt_minus_cached' or\n"
        "     'prompt_minus_cached_minus_writes'), adjudicated_by (short\n"
        "     single-line string), count_reconciliation (null when every model's\n"
        "     counts are determinate — REQUIRED null; otherwise an object mapping\n"
        "     EXACTLY the indeterminate models to their cause codes as printed in\n"
        "     conservation_facts: 'wire_omitted_total_tokens' /\n"
        "     'cold_warm_pair_incoherent' / 'total_matches_neither_equation' —\n"
        "     the gate re-derives the causes from fixture bytes and requires the\n"
        "     mapping to match), raw_export_sha256 (hash of the local raw export),\n"
        "     window_utc {start_epoch, end_epoch} covering this capture (<=24h), and\n"
        "     models with EXACTLY one entry per declared full_matrix_models model,\n"
        "     each\n"
        "     {cold_response_id, warm_response_id, billed_fresh_input_tokens,\n"
        "     billed_cache_write_tokens} — response IDs from the fixtures, billed\n"
        "     counts from the export. No project/org/key identifiers, no dollar\n"
        "     amounts.\n"
        "  3. Fill manifest.conservation_adjudication with adjudication_file,\n"
        "     adjudication_sha256, AND raw_export_file ('raw/<filename>' of the\n"
        "     step-1 export) — the gate hashes the raw export's ACTUAL bytes\n"
        "     against the artifact's raw_export_sha256, so the export must exist\n"
        "     locally at admission.\n"
        "The gate re-verifies everything against fixture bytes (response IDs, window\n"
        "vs created, billed classes vs wire counts under the adjudicated equation) and\n"
        "recomputes count support: contrary counts refuse admission outright. If the\n"
        "verdict is minus_writes, read_usage() must change BEFORE any scorecard run."
    )
    return 0


def _run_dry() -> int:
    """Free prefix: builds every PRODUCTION request body; no network, no key use."""
    print(f"base_url={OPENAI_PROFILE.base_url} (profile-pinned)  cache_floor={_CACHE_FLOOR}")
    digest16 = OPENAI_PROFILE.profile_contract_digest[:16]
    # trace/patch rows ride the real node requests: no response schema on the
    # wire (the parser is the contract) and their own template versions in the
    # production-derived cache key. Emission rows ride the production analyze
    # prompt (analyze VERSION, schema on the wire).
    version_by_kind = {
        "trace": trace_prompt.VERSION,
        "patch": patch_prompt.VERSION,
        "trace_emission": analyze_prompt.VERSION,
    }
    problems = 0
    for tag, required, kwargs in _plan():
        kind = tag.rsplit(":", 1)[1]
        schema_row = kind not in ("trace", "patch")
        expected_cache_key = f"outrider:{digest16}:{version_by_kind.get(kind, _PROMPT_VERSION)}"
        body_bytes = sum(len(str(m["content"]).encode("utf-8")) for m in kwargs["messages"])
        checks = {
            "response_format": (
                kwargs.get("response_format") == {"type": "json_object"}
                if schema_row
                else "response_format" not in kwargs
            ),
            "tier_default": kwargs.get("service_tier") == "default",
            "reasoning_none": kwargs.get("reasoning_effort") == "none",
            "prod_cache_key": kwargs.get("prompt_cache_key") == expected_cache_key,
            "no_store": "store" not in kwargs,
            "under_ceiling": body_bytes + 1024 <= 272_000,
        }
        if kind == "trace_emission":
            # Production-equivalence pin: the prompt must NOT hand the model
            # the answer — the import module string appears nowhere in the
            # request (production shows scope bodies + clipped hunks only;
            # the template prose may say "import", but the MODULE may not
            # appear).
            user_content = str(kwargs["messages"][1]["content"])
            checks["answer_withheld"] = all(
                module not in user_content for module in _EMISSION_FROM_IMPORTS.values()
            )
        bad = [name for name, ok in checks.items() if not ok]
        problems += len(bad)
        req = "required" if required else "recorded"
        print(f"  {tag} [{req}]: bytes={body_bytes} " + ("OK" if not bad else f"FAIL={bad}"))
    prefix_bytes = len(_STABLE_PREFIX.encode("utf-8"))
    floor_ok = prefix_bytes // 4 >= _CACHE_FLOOR
    print(
        f"  cold/warm prefix: {prefix_bytes} bytes (~{prefix_bytes // 4} tokens vs floor "
        f"{_CACHE_FLOOR}: {'OK' if floor_ok else 'BELOW FLOOR — enlarge'})"
    )
    if not floor_ok:
        problems += 1
    print(f"dry-run {'clean' if not problems else f'FOUND {problems} problems'}; no calls made.")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    # Dispatch on the PARSED flag, never a raw argv string: `--dry-run=true`
    # is rejected at parse time, so a malformed dry-run request can never
    # fall through to the paid path.
    if "dry-run" in _FLAGS:
        raise SystemExit(_run_dry())
    raise SystemExit(asyncio.run(_run_paid()))
