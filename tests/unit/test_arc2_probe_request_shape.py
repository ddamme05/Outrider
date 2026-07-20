"""Arc 2 — the probe sends PRODUCTION's request, with exactly one deliberate delta.

Without this file the probe could prove something true about a *neighbouring*
request rather than the production candidate: a different token-limit kwarg, a
dropped `prompt_cache_key`, a different service tier, and the paid row would
answer a question nobody asked.

Also pins Gate 1: paid mode is EXECUTABLE-refused, not documented-refused.

See `specs/2026-07-20-arc2-strict-schema-feasibility.md`.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from spikes.openai import strict_schema_probe as probe
from spikes.openai.arc2 import plan
from spikes.openai.arc2.classifier import ROW_ORDER, RowId
from spikes.openai.arc2.contracts import (
    STRICT_PROBE_CONTRACT_VERSION,
    DryRunManifest,
    PaidProbeContract,
    RowMeasurement,
)

from outrider.schemas.llm.analyze import ANALYZE_RESPONSE_SCHEMA


def _pair() -> dict[str, Any]:
    """Build a request pair through the SAME production assembly the probe uses,
    so this test cannot drift from what is actually sent."""
    s = probe.CLEAN_SCENARIO
    system, user = plan.analyze_prompt_pair(s, probe.derive_fired_query_match_ids(s))
    context = (
        plan._context_entry(
            file_path=s.file_path,
            scope_unit_name=s.scope_name,
            lines=(s.scope_line_start, s.scope_line_end),
        ),
    )
    return {"system": system, "user": user, "context": context}


def test_prompt_bytes_come_from_production_assembly() -> None:
    """The probe's prompt must BE the analyze node's prompt.

    An earlier version passed bare source as `scope_unit_context` and the literal
    `"(none)"` as the query list. The kwargs-equality test below could not see it,
    because both sides of that comparison were built from the same hand-assembled
    inputs — it proved provider-kwargs equality, not production-node equivalence.
    """
    from outrider.agent.nodes.analyze import (
        _assemble_query_match_id_list,
        _assemble_scope_unit_context,
        _concat_clipped_hunks,
    )
    from outrider.prompts import analyze as analyze_prompt

    s = probe.DEFECT_SCENARIO
    fired = probe.derive_fired_query_match_ids(s)
    expected = analyze_prompt.render(
        file_path=s.file_path,
        scope_unit_context=_assemble_scope_unit_context(
            included_scope_units=(s.scope_unit(),),
            source_bytes=s.source.encode("utf-8"),
        ),
        query_match_id_list=_assemble_query_match_id_list(fired),
        diff_hunks=_concat_clipped_hunks(((s.diff_hunk,),)),
        pass_index=0,
    )
    system, user = plan.analyze_prompt_pair(s, fired)
    assert system == expected.system_prompt
    assert user == expected.user_prompt
    # And the literal placeholder that used to stand in for the fired-id block is gone.
    assert "(none)" not in user


def test_fired_query_ids_are_derived_not_asserted() -> None:
    """The probe's fired set comes from production's builder + scope filter.

    It was a hand-written literal `py.sql.string_concat` — not a registered id at
    all (the registry's is `python.sql_injection_string_concat`), so the probe's
    own "authentic OBSERVED proof" was fabricated: an
    `evidence-tier-schema-enforced` violation inside the instrument built to
    detect exactly that.
    """
    from outrider.queries import registry

    fired = probe.derive_fired_query_match_ids(probe.DEFECT_SCENARIO)
    assert fired == probe.FIRED_QUERY_MATCH_IDS
    assert fired, "the defect scenario must fire at least one registered query"
    known = registry.structural_query_ids_for("python")
    assert fired <= known, f"derived ids not in the registry: {sorted(fired - known)}"


def test_an_authentic_observed_claim_admits_and_a_fabricated_one_rejects() -> None:
    """The distinction the whole arc rests on, over the REAL derived set."""
    authentic_id = sorted(probe.FIRED_QUERY_MATCH_IDS)[0]
    ok = probe.run_real_parser(
        probe._finding_body(evidence_tier="observed", query_match_id=authentic_id)
    )
    assert ok.rejection_reasons == ()
    assert [f.query_match_id for f in ok.admitted] == [authentic_id]

    bad = probe.run_real_parser(
        probe._finding_body(evidence_tier="observed", query_match_id="python.not_registered")
    )
    assert "query_match_id_not_in_registry" in bad.rejection_reasons
    assert bad.admitted == ()


def test_strict_kwargs_equal_production_except_response_format() -> None:
    """The ONLY difference between the probe wire and the production wire is
    `response_format`. Everything else is `_build_sdk_kwargs` output verbatim."""
    kwargs = _pair()
    production = plan.production_kwargs(**kwargs)
    strict = plan.strict_kwargs(**kwargs)

    assert set(strict) == set(production)
    differing = {k for k in production if strict[k] != production[k]}
    assert differing == {"response_format"}


def test_production_baseline_is_json_object_and_probe_is_json_schema() -> None:
    """Names the delta explicitly, so a future production switch to strict mode
    makes this test fail rather than silently making the probe a no-op."""
    kwargs = _pair()
    assert plan.production_kwargs(**kwargs)["response_format"] == {"type": "json_object"}

    strict_format = plan.strict_kwargs(**kwargs)["response_format"]
    assert strict_format["type"] == "json_schema"
    assert strict_format["json_schema"]["strict"] is True
    assert strict_format["json_schema"]["schema"] == probe.derive_strict_analyze_schema()


def test_probe_model_is_derived_from_host_config_not_hardcoded() -> None:
    """`model-strings-from-config-not-hardcoded`. A literal slug would keep probing
    the old model after the host's analyze default moved, so the capture would
    describe a model the production path no longer uses."""
    from outrider.llm.host_profiles import HOST_DEFAULT_MODELS

    assert HOST_DEFAULT_MODELS["openai"]["analyze_model"] == plan.MODEL
    # And it is the DEEP-analyze role specifically, not some other node's model.
    assert HOST_DEFAULT_MODELS["openai"]["triage_model"] != plan.MODEL


def test_probe_model_is_validated_against_the_host_profile() -> None:
    """A config value outside the host's slug family must fail OFFLINE, not as a
    400 on the schema-admission row — which the positional classifier would then
    be obliged to read as `STOP-shape`, blaming the schema for a bad slug."""
    from outrider.llm.host_profiles import OPENAI_PROFILE

    OPENAI_PROFILE.validate_model_slug(plan.MODEL)  # must not raise
    with pytest.raises(ValueError, match="slug pattern"):
        OPENAI_PROFILE.validate_model_slug("claude-sonnet-4-6")


def test_probe_model_is_not_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model is part of evidence identity; an ambient variable must not be
    able to silently change what the paid session bought."""
    for name in ("OUTRIDER_LLM_MODEL", "OUTRIDER_ANALYZE_MODEL", "ARC2_MODEL", "MODEL"):
        monkeypatch.setenv(name, "gpt-5.6-luna")
    assert plan._resolve_probe_model() == plan.MODEL


def test_derived_model_is_bound_into_generation_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derived model is part of the CONTRACTS, not just of the outgoing call.

    The previous version ended its first assertion with `or True`, so it could not
    fail no matter what the contract held, and the surviving assertion checked only
    request kwargs — leaving the actual claim ("bound into generation identity")
    untested. The contracts are what a reviewer authorizes and what replay refuses a
    model mismatch against, so they are where the binding has to be asserted.

    The elicitations are patched to distinct non-placeholder values because
    `locked_contract()` refuses to build while they hold the placeholder — that
    refusal is a different guarantee, pinned separately by
    `test_locked_contract_is_unbuildable_while_prompts_are_placeholders`.
    """
    # Patched on `plan`, which is where the elicitations now live — patching a
    # re-export on `probe` would leave `registered_locked_contract()` reading the
    # originals and quietly make this test vacuous again.
    monkeypatch.setattr(
        plan,
        "_REFUSAL_ELICITATIONS",
        {
            RowId.REFUSAL_1: "frozen elicitation one",
            RowId.REFUSAL_2: "frozen elicitation two",
            RowId.REFUSAL_3: "frozen elicitation three",
        },
    )
    locked = plan.registered_locked_contract()
    assert locked.model == plan.MODEL

    dry = DryRunManifest(
        contract_version=STRICT_PROBE_CONTRACT_VERSION,
        locked_contract_digest=locked.digest,
        measurements=tuple(
            RowMeasurement(
                row_id=row,
                prompt_bytes=32_000,
                estimated_tokens=8_000,
                request_body_digest=hashlib.sha256(f"req-{row.value}".encode()).hexdigest(),
            )
            for row in ROW_ORDER
        ),
    )
    paid = PaidProbeContract.from_reviewed(
        locked=locked, dry=dry, caps=dict.fromkeys(ROW_ORDER, 40_000)
    )
    assert paid.model == plan.MODEL

    # And the outgoing request agrees with the identity that was authorized.
    assert plan.strict_kwargs(**_pair())["model"] == plan.MODEL


def test_probe_keeps_the_production_token_limit_kwarg_name() -> None:
    """SHAPER v3: the GPT-5.6 wire 400s on `max_tokens`, so the ceiling must ride
    under the profile-declared `max_completion_tokens`. A probe that sent the
    wrong kwarg would 400 on row 1 and be misread as a schema rejection."""
    strict = plan.strict_kwargs(**_pair())
    assert "max_completion_tokens" in strict
    assert "max_tokens" not in strict


def test_probe_preserves_cache_key_and_service_tier() -> None:
    """Both are wire-affecting host DATA; dropping either would change what the
    paid row actually measures.

    Asserts PRESENCE first rather than `if key in production: assert ...` — the
    guarded form silently asserts nothing if production ever stops emitting the
    key, which is the `if X in result: assert` anti-pattern.
    """
    kwargs = _pair()
    production = plan.production_kwargs(**kwargs)
    strict = plan.strict_kwargs(**kwargs)
    for key in ("prompt_cache_key", "service_tier", "model", "temperature", "messages"):
        assert key in production, f"production wire no longer sends {key!r}"
        assert strict[key] == production[key]


def test_all_five_rows_build_offline_and_are_bounded() -> None:
    """Every row builds with no network access, and the matrix is exactly the
    five pre-registered rows in the frozen execution order."""
    rows = plan.build_rows()
    assert set(rows) == set(ROW_ORDER)
    assert len(rows) == plan.MAX_PAID_CALLS
    assert ROW_ORDER[0] is RowId.ACCEPTANCE_CLEAN  # positional classification depends on this


def test_probe_dry_run_makes_no_network_call(tmp_path: Any) -> None:
    """The free prefix is genuinely free — asserted at the SOCKET, not by patching.

    An earlier version patched a client class the probe holds no reference to, so
    the assertion was inert. A PEP-578 audit hook tests the PROPERTY rather than
    one route to violating it.
    """
    events: list[str] = []

    def _audit(event: str, _args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.getaddrinfo", "urllib.Request"}:
            events.append(event)

    sys.addaudithook(_audit)
    with patch.object(probe, "_OUT_DIR", tmp_path):
        assert probe.run_dry() == 0
    assert events == [], f"--dry-run attempted network activity: {events}"


def test_placeholder_dry_run_is_preview_only_and_gate2_ineligible(tmp_path: Any) -> None:
    """A dry run over PLACEHOLDER prompts must not be usable as a Gate-2 source.

    It measures prompts that will not be the ones sent, so caps derived from it
    would describe a different experiment. The artifact says so mechanically
    rather than relying on the operator noticing.
    """
    assert plan.refusal_freeze_error() is not None  # placeholders still in place
    with patch.object(probe, "_OUT_DIR", tmp_path):
        probe.run_dry()

    assert not (tmp_path / "dry_run_preview_only.json").with_suffix(".missing").exists()
    preview = json.loads((tmp_path / "dry_run_preview_only.json").read_text())
    assert preview["preview_only"] is True
    assert "placeholder" in preview["ineligible_reason"]
    # No content-addressed DryRunManifest was written.
    assert not list(tmp_path.glob("dry_run_[0-9a-f]*.json"))


def test_paid_mode_is_an_unconditional_stub_at_gate_one() -> None:
    """`run_paid()` refuses unconditionally and consumes no contract. Both halves.

    The gate at Gate 1 is a STUB, not a contract check — the earlier name of this
    test ("requires a reviewed contract, not a flag") claimed a mechanism the
    function does not implement: it accepts no contract and loads none. What is
    true is that no paid runner exists, so nothing can spend.

    The absence assertions remain meaningful for a different reason: they pin that
    the superseded flag-shaped design (`PAID_MODE_UNLOCKED` plus nullable
    `FROZEN_*` caps) has not crept back as an alternative unlock path.
    """
    import inspect

    assert not hasattr(probe, "PAID_MODE_UNLOCKED")
    assert not hasattr(probe, "FROZEN_MAX_COMPLETION_TOKENS")
    assert not hasattr(probe, "FROZEN_PER_ROW_PROMPT_BYTE_CAP")

    # Takes no arguments: there is no contract to pass, by construction.
    assert not inspect.signature(probe.run_paid).parameters
    assert probe.run_paid() == 2
    # And refuses irrespective of the freeze state — the refusal is unconditional,
    # not a consequence of the placeholder elicitations.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            plan,
            "_REFUSAL_ELICITATIONS",
            {
                RowId.REFUSAL_1: "frozen one",
                RowId.REFUSAL_2: "frozen two",
                RowId.REFUSAL_3: "frozen three",
            },
        )
        assert plan.refusal_freeze_error() is None  # freeze satisfied...
        assert probe.run_paid() == 2  # ...and it still refuses


def test_completion_limit_is_single_sourced_onto_the_wire() -> None:
    """ONE value reaches the request, the measurement, and the contract.

    The limit was previously recorded in the artifact as a nullable constant while
    the request sent `analyze_prompt.MAX_TOKENS` — so changing the recorded value
    rotated the manifest without changing the wire at all.
    """
    kwargs = plan.strict_kwargs(**_pair())
    assert kwargs["max_completion_tokens"] == plan.MAX_COMPLETION_TOKENS


def test_locked_contract_is_unbuildable_while_prompts_are_placeholders() -> None:
    """The identity contract itself refuses placeholder prompts.

    All three placeholders are the same string, and identical elicitations are ONE
    observation billed three times — so the contract that would authorize a paid
    run cannot even be constructed. That is a stronger gate than a boolean: there
    is no state in which the artifact exists but is "not yet unlocked".
    """
    from pydantic import ValidationError

    assert plan.refusal_freeze_error() is not None
    with pytest.raises(ValidationError, match="not distinct"):
        plan.registered_locked_contract()


def test_evaluation_contract_holds_no_observed_counts() -> None:
    """Identity only. A count is a response-derived observation, and an artifact
    that asserts its own observations attests to itself."""
    fields = set(type(plan.registered_evaluation_contract()).model_fields)
    for observed in ("retained_trace_candidate_count", "trace_candidate_count", "finding_count"):
        assert observed not in fields


def test_probe_does_not_import_the_openai_sdk() -> None:
    """Trust boundary #8: the SDK is owned by `llm/raw_openai_capture`. The probe
    and every `arc2/` module must reach the wire only through that helper.

    Parsed with `ast`, not a substring scan. A substring check for
    `"import openai"` misses `from openai import ...`, `import openai as x`, and
    `importlib.import_module("openai")` — and `spikes/` is outside BOTH automated
    enforcement layers (`scripts/check_import_boundaries.py` scans
    `src/outrider/**`; the `check-trust-boundaries` skill fires on `src/`,
    `tests/`, `scripts/`), so this test is the only guard that exists here.
    """
    probe_path = Path(probe.__file__ or "")
    assert probe_path.is_file()
    targets = [probe_path, *sorted(probe_path.parent.joinpath("arc2").glob("*.py"))]
    assert len(targets) >= 4  # probe + __init__ + the three arc2 modules

    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] != "openai", f"{path.name}: {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] != "openai", f"{path.name}: {node.module}"
            elif isinstance(node, ast.Call):
                # importlib.import_module("openai") / __import__("openai")
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in {"import_module", "__import__"}:
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            assert arg.value.split(".")[0] != "openai", f"{path.name}: dynamic"


def test_probe_schema_is_derived_from_canonical_not_hand_copied() -> None:
    """If canonical gains a property, the derived branches gain it too. A
    hand-copied schema would silently keep probing the old shape."""
    canonical_props = set(ANALYZE_RESPONSE_SCHEMA["properties"]["findings"]["items"]["properties"])
    schema = probe.derive_strict_analyze_schema()
    for branch in schema["properties"]["findings"]["items"]["anyOf"]:
        assert set(branch["properties"]) == canonical_props


# --------------------------------------------------------------------------
# Scenario authority — one coordinate frame, verified END TO END.
# --------------------------------------------------------------------------


def test_scenario_is_the_single_coordinate_authority() -> None:
    """Prompt, scope, file content, and expected identity all read from one place.

    They previously disagreed three ways at once: the diff presented lines 10–14,
    the parser saw the scope text as a complete 4-line file, and the expected
    finding named lines 1–2 — while the vulnerable SQL is on line 3.
    """
    s = probe.DEFECT_SCENARIO
    lines = s.source.splitlines()
    assert s.scope_line_end == len(lines)
    assert s.defect_line is not None
    # The declared defect line really is the vulnerable SQL construction.
    assert "+ user_id +" in lines[s.defect_line - 1]
    # And the expected identity names exactly that line.
    assert probe.EXPECTED_FINDING.line_start == s.defect_line
    assert probe.EXPECTED_FINDING.line_end == s.defect_line
    # The hunk header is in the SAME frame as the source, not a different one.
    assert s.diff_hunk.startswith(f"@@ -0,0 +{s.scope_line_start},{s.scope_line_end} @@")


def test_planted_defect_line_is_admitted_by_the_real_parser() -> None:
    """END TO END: a response naming the planted line is ADMITTED by the real
    `parse_analyze_response` AND satisfies the expected identity.

    This is the test that would have caught the three-frame disagreement: under
    the old coordinates a locally correct line-3 answer was rejected
    (`span_outside_scope_unit`) while the 1–2 fixture matched an expectation that
    described no real line.
    """
    from spikes.openai.arc2.classifier import derive_assessment

    body = probe._finding_body()
    outcome = probe.run_real_parser(body)

    assert outcome.rejection_reasons == (), (
        f"the planted defect line was rejected: {outcome.rejection_reasons}"
    )
    assert len(outcome.admitted) == 1

    assessment = derive_assessment(
        body=body,
        parser=outcome,
        fired_query_match_ids=probe.FIRED_QUERY_MATCH_IDS,
        expected=probe.EXPECTED_FINDING,
    )
    assert assessment.returned_any_finding is True
    assert assessment.nonempty_trace_candidates is False


def test_a_request_visible_coordinate_answer_is_not_silently_admitted() -> None:
    """The failure mode the old frames created: an answer in DIFFERENT coordinates
    must not match the expected identity. It is INCONCLUSIVE, not a pass."""
    from spikes.openai.arc2.classifier import derive_assessment

    body = probe._finding_body(line_start=12, line_end=12)
    outcome = probe.run_real_parser(body)
    assessment = derive_assessment(
        body=body,
        parser=outcome,
        fired_query_match_ids=probe.FIRED_QUERY_MATCH_IDS,
        expected=probe.EXPECTED_FINDING,
    )
    assert assessment.returned_any_finding is False


def test_prompt_and_context_entry_use_the_scenario_range() -> None:
    """The packed-context manifest must name the scope the source actually has."""
    rows = plan.build_rows()
    s = probe.DEFECT_SCENARIO
    assert s.defect_line is not None
    user_prompt = rows[RowId.ACCEPTANCE_FINDING]["messages"][1]["content"]
    # The vulnerable line is actually shown to the model, in the frame the parser
    # will validate the answer against.
    assert s.source.splitlines()[s.defect_line - 1].strip() in user_prompt
    assert f"@@ -0,0 +{s.scope_line_start},{s.scope_line_end} @@" in user_prompt
