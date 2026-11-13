"""Python package-relative candidates survive admission and native trace probes."""

from uuid import uuid4

import pytest
from test_analyze_parser import _build_response_json, _call_parser, _minimal_proposal
from test_trace_node import _fake_fetch_for

from outrider.agent.nodes import trace as trace_module
from outrider.agent.nodes.trace import _ProbeBudget, _resolve_via_probes
from outrider.audit.events import TraceDecisionEvent
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import TraceDecision


def _candidates(import_string, source_file, *, form="module"):
    source = "def read(value):\n    return value\n"
    proposal = _minimal_proposal(
        finding_type="correctness_bug",
        evidence="    return value",
        line_start=2,
        line_end=2,
        trace_candidates=[{"import_string_raw": import_string, "reason": "check imported helper"}],
    )
    result = _call_parser(
        _build_response_json(proposal),
        file_path=source_file,
        file_content=source,
        file_byte_length=len(source.encode()),
        primary_line_ranges=((1, 2),),
        active_policy_version=ACTIVE_POLICY_VERSION,
        trace_candidate_form=form,
    )
    assert result.response_rejection is None
    assert len(result.admitted_findings) == 1
    return result


async def _resolve(monkeypatch, result, source_file, real_files):
    fetch, probes = _fake_fetch_for(real_files)
    monkeypatch.setattr(trace_module, "fetch_file_content_at", fetch)
    budget = _ProbeBudget(remaining=8)
    outcome = await _resolve_via_probes(
        candidates=result.trace_candidates,
        source_finding_file=source_file,
        gh_client=object(),
        owner="owner",
        repo="repo",
        head_sha="a" * 40,
        probe_memo={},
        probe_budget=budget,
    )
    assert budget.remaining == 8 - len(probes)
    return outcome, probes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("import_string", "source_file", "target", "content"),
    [
        (
            ".codec",
            "pkg/readers/worker.py",
            "pkg/readers/codec.py",
            b"def decode(value):\n    return value\n",
        ),
        (
            "..shared.decode",
            "src/pkg/jobs/worker.py",
            "src/pkg/shared.py",
            b"def decode(value):\n    return value\n",
        ),
    ],
)
async def test_relative_candidate_reaches_native_probe_consumer(
    monkeypatch, import_string, source_file, target, content
):
    result = _candidates(import_string, source_file)
    outcome, probes = await _resolve(monkeypatch, result, source_file, {target: content})
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == target
    assert outcome.resolved_candidate_paths == (target,)
    assert target in probes
    assert result.counters.n_trace_candidates_dropped_malformed == 0


@pytest.mark.asyncio
async def test_relative_module_and_package_remain_ambiguous(monkeypatch):
    source_file = "pkg/jobs/worker.py"
    result = _candidates(".helpers", source_file)
    files = {"pkg/jobs/helpers.py": b"VALUE = 1\n", "pkg/jobs/helpers/__init__.py": b"VALUE = 2\n"}
    outcome, _ = await _resolve(monkeypatch, result, source_file, files)
    assert outcome.resolution_status == "ambiguous"
    assert outcome.target_file is None
    assert set(outcome.resolved_candidate_paths) == set(files)


@pytest.mark.parametrize(
    ("import_string", "source_file", "form"),
    [
        ("..outside", "pkg/worker.py", "module"),
        (".helper..decode", "pkg/worker.py", "module"),
        (".helper/../../outside", "pkg/worker.py", "module"),
        (".helpers", "pkg/worker.ts", "specifier"),
    ],
)
def test_escape_invalid_and_wrong_language_candidates_stay_rejected(
    import_string, source_file, form
):
    result = _candidates(import_string, source_file, form=form)
    assert result.trace_candidates == ()
    assert result.counters.n_trace_candidates_dropped_malformed == 1


@pytest.mark.asyncio
async def test_javascript_relative_specifier_keeps_existing_resolution(monkeypatch):
    source_file = "pkg/jobs/worker.ts"
    result = _candidates("./helpers", source_file, form="specifier")
    target = "pkg/jobs/helpers.ts"
    outcome, _ = await _resolve(
        monkeypatch, result, source_file, {target: b"export const value = 1;\n"}
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == target


def test_relative_imports_have_matching_canonical_state_and_audit_bytes():
    result = _candidates(".cafe\u0301", "pkg/jobs/worker.py")
    (candidate,) = result.trace_candidates
    assert candidate.import_string == ".caf\u00e9"
    fields = dict(
        source_finding_id=uuid4(),
        target_file="pkg/jobs/caf\u00e9.py",
        resolution_status="resolved",
        reason="check helper",
        proposed_import_strings=(candidate.import_string,),
        resolved_candidate_paths=("pkg/jobs/caf\u00e9.py",),
    )
    decision = TraceDecision(**fields)
    event = TraceDecisionEvent(review_id=uuid4(), **fields)
    assert (
        decision.proposed_import_strings
        == event.proposed_import_strings
        == (candidate.import_string,)
    )
