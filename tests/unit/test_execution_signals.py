"""Separate structural execution facts from contextual vulnerability findings."""

from __future__ import annotations

import difflib
import json
from dataclasses import replace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import test_analyze_node as recorders
from pydantic import ValidationError

from outrider.agent.nodes.analyze_observed import (
    compute_observed_skip_shadow,
    produce_observed_findings,
    run_observed_matches,
)
from outrider.ast_facts import parse_python
from outrider.coordinates import lookup_patched_file
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.queries.observed import ObservedQuery, QueryClass
from outrider.queries.registry import OBSERVED_QUERIES

_PATH = "app/reports.py"
_BASE = "import subprocess\nfrom flask import request\n"
_HEAD = _BASE + (
    "\ndef report():\n"
    '    """Public endpoint accepting arbitrary report labels."""\n'
    '    name = request.args["name"]\n'
    '    return subprocess.run("report " + name, shell=True)\n'
)


def _matches(source: str):
    parsed = parse_python(source.encode(), _PATH, MagicMock())
    return run_observed_matches(
        file_path=_PATH,
        head_content=source,
        included_scope_units=parsed.scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
    )


def test_execution_facts_do_not_automatically_claim_injection() -> None:
    source = (
        "import subprocess\n"
        "def local_command():\n"
        '    return subprocess.run("echo ready", shell=True)\n'
        "def caller_command(cmd):\n"
        "    return subprocess.run(cmd, shell=True)\n"
    )
    matches = _matches(source)
    assert len(matches) == 2
    assert all(
        match.query_match_id == "python.command_injection_subprocess_shell" for match in matches
    )
    assert [match.line_start for match in matches] == [3, 5]
    assert all("shell=True" in match.evidence for match in matches)
    assert (
        produce_observed_findings(
            matches,
            file_path=_PATH,
            review_id=uuid4(),
            installation_id=1,
            active_policy_version=ACTIVE_POLICY_VERSION,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_judged_injection_keeps_its_context_and_critical_policy() -> None:
    evidence = '    return subprocess.run("report " + name, shell=True)'
    proposal = {
        "finding_type": "command_injection",
        "evidence_tier": "judged",
        "query_match_id": None,
        "trace_path": None,
        "title": "Report label reaches the shell",
        "description": (
            "The public request label is concatenated into shell source; a label containing "
            "shell operators executes a second command under the service account."
        ),
        "evidence": evidence,
        "line_start": 7,
        "line_end": 7,
        "trace_candidates": [],
    }
    changed = recorders._build_changed_file(
        path=_PATH,
        content=_HEAD.encode(),
        content_base=_BASE,
        patch="".join(
            difflib.unified_diff(
                _BASE.splitlines(keepends=True),
                _HEAD.splitlines(keepends=True),
                fromfile="a/" + _PATH,
                tofile="b/" + _PATH,
            )
        ),
    )
    state = recorders._build_review_state(
        pr_context=recorders._build_pr_context(changed_files=(changed,)),
        triage_result=recorders._build_triage_result(file_tiers={_PATH: recorders.ReviewTier.DEEP}),
    )
    deps = recorders.deps.__wrapped__()
    provider = recorders._StubLLMProvider(json.dumps({"findings": [proposal]}))
    result = await recorders.run_analyze_pass(state, {**deps, "provider": provider})
    (finding,) = result["analysis_rounds"][0].findings
    assert finding.finding_type is FindingType.COMMAND_INJECTION
    assert finding.severity is FindingSeverity.CRITICAL
    assert finding.evidence_tier is EvidenceTier.JUDGED
    assert (finding.title, finding.description, finding.evidence) == (
        proposal["title"],
        proposal["description"],
        evidence,
    )
    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.producer_finding_ids == ()
    assert len(provider.calls) == 1


def test_context_required_catalog_is_explicit() -> None:
    assert {
        query.query_match_id for query in OBSERVED_QUERIES.values() if query.requires_context
    } == {
        "python.command_injection_subprocess_shell",
        "python.command_injection_os_system",
        "python.command_injection_eval_exec",
        "javascript.command_injection_child_process",
        "javascript.command_injection_eval",
    }


@pytest.mark.parametrize(
    "extra", [{"query_class": QueryClass.SKIP_SAFE}, {"module_scope_eligible": True}]
)
def test_context_required_query_cannot_authorize_structural_shortcuts(extra) -> None:
    source = OBSERVED_QUERIES["python.command_injection_subprocess_shell"].model_dump()
    with pytest.raises(ValidationError, match="requires_context"):
        ObservedQuery(**{**source, "requires_context": True, **extra})


def test_direct_match_cannot_fund_skip_coverage() -> None:
    source = "import subprocess\ndef run(cmd):\n    subprocess.run(cmd, shell=True)\n"
    (match,) = _matches(source)
    injected = replace(match, query_class=QueryClass.SKIP_SAFE)
    parsed = parse_python(source.encode(), _PATH, MagicMock())
    patch = lookup_patched_file(
        "--- a/app/reports.py\n+++ b/app/reports.py\n@@ -1,2 +1,3 @@\n"
        " import subprocess\n def run(cmd):\n+    subprocess.run(cmd, shell=True)\n",
        _PATH,
    )
    assert patch is not None
    event = compute_observed_skip_shadow(
        (injected,),
        file_path=_PATH,
        included_scope_units=parsed.scope_units,
        patched_file=patch,
        head_source=source,
        base_source="import subprocess\ndef run(cmd):\n",
        review_id=uuid4(),
        is_eval=True,
    )
    assert event is not None and event.outcome == "not_eligible"
    assert event.covering_matches == ()
