"""Prioritize helpers of the actual changed scope in recorded and authored inputs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import test_analyze_node as recorders

from outrider.ast_facts.review_context import build_called_function_index, called_function_context
from outrider.llm.base import INCLUDE_TEXT_OPT_IN
from outrider.schemas.review_state import ReviewState

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "helper_call_priority"
_TARGET = "src/trio/_core/_run.py"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["trio-2938", "trio-3394"])
async def test_recorded_changed_method_receives_its_factory(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay actual packets with saved empty answers, measuring source delivery only."""
    fixture = _FIXTURES / case
    original = json.loads((fixture / "graph-state.json").read_text())
    state = ReviewState.model_validate(
        {
            **original,
            "analysis_rounds": [],
            "analyze_worker_outcomes": [],
            "analyze_pass_started_at": None,
            "review_report": None,
            "publish_result": None,
        }
    )
    requests = {}
    replies = {}
    for path in sorted((fixture / "logical").glob("*.request.json")):
        request = json.loads(path.read_text())
        file_path = request["user_prompt"].split("File: ", 1)[1].split("\n", 1)[0]
        response_path = path.with_name(path.name.replace(".request.json", ".response.json"))
        response = json.loads(response_path.read_text())
        assert json.loads(response["text"]) == {"findings": []}
        requests[file_path] = request
        replies[file_path] = response["text"]
    deps = recorders.deps.__wrapped__()
    deps.update(analyze_model="claude-sonnet-5-5", standard_analyze_model="claude-sonnet-5-5")
    deps["import_path_resolver"].resolve_candidate_paths.return_value = []
    deps["import_path_resolver"].resolve_specifier_candidate_paths.return_value = []
    provider = recorders._StubLLMProvider("")
    complete = provider.complete

    async def saved_answer(request):
        file_path = request.user_prompt.split("File: ", 1)[1].split("\n", 1)[0]
        provider.response_text = replies[file_path]
        return await complete(request)

    monkeypatch.setattr(provider, "complete", saved_answer)
    deps["provider"] = provider
    baseline = await recorders.run_analyze_pass(state, deps)
    for index, request in enumerate(provider.calls):
        actual = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
        (tmp_path / f"baseline-{index}.request.json").write_text(json.dumps(actual, indent=2))
        file_path = actual["user_prompt"].split("File: ", 1)[1].split("\n", 1)[0]
        assert actual == requests[file_path]
    assert len(provider.calls) == len(requests) == 2
    baseline_calls = {request.phase_key: request for request in provider.calls}
    provider.calls.clear()
    candidate = await recorders.run_analyze_pass(
        state, {**deps, "called_function_context_enabled": True}
    )
    for index, request in enumerate(provider.calls):
        actual = request.model_dump(mode="json", context=INCLUDE_TEXT_OPT_IN)
        (tmp_path / f"helper-{index}.request.json").write_text(json.dumps(actual, indent=2))
        before = baseline_calls[request.phase_key].model_dump(
            mode="json", context=INCLUDE_TEXT_OPT_IN
        )
        assert {key for key in actual if actual[key] != before[key]} <= {
            "context_summary",
            "user_prompt",
        }
        assert tuple(
            e for e in request.context_summary if e.inclusion_reason == "changed_scope"
        ) == (
            tuple(
                e
                for e in baseline_calls[request.phase_key].context_summary
                if e.inclusion_reason == "changed_scope"
            )
        )
    target = next(r for r in provider.calls if f"File: {_TARGET}\n" in r.user_prompt)
    references = [
        entry for entry in target.context_summary if entry.inclusion_reason == "captured_reference"
    ]
    (tmp_path / "helper-references.json").write_text(
        json.dumps([entry.model_dump(mode="json") for entry in references], indent=2)
    )
    assert len(references) <= 2
    assert any(entry.scope_unit_name.startswith("open_nursery@") for entry in references), (
        "The changed Nursery.start factory must take priority over unchanged sibling methods",
        [entry.scope_unit_name for entry in references],
    )
    assert "def open_nursery(" in target.user_prompt
    assert "return NurseryManager(" in target.user_prompt
    assert "class NurseryManager:" not in target.user_prompt
    assert (
        len(candidate["analyze_worker_outcomes"]) == len(baseline["analyze_worker_outcomes"]) == 2
    )
    assert [o.skip_reason for o in candidate["analyze_worker_outcomes"]] == [None, None]
    assert not candidate["analysis_rounds"][0].findings


@pytest.mark.parametrize("suffix", [".py", ".js", ".ts"])
def test_authored_changed_owner_precedes_sibling_and_nested_calls(suffix: str) -> None:
    """Authored control for ownership across registered language parsers."""
    if suffix == ".py":
        source = """def early_one():
    return 1

def early_two():
    return 2

def needed():
    return 3

class Service:
    def sibling(self):
        return early_one()

    def changed(self):
        def nested():
            return early_two()
        return needed() + 1
"""
    else:
        source = """function early_one() {
    return 1;
}
function early_two() {
    return 2;
}
function needed() {
    return 3;
}
class Service {
    sibling() {
        return early_one();
    }
    changed() {
        function nested() {
            return early_two();
        }
        return needed() + 1;
    }
}
"""
    path = "app/service" + suffix
    changed_line = next(
        index for index, line in enumerate(source.splitlines(), 1) if "return needed()" in line
    )
    index = build_called_function_index({path: source}, import_path_resolver=MagicMock())
    references = called_function_context(
        index, file_path=path, changed_lines=frozenset({changed_line})
    )
    assert [r.excerpt.name.split("@", 1)[0] for r in references] == ["needed", "early_one"]
