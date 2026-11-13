"""Append bounded constructor callers without changing primary coverage."""

from __future__ import annotations

from difflib import unified_diff
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import test_analyze_node as recorders
from test_analyze_cache_key import _BASE_KWARGS

from outrider.agent.nodes import analyze as node
from outrider.agent.nodes import analyze_context as context
from outrider.ast_facts import parse_python
from outrider.cache import compute_analyze_cache_key
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER

_PATH = "app/records.py"
_HEAD = """def normalize(value):
    return value


class Base:
    def __init__(self, value):
        self.value = value


class Record(Base):
    def __init__(self, value=2):
        super().__init__(value)

    def normalize(self):
        return normalize(self.value)


def prior_caller(value):
    return normalize(value)


def make_record(value):
    return Record(value)
"""


def _inputs(source=_HEAD):
    before = source.replace("value=2", "value=1")
    changed = recorders._build_changed_file(
        path=_PATH,
        content=source.encode(),
        content_base=before,
        patch="".join(
            unified_diff(
                before.splitlines(keepends=True),
                source.splitlines(keepends=True),
                fromfile="a/" + _PATH,
                tofile="b/" + _PATH,
            )
        ),
    )
    prepared = node._select_file_review(
        changed_file=changed, import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER
    )
    prior = context.prepare_review_context(
        file_path=_PATH,
        content=source,
        patch=changed.patch,
        changed_files=(changed,),
        import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER,
    )
    for add in (
        context.add_parent_constructor_context,
        context.add_parent_class_context,
        context.add_parent_reference_declarations,
    ):
        prior = add(
            prior,
            file_path=_PATH,
            head_sha="b" * 40,
            source=source,
            selected_scope_units=prepared.decision.included_scope_units,
        )
    return changed, prepared, prior


async def _capture(
    source=_HEAD, *, budget=node.DEFAULT_REVIEW_BUDGET_TOKENS, response='{"findings":[]}'
):
    changed, prepared, prior = _inputs(source)
    state = recorders._build_review_state(
        pr_context=recorders._build_pr_context(changed_files=(changed,)),
        triage_result=recorders._build_triage_result(file_tiers={_PATH: recorders.ReviewTier.DEEP}),
    )
    deps = recorders.deps.__wrapped__()
    deps.update(provider=recorders._StubLLMProvider(response), total_review_budget_tokens=budget)
    command = await node.analyze(state, **deps)
    (send,) = command.goto
    worker = await node.analyze_file(
        send.arg, **{key: deps[key] for key in recorders._WORKER_DEP_KEYS if key in deps}
    )
    (request,) = deps["provider"].calls
    baseline = node._prepare_file_prompt(
        prepared, file_path=_PATH, pass_index=0, review_context=prior.text
    )
    return SimpleNamespace(
        payload=send.arg,
        request=request,
        prior=prior,
        baseline=baseline,
        worker=worker,
        prepared=prepared,
        sink=deps["analyze_event_sink"],
    )


def _cache_key(parts):
    return compute_analyze_cache_key(
        **{**_BASE_KWARGS, "system_prompt": parts.system_prompt, "user_prompt": parts.user_prompt}
    )


@pytest.mark.asyncio
async def test_native_constructor_caller_appends_after_existing_references() -> None:
    result = await _capture()
    assert "def make_record(value):" in result.payload.review_context
    assert "def prior_caller(value):" in result.prior.text
    assert result.payload.review_context.startswith(result.prior.text)
    assert (
        result.payload.supplemental_context_manifest[: len(result.prior.manifest)]
        == result.prior.manifest
    )
    assert result.request.system_prompt == result.baseline.prompt_parts.system_prompt
    assert (
        result.request.user_prompt.split("## Additional bounded review context")[0]
        == result.baseline.prompt_parts.user_prompt.split("## Additional bounded review context")[0]
    )
    assert _cache_key(result.request) != _cache_key(result.baseline.prompt_parts)
    assert any(
        entry.scope_unit_name.startswith("make_record@")
        and entry.inclusion_reason == "same_file_context"
        for entry in result.request.context_summary
    )


@pytest.mark.asyncio
async def test_scarce_budget_keeps_existing_context_and_cache_identity() -> None:
    _, prepared, prior = _inputs()
    estimate = node._prepare_file_prompt(
        prepared, file_path=_PATH, pass_index=0, review_context=prior.text
    ).estimated_tokens
    result = await _capture(budget=estimate * 4)
    assert result.payload.allocation_tokens == estimate
    assert result.payload.review_context == result.prior.text
    assert result.request.user_prompt == result.baseline.prompt_parts.user_prompt
    assert _cache_key(result.request) == _cache_key(result.baseline.prompt_parts)


@pytest.mark.asyncio
async def test_existing_two_caller_limit_is_not_extended() -> None:
    source = _HEAD.replace(
        "def make_record(value):",
        "def second_caller(value):\n    return normalize(value)\n\n\ndef make_record(value):",
    )
    result = await _capture(source)
    assert "def prior_caller(value):" in result.prior.text
    assert "def second_caller(value):" in result.prior.text
    assert result.payload.review_context == result.prior.text


@pytest.mark.asyncio
async def test_appended_caller_remains_outside_primary_finding_locations() -> None:
    line = _HEAD.splitlines().index("    return Record(value)") + 1
    response = recorders._build_finding_proposal_json(line_start=line, line_end=line)
    result = await _capture(response=response)
    assert "def make_record(value):" in result.payload.review_context
    assert not result.worker["analyze_worker_outcomes"][0].admitted_findings
    assert [entry.rejection_reason for entry in result.sink.proposal_rejections] == [
        "span_outside_scope_unit"
    ]


def test_new_constructor_references_reject_obvious_ambiguity_and_shadows() -> None:
    from outrider.ast_facts.review_context import python_constructor_caller_context

    cls = "class Record:\n    def __init__(self, value):\n        self.value = value\n"
    caller = "\ndef make_record(value):\n    return Record(value)\n"

    def select(source, *, ordinary_method=False, supplied_names=frozenset()):
        parsed = parse_python(source.encode(), _PATH, MagicMock())
        selected = tuple(
            scope
            for scope in parsed.scope_units
            if scope.qualified_name
            == ("Record.normalize" if ordinary_method else "Record.__init__")
        )
        return python_constructor_caller_context(
            source,
            file_path=_PATH,
            selected_scope_units=selected,
            supplied_names=supplied_names,
            supplied_spans=(),
        )

    assert len(select(cls + caller)) == 1
    invalid = (
        "@decorate\n" + cls + caller,
        cls.replace("class Record:", "class Record(metaclass=Meta):") + caller,
        cls + "Record = factory()\n" + caller,
        cls + "Record.__new__ = replacement\n" + caller,
        cls + "    def __new__(cls, value):\n        return object.__new__(cls)\n" + caller,
        cls + "\n" + cls + caller,
        cls + "\ndef make_record(Record, value):\n    return Record(value)\n",
        cls + "\ndef make_record(value):\n    Record = factory()\n    return Record(value)\n",
        cls + "\ndef make_record(value):\n    return (lambda Record: Record(value))(factory)\n",
        cls + caller.replace("    return", "    # " + "x" * 1000 + "\n    return"),
    )
    for source in invalid:
        assert select(source) == (), source
    assert select(_HEAD, ordinary_method=True) == ()
    parsed = parse_python(_HEAD.encode(), _PATH, MagicMock())
    class_only = tuple(scope for scope in parsed.scope_units if scope.qualified_name == "Record")
    assert (
        len(
            python_constructor_caller_context(
                _HEAD,
                file_path=_PATH,
                selected_scope_units=class_only,
                supplied_names=frozenset(),
                supplied_spans=(),
            )
        )
        == 1
    )
    unrelated = class_only + tuple(
        scope for scope in parsed.scope_units if scope.qualified_name == "Record.normalize"
    )
    assert (
        python_constructor_caller_context(
            _HEAD,
            file_path=_PATH,
            selected_scope_units=unrelated,
            supplied_names=frozenset(),
            supplied_spans=(),
        )
        == ()
    )
