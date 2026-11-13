"""Execute synthetic case programs to prove labels without model outputs."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import TYPE_CHECKING

import pytest

from outrider.agent.eval_driver import EvalFixture

from .grounded_cases import ROOT, context_for_case, dataset_inputs, load_cases

if TYPE_CHECKING:
    from pathlib import Path


def program(case_id: str):
    case = next(case for case in load_cases() if case.id == case_id)
    fixture = EvalFixture.model_validate_json((ROOT / case.fixture_path).read_bytes())
    namespace = {}
    # Only hash-approved local programs run here, never model-generated code.
    exec(compile(fixture.files[0].content_head, fixture.files[0].path, "exec"), namespace)  # noqa: S102
    return namespace["handle_request"]


def test_request_sql_pair_is_proved_by_the_database() -> None:
    interpolate = program("request_sql_interpolation")
    bind = program("request_sql_binding")
    for endpoint in (interpolate, bind):
        assert endpoint({"name": "alice"}) == [1]
        assert endpoint({"name": "absent"}) == []
    attack = {"name": "' OR 1=1 --"}
    assert interpolate(attack) == [1, 2]
    assert bind(attack) == []


def test_json_pair_is_proved_at_the_response_boundary() -> None:
    escape = program("json_decode_escape")
    respond = program("json_decode_response")
    for endpoint in (escape, respond):
        assert endpoint({"body": '{"value": 7}'}) == (200, {"value": 7})
        assert endpoint({"body": b"\xff"}) == (400, None)
    with pytest.raises(json.JSONDecodeError):
        escape({"body": "{"})
    assert respond({"body": "{"}) == (400, None)


def test_json_depth_pair_is_proved_at_the_response_boundary() -> None:
    escape = program("json_depth_escape")
    respond = program("json_depth_response")
    for endpoint in (escape, respond):
        for body in ('{"value": 7}', b'{"value": 7}', bytearray(b'{"value": 7}')):
            assert endpoint({"body": body}) == (200, {"value": 7})
        assert endpoint({"body": "{"}) == (400, None)
        assert endpoint({"body": b"\xff"}) == (400, None)
        for body in (None, 7, [], {}):
            assert endpoint({"body": body}) == (400, None)
    depth = 10_000
    body = "[" * depth + "0" + "]" * depth
    with pytest.raises(RecursionError):
        escape({"body": body})
    assert respond({"body": body}) == (400, None)


def test_account_pair_is_proved_by_denied_credentials() -> None:
    bypass = program("account_check_bypass")
    require = program("account_check_required")
    accounts = {"admin": {"balance": 20}, "alice": {"balance": 10}}
    calls = []

    def credential_check(username, password):
        calls.append((username, password))
        return username in accounts and password == "accepted-test-password"  # noqa: S105 — oracle value

    for endpoint in (bypass, require):
        assert endpoint(
            {"username": "alice", "password": "accepted-test-password"}, credential_check, accounts
        ) == (200, accounts["alice"])
        assert endpoint({"username": "alice", "password": "wrong"}, credential_check, accounts) == (
            403,
            None,
        )
        assert endpoint({}, credential_check, accounts) == (400, None)
    calls.clear()
    request = {"username": "admin", "password": "wrong"}
    assert bypass(request, credential_check, accounts) == (200, accounts["admin"])
    assert calls == []
    assert require(request, credential_check, accounts) == (403, None)
    assert calls == [("admin", "wrong")]


def test_async_pair_is_proved_by_other_tasks_making_progress() -> None:
    async def observe_schedule(endpoint):
        events = []

        async def observer():
            events.append("observer")

        results = await asyncio.gather(endpoint(events), observer())
        assert results[0] == "ok"
        return events

    blocked = asyncio.run(observe_schedule(program("async_wait_blocks")))
    cooperative = asyncio.run(observe_schedule(program("async_wait_yields")))
    assert blocked == ["started", "finished", "observer"]
    assert cooperative == ["started", "observer", "finished"]


def test_case_identity_has_no_scripted_answers_and_keeps_pairs_exposed() -> None:
    cases = load_cases()
    families = Counter(case.provenance_family for case in cases)
    assert families and set(families.values()) == {2}
    for family in families:
        assert {case.track for case in cases if case.provenance_family == family} == {
            "grounded-positive",
            "grounded-safe",
        }
    for case in cases:
        fixture = EvalFixture.model_validate_json((ROOT / case.fixture_path).read_bytes())
        assert not fixture.llm_responses
        assert fixture.analyze_responses_by_path is None
        assert case.exposure == "development-exposed"
        assert len(case.expected) == int(case.track == "grounded-positive")
        if case.expected:
            expected = case.expected[0]
            assert context_for_case(case.fixture_path, expected)[0] == "supported"
            assert (
                1
                <= expected.line_start
                <= expected.line_end
                <= len(fixture.files[0].content_head.splitlines())
            )
    pinned = dataset_inputs()
    assert {case.fixture_path for case in cases} <= pinned.keys()
    assert "tests/eval/grounded_cases.py" in pinned
    assert "tests/eval/test_grounded_cases.py" in pinned


def test_changed_grounded_copy_cannot_keep_context_approval(tmp_path: Path) -> None:
    case = next(case for case in load_cases() if case.expected)
    copy = tmp_path / case.fixture_path
    copy.parent.mkdir(parents=True)
    copy.write_bytes((ROOT / case.fixture_path).read_bytes())
    assert context_for_case(copy, case.expected[0])[0] == "supported"
    changed = case.expected[0].model_copy(
        update={
            "line_start": case.expected[0].line_start + 1,
            "line_end": case.expected[0].line_end + 1,
        }
    )
    assert context_for_case(copy, changed)[0] == "unknown"
    copy.write_bytes(copy.read_bytes() + b" ")
    assert context_for_case(copy, case.expected[0])[0] == "unknown"
