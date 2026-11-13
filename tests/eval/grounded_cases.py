"""Load exposed synthetic bug/fix pairs with executable label oracles.

Fixture hashes bind context approval. Expectations locate the relevant source
operation; the separate tests execute its behavior without model responses.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from outrider.agent.eval_driver import EvalFixture
from outrider.policy import FindingType, lookup_severity

from .grading import ExpectedFinding

if TYPE_CHECKING:
    from .review_quality import ContextStatus

ROOT = Path(__file__).resolve().parents[2]
DATASET_VERSION = "grounded-development-v3"
EXPOSURE = "development-exposed"
_DIRECTORY = "tests/eval/corpus/grounded"
_HASHES = {
    "request_sql_interpolation": "f56b86bd35516c87beba70e1a968e9186054c3d3b6fdf76d98c1676056a8d42a",
    "request_sql_binding": "d8d90fc3b92736b2952b96fe546926633d59cb1e3270e1aad55a5c32615204aa",
    "json_decode_escape": "940030fc73abb6678ef1b8fdd3b481b9e33f6760b97a77a831d90e131dde1b28",
    "json_decode_response": "542d6ded794ab137e1a60ee9ff6603d481aab42e3dd2e4f6b369388c9395088a",
    "account_check_bypass": "b27ae644d69aecc0f57371990f92c426bc45395212b9a2110bdf4638a9167a77",
    "account_check_required": "0a2a253e3ce45b74b75cd225425f0f2de2e841014cd2ebfb0b0cd57706127cca",
    "async_wait_blocks": "92361954800103320fe42138ed8c254c403547cbb19e07ded9c268b7d29829ef",
    "async_wait_yields": "89d34698473edaa27f9d3400dc34014b0af386696bd8d1ea1e7a3e32c56588ca",
    "json_depth_escape": "580781e808f3cfa2cc3584183ff697a483fb473c26e7aa6c969ed69e6ef2f13c",
    "json_depth_response": "0c5f6849667ebfd9bdcf1f965db7bb85a39bab1cbc6ab6d1ebbd987eece90622",
}
_PAID_SOURCE = (
    "reports/model-refresh/20261031T175134Z-bf6fb496584a/results.json"
    "#sha256=1978336e91cd42d2d4002d69899758802fad8f6c0d548ada57b7b4483135218e"
)
_AUTH_SOURCE = (
    "tests/eval/fixtures/mock_github/pygoat_auth_bypass.json"
    "#sha256=a70cf6663f6e2f2d214be7b6185f323c3a85bef27aef6fbe019222b6fe5fc6cb"
    "; inspiration=context-gap"
)


@dataclass(frozen=True, slots=True)
class GroundedCase:
    id: str
    fixture_path: str
    expected: tuple[ExpectedFinding, ...]
    track: Literal["grounded-positive", "grounded-safe"]
    context_status: ContextStatus
    context_reason: str
    provenance_family: str
    provenance_source: str
    fixture_sha256: str
    exposure: str = EXPOSURE


def _expected_site(source: str, kind: FindingType) -> tuple[int, int]:
    tree = ast.parse(source)
    sites = []
    for node in ast.walk(tree):
        if kind is FindingType.AUTH_BYPASS:
            if isinstance(node, ast.If) and ast.unparse(node.test) == "username == 'admin'":
                sites.append(node)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            sql_site = (
                kind is FindingType.SQL_INJECTION
                and node.func.attr == "execute"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "query"
            )
            json_site = (
                kind is FindingType.MISSING_ERROR_HANDLING
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "json"
                and node.func.attr == "loads"
            )
            blocking_site = (
                kind is FindingType.BLOCKING_CALL_IN_ASYNC
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "time"
                and node.func.attr == "sleep"
            )
            if sql_site or json_site or blocking_site:
                sites.append(node)
    if len(sites) != 1:
        raise ValueError("grounded expectation requires one identifiable source operation")
    return sites[0].lineno, sites[0].end_lineno


def _case(
    case_id: str, *, kind: FindingType, positive: bool, family: str, reason: str, source: str
) -> GroundedCase:
    relative = f"{_DIRECTORY}/{case_id}.json"
    path = ROOT / relative
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != _HASHES[case_id]:
        raise ValueError(f"grounded fixture changed after context review: {relative}")
    fixture = EvalFixture.model_validate_json(content)
    if (
        len(fixture.files) != 1
        or fixture.llm_responses
        or fixture.analyze_responses_by_path is not None
    ):
        raise ValueError("grounded fixtures require one file and no scripted model responses")
    file = fixture.files[0]
    if file.content_head is None:
        raise ValueError("grounded fixtures require executable head source")
    expected = ()
    if positive:
        start, end = _expected_site(file.content_head, kind)
        expected = (
            ExpectedFinding(
                file_path=file.path,
                line_start=start,
                line_end=end,
                finding_type=kind,
                severity=lookup_severity(kind),
            ),
        )
    return GroundedCase(
        id=case_id,
        fixture_path=relative,
        expected=expected,
        track="grounded-positive" if positive else "grounded-safe",
        context_status="supported",
        context_reason=reason,
        provenance_family=family,
        provenance_source=source,
        fixture_sha256=_HASHES[case_id],
    )


def load_cases() -> tuple[GroundedCase, ...]:
    """Return reviewed pairs; keep each family together in any data split."""
    return (
        _case(
            "request_sql_interpolation",
            kind=FindingType.SQL_INJECTION,
            positive=True,
            family="synthetic-request-sql",
            reason=(
                "Request name reaches SQLite query text; "
                "the executable oracle demonstrates additional account rows."
            ),
            source=_PAID_SOURCE + "; inspiration=pygoat_sql_injection",
        ),
        _case(
            "request_sql_binding",
            kind=FindingType.SQL_INJECTION,
            positive=False,
            family="synthetic-request-sql",
            reason=(
                "The request name is a bound SQLite value; "
                "the same attack returns no unrelated account rows."
            ),
            source=_PAID_SOURCE + "; inspiration=pygoat_sql_injection",
        ),
        _case(
            "json_decode_escape",
            kind=FindingType.MISSING_ERROR_HANDLING,
            positive=True,
            family="synthetic-json-request",
            reason=(
                "The HTTP response function handles decoding errors, "
                "but malformed JSON escapes its wrong exception handler."
            ),
            source=_PAID_SOURCE + "; inspiration=missing_error_handling",
        ),
        _case(
            "json_decode_response",
            kind=FindingType.MISSING_ERROR_HANDLING,
            positive=False,
            family="synthetic-json-request",
            reason=(
                "The same HTTP response function catches JSON decoding errors "
                "and returns its client-error response."
            ),
            source=_PAID_SOURCE + "; inspiration=missing_error_handling",
        ),
        _case(
            "account_check_bypass",
            kind=FindingType.AUTH_BYPASS,
            positive=True,
            family="synthetic-account-access",
            reason=(
                "The admin branch returns an account record despite denied credentials; "
                "the executable oracle observes disclosure."
            ),
            source=_AUTH_SOURCE,
        ),
        _case(
            "account_check_required",
            kind=FindingType.AUTH_BYPASS,
            positive=False,
            family="synthetic-account-access",
            reason=(
                "The same request receives an account record after its credential check "
                "succeeds; denied credentials receive 403."
            ),
            source=_AUTH_SOURCE,
        ),
        _case(
            "async_wait_blocks",
            kind=FindingType.BLOCKING_CALL_IN_ASYNC,
            positive=True,
            family="synthetic-async-scheduling",
            reason=(
                "The async job pauses with time.sleep; the concurrent observer "
                "cannot run until the job finishes."
            ),
            source="https://docs.python.org/3.13/library/asyncio-task.html#sleeping",
        ),
        _case(
            "async_wait_yields",
            kind=FindingType.BLOCKING_CALL_IN_ASYNC,
            positive=False,
            family="synthetic-async-scheduling",
            reason=(
                "The async job awaits asyncio.sleep; the same observer runs "
                "before the job finishes. This establishes cooperative scheduling only."
            ),
            source="https://docs.python.org/3.13/library/asyncio-task.html#sleeping",
        ),
        _case(
            "json_depth_escape",
            kind=FindingType.MISSING_ERROR_HANDLING,
            positive=True,
            family="synthetic-json-depth-response",
            reason=(
                "The HTTP response function validates supported body types and handles "
                "decoding errors, but valid deeply nested JSON raises RecursionError "
                "outside that recovery boundary."
            ),
            source=(
                "tests/eval/test_grounded_cases.py"
                "::test_json_depth_pair_is_proved_at_the_response_boundary"
            ),
        ),
        _case(
            "json_depth_response",
            kind=FindingType.MISSING_ERROR_HANDLING,
            positive=False,
            family="synthetic-json-depth-response",
            reason=(
                "The same response function catches RecursionError and returns 400. "
                "This establishes decoder-depth recovery only, not whole-handler safety."
            ),
            source=(
                "tests/eval/test_grounded_cases.py"
                "::test_json_depth_pair_is_proved_at_the_response_boundary"
            ),
        ),
    )


def context_for_case(
    fixture_path: str | Path, expected: ExpectedFinding
) -> tuple[ContextStatus, str] | None:
    """Find exact reviewed context, including an archived copy of a fixture."""
    path = Path(fixture_path)
    matches = [
        key
        for key in _HASHES
        if path.as_posix().endswith(f"/{_DIRECTORY}/{key}.json")
        or path.as_posix() == f"{_DIRECTORY}/{key}.json"
    ]
    if not matches:
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != _HASHES[matches[0]]:
        return "unknown", "Grounded fixture bytes changed after context review."
    case = next(case for case in load_cases() if case.id == matches[0])
    if expected not in case.expected:
        return (
            "unknown",
            "Grounded expectation identity differs from the reviewed source operation.",
        )
    return case.context_status, case.context_reason


def dataset_inputs() -> dict[str, str]:
    """Pin fixture programs, labels, context approvals and executable oracle code."""
    paths = (
        "tests/eval/grounded_cases.py",
        "tests/eval/test_grounded_cases.py",
        f"{_DIRECTORY}/README.md",
        *(f"{_DIRECTORY}/{key}.json" for key in _HASHES),
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
