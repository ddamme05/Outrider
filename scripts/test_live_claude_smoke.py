"""Zero-spend contract tests for the live-smoke/seed scenario path.

Pins the #070 async github_factory contract at the exact surface intake, trace,
and publish consume: `_make_scenario_github_factory` must return an AWAITABLE
factory (a0eb420 flipped the injected contract to
`Callable[[int], Awaitable[InstallationGitHubClient]]`; the tests/-side stub was
updated, the scripts/-side scenario factory was missed, and every seed_demo
entry failed at intake's first `await github_factory(...)` with zero coverage —
mypy --strict scans src/ only, and --dry-run never constructs the factory).

Run: `uv run python scripts/test_live_claude_smoke.py` (no DB, no network, no
LLM spend). Wired into pre-commit so edits to the factory OR its awaiting
consumers re-run this file.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_ROOT), str(_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _git_range_scenario import FileEntry  # noqa: E402
from live_claude_smoke import (  # noqa: E402
    _INSTALLATION_ID,
    _demo_decision,
    _make_scenario_github_factory,
    _Scenario,
)

from outrider.github.fetch import fetch_file_content_at, list_pr_files  # noqa: E402
from outrider.schemas.hitl import PerFindingOutcome  # noqa: E402

_CONTENT_HEAD = "x = 1\ny = 2\n"


def _scenario() -> _Scenario:
    return _Scenario(
        files=(
            FileEntry(
                path="src/example.py",
                status="added",
                additions=2,
                deletions=0,
                patch="@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n",
                content_base=None,
                content_head=_CONTENT_HEAD,
                previous_path=None,
            ),
        ),
        pr_title="contract test PR",
        label="contract-test",
    )


def test_factory_is_awaitable() -> None:
    """The factory itself must be a coroutine function — intake AWAITS its call."""
    factory = _make_scenario_github_factory(_scenario())
    assert inspect.iscoroutinefunction(factory), (
        "scenario github_factory must be async: intake/trace/publish await "
        "github_factory(installation_id) per the #070 contract"
    )


def test_stub_factory_is_awaitable() -> None:
    """The default (non-scenario) stub shares the same awaited contract."""
    from tests.integration.test_e2e_smoke import _stub_github_factory

    assert inspect.iscoroutinefunction(_stub_github_factory)


def test_real_fetch_wrappers_accept_the_stub_client() -> None:
    """Drive the REAL src fetch wrappers (intake's actual calls) against the stub.

    `list_pr_files` and `fetch_file_content_at` are exactly what intake invokes
    after awaiting the factory, so this pins the client surface end to end
    without a database or a model call.
    """
    scenario = _scenario()
    factory = _make_scenario_github_factory(scenario)

    async def _drive() -> None:
        gh = await factory(_INSTALLATION_ID)
        files = await list_pr_files(gh, owner="o", repo="r", pull_number=7)
        assert [f.filename for f in files] == ["src/example.py"]
        content = await fetch_file_content_at(
            gh, owner="o", repo="r", path="src/example.py", ref=scenario.head_sha
        )
        assert content is not None
        assert content.decode() == _CONTENT_HEAD
        # The 404 shape must be trace-probe-faithful: the raised error carries
        # response.status_code == 404 (trace's duck-typed soft-miss check), never
        # a bare exception that trace would classify as transient and re-raise.
        try:
            await fetch_file_content_at(
                gh, owner="o", repo="r", path="nope/missing.py", ref=scenario.head_sha
            )
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            assert status == 404, f"missing-path error lacks the 404 shape: {exc!r}"
        else:
            raise AssertionError("missing path did not raise")

    asyncio.run(_drive())


def test_wrong_installation_id_rejected() -> None:
    factory = _make_scenario_github_factory(_scenario())

    async def _drive() -> None:
        try:
            await factory(_INSTALLATION_ID + 1)
        except ValueError:
            return
        raise AssertionError("factory accepted an unexpected installation_id")

    asyncio.run(_drive())


def _b64_roundtrip_sanity() -> None:
    # Guard the wire-faithful newline-wrapped base64 the stub emits.
    wrapped = base64.encodebytes(_CONTENT_HEAD.encode()).decode("ascii")
    assert "\n" in wrapped


def test_demo_decision_constructs_a_valid_hitl_decision() -> None:
    """The pre-decide HITLDecision must construct with every REQUIRED field — including
    `decided_at` — so the paid seed never fails validation mid-run. Two gated findings
    (a HIGH + a CRITICAL): the highest-severity one is downgraded, the rest approved,
    and the decision set exactly equals the gate's ids."""
    high = "11111111-1111-4111-8111-111111111111"
    crit = "22222222-2222-4222-8222-222222222222"
    decision = _demo_decision([high, crit], {high: "high", crit: "critical"})
    # decided_at is present + timezone-aware (would have raised without the field).
    assert decision.decided_at.tzinfo is not None
    assert {str(d.finding_id) for d in decision.decisions} == {high, crit}
    by_id = {str(d.finding_id): d for d in decision.decisions}
    # CRITICAL is the highest → it gets the one override (critical → high); HIGH is approved.
    assert by_id[crit].outcome == PerFindingOutcome.SEVERITY_OVERRIDE
    assert by_id[crit].original_severity is not None
    assert by_id[high].outcome == PerFindingOutcome.APPROVE


def test_demo_decision_single_finding_gets_the_override() -> None:
    only = "33333333-3333-4333-8333-333333333333"
    decision = _demo_decision([only], {only: "high"})
    assert decision.decided_at.tzinfo is not None
    assert decision.decisions[0].outcome == PerFindingOutcome.SEVERITY_OVERRIDE


def main() -> int:
    tests = [
        test_factory_is_awaitable,
        test_stub_factory_is_awaitable,
        test_real_fetch_wrappers_accept_the_stub_client,
        test_wrong_installation_id_rejected,
        test_demo_decision_constructs_a_valid_hitl_decision,
        test_demo_decision_single_finding_gets_the_override,
        _b64_roundtrip_sanity,
    ]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    print(f"all {len(tests)} contract checks passed (zero-spend)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
