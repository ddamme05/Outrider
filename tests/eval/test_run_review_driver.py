"""Check that the fixture driver runs a review through the real graph to publication.

Responses and external services are substituted. The graph case requires
--is-eval and postgres-test; the driver creates its own isolated database.
The unsafe-database rejection check runs without a database."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from outrider.agent import eval_driver, run_review
from outrider.agent.eval_driver import EvalRunResult
from outrider.eval_support import EvalModeNotEnabledError

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

    from outrider.llm.base import LLMExchangePersister, LLMRequest, LLMResponse

_FILE = "src/handler.py"
_BASE = "def existing():\n    return 1\n"
_HEAD = "def existing():\n    return 1\n\ndef vulnerable(user_input):\n    return user_input\n"
# Hunks-only — real GitHub /pulls/{n}/files wire shape (coordinates wraps it
# with synthesized headers; post-wrap identical to the smoke test's patch).
_PATCH = (
    "@@ -1,2 +1,5 @@\n"
    " def existing():\n"
    "     return 1\n"
    "+\n"
    "+def vulnerable(user_input):\n"
    "+    return user_input\n"
)
# 1-indexed line of "    return user_input" in _HEAD (the finding target).
_FINDING_LINE = _HEAD[: _HEAD.index("    return user_input")].count("\n") + 1


def _smoke_fixture() -> dict[str, object]:
    return {
        "installation_id": 12345,
        "owner": "acme",
        "repo": "widget",
        "pr_number": 7,
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "pr_title": "Add vulnerable handler",
        "author": "someone",
        "total_additions": 3,
        "total_deletions": 0,
        "files": [
            {
                "path": _FILE,
                "status": "modified",
                "additions": 3,
                "deletions": 0,
                "patch": _PATCH,
                "content_base": _BASE,
                "content_head": _HEAD,
            }
        ],
        "llm_responses": {
            "triage": [
                json.dumps(
                    {
                        "file_tiers": {_FILE: "deep"},
                        "overall_risk": "medium",
                        "relevant_dimensions": ["security"],
                        "reasoning": "deep-review the changed handler.",
                    }
                )
            ],
            "analyze": [
                json.dumps(
                    {
                        "findings": [
                            {
                                "finding_type": "missing_input_validation",
                                "evidence_tier": "judged",
                                "query_match_id": None,
                                "trace_path": None,
                                "title": "Unvalidated user input returned directly",
                                "description": "vulnerable() returns user_input unvalidated.",
                                "evidence": "    return user_input",
                                "line_start": _FINDING_LINE,
                                "line_end": _FINDING_LINE,
                                "trace_candidates": [],
                            }
                        ]
                    }
                )
            ],
            "synthesize": ["One input-validation finding on the new function."],
        },
    }


def _write_fixture(tmp_path: Path, payload: dict[str, object]) -> str:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_run_review_requires_eval_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail-closed before any DB work when OUTRIDER_IS_EVAL is unset (no DB needed)."""
    monkeypatch.delenv("OUTRIDER_IS_EVAL", raising=False)
    fixture_path = _write_fixture(tmp_path, _smoke_fixture())
    with pytest.raises(EvalModeNotEnabledError):
        run_review(fixture_path)


def test_run_review_drives_graph_and_returns_result(tmp_path: Path) -> None:
    """run_review drives the real graph to publish and returns a populated result.

    Needs postgres-test (run_review carves its own ephemeral DB). The internal
    is_eval integrity gate runs inside run_review; a violation would raise here.
    """
    fixture_path = _write_fixture(tmp_path, _smoke_fixture())

    diagnostics: list[dict[str, Any]] = []
    result = run_review(fixture_path, diagnostic_events=diagnostics)

    assert diagnostics
    assert all(row["kind"] == "persisted-audit-event" for row in diagnostics)
    assert all(row["review_id"] == str(result.review_id) and row["is_eval"] for row in diagnostics)
    assert "llm_call" not in {row["event_type"] for row in diagnostics}
    assert "analyze_completed" in {row["event_type"] for row in diagnostics}
    assert [(row["timestamp"], row["event_id"]) for row in diagnostics] == sorted(
        (row["timestamp"], row["event_id"]) for row in diagnostics
    )
    assert isinstance(result, EvalRunResult)
    # MEDIUM finding -> no HITL gate -> publish ran.
    assert result.hitl_gated is False
    # Finding present + iterable/len contract holds.
    assert len(result) >= 1
    assert any(f.finding_type.value == "missing_input_validation" for f in result.findings)
    # Sub-HIGH finding on a changed line -> routed inline -> published.
    assert len(result.published_comments) >= 1
    assert result.review_id is not None

    # FUP-093 Finding-2 pin: the eval driver emits faithful LLMCallEvent rows, and
    # synthesize populates ReviewMetrics by SUMming them — so the aggregates are
    # NON-ZERO, not the false-zero a non-emitting provider double would produce.
    # Each scripted call carries fixed sentinels (100 in / 50 out), so the totals
    # are exact multiples of the call count: the `== 100 * calls` form cannot pass
    # at zero. This is the guard that the driver can't silently report false-zero
    # metrics (and that synthesize's POST-call query counted synthesize's own call,
    # not just the earlier scripted ones).
    metrics = result.review_metrics
    assert metrics is not None
    assert metrics.llm_calls_made is not None and metrics.llm_calls_made > 0
    assert metrics.total_input_tokens == 100 * metrics.llm_calls_made
    assert metrics.total_output_tokens == 50 * metrics.llm_calls_made
    assert metrics.total_cost_usd is not None and metrics.total_cost_usd > 0


def test_failed_review_retains_partial_diagnostics_and_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _smoke_fixture()
    fixture_path = _write_fixture(tmp_path, fixture)
    failure = RuntimeError("injected provider failure")
    diagnostics: list[dict[str, Any]] = []
    verified_gates: list[bool] = []
    original_gate = eval_driver._run_is_eval_gate
    original_export = eval_driver._export_review_diagnostics

    def provider_factory(persister: LLMExchangePersister):
        scripted = eval_driver._FixtureScriptedProvider(
            fixture["llm_responses"],
            persister=persister,
            triage_paths=tuple(file["path"] for file in fixture["files"]),
        )

        class FailingProvider:
            async def complete(self, request: LLMRequest) -> LLMResponse:
                if request.node_id == "analyze":
                    raise failure
                return await scripted.complete(request)

            async def aclose(self) -> None:
                await scripted.aclose()

        return FailingProvider()

    async def verified_gate(engine: AsyncEngine) -> None:
        await original_gate(engine)
        verified_gates.append(True)

    async def export_then_fail(
        engine: AsyncEngine, review_id: UUID, output: list[dict[str, Any]]
    ) -> None:
        await original_export(engine, review_id, output)
        raise RuntimeError("postgresql://fake-user:fake-password@localhost/fake-db")

    monkeypatch.setattr(eval_driver, "_run_is_eval_gate", verified_gate)
    monkeypatch.setattr(eval_driver, "_export_review_diagnostics", export_then_fail)
    with pytest.raises(RuntimeError) as raised:
        run_review(fixture_path, provider_factory=provider_factory, diagnostic_events=diagnostics)

    assert raised.value is failure
    assert verified_gates == [True]
    persisted = [row for row in diagnostics if row["kind"] == "persisted-audit-event"]
    assert persisted and all(row["is_eval"] for row in persisted)
    assert len({row["review_id"] for row in persisted}) == 1
    assert "file_examination" in {row["event_type"] for row in persisted}
    assert "llm_call" not in {row["event_type"] for row in persisted}
    assert "analyze_completed" not in {row["event_type"] for row in persisted}
    phases = [
        row["payload"]["marker"]
        for row in persisted
        if row["event_type"] == "review_phase" and row["phase_key"] == f"file:{_FILE}#0"
    ]
    assert phases == ["start"]
    assert diagnostics[-1] == {
        "kind": "diagnostic-export-error",
        "review_id": persisted[0]["review_id"],
        "error_type": "RuntimeError",
    }
    assert "fake-password" not in json.dumps(diagnostics)


def test_diagnostic_export_excludes_same_review_non_eval_metadata(tmp_path: Path) -> None:
    from outrider.audit.events import ReviewPhaseEvent
    from outrider.eval_support import EvalIsolationViolationError

    fixture = _smoke_fixture()
    fixture_path = _write_fixture(tmp_path, fixture)
    diagnostics: list[dict[str, Any]] = []
    injected: list[ReviewPhaseEvent] = []

    def provider_factory(persister):
        scripted = eval_driver._FixtureScriptedProvider(
            fixture["llm_responses"],
            persister=persister,
            triage_paths=tuple(file["path"] for file in fixture["files"]),
        )

        class ContaminatingProvider:
            async def complete(self, request: LLMRequest) -> LLMResponse:
                if not injected:
                    event = ReviewPhaseEvent(
                        review_id=request.review_id,
                        phase_id="non-eval-export-control",
                        node_id="triage",
                        marker="start",
                        phase_key="non-eval-export-control",
                        is_eval=False,
                    )
                    await persister.emit_phase(event)
                    injected.append(event)
                return await scripted.complete(request)

            async def aclose(self) -> None:
                await scripted.aclose()

        return ContaminatingProvider()

    with pytest.raises(EvalIsolationViolationError, match="is_eval discipline violation"):
        run_review(fixture_path, provider_factory=provider_factory, diagnostic_events=diagnostics)

    assert len(injected) == 1
    assert diagnostics and all(row["kind"] == "persisted-audit-event" for row in diagnostics)
    assert all(row["review_id"] == str(injected[0].review_id) for row in diagnostics)
    assert all(row["is_eval"] is True for row in diagnostics)
    assert str(injected[0].event_id) not in {row["event_id"] for row in diagnostics}
    assert "non-eval-export-control" not in json.dumps(diagnostics)
