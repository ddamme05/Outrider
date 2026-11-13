"""Collect changed questions once while keeping the paid control immutable."""

from __future__ import annotations

import json
import shutil
from decimal import Decimal

import pytest

from outrider.agent import candidate_checks

from . import review_check_comparison as comparison
from . import review_check_retry as retry
from .revised_check_inputs import load_previous_checks
from .test_review_check_retry import RecordingCheckProvider, _source


async def _previous(tmp_path, monkeypatch):
    source, plan, _ = _source(tmp_path, monkeypatch)
    directory = tmp_path / "previous"
    await retry.collect_checks(
        source,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=2,
        output=directory,
        provider_factory=lambda _target, recorder: RecordingCheckProvider(recorder),
    )
    return directory


@pytest.mark.asyncio
async def test_unchanged_questions_refuse_and_changed_questions_keep_stable_identity(
    tmp_path, monkeypatch
):
    directory = await _previous(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="unchanged questions"):
        comparison.build_plan(load_previous_checks(directory))
    monkeypatch.setattr(
        candidate_checks, "_SYSTEM", candidate_checks._SYSTEM + "\nRevised wording."
    )
    plan = comparison.build_plan(load_previous_checks(directory))
    copy = tmp_path / "renamed-control"
    shutil.copytree(directory, copy)
    assert (
        comparison.build_plan(load_previous_checks(copy))["comparison_id"] == plan["comparison_id"]
    )
    assert len(plan["pairs"]) == 2


@pytest.mark.asyncio
async def test_collection_bills_only_revisions_preserves_old_and_refuses_repeat(
    tmp_path, monkeypatch
):
    directory = await _previous(tmp_path, monkeypatch)
    original = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*.json")}
    monkeypatch.setattr(
        candidate_checks, "_SYSTEM", candidate_checks._SYSTEM + "\nRevised wording."
    )
    previous = load_previous_checks(directory)
    plan = comparison.build_plan(previous)
    providers = []

    def factory(_target, recorder):
        provider = RecordingCheckProvider(recorder)
        providers.append(provider)
        return provider

    output = tmp_path / "comparison"
    result = await comparison.collect_comparison(
        previous,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=2,
        output=output,
        provider_factory=factory,
    )
    assert result["status"] == "collected"
    assert len(providers) == 1 and len(providers[0].requests) == 2
    assert result["new_check_budget"]["calls"] == 2
    assert result["previous_check_cost_usd"] == "0.0014"
    assert all(
        pair["previous_assessment"] and pair["revised_assessment"] for pair in result["pairs"]
    )
    assert {
        str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*.json")
    } == original
    with pytest.raises(FileExistsError):
        await comparison.collect_comparison(
            previous,
            plan,
            maximum_usd=Decimal("3"),
            maximum_calls=2,
            output=output,
            provider_factory=factory,
        )
    assert len(providers) == 1
    receipts = json.loads((output / "receipts.json").read_text())
    assert "revised-checks/receipts.json" in receipts
    assert "receipts.json" not in receipts


@pytest.mark.asyncio
async def test_free_cli_and_insufficient_cap_do_not_construct_provider(tmp_path, monkeypatch):
    directory = await _previous(tmp_path, monkeypatch)
    monkeypatch.setattr(
        candidate_checks, "_SYSTEM", candidate_checks._SYSTEM + "\nRevised wording."
    )
    previous = load_previous_checks(directory)
    plan = comparison.build_plan(previous)
    monkeypatch.setattr(comparison, "ROOT", tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("free or insufficient-cap collection must not run")

    collect = comparison.collect_comparison
    monkeypatch.setattr(comparison, "collect_comparison", forbidden)
    assert comparison.main(["--previous", str(directory), "--dry-run"]) == 0
    with pytest.raises(ValueError, match="conservative guard"):
        await collect(
            previous,
            plan,
            maximum_usd=Decimal("0.25"),
            maximum_calls=2,
            output=tmp_path / "new",
            provider_factory=forbidden,
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.asyncio
async def test_control_drift_after_first_answer_stops_before_next_paid_attempt(
    tmp_path, monkeypatch
):
    directory = await _previous(tmp_path, monkeypatch)
    monkeypatch.setattr(
        candidate_checks, "_SYSTEM", candidate_checks._SYSTEM + "\nRevised wording."
    )
    previous = load_previous_checks(directory)
    plan = comparison.build_plan(previous)
    providers = []

    class DriftingProvider(RecordingCheckProvider):
        async def complete(self, request):
            response = await super().complete(request)
            if len(self.requests) == 1:
                path = directory / "results.json"
                path.write_text(path.read_text() + "\n")
            return response

    def factory(_target, recorder):
        provider = DriftingProvider(recorder)
        providers.append(provider)
        return provider

    output = tmp_path / "comparison"
    result = await comparison.collect_comparison(
        previous,
        plan,
        maximum_usd=Decimal("3"),
        maximum_calls=2,
        output=output,
        provider_factory=factory,
    )
    assert result["status"] == "incomplete"
    assert len(providers[0].requests) == result["new_check_budget"]["calls"] == 1
    assert Decimal(result["new_check_budget"]["known_usage_based_usd"]) == Decimal("0.0007")
    assert not (output / "revised-checks/calls/000002.started.json").exists()
