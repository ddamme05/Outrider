"""Offline unit tests for the analyze-EXEMPLARS baseline harness DETERMINISTIC CORE
(`tests/eval/exemplar_baseline.py`) PLUS the opt-in PAID runner that drives the four real providers
and freezes / gates the baseline.

The offline tests pin the pre-registered contract so the accept rule cannot drift: N=3, the exact
acceptance set, ≥2/3 majority, the run/provider comparability (fixture identity + SEMANTIC digests,
per-type totals, model/profile identity, role), the fail-closed prompt-identity gate, the persisted
token evidence, and the role-aware ε=0 gate. They run with no models.

The PAID runner (`_collect_real_observations` + `test_freeze_exemplar_baseline` /
`test_gate_shrunk_prompt_against_frozen_baseline`) is gated on `OUTRIDER_EVAL_REAL_MODELS=1` exactly
like `tests/eval/test_glm_scorecard.py`; it spends API tokens and must run under `op run`.

Run offline: uv run pytest tests/eval/test_exemplar_baseline.py --is-eval -v
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest

from .exemplar_baseline import (
    ACCEPTANCE,
    BASELINE_DIR,
    BASETEN_GLM,
    CLAUDE_DEEP,
    CLAUDE_STANDARD,
    FIREWORKS_GLM,
    MEASUREMENT_CONTRACT,
    PRECISION,
    RECALL,
    REQUIRED_REPS,
    SUPPORTING,
    Observation,
    ProviderMeta,
    RunMeta,
    TokenUsage,
    aggregate,
    authoritative_attempt,
    compare,
    cost_objective,
    fixture_content_digest,
    harness_source_digest,
    majority_threshold,
    preflight_comparability,
    provenance_notes,
    read_baseline,
    render_comparison_html,
    render_run_html,
    run_validity,
    token_delta,
    write_attempt,
    write_baseline,
    write_report,
)

_SQLI = "sqli_fx"
_SAFE = "safe_fx"
_DIGESTS = {_SQLI: "d-sqli", _SAFE: "d-safe"}
_ACCEPT = (CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM)


def _usage(total: int | None) -> TokenUsage | None:
    """A TokenUsage summing to `total`, split across all three input-side classes so tests exercise
    the class-aware path (a Claude-shaped call puts most of the prefix in cache_read)."""
    if total is None:
        return None
    cache_read = total // 2
    cache_write = total // 10
    return TokenUsage(
        input_tokens=total - cache_read - cache_write,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


def _obs(provider: str, sqli: int, safe: int, *, tokens: int | None = None) -> list[Observation]:
    """`sqli`/3 detected reps on the recall fixture, `safe`/3 flagged reps on the safe fixture.
    `tokens` (if set) is the per-rep input-side TOTAL, split across the three classes."""
    u = _usage(tokens)
    hits = [Observation(provider, _SQLI, RECALL, "sql_injection", i < sqli, u) for i in range(3)]
    flags = [Observation(provider, _SAFE, PRECISION, "", i < safe, u) for i in range(3)]
    return hits + flags


def _pmeta(
    role: str, model: str = "m1", profile: str = "pc1", accounting: str = "prompt_excludes_cached"
) -> ProviderMeta:
    return ProviderMeta(role, model, profile, accounting)


def _meta(
    prompt: str = "v10",
    *,
    version: str | None = None,
    digest: str | None = None,
    include_baseten: bool = True,
    models: dict[str, str] | None = None,
    drop: str | None = None,
    extra: dict[str, ProviderMeta] | None = None,
) -> RunMeta:
    """`prompt` sets both the version and content digest (distinct namespaces so a mismatch is
    unambiguous); override either independently via `version=` / `digest=`."""
    models = models or {}
    providers: dict[str, ProviderMeta] = {}
    for p in _ACCEPT:
        if p != drop:
            providers[p] = _pmeta(ACCEPTANCE, model=models.get(p, "m1"))
    if include_baseten:
        providers[BASETEN_GLM] = _pmeta(
            SUPPORTING, model=models.get(BASETEN_GLM, "mb"), profile="pcb"
        )
    if extra:
        providers.update(extra)
    return RunMeta(
        n_reps=3,
        prompt_version=version if version is not None else f"ver-{prompt}",
        prompt_digest=digest if digest is not None else f"dig-{prompt}",
        fixture_digests=dict(_DIGESTS),
        providers=providers,
        harness_digest="h-digest",
    )


def _run(meta: RunMeta, spec: dict[str, tuple[int, int]] | None = None) -> dict:
    """Build a full, valid run: every meta provider gets (sqli, safe) counts from `spec`
    (default 3,0 = full recall, no false positive)."""
    spec = spec or {}
    obs: list[Observation] = []
    for p in meta.providers:
        sqli, safe = spec.get(p, (3, 0))
        obs += _obs(p, sqli, safe)
    return aggregate(obs, meta)


# --- majority + aggregate freeze-time contract ---------------------------------------------------
def test_majority_threshold_is_two_thirds_ceil() -> None:
    assert majority_threshold(1) == 1
    assert majority_threshold(3) == 2
    assert majority_threshold(5) == 4
    with pytest.raises(ValueError):
        majority_threshold(0)


def test_aggregate_stores_provenance_and_majority() -> None:
    base = _run(_meta("v10"), {CLAUDE_DEEP: (2, 0), CLAUDE_STANDARD: (1, 2)})
    assert base["schema_version"] == 3
    assert base["measurement_contract"] == MEASUREMENT_CONTRACT
    assert base["harness_digest"] == "h-digest"
    assert base["n_reps"] == 3
    assert base["prompt_version"] == "ver-v10"
    assert base["prompt_digest"] == "dig-v10"
    assert base["fixture_digests"] == _DIGESTS
    deep = base["providers"][CLAUDE_DEEP]  # type: ignore[index]
    assert (
        deep["role"] == ACCEPTANCE and deep["model"] == "m1" and deep["profile_contract"] == "pc1"
    )
    assert deep["recall_by_type"]["sql_injection"] == {
        "passed": 1,
        "total": 1,
    }  # 2/3 majority -> pass
    std = base["providers"][CLAUDE_STANDARD]  # type: ignore[index]
    assert std["recall_by_type"]["sql_injection"] == {"passed": 0, "total": 1}  # 1/3 -> miss
    assert std["fp_count"] == 1  # safe flagged 2/3 -> majority FP


def test_aggregate_requires_exactly_three_reps() -> None:
    meta = _meta()
    bad = RunMeta(
        n_reps=1,
        prompt_version="ver-v10",
        prompt_digest="dig-v10",
        fixture_digests=meta.fixture_digests,
        providers=meta.providers,
    )
    with pytest.raises(ValueError, match="exactly 3"):
        aggregate(_obs(CLAUDE_DEEP, 1, 0)[:1], bad)  # any obs; n_reps check fires first


def test_aggregate_requires_exact_acceptance_set() -> None:
    with pytest.raises(ValueError, match="acceptance set must equal"):
        _run(_meta(drop=FIREWORKS_GLM))  # missing an acceptance provider


def test_aggregate_rejects_unexpected_provider() -> None:
    extra = {"deepinfra-glm": _pmeta(SUPPORTING)}
    with pytest.raises(ValueError, match="unexpected provider"):
        _run(_meta(extra=extra))


def test_aggregate_rejects_fixture_digest_domain_mismatch() -> None:
    meta = _meta()
    meta = meta._replace(fixture_digests={_SQLI: "d-sqli"})  # omit the safe fixture's digest
    with pytest.raises(ValueError, match="fixture_digests must cover"):
        _run(meta)


def test_aggregate_rejects_wrong_rep_count_per_cell() -> None:
    meta = _meta(include_baseten=False, drop=None)
    # give CLAUDE_DEEP only 2 reps on _SQLI; others fine
    obs: list[Observation] = []
    for p in meta.providers:
        obs += _obs(p, 3, 0)
    obs = [o for o in obs if not (o.provider == CLAUDE_DEEP and o.fixture == _SQLI)]
    obs += [Observation(CLAUDE_DEEP, _SQLI, RECALL, "sql_injection", True)] * 2
    with pytest.raises(ValueError, match="expected exactly 3"):
        aggregate(obs, meta)


def test_aggregate_rejects_incomplete_provider_grid() -> None:
    meta = _meta(include_baseten=False)
    obs: list[Observation] = []
    for p in meta.providers:
        obs += _obs(p, 3, 0)
    obs = [
        o for o in obs if not (o.provider == CLAUDE_DEEP and o.fixture == _SAFE)
    ]  # deep skips safe
    with pytest.raises(ValueError, match="must run every fixture"):
        aggregate(obs, meta)


# --- compare: ε=0 gate + run/provider integrity --------------------------------------------------
def test_compare_passes_when_no_regression_and_prompt_differs() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"))  # same quality, different prompt
    v = compare(base, cand)
    assert v["passed"] is True
    assert v["regressions"] == []
    assert v["advisories"] == []


def test_compare_fails_closed_when_prompt_fully_unchanged() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v10"))  # identical VERSION AND content -> no real prompt change under test
    v = compare(base, cand)
    assert v["passed"] is False  # was advisory; now gating (fail-closed per Codex finding 1)
    details = [r["detail"] for r in v["regressions"]]  # type: ignore[union-attr]
    assert any("prompt_version identical" in d for d in details)
    assert any("prompt_digest identical" in d for d in details)


def test_compare_fails_on_reused_version_even_if_content_changed() -> None:
    base = _run(_meta("v10"))  # ver-v10 / dig-v10
    cand = _run(_meta(version="ver-v10", digest="dig-v11"))  # content bumped, VERSION reused
    v = compare(base, cand)
    assert v["passed"] is False  # cache-key discipline: content change MUST bump VERSION
    assert any("prompt_version identical" in r["detail"] for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_reused_content_even_if_version_bumped() -> None:
    base = _run(_meta("v10"))  # ver-v10 / dig-v10
    cand = _run(_meta(version="ver-v11", digest="dig-v10"))  # VERSION bumped, content identical
    v = compare(base, cand)
    assert v["passed"] is False  # a VERSION bump with no content change is not a real change
    assert any("prompt_digest identical" in r["detail"] for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_acceptance_recall_regression() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"), {CLAUDE_DEEP: (1, 0)})  # deep recall 1/3 -> miss
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["provider"] == CLAUDE_DEEP and r["kind"] == "recall" for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_acceptance_fp_increase() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"), {FIREWORKS_GLM: (3, 2)})  # fireworks new FP (2/3 majority)
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["kind"] == "false_positive" for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_supporting_regression_is_advisory_not_gating() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"), {BASETEN_GLM: (1, 3)})  # baseten worse: recall down + FP up
    v = compare(base, cand)
    assert v["passed"] is True  # supporting never vetoes
    assert v["regressions"] == []
    assert len(v["advisories"]) >= 1  # type: ignore[arg-type]
    assert v["providers"][BASETEN_GLM]["ok"] is False  # type: ignore[index]


def test_compare_supporting_missing_is_tolerated() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11", include_baseten=False))  # no baseten in candidate
    v = compare(base, cand)
    assert v["passed"] is True
    assert v["providers"][BASETEN_GLM]["ok"] is None  # type: ignore[index]


def test_compare_fails_on_missing_acceptance_provider() -> None:
    base = _run(_meta("v10"))
    cand = copy.deepcopy(base)
    del cand["providers"][FIREWORKS_GLM]  # candidate dropped an acceptance surface
    v = compare(base, cand)
    assert v["passed"] is False


def test_compare_fails_on_fixture_digest_mismatch() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"))
    cand["fixture_digests"][_SQLI] = "CHANGED"  # a fixture's content changed under a stable label
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(
        r["kind"] == "integrity" and "fixture_digests" in r["detail"] for r in v["regressions"]
    )  # type: ignore[union-attr]


def test_compare_fails_on_model_identity_mismatch() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11", models={CLAUDE_DEEP: "m2"}))  # deep swapped model under stable label
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["provider"] == CLAUDE_DEEP and r["kind"] == "integrity" for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_n_reps_mismatch() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"))
    cand["n_reps"] = 5  # corrupted / different rep count
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["kind"] == "integrity" and "n_reps" in r["detail"] for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_per_type_total_mismatch() -> None:
    base = _run(_meta("v10"))
    cand = _run(_meta("v11"))
    cand["providers"][CLAUDE_DEEP]["recall_by_type"]["sql_injection"]["total"] = 2  # ran fewer/more
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["provider"] == CLAUDE_DEEP and r["kind"] == "integrity" for r in v["regressions"])  # type: ignore[union-attr]


def test_compare_fails_on_role_mismatch() -> None:
    base = _run(_meta("v10"))
    cand = copy.deepcopy(base)
    cand["providers"][CLAUDE_DEEP]["role"] = (
        SUPPORTING  # role flipped (also breaks the acceptance set)
    )
    v = compare(base, cand)
    assert v["passed"] is False


def test_compare_flags_unexpected_candidate_provider() -> None:
    base = _run(_meta("v10"))
    cand = copy.deepcopy(base)
    cand["providers"]["deepinfra-glm"] = copy.deepcopy(cand["providers"][BASETEN_GLM])
    v = compare(base, cand)
    assert v["passed"] is False
    assert any("unexpected candidate providers" in r["detail"] for r in v["regressions"])  # type: ignore[union-attr]


# --- fixture_content_digest: commits to the full semantic contract (Codex finding 2) ------------
def test_fixture_digest_is_deterministic() -> None:
    a = fixture_content_digest(source="x = 1", expected_types=["sql_injection"], is_safe=False)
    b = fixture_content_digest(source="x = 1", expected_types=["sql_injection"], is_safe=False)
    assert a == b
    # order of expected_types must not matter (sorted internally)
    c = fixture_content_digest(source="s", expected_types=["b", "a"], is_safe=False)
    d = fixture_content_digest(source="s", expected_types=["a", "b"], is_safe=False)
    assert c == d


def test_fixture_digest_changes_when_source_changes() -> None:
    base = fixture_content_digest(source="x = 1", expected_types=["t"], is_safe=False)
    assert base != fixture_content_digest(source="x = 2", expected_types=["t"], is_safe=False)


def test_fixture_digest_changes_when_expected_types_change() -> None:
    base = fixture_content_digest(source="s", expected_types=["sql_injection"], is_safe=False)
    relabel = fixture_content_digest(source="s", expected_types=["xss"], is_safe=False)
    assert base != relabel  # relabeling a positive must NOT pass a stable identity


def test_fixture_digest_changes_when_safe_classification_flips() -> None:
    unsafe = fixture_content_digest(source="s", expected_types=[], is_safe=False)
    safe = fixture_content_digest(source="s", expected_types=[], is_safe=True)
    assert unsafe != safe  # reclassifying safe<->unsafe must NOT pass a stable identity


# --- input-side token telemetry persisted in v2, split by class ----------------------------------
def _run_with_tokens(token_by_provider: dict[str, int | None]) -> dict:
    meta = _meta("v10")
    obs: list[Observation] = []
    for p in meta.providers:
        obs += _obs(p, 3, 0, tokens=token_by_provider.get(p))
    return aggregate(obs, meta)


def test_aggregate_persists_per_fixture_and_per_provider_tokens() -> None:
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 400})
    deep = base["providers"][CLAUDE_DEEP]  # type: ignore[index]
    # each provider ran 2 fixtures x 3 reps = 6 calls; per-fixture is 3 reps. _usage(500) splits as
    # input=200 / cache_read=250 / cache_write=50 — the class split is what makes a Claude-shaped
    # (cached-prefix) saving visible at all.
    assert deep["input_side_tokens"] == {
        "expected": 6,
        "observed": 6,
        "missing": 0,
        "total": 3000,
        "by_class": {"input": 1200, "cache_read": 1500, "cache_write": 300},
    }
    assert deep["per_fixture"][_SQLI]["input_side_tokens"] == {
        "expected": 3,
        "observed": 3,
        "missing": 0,
        "total": 1500,
        "by_class": {"input": 600, "cache_read": 750, "cache_write": 150},
        "values": [500, 500, 500],
    }
    assert base["providers"][BASETEN_GLM]["input_side_tokens"]["total"] == 2400  # type: ignore[index]


def test_aggregate_counts_all_three_input_side_classes_not_just_input() -> None:
    # THE Claude case: the cached prefix lands in cache_read and is NET of input_tokens, so an
    # input-only measure would under-count the shrink to ~nothing. `total` must include all three.
    meta = _meta("v10")
    obs: list[Observation] = []
    cached_shaped = TokenUsage(input_tokens=10, cache_read_tokens=5000, cache_write_tokens=0)
    for p in meta.providers:
        obs += [Observation(p, _SQLI, RECALL, "sql_injection", True, cached_shaped)] * 3
        obs += [Observation(p, _SAFE, PRECISION, "", False, cached_shaped)] * 3
    data = aggregate(obs, meta)
    deep = data["providers"][CLAUDE_DEEP]  # type: ignore[index]
    assert deep["input_side_tokens"]["total"] == 6 * 5010  # not 6*10
    assert deep["input_side_tokens"]["by_class"]["cache_read"] == 6 * 5000


def test_aggregate_drops_none_tokens_not_counts_as_zero() -> None:
    # telemetry-absent must NOT deflate the mean to 0 — it is dropped, and `missing` records the
    # gap so a downstream reader can never mistake the shortfall for a measured saving
    base = _run_with_tokens({p: None for p in _ACCEPT} | {BASETEN_GLM: 100})
    deep = base["providers"][CLAUDE_DEEP]  # type: ignore[index]
    assert deep["input_side_tokens"] == {
        "expected": 6,
        "observed": 0,
        "missing": 6,
        "total": 0,
        "by_class": {"input": 0, "cache_read": 0, "cache_write": 0},
    }
    assert deep["per_fixture"][_SQLI]["input_side_tokens"] == {
        "expected": 3,
        "observed": 0,
        "missing": 3,
        "total": 0,
        "by_class": {"input": 0, "cache_read": 0, "cache_write": 0},
        "values": [],
    }


# --- token_delta: prices the saving ONLY on complete paired coverage (Codex finding 1) -----------
def test_token_delta_measures_saving_on_complete_coverage() -> None:
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: 400 for p in _ACCEPT} | {BASETEN_GLM: 400})
    d = token_delta(base, cand)
    deep = d[CLAUDE_DEEP]
    assert deep["status"] == "measured"
    assert deep["baseline_mean_per_call"] == 500
    assert deep["candidate_mean_per_call"] == 400
    assert deep["delta_per_call"] == -100  # negative = the shrink saved input tokens


def test_token_delta_is_inconclusive_when_candidate_telemetry_is_missing() -> None:
    # THE misleading-savings case: candidate reported no usage. total=0 would read as a 100%
    # saving; the completeness gate must refuse to price it instead.
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: None for p in _ACCEPT} | {BASETEN_GLM: None})
    d = token_delta(base, cand)
    assert d[CLAUDE_DEEP]["status"] == "inconclusive"
    assert "incomplete token telemetry" in d[CLAUDE_DEEP]["reason"]
    assert "delta_per_call" not in d[CLAUDE_DEEP]


def test_token_delta_is_inconclusive_when_baseline_telemetry_is_missing() -> None:
    base = _run_with_tokens({p: None for p in _ACCEPT} | {BASETEN_GLM: None})
    cand = _run_with_tokens({p: 400 for p in _ACCEPT} | {BASETEN_GLM: 400})
    d = token_delta(base, cand)
    assert d[CLAUDE_DEEP]["status"] == "inconclusive"


def test_token_delta_is_inconclusive_on_partial_coverage() -> None:
    # one rep of three reported no usage -> the mean is not trustworthy; refuse to price it
    meta = _meta("v10")
    obs: list[Observation] = []
    for p in meta.providers:
        obs += [
            Observation(p, _SQLI, RECALL, "sql_injection", True, _usage(500)),
            Observation(p, _SQLI, RECALL, "sql_injection", True, _usage(500)),
            Observation(p, _SQLI, RECALL, "sql_injection", True, None),  # telemetry gap
            *[Observation(p, _SAFE, PRECISION, "", False, _usage(500)) for _ in range(3)],
        ]
    partial = aggregate(obs, meta)
    assert partial["providers"][CLAUDE_DEEP]["input_side_tokens"]["missing"] == 1  # type: ignore[index]
    full = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    assert token_delta(full, partial)[CLAUDE_DEEP]["status"] == "inconclusive"


def test_token_differences_are_not_gated() -> None:
    # a candidate that uses FEWER tokens (the whole point of the shrink) must not trip any gate,
    # and MORE tokens is also not a correctness regression — tokens are evidence, never a veto
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    meta_after = _meta("v11")
    obs: list[Observation] = []
    for p in meta_after.providers:
        obs += _obs(p, 3, 0, tokens=9999)  # tokens way up
    cand = aggregate(obs, meta_after)
    v = compare(base, cand)
    assert v["passed"] is True
    assert v["regressions"] == []


# --- cost_objective: independent of the quality gate, never inferred from it ---------------------
def test_cost_objective_proven_on_complete_coverage_with_reduction() -> None:
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: 400 for p in _ACCEPT} | {BASETEN_GLM: 400})
    assert cost_objective(base, cand)["status"] == "proven"


def test_cost_objective_not_met_when_tokens_did_not_drop() -> None:
    # quality can be identical and the ε=0 gate can pass, yet the shrink saved nothing
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    obj = cost_objective(base, cand)
    assert obj["status"] == "not_met"
    assert "no measured per-call reduction" in obj["reason"]


def test_cost_objective_inconclusive_when_acceptance_telemetry_incomplete() -> None:
    # the misleading case: no usage reported would look like a 100% saving — must never be "proven"
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: None for p in _ACCEPT} | {BASETEN_GLM: 400})
    assert cost_objective(base, cand)["status"] == "inconclusive"


def test_cost_objective_ignores_supporting_provider() -> None:
    # Baseten is advisory: its cost evidence never decides the objective either way
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    cand = _run_with_tokens({p: 400 for p in _ACCEPT} | {BASETEN_GLM: None})
    assert cost_objective(base, cand)["status"] == "proven"  # despite Baseten telemetry missing


def test_quality_pass_does_not_imply_cost_objective() -> None:
    # the pinned separation: ε=0 passes (no regression) while the cost objective is NOT met
    base = _run_with_tokens({p: 500 for p in _ACCEPT} | {BASETEN_GLM: 500})
    meta_after = _meta("v11")
    obs: list[Observation] = []
    for p in meta_after.providers:
        obs += _obs(p, 3, 0, tokens=600)  # same quality, MORE tokens
    cand = aggregate(obs, meta_after)
    assert compare(base, cand)["passed"] is True
    assert cost_objective(base, cand)["status"] == "not_met"


# --- HTML reports: derived views, never evidence -------------------------------------------------
def test_render_run_html_reports_the_run_facts() -> None:
    data = _run_full(_meta("v10"))
    out = render_run_html(data, title="baseline analyze-v10")
    assert out.startswith("<!doctype html>")
    assert "baseline analyze-v10" in out
    assert "VALID EVIDENCE" in out  # complete telemetry
    assert "ver-v10" in out
    assert "sql_injection" in out  # the recall-by-type matrix
    assert "cache read" in out  # the class split is surfaced, not just the total
    for provider in (CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM, BASETEN_GLM):
        assert provider in out


def test_render_run_html_marks_a_void_run() -> None:
    out = render_run_html(_run_full(_meta("v10"), tokens=None), title="t")
    assert "VOID" in out and "VALID EVIDENCE" not in out


def test_render_run_html_yield_column_distinguishes_unrecorded_from_zero() -> None:
    data = _run_full(_meta("v10"))
    out = render_run_html(data, title="t")
    assert "yield (accepted/attempts)" in out
    assert "6/6" in out  # v3 run: 2 fixtures x 3 reps, none rejected
    v2ish = copy.deepcopy(data)
    for p in v2ish["providers"].values():
        p["structured_output"] = None  # what read_baseline's v2 upgrade produces
    assert "unrecorded (v2)" in render_run_html(v2ish, title="t")


def test_render_html_escapes_untrusted_content() -> None:
    # model ids / fixture paths / reasons flow into the report; none may inject markup
    meta = _meta("v10", models={CLAUDE_DEEP: "<script>alert(1)</script>"})
    out = render_run_html(_run_full(meta), title="t")
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_render_comparison_html_accepts_only_on_both_verdicts() -> None:
    base = _run_full(_meta("v10"))
    good = _run_full(_meta("v11"), tokens=400)  # quality held, tokens down
    assert "ACCEPTED" in render_comparison_html(base, good, title="t")
    assert "NOT ACCEPTED" not in render_comparison_html(base, good, title="t")
    # quality holds but no saving -> cost not_met -> NOT accepted
    no_saving = _run_full(_meta("v11"), tokens=500)
    out = render_comparison_html(base, no_saving, title="t")
    assert "NOT ACCEPTED" in out
    assert "NOT_MET" in out


def test_render_comparison_html_shows_quality_regression() -> None:
    base = _run_full(_meta("v10"))
    worse = _run_full(_meta("v11"), tokens=400, spec={CLAUDE_DEEP: (1, 0)})
    out = render_comparison_html(base, worse, title="t")
    assert "NOT ACCEPTED" in out
    assert "FAIL" in out


def test_write_report_is_overwritable_because_a_view_is_not_evidence(tmp_path, monkeypatch) -> None:
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "REPORT_DIR", tmp_path)
    p1 = write_report("<p>one</p>", label="run")
    p2 = write_report("<p>two</p>", label="run")  # re-render: allowed, unlike a baseline
    assert p1 == p2
    assert p2.read_text(encoding="utf-8") == "<p>two</p>"


# --- structured-output yield capture (schema v3, FUP-219) ----------------------------------------
def test_aggregate_records_structured_output_raw_counts() -> None:
    # RAW counts, never a derived rate: reps whose output was rejected land in `rejected`, the
    # rest in `accepted`; attempts = one single-file attempt per rep.
    meta = _meta("v10")
    obs: list[Observation] = []
    for p in meta.providers:
        rejected = 2 if p == FIREWORKS_GLM else 0
        obs += [
            Observation(p, _SQLI, RECALL, "sql_injection", i >= rejected, None, i < rejected)
            for i in range(3)
        ]
        obs += [Observation(p, _SAFE, PRECISION, "", False, None) for i in range(3)]
    data = aggregate(obs, meta)
    fw = data["providers"][FIREWORKS_GLM]
    assert fw["per_fixture"][_SQLI]["structured_output"] == {
        "attempts": 3,
        "accepted": 1,
        "rejected": 2,
        "void": 0,
    }
    assert fw["structured_output"] == {"attempts": 6, "accepted": 4, "rejected": 2, "void": 0}
    clean = data["providers"][CLAUDE_DEEP]
    assert clean["structured_output"] == {"attempts": 6, "accepted": 6, "rejected": 0, "void": 0}


def test_aggregate_rejects_multi_attempt_reps() -> None:
    # n_rejected > 1 means a multi-file fixture reached the single-attempt cell model
    meta = _meta("v10")
    obs: list[Observation] = []
    for p in meta.providers:
        obs += _obs(p, 3, 0)
    obs[0] = obs[0]._replace(n_rejected=2)
    with pytest.raises(ValueError, match="one structured-output attempt per rep"):
        aggregate(obs, meta)


def test_aggregate_requires_harness_digest() -> None:
    # loud-failure: an artifact that cannot state its producing harness must not freeze (FUP-238)
    meta = _meta("v10")._replace(harness_digest="")
    with pytest.raises(ValueError, match="harness_digest"):
        _run(meta)


def test_aggregate_requires_measurement_contract() -> None:
    meta = _meta("v10")._replace(measurement_contract="")
    with pytest.raises(ValueError, match="measurement_contract"):
        _run(meta)


def test_compare_gates_on_measurement_contract() -> None:
    # provenance (harness_digest) never gates, but the measurement-semantics identity ALWAYS does:
    # two runs collected under different aggregation/grading/majority semantics must not ε=0-compare
    base = _run(_meta("v10"))
    rotated = _run(_meta("v11")._replace(measurement_contract="exemplar-mc-2"))
    v = compare(base, rotated)
    assert v["passed"] is False
    assert any(
        r["kind"] == "integrity" and "measurement_contract" in r["detail"] for r in v["regressions"]
    )


def test_preflight_catches_measurement_contract_drift() -> None:
    base = _run_full(_meta("v10"))
    drifted = _meta("v11")._replace(measurement_contract="exemplar-mc-2")
    reasons = preflight_comparability(base, drifted)
    assert any("measurement_contract" in r for r in reasons)


def test_harness_source_digest_is_stable_sha256() -> None:
    d = harness_source_digest()
    assert d == harness_source_digest()  # deterministic over the on-disk source
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)


def test_compare_never_gates_yield_or_provenance() -> None:
    # yield counts and harness digest are evidence/provenance — a candidate with WORSE yield and a
    # DIFFERENT producing harness still passes ε=0 (recall/FP are the acceptance criteria)
    base = _run(_meta("v10"))
    meta = _meta("v11")._replace(harness_digest="another-harness")
    obs: list[Observation] = []
    for p in meta.providers:
        obs += [
            Observation(p, _SQLI, RECALL, "sql_injection", True, None, n_rejected=1)
            for _ in range(3)
        ]
        obs += [Observation(p, _SAFE, PRECISION, "", False, None) for _ in range(3)]
    worse_yield = aggregate(obs, meta)
    assert worse_yield["providers"][CLAUDE_DEEP]["structured_output"]["rejected"] == 3
    assert compare(base, worse_yield)["passed"] is True


def test_provenance_notes_surface_without_gating() -> None:
    base = _run(_meta("v10"))
    assert provenance_notes(base, _meta("v11")) == []  # same digest -> silent
    mismatch = provenance_notes(base, _meta("v11")._replace(harness_digest="other-harness"))
    assert len(mismatch) == 1 and "harness digest differs" in mismatch[0]
    unrecorded = dict(base)
    unrecorded["harness_digest"] = None  # what a v2 artifact upgrades to
    notes = provenance_notes(unrecorded, _meta("v11"))
    assert len(notes) == 1 and "does not record its producing harness" in notes[0]


def test_run_validity_requires_the_full_acceptance_set() -> None:
    # a partial/hand-made dict must never read as valid evidence — it could otherwise become the
    # authoritative attempt for its prompt identity
    assert run_validity({"providers": {}})["valid"] is False
    partial = _run_full(_meta("v11"))
    partial["providers"].pop(FIREWORKS_GLM)
    verdict = run_validity(partial)
    assert verdict["valid"] is False
    assert "not a complete run" in verdict["reason"]


# --- persistence ---------------------------------------------------------------------------------
def test_read_baseline_rejects_wrong_schema_version(tmp_path, monkeypatch) -> None:
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    stale = _run(_meta("v10"))
    stale["schema_version"] = 999
    (tmp_path / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_baseline("stale")


def _as_v2(data: dict) -> dict:
    """Strip a v3 run down to the exact v2 shape (what aggregate() emitted before the bump)."""
    v2 = copy.deepcopy(data)
    v2["schema_version"] = 2
    del v2["harness_digest"]
    del v2["measurement_contract"]
    for p in v2["providers"].values():
        del p["structured_output"]
        for fx in p["per_fixture"].values():
            del fx["structured_output"]
    return v2


def test_read_baseline_upgrades_v2_in_memory_never_on_disk(tmp_path, monkeypatch) -> None:
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    v2 = _as_v2(_run(_meta("v10")))
    raw = json.dumps(v2, indent=2, sort_keys=True)
    (tmp_path / "frozen-v2.json").write_text(raw, encoding="utf-8")
    up = read_baseline("frozen-v2")
    # upgraded in memory: new fields exist as None = UNRECORDED (distinct from a measured zero) —
    # except measurement_contract, which is the reviewed DECLARATION that v2 semantics == mc-1
    assert up["schema_version"] == 3
    assert up["measurement_contract"] == MEASUREMENT_CONTRACT
    assert up["harness_digest"] is None
    for p in up["providers"].values():
        assert p["structured_output"] is None
        assert all(fx["structured_output"] is None for fx in p["per_fixture"].values())
    # ...and the frozen evidence bytes are untouched
    assert (tmp_path / "frozen-v2.json").read_text(encoding="utf-8") == raw
    # a v2 baseline gates cleanly against a v3 candidate — the upgrade is what keeps the frozen
    # bar usable without a paid re-freeze
    assert compare(up, _run(_meta("v11")))["passed"] is True


def test_frozen_v10_artifact_reads_under_v3_and_is_unchanged_on_disk() -> None:
    # pin against the REAL committed evidence: the immutable v2 artifact must stay readable
    raw = json.loads((BASELINE_DIR / "analyze-v10.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2  # the on-disk artifact is still v2 — never rewritten
    up = read_baseline("analyze-v10")
    assert up["schema_version"] == 3
    assert up["measurement_contract"] == MEASUREMENT_CONTRACT  # the reviewed v2 declaration
    assert up["harness_digest"] is None
    # the frozen quality cells survive the upgrade byte-for-byte (recomputed 2026-07-15)
    fp = {p: m["fp_count"] for p, m in up["providers"].items()}
    assert fp == {CLAUDE_DEEP: 2, CLAUDE_STANDARD: 3, FIREWORKS_GLM: 0, BASETEN_GLM: 0}
    assert all(m["structured_output"] is None for m in up["providers"].values())


def test_baseline_round_trips_to_tracked_dir(tmp_path, monkeypatch) -> None:
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    data = _run(_meta("v10"))
    path = write_baseline(data, label="analyze-v11")
    assert path.name == "analyze-v11.json"
    assert read_baseline("analyze-v11") == data


def test_write_baseline_is_create_once(tmp_path, monkeypatch) -> None:
    # a frozen baseline IS the preregistered bar — re-freezing after results are visible would
    # move the goalposts, so a second write under the same label must fail loudly
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    write_baseline(_run(_meta("v10")), label="analyze-v10")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_baseline(_run(_meta("v10")), label="analyze-v10")


def test_write_attempt_preserves_every_attempt(tmp_path, monkeypatch) -> None:
    # no-cherry-picking: re-running a candidate must NOT bury the earlier result. Each completed
    # attempt lands under its own ordinal, so the full history is the evidence.
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    first = _run(_meta("v11"), {CLAUDE_DEEP: (1, 0)})  # a bad run
    second = _run(_meta("v11"))  # a re-run that looks better
    p1 = write_attempt(first, label_prefix="analyze-v11-candidate")
    p2 = write_attempt(second, label_prefix="analyze-v11-candidate")
    assert p1.name == "analyze-v11-candidate-attempt-1.json"
    assert p2.name == "analyze-v11-candidate-attempt-2.json"
    # the worse first attempt is still on disk, unmodified — it cannot be quietly replaced
    assert read_baseline("analyze-v11-candidate-attempt-1") == first
    assert read_baseline("analyze-v11-candidate-attempt-2") == second


def _run_full(meta: RunMeta, *, tokens: int | None = 500, spec=None) -> dict:
    """A run with per-rep telemetry (so it is VALID) and an optional quality spec."""
    spec = spec or {}
    obs: list[Observation] = []
    for p in meta.providers:
        sqli, safe = spec.get(p, (3, 0))
        obs += _obs(p, sqli, safe, tokens=tokens)
    return aggregate(obs, meta)


def test_run_validity_is_about_measuring_not_about_the_result() -> None:
    # an unfavorable-but-complete run IS a result and stands; a run that failed to MEASURE is not
    unfavorable = _run_full(_meta("v11"), spec={CLAUDE_DEEP: (1, 0)})  # bad recall, full telemetry
    assert run_validity(unfavorable)["valid"] is True
    unmeasured = _run_full(_meta("v11"), tokens=None)  # perfect quality, no telemetry
    assert run_validity(unmeasured)["valid"] is False
    assert "incomplete token telemetry" in run_validity(unmeasured)["reason"]


def test_run_validity_ignores_supporting_provider_telemetry() -> None:
    # Baseten never decides anything, including whether the run measured
    meta = _meta("v11")
    obs: list[Observation] = []
    for p in meta.providers:
        obs += _obs(p, 3, 0, tokens=None if p == BASETEN_GLM else 500)
    assert run_validity(aggregate(obs, meta))["valid"] is True


def test_authoritative_attempt_is_the_first_valid_not_the_latest(tmp_path, monkeypatch) -> None:
    # preserving attempts is NOT enough: without a decision rule, a failed attempt-1 could be
    # re-run until attempt-2 came back green. The FIRST VALID attempt is what decides.
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    assert authoritative_attempt("analyze-v11-candidate") is None  # nothing decided yet
    write_attempt(
        _run_full(_meta("v11"), spec={CLAUDE_DEEP: (1, 0)}), label_prefix="analyze-v11-candidate"
    )
    write_attempt(_run_full(_meta("v11")), label_prefix="analyze-v11-candidate")  # greener re-run
    decided = authoritative_attempt("analyze-v11-candidate")
    assert decided is not None
    assert decided.name == "analyze-v11-candidate-attempt-1.json"  # NOT attempt-2


def test_authoritative_attempt_skips_invalid_attempts(tmp_path, monkeypatch) -> None:
    # the valid-vs-void distinction: a run that failed to MEASURE must not permanently decide the
    # experiment, so it is preserved but skipped and the next VALID attempt becomes authoritative
    from . import exemplar_baseline as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "BASELINE_DIR", tmp_path)
    write_attempt(_run_full(_meta("v11"), tokens=None), label_prefix="analyze-v11-candidate")
    assert authoritative_attempt("analyze-v11-candidate") is None  # void -> still undecided
    write_attempt(_run_full(_meta("v11")), label_prefix="analyze-v11-candidate")
    decided = authoritative_attempt("analyze-v11-candidate")
    assert decided is not None
    assert decided.name == "analyze-v11-candidate-attempt-2.json"  # the first VALID one
    assert (tmp_path / "analyze-v11-candidate-attempt-1.json").exists()  # void one still preserved


# --- preflight_comparability: reject static drift BEFORE the paid loop ----------------------------
def test_preflight_passes_on_a_comparable_planned_run() -> None:
    base = _run_full(_meta("v10"))
    assert preflight_comparability(base, _meta("v11")) == []


def test_preflight_catches_unbumped_prompt_identity() -> None:
    base = _run_full(_meta("v10"))
    reasons = preflight_comparability(base, _meta("v10"))  # nothing changed
    assert any("prompt_version" in r for r in reasons)
    assert any("prompt_digest" in r for r in reasons)


def test_preflight_catches_model_and_fixture_drift() -> None:
    base = _run_full(_meta("v10"))
    swapped = _meta("v11", models={CLAUDE_DEEP: "m2"})
    assert any("claude-deep.model" in r for r in preflight_comparability(base, swapped))
    moved = _meta("v11")._replace(fixture_digests={_SQLI: "CHANGED", _SAFE: "d-safe"})
    assert any("fixture_digests" in r for r in preflight_comparability(base, moved))


def test_preflight_catches_token_accounting_drift() -> None:
    base = _run_full(_meta("v10"))
    meta = _meta("v11")
    flipped = dict(meta.providers)
    flipped[CLAUDE_DEEP] = _pmeta(ACCEPTANCE, accounting="prompt_includes_cached")  # noqa: S106
    reasons = preflight_comparability(base, meta._replace(providers=flipped))
    assert any("token_accounting" in r for r in reasons)


def test_compare_fails_on_token_accounting_mismatch() -> None:
    # a §8a mode flip between runs means the token classes were derived differently — the runs
    # are not comparable, even though every other identity matches
    base = _run(_meta("v10"))
    cand = copy.deepcopy(base)
    cand["prompt_version"] = "ver-v11"
    cand["prompt_digest"] = "dig-v11"
    cand["providers"][CLAUDE_DEEP]["token_accounting"] = "prompt_includes_cached"  # noqa: S105
    v = compare(base, cand)
    assert v["passed"] is False
    assert any(r["provider"] == CLAUDE_DEEP and r["kind"] == "integrity" for r in v["regressions"])  # type: ignore[union-attr]


# --- PAID RUNNER: freeze / gate the real baseline (opt-in, real spend) ---------------------------
# Drives the four providers over the 16 recall + 4 safe fixtures × REQUIRED_REPS reps, capturing
# provider-reported input_tokens per call, and feeds aggregate() (which raises unless the run
# satisfies the pre-registration contract). Built against the real scorecard machinery
# (run_analyze_under_model + state_from_eval_fixture + grade) — NOT the pairwise
# compare_models_on_scenario, since per-provider single-model detection is what the baseline needs.
_REAL_MODELS = os.environ.get("OUTRIDER_EVAL_REAL_MODELS") == "1"
_REAL_SKIP = "spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 and run under `op run` to execute"


def _prompt_identity() -> tuple[str, str]:
    """(analyze VERSION, sha256 of the prompt CONTENT) — pure module reads, NO spend.

    Computable before the paid run so both paid tests can pre-flight their "this identity is already
    decided" guards without first burning ~240 calls.
    """
    from outrider.prompts import analyze as analyze_prompt  # noqa: PLC0415

    # Same recipe as llm.base._canonical_system_prompt_hash.
    digest = hashlib.sha256(analyze_prompt.SYSTEM_PROMPT_STABLE_PREFIX.encode("utf-8")).hexdigest()
    return analyze_prompt.VERSION, digest


def _candidate_prefix(version: str, digest: str) -> str:
    """Attempt label prefix, content-addressed by the candidate's own prompt identity."""
    return f"{version}-{digest[:12]}-candidate"


class _InputSideTokenRecorder:
    """Records all THREE input-side token classes per real analyze call, from the `LLMResponse`.

    Reads the RESPONSE, not the event: `LLMCallEvent` carries only input/output/cached and has no
    cache-WRITE field, while `LLMResponse` carries `input_tokens` + `cache_read_tokens` +
    `cache_write_tokens`. The cached analyze prefix (`#042`) lands in the cache classes on Claude
    and is net of `input_tokens`, so an event-sourced (input-only or input+cached) recorder cannot
    measure this shrink on the Claude tiers. Only REAL providers call `persist`.
    """

    def __init__(self) -> None:
        self.records: list[TokenUsage] = []

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        self.records.append(
            TokenUsage(
                input_tokens=getattr(response, "input_tokens", 0),
                cache_read_tokens=getattr(response, "cache_read_tokens", 0),
                cache_write_tokens=getattr(response, "cache_write_tokens", 0),
            )
        )


def _build_run_context() -> tuple[list[tuple], RunMeta]:
    """Everything the paid loop needs, built with NO SPEND: the four providers + their recorders,
    the semantic fixture digests, and the full `RunMeta`.

    Split from the paid loop so every STATIC field (schema/N contract, fixture + ground-truth
    digests, provider set, roles, model ids, profile-contract digests, token-accounting modes) is
    known BEFORE the ~240 calls — `preflight_comparability()` can then reject a drifted run for
    free instead of surfacing it as an integrity regression after the spend.

    Skips (not fails) if any provider key is missing/unresolved.
    """
    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.host_profiles import (  # noqa: PLC0415
        ANTHROPIC_CONTRACT_DIGEST,
        BASETEN_PROFILE,
        FIREWORKS_PROFILE,
        TokenAccounting,
    )
    from outrider.llm.openai_compatible_provider import (  # noqa: PLC0415
        GLM_MODEL_ID,
        GLMProvider,
        OpenAICompatibleProvider,
    )

    from .test_model_comparison import (  # noqa: PLC0415
        _GROUND_TRUTH_BY_FIXTURE,
        _SAFE_CODE_FIXTURES,
    )

    keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "FIREWORKS_API_KEY": os.environ.get("FIREWORKS_API_KEY"),
        "BASETEN_API_KEY": os.environ.get("BASETEN_API_KEY"),
    }
    for name, val in keys.items():
        if not val or val.startswith("op://"):
            pytest.skip(f"{name} (resolved, not an op:// ref) required; run under `op run`")

    cfg = ModelConfig()
    fw_model = ModelConfig.for_host("fireworks").analyze_model  # accounts/fireworks/models/glm-5p2

    # Each logical provider gets its OWN instance + token recorder, so per-rep input_tokens
    # attribute cleanly by snapshot-slicing the recorder around each run. CLAUDE_DEEP and
    # CLAUDE_STANDARD share the AnthropicProvider shape but differ by the per-call model.
    recs = {
        k: _InputSideTokenRecorder()
        for k in (CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM, BASETEN_GLM)
    }
    # Anthropic is not a HostProfile (native path): its `usage.input_tokens` excludes the cache
    # classes, which is exactly the PROMPT_EXCLUDES_CACHED mode. Recorded as provenance only — the
    # wrapper has already normalized every host to the same disjoint representation.
    anthropic_accounting = TokenAccounting.PROMPT_EXCLUDES_CACHED.value
    specs = [
        (
            CLAUDE_DEEP,
            ACCEPTANCE,
            AnthropicProvider(
                api_key=SecretStr(keys["ANTHROPIC_API_KEY"]),
                model_config=cfg,
                persister=recs[CLAUDE_DEEP],
            ),
            cfg.analyze_model,
            ANTHROPIC_CONTRACT_DIGEST,
            anthropic_accounting,
            recs[CLAUDE_DEEP],
        ),
        (
            CLAUDE_STANDARD,
            ACCEPTANCE,
            AnthropicProvider(
                api_key=SecretStr(keys["ANTHROPIC_API_KEY"]),
                model_config=cfg,
                persister=recs[CLAUDE_STANDARD],
            ),
            cfg.standard_analyze_model,
            ANTHROPIC_CONTRACT_DIGEST,
            anthropic_accounting,
            recs[CLAUDE_STANDARD],
        ),
        (
            FIREWORKS_GLM,
            ACCEPTANCE,
            OpenAICompatibleProvider(
                api_key=SecretStr(keys["FIREWORKS_API_KEY"]),
                profile=FIREWORKS_PROFILE,
                persister=recs[FIREWORKS_GLM],
                models=(fw_model,),
            ),
            fw_model,
            FIREWORKS_PROFILE.profile_contract_digest,
            FIREWORKS_PROFILE.token_accounting.value,
            recs[FIREWORKS_GLM],
        ),
        (
            BASETEN_GLM,
            SUPPORTING,
            GLMProvider(api_key=SecretStr(keys["BASETEN_API_KEY"]), persister=recs[BASETEN_GLM]),
            GLM_MODEL_ID,
            BASETEN_PROFILE.profile_contract_digest,
            BASETEN_PROFILE.token_accounting.value,
            recs[BASETEN_GLM],
        ),
    ]

    recall_items = list(_GROUND_TRUTH_BY_FIXTURE.items())  # fixture -> (ExpectedFinding, ...)
    safe_fixtures = list(_SAFE_CODE_FIXTURES)

    # SEMANTIC fixture digests (source + ground-truth types + safe classification) — see Codex
    # finding 2: a source-only digest would let expected labels drift under a stable identity.
    fixture_digests: dict[str, str] = {}
    for fx, gt in recall_items:
        types = sorted({ef.finding_type.value for ef in gt})
        fixture_digests[fx] = fixture_content_digest(
            source=Path(fx).read_text(encoding="utf-8"), expected_types=types, is_safe=False
        )
    for fx in safe_fixtures:
        fixture_digests[fx] = fixture_content_digest(
            source=Path(fx).read_text(encoding="utf-8"), expected_types=[], is_safe=True
        )

    providers_meta: dict[str, ProviderMeta] = {
        key: ProviderMeta(
            role=role, model=model, profile_contract=contract, token_accounting=accounting
        )
        for key, role, _provider, model, contract, accounting, _rec in specs
    }
    version, digest = _prompt_identity()
    meta = RunMeta(
        n_reps=REQUIRED_REPS,
        prompt_version=version,
        prompt_digest=digest,
        fixture_digests=fixture_digests,
        providers=providers_meta,
        harness_digest=harness_source_digest(),
    )
    return specs, meta


async def _collect_real_observations(specs: list[tuple]) -> list[Observation]:
    """THE PAID LOOP: drive all four providers over the fixtures × REQUIRED_REPS reps.

    Per (provider, fixture, rep): run one analyze pass, grade it, and record the summed
    provider-reported input-side usage for that run's LLM calls (snapshot-sliced from the provider's
    own recorder so reps attribute cleanly) plus the structured-output rejection count
    (`n_rejected`, the FUP-219 yield signal — previously discarded). Recall detection = the
    expected finding matched (`not grade.missed`); safe-fixture detection = a false positive was
    produced (`grade.n_false_positives > 0`).

    Each rep is a REAL independent provider call: `run_analyze_under_model` never passes an
    `analyze_cache_store`, and every analyze cache path (scope-resolve / lookup / write) is
    `analyze_cache_store is not None`-guarded, so nothing replays rep 0 for reps 1-2 and the >=2/3
    majority measures real model variance. If a cache store is ever threaded into that helper, the
    N=3 contract breaks and this harness must namespace or disable it.
    """
    from .grading import grade  # noqa: PLC0415
    from .model_comparison import run_analyze_under_model, state_from_eval_fixture  # noqa: PLC0415
    from .test_model_comparison import (  # noqa: PLC0415
        _GROUND_TRUTH_BY_FIXTURE,
        _SAFE_CODE_FIXTURES,
    )

    recall_items = list(_GROUND_TRUTH_BY_FIXTURE.items())
    safe_fixtures = list(_SAFE_CODE_FIXTURES)

    def _run_tokens(rec: _InputSideTokenRecorder, before: int) -> TokenUsage | None:
        """Sum this rep's calls per class. None (not zero) when the host reported no usage."""
        sliced = rec.records[before:]
        if not sliced:
            return None
        return TokenUsage(
            input_tokens=sum(u.input_tokens for u in sliced),
            cache_read_tokens=sum(u.cache_read_tokens for u in sliced),
            cache_write_tokens=sum(u.cache_write_tokens for u in sliced),
        )

    def _single_file_state(fx: str):  # noqa: ANN202
        # The yield accounting records ONE structured-output attempt per rep (`n_rejected` is
        # per-file), so a multi-file fixture would silently skew accepted/rejected — fail loud,
        # same rule as the one-finding-type cell check below.
        state = state_from_eval_fixture(fx)
        n_files = len(state.pr_context.changed_files)
        if n_files != 1:
            raise AssertionError(
                f"{fx} has {n_files} changed files; the yield accounting requires single-file "
                "fixtures (one structured-output attempt per rep)"
            )
        return state

    observations: list[Observation] = []
    for key, _role, provider, model, _contract, _accounting, rec in specs:
        for _rep in range(REQUIRED_REPS):
            for fx, gt in recall_items:
                types = {ef.finding_type.value for ef in gt}
                # The core keys ONE finding_type per (provider, fixture) cell, and detection is
                # `not grade.missed` (all-or-nothing over gt). A multi-type fixture would grade
                # against every type but be recorded under one — fail loud rather than mislabel.
                if len(types) != 1:
                    raise AssertionError(
                        f"{fx} has expected finding-types {sorted(types)}; the baseline cell model "
                        "requires exactly one type per recall fixture"
                    )
                ftype = next(iter(types))
                before = len(rec.records)
                findings, n_rejected = await run_analyze_under_model(
                    _single_file_state(fx), provider=provider, model=model
                )
                gr = grade(findings, gt)
                observations.append(
                    Observation(
                        key,
                        fx,
                        RECALL,
                        ftype,
                        not gr.missed,
                        _run_tokens(rec, before),
                        n_rejected=n_rejected,
                    )
                )
            for fx in safe_fixtures:
                before = len(rec.records)
                findings, n_rejected = await run_analyze_under_model(
                    _single_file_state(fx), provider=provider, model=model
                )
                gr = grade(findings, ())
                observations.append(
                    Observation(
                        key,
                        fx,
                        PRECISION,
                        "",
                        gr.n_false_positives > 0,
                        _run_tokens(rec, before),
                        n_rejected=n_rejected,
                    )
                )
    return observations


@pytest.mark.skipif(not _REAL_MODELS, reason=_REAL_SKIP)
@pytest.mark.asyncio
async def test_freeze_exemplar_baseline() -> None:
    """PAID pre-registration (spec step 1): freeze the CURRENT-prompt baseline BEFORE any shrink.

    aggregate() raises unless the run satisfies the pre-registration contract. Writes the tracked
    artifact labeled by the analyze VERSION; commit it BEFORE editing the prompt.
    """
    # Second explicit opt-in beyond OUTRIDER_EVAL_REAL_MODELS: this spends ~240 calls AND writes the
    # tracked baselines/ tree, so it must not fire during a broad real-models eval sweep.
    if os.environ.get("OUTRIDER_FREEZE_EXEMPLAR_BASELINE") != "1":
        pytest.skip(
            "set OUTRIDER_FREEZE_EXEMPLAR_BASELINE=1 to freeze (spends + writes tracked tree)"
        )
    # Pre-flight BEFORE spending: the identity is a pure module read, so a re-freeze fails here for
    # free instead of after ~240 paid calls.
    version, _digest = _prompt_identity()
    if (BASELINE_DIR / f"{version}.json").exists():
        pytest.fail(
            f"a frozen baseline for {version!r} already exists — it is the preregistered bar and "
            "cannot be re-frozen. To measure a different prompt, bump the analyze VERSION."
        )
    specs, meta = _build_run_context()
    observations = await _collect_real_observations(specs)
    data = aggregate(observations, meta)
    # An INVALID run (incomplete acceptance telemetry) never becomes the bar: it failed to measure,
    # so it is not evidence. Preserve it for the record under a distinct label and leave the
    # canonical label free for a clean re-run — the same void-and-re-run rule the spec applies to
    # an errored rep. A VALID run is frozen and is authoritative from then on.
    validity = run_validity(data)
    if not validity["valid"]:
        void = write_attempt(data, label_prefix=f"{meta.prompt_version}-void")
        pytest.fail(
            f"refusing to freeze an invalid baseline: {validity['reason']}. The run failed to "
            f"MEASURE, so it is not the bar; preserved as {void.name}. Re-run to freeze."
        )
    path = write_baseline(data, label=meta.prompt_version)
    report = write_report(
        render_run_html(data, title=f"Frozen baseline — {meta.prompt_version}"),
        label=meta.prompt_version,
    )
    print(f"\nfrozen baseline: {path}\nreport: {report}")
    assert path.exists()
    assert read_baseline(meta.prompt_version) == data


@pytest.mark.skipif(not _REAL_MODELS, reason=_REAL_SKIP)
@pytest.mark.asyncio
async def test_gate_shrunk_prompt_against_frozen_baseline() -> None:
    """PAID gate (spec step 3): compare the CURRENT (post-shrink) run to the frozen pre-shrink
    baseline named by OUTRIDER_EXEMPLAR_BASELINE_LABEL.

    Fails if any ACCEPTANCE provider regresses (ε=0). The current VERSION+content must both differ
    from the frozen baseline (compare() fails closed otherwise). Skips if the label is unset.

    ACCEPTANCE requires BOTH verdicts, computed independently: the ε=0 quality gate
    (`compare().passed`) AND the cost objective (`cost_objective().status == "proven"`). Quality
    passing alone never accepts — a shrink that holds quality but shows `inconclusive`/`not_met`
    cost has not met its reason for existing. Both dispositions' artifacts are preserved either way.
    """
    label = os.environ.get("OUTRIDER_EXEMPLAR_BASELINE_LABEL")
    if not label:
        pytest.skip("set OUTRIDER_EXEMPLAR_BASELINE_LABEL to the frozen pre-shrink baseline")
    baseline = read_baseline(label)
    # Both guards pre-flight BEFORE spending — the prompt identity is a pure module read.
    version, digest = _prompt_identity()
    if version == label:
        pytest.fail(
            f"analyze VERSION ({version!r}) still equals the frozen baseline label ({label!r}) — "
            "bump the VERSION (and shrink the prompt) before gating"
        )
    # First-VALID-attempt-wins: if this prompt identity already has a valid attempt, THAT one
    # decides — permanently. Re-running and asserting on a fresh result is exactly the
    # cherry-picking the pre-registration forbids. Invalid attempts (failed to measure) are skipped
    # by `authoritative_attempt`, so the legitimate re-run needs no delete-the-file escape hatch.
    prior = authoritative_attempt(_candidate_prefix(version, digest))
    if prior is not None:
        pytest.fail(
            f"{prior.name} is the authoritative attempt for this prompt identity "
            f"({version} / {digest[:12]}) — first-VALID-attempt-wins, permanently. A re-run "
            "cannot supersede it, and deleting it is not a re-decision path (these artifacts "
            "are untracked until committed, so a delete leaves no trace). To measure a "
            "different prompt, bump the VERSION and change the content; to change the rule, "
            "amend the pre-registration."
        )
    # Every STATIC comparability field is knowable now — reject drift for free, not after the spend.
    specs, meta = _build_run_context()
    drift = preflight_comparability(baseline, meta)
    if drift:
        pytest.fail(
            "planned run is not comparable to the frozen baseline:\n  - " + "\n  - ".join(drift)
        )
    # Provenance is surfaced, never gated: an immutable baseline can't be re-frozen to chase
    # harness edits, so a digest mismatch is a fact for the reader, not an integrity failure.
    for note in provenance_notes(baseline, meta):
        print(f"\nprovenance: {note}")
    observations = await _collect_real_observations(specs)
    candidate = aggregate(observations, meta)
    # Persist under a label that is never a canonical baseline name, so a regressed candidate can't
    # masquerade as an accepted baseline. Nothing is overwritten and every attempt is preserved;
    # the guard above is what makes the FIRST one authoritative.
    attempt = write_attempt(candidate, label_prefix=_candidate_prefix(version, digest))
    # The spec's OTHER deliverable, computed as its OWN verdict: `proven` only on complete coverage
    # AND a measured per-call reduction. Never inferred from the quality gate.
    cost = cost_objective(baseline, candidate)
    verdict = compare(baseline, candidate)
    report = write_report(
        render_comparison_html(
            baseline, candidate, title=f"{label} → {meta.prompt_version} (ε=0 gate)"
        ),
        label=f"{meta.prompt_version}-vs-{label}",
    )
    print(f"\nattempt: {attempt.name}\nreport: {report}")
    print(f"quality gate: {'passed' if verdict['passed'] else 'FAILED'}")
    print(f"cost objective: {cost['status']} — {cost['reason']}")
    print(json.dumps(cost["per_provider"], indent=2, default=str))
    # ACCEPTANCE = both. inconclusive/not_met keep their artifacts but never accept.
    assert verdict["passed"], f"ε=0 quality gate failed: {verdict['regressions']}"
    assert cost["status"] == "proven", (
        f"cost objective not proven: {cost['status']} — {cost['reason']}"
    )


# --- runner WIRING guard: exercises the whole free prefix with NO spend --------------------------
# Monkeypatches the ONLY paid call (run_analyze_under_model) and drives the rest — provider
# construction with dummy keys, config accessors, semantic digests, fixture reads, the 4x3x20 loop,
# and aggregate() acceptance. Catches scorecard-signature / constructor drift (the FUP-140 class:
# a wiring break the eval tier catches when unit+integration stay green) without an API call.
@pytest.mark.asyncio
async def test_paid_runner_wiring_is_valid_without_spend(monkeypatch) -> None:
    from . import model_comparison as mc  # noqa: PLC0415

    # Key on the PROVIDER INSTANCE (+ resolved model), never the model alone: each logical provider
    # is its own instance, so instance identity already subsumes the #056 host/profile contract —
    # and a model-only key would spuriously fail a correct runner if a config ever pointed two
    # providers at one slug (e.g. OUTRIDER_MODEL_ANALYZE_MODEL collapsing both Claude tiers).
    calls: list[tuple[int, str]] = []  # (provider instance id, resolved model) per invocation

    # One fixture "rejects" its structured output on every rep, so the guard proves the runner
    # THREADS n_rejected into the right per-fixture cell — a uniform zero would pass even if the
    # runner still discarded the count (Observation.n_rejected defaults to 0).
    rejected_fx = "tests/eval/fixtures/mock_github/cmd_injection_eval_indirect.json"
    rejected_path = "app/calc.py"  # that fixture's single changed file (paths are fixture-unique)

    async def _fake_run(state, *, provider, model):  # noqa: ANN001, ANN202
        calls.append((id(provider), model))
        n_rejected = 1 if state.pr_context.changed_files[0].path == rejected_path else 0
        return (), n_rejected  # no findings — and crucially, no network call / no spend

    monkeypatch.setattr(mc, "run_analyze_under_model", _fake_run)
    for name in ("ANTHROPIC_API_KEY", "FIREWORKS_API_KEY", "BASETEN_API_KEY"):
        monkeypatch.setenv(name, "test-not-a-real-key")

    specs, meta = _build_run_context()
    observations = await _collect_real_observations(specs)

    # 4 providers x REQUIRED_REPS reps x (16 recall + 4 safe) fixtures
    assert len(observations) == 4 * REQUIRED_REPS * 20
    # N=3 must mean THREE separate invocations — not one call reused for the other two. One call
    # per observation, and each of the FOUR distinct providers ran every fixture x REQUIRED_REPS.
    # Combined with aggregate()'s exactly-REQUIRED_REPS-per-(provider,fixture) check below, that
    # pins 3/cell per provider — no host's shortfall can be absorbed by another's surplus.
    # Independence downstream holds because no analyze_cache_store is injected (see the runner).
    assert len(calls) == len(observations)
    by_provider = Counter(calls)
    assert len(by_provider) == 4  # four distinct (instance, model) identities, none conflated
    assert set(by_provider.values()) == {REQUIRED_REPS * 20}
    assert meta.prompt_version  # analyze VERSION resolved
    assert len(meta.prompt_digest) == 64  # sha256 hex of SYSTEM_PROMPT_STABLE_PREFIX
    assert len(meta.fixture_digests) == 20
    assert set(meta.providers) == {CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM, BASETEN_GLM}
    assert meta.providers[BASETEN_GLM].role == SUPPORTING
    assert meta.providers[CLAUDE_DEEP].role == ACCEPTANCE
    # the runner's output must satisfy the freeze-time contract end-to-end
    data = aggregate(observations, meta)
    assert data["schema_version"] == 3
    assert data["harness_digest"] == harness_source_digest()  # the artifact self-records its code
    # faked run makes no LLM calls, so token telemetry is legitimately absent (not zero-faked) and
    # `missing` records every expected rep — which is what makes token_delta refuse to price it
    assert data["providers"][CLAUDE_DEEP]["input_side_tokens"] == {
        "expected": 60,
        "observed": 0,
        "missing": 60,
        "total": 0,
        "by_class": {"input": 0, "cache_read": 0, "cache_write": 0},
    }
    assert token_delta(data, data)[CLAUDE_DEEP]["status"] == "inconclusive"
    # The injected rejection landed in ITS cell (3 reps × 1) and nowhere else, per provider —
    # raw counts, exactly one single-file attempt per rep.
    for prov in (CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM, BASETEN_GLM):
        per_fixture = data["providers"][prov]["per_fixture"]
        assert per_fixture[rejected_fx]["structured_output"] == {
            "attempts": 3,
            "accepted": 0,
            "rejected": 3,
            "void": 0,
        }
        others = sum(
            fx["structured_output"]["rejected"]
            for name, fx in per_fixture.items()
            if name != rejected_fx
        )
        assert others == 0
        assert data["providers"][prov]["structured_output"] == {
            "attempts": 60,
            "accepted": 57,
            "rejected": 3,
            "void": 0,
        }
