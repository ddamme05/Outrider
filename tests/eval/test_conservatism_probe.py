"""Broad-conservatism probe (opt-in, analyze-direct, real-model spend).

The v7 scorecard showed the SSRF fix worked but the over-flag REMAPPED rather than
vanished: Haiku still manufactures a finding on clean-ish code, just at lower severity
(`ssrf` HIGH -> `missing_input_validation` MEDIUM / `missing_error_handling` LOW). The
remaining precision tax is NOT SSRF-specific — it is Haiku's baseline over-eagerness, and
a per-finding-type prompt fix just plays whack-a-mole.

This probe tests whether a BROAD "don't manufacture findings on clean code" rule reduces
over-flagging ACROSS finding types WITHOUT losing recall. The recall gate is the point:
"don't manufacture" can easily become "miss subtle issues", so unlike the SSRF probe (one
true-positive fixture) this guards recall with a DIVERSE set of real-finding fixtures.

Metric, per variant, across both models and all reps:
- FALSE POSITIVES — every finding on a CLEAN fixture (those expect none), plus every
  non-expected finding on a TRUE-POSITIVE fixture (the over-flag also rides along there).
- RECALL — each true-positive fixture's expected finding-type must STILL be present.

A variant WINS only if it cuts total false positives vs the v7 control AND loses ZERO
recall (every expected type caught, both models, every rep). A winner is a candidate for an
`analyze-v8` bump ONLY after a 5-rep reconfirm; if nothing clears cleanly, the over-eagerness
is not cheaply prompt-reducible — stop churning and move to the next roadmap item (GLM).

Test-tier: candidate rules are injected by APPENDING them to
`outrider.prompts.analyze.SYSTEM_PROMPT_STABLE_PREFIX` at runtime. Production
`analyze.VERSION` is never touched; promotion to v8 is a separate, evidence-gated step.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest import mock

import pytest

import outrider.prompts.analyze as analyze_prompt
from outrider.policy import FindingType

from .model_comparison import run_analyze_under_model, state_from_eval_fixture

if TYPE_CHECKING:
    from collections.abc import Iterable

    from outrider.schemas.review_finding import ReviewFinding

_REPS = 3  # first-pass; a winner is reconfirmed at 5 reps before any v8 bump

# Clean fixtures — these expect NO findings; ANY finding is a false positive.
_CLEAN_FIXTURES = (
    "safe_refactor.json",
    "ssrf_fixed_host_safe.json",
    "safe_parameterized_query.json",
)

# True-positive fixtures — the RECALL GUARD. The expected finding-type must survive the
# conservatism rule; a deliberately DIVERSE spread (injection / authz / ssrf / crypto /
# perf) so a rule that suppresses one class of real finding is caught.
_TP_FIXTURES: tuple[tuple[str, FindingType], ...] = (
    ("pygoat_sql_injection.json", FindingType.SQL_INJECTION),
    ("pygoat_auth_bypass.json", FindingType.AUTH_BYPASS),
    ("ssrf_user_host.json", FindingType.SSRF),
    ("weak_password_hash_md5.json", FindingType.WEAK_PASSWORD_HASH),
    ("n_plus_one_query.json", FindingType.N_PLUS_ONE_QUERY),
)

# --- candidate conservatism rules (appended to the live prefix at runtime) ----
# None of these name a finding-type; they target the over-eagerness itself. No `{`/`}`
# (the prefix placeholder invariant forbids them).

_RULE_CLEAN_IS_COMMON = (
    "\n\nBEFORE YOU FINISH — calibration: most code under review is fine. An EMPTY findings "
    "list is a valid, common, and CORRECT result; clean code should produce no findings. Do "
    "not manufacture a finding to have something to report. If nothing in the diff is "
    "concretely wrong, return no findings.\n"
)

_RULE_DEFENSIBLE = (
    "\n\nBEFORE EMITTING ANY FINDING, require a concrete trigger: name the specific line AND "
    "the input or state that makes it wrong. A generic 'this could be better', 'this lacks "
    "validation', or 'consider handling X' with no concrete failure or exploit path is NOT a "
    "finding — drop it. Real findings have a specific trigger; speculative improvements do "
    "not.\n"
)

_RULE_COMBINED = _RULE_CLEAN_IS_COMMON + _RULE_DEFENSIBLE

# (label, appended_rule | None). None is the v7 control — the live prompt, no append.
_VARIANTS: tuple[tuple[str, str | None], ...] = (
    ("v7-control", None),
    ("clean-is-common", _RULE_CLEAN_IS_COMMON),
    ("defensible-evidence", _RULE_DEFENSIBLE),
    ("combined", _RULE_COMBINED),
)


def _finding_types(findings: Iterable[ReviewFinding]) -> list[FindingType]:
    return [f.finding_type for f in findings]


class _NoOpExchangePersister:
    """No-op `LLMExchangePersister`: the provider is fail-closed on `persister=None`; this
    probe reads findings off the analyze return, so the exchange persist is discarded."""

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        return None


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="conservatism probe spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
async def test_conservatism_probe() -> None:
    """OPT-IN real API spend — emits reports/probe/conservatism.{json,md}.

    Analyze-direct only (no run_review, no cost pass, no DB). For each variant × model ×
    fixture × rep, run the real analyze node with the candidate rule appended, then score
    false positives (clean + non-expected) and recall (expected type present). Report-only:
    the assertion is only that the matrix COMPLETED; the verdict is the artifact + the
    printed decision, read by a human.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the conservatism probe")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415

    cfg = ModelConfig()
    models = (cfg.analyze_model, "claude-haiku-4-5")  # Sonnet baseline + Haiku candidate
    mock_dir = Path("tests/eval/fixtures/mock_github")
    live_prefix = analyze_prompt.SYSTEM_PROMPT_STABLE_PREFIX

    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )

    rows: list[dict[str, object]] = []
    try:
        for label, rule in _VARIANTS:
            prefix = live_prefix if rule is None else live_prefix + rule
            with mock.patch.object(analyze_prompt, "SYSTEM_PROMPT_STABLE_PREFIX", prefix):
                for model in models:
                    for fx in _CLEAN_FIXTURES:
                        for rep in range(_REPS):
                            state = state_from_eval_fixture(mock_dir / fx)
                            found = await run_analyze_under_model(
                                state, provider=provider, model=model
                            )
                            rows.append(
                                {
                                    "variant": label,
                                    "model": model,
                                    "fixture": fx,
                                    "kind": "clean",
                                    "rep": rep,
                                    "fp": len(list(found)),
                                    "recall_hit": True,  # n/a for clean; keeps the schema uniform
                                }
                            )
                    for fx, expected in _TP_FIXTURES:
                        for rep in range(_REPS):
                            state = state_from_eval_fixture(mock_dir / fx)
                            found = await run_analyze_under_model(
                                state, provider=provider, model=model
                            )
                            types = _finding_types(found)
                            rows.append(
                                {
                                    "variant": label,
                                    "model": model,
                                    "fixture": fx,
                                    "kind": "tp",
                                    "rep": rep,
                                    "fp": sum(
                                        1 for t in types if t != expected
                                    ),  # extra over-flags
                                    "recall_hit": expected in types,
                                }
                            )
    finally:
        await provider.aclose()

    # --- per-variant verdict vs the v7 control --------------------------------
    verdicts: dict[str, dict[str, object]] = {}
    for label, _rule in _VARIANTS:
        vr = [r for r in rows if r["variant"] == label]
        clean_fp = sum(cast("int", r["fp"]) for r in vr if r["kind"] == "clean")
        tp_extra_fp = sum(cast("int", r["fp"]) for r in vr if r["kind"] == "tp")
        recall_misses = sum(1 for r in vr if r["kind"] == "tp" and not r["recall_hit"])
        verdicts[label] = {
            "clean_fp": clean_fp,
            "tp_extra_fp": tp_extra_fp,
            "total_fp": clean_fp + tp_extra_fp,
            "recall_misses": recall_misses,
        }

    control_fp = cast("int", verdicts["v7-control"]["total_fp"])
    for label, v in verdicts.items():
        v["wins"] = bool(
            label != "v7-control"
            and cast("int", v["total_fp"]) < control_fp  # fewer false positives than v7
            and cast("int", v["recall_misses"]) == 0  # and ZERO recall loss
        )

    winners = [label for label, v in verdicts.items() if v["wins"]]
    decision = (
        f"WINNER(S): {winners} — fewer false positives than v7 with no recall loss. RECONFIRM "
        "at 5 reps before any analyze-v8 bump (recall loss can hide in a rare rep)."
        if winners
        else "NO WINNER — no rule cut false positives without losing recall. The over-eagerness "
        "is not cheaply prompt-reducible; stop churning, lean on HITL + the relative gate, and "
        "move to the next roadmap item (GLM)."
    )

    out_dir = Path("reports") / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "analyze_base_version": analyze_prompt.VERSION,
        "models": list(models),
        "reps": _REPS,
        "clean_fixtures": list(_CLEAN_FIXTURES),
        "tp_fixtures": {fx: t.value for fx, t in _TP_FIXTURES},
        "rows": rows,
        "verdicts": verdicts,
        "decision": decision,
    }
    (out_dir / "conservatism.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "conservatism.md").write_text(_render_md(artifact), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nCONSERVATISM PROBE — REPORT ONLY: wrote {out_dir}/conservatism.{{json,md}}\n{decision}"
    )
    # Report-only: assert only that the full matrix ran (a row per cell).
    n_fixtures = len(_CLEAN_FIXTURES) + len(_TP_FIXTURES)
    assert len(rows) == len(_VARIANTS) * len(models) * n_fixtures * _REPS


def test_conservatism_variants_are_placeholder_free() -> None:
    """Non-paid guard (runs in the normal eval suite): every candidate rule appends to the
    live prefix without introducing `{`/`}` placeholder markers (the prefix invariant), and
    every recall-guard fixture maps to a real FindingType. Fails before any paid run."""
    for _label, rule in _VARIANTS:
        if rule is None:
            continue
        assert "{" not in rule and "}" not in rule, "a rule must not add placeholder markers"
        assert rule.strip(), "a rule must be non-empty"
    for _fx, expected in _TP_FIXTURES:
        assert isinstance(expected, FindingType)
    assert len(set(_CLEAN_FIXTURES)) == len(_CLEAN_FIXTURES)  # no duplicate fixtures


def _render_md(artifact: dict[str, object]) -> str:
    """Human-glance markdown: per-variant verdict + a per-fixture FP/recall breakdown."""
    lines = [
        "# Broad-conservatism probe",
        "",
        f"- generated: `{artifact['generated_at']}`  ·  base: `{artifact['analyze_base_version']}`",
        f"- models: {artifact['models']}  ·  reps: {artifact['reps']}",
        f"- clean fixtures (any finding = FP): {artifact['clean_fixtures']}",
        f"- recall-guard fixtures: {artifact['tp_fixtures']}",
        "",
        "## Decision",
        "",
        f"**{artifact['decision']}**",
        "",
        "## Per-variant verdict (win = fewer total FP than v7 control AND zero recall miss)",
        "",
        "| variant | clean FP | tp extra FP | total FP | recall misses | wins |",
        "|---|---|---|---|---|---|",
    ]
    verdicts: dict[str, dict[str, object]] = artifact["verdicts"]  # type: ignore[assignment]
    for label, v in verdicts.items():
        lines.append(
            f"| {label} | {v['clean_fp']} | {v['tp_extra_fp']} | {v['total_fp']} | "
            f"{v['recall_misses']} | {'✅' if v['wins'] else '❌'} |"
        )
    lines += [
        "",
        "## Per-(variant, model, fixture): FP count / recall",
        "",
        "| variant | model | fixture | kind | FP (sum) | recall (hits/reps) |",
        "|---|---|---|---|---|---|",
    ]
    rows: list[dict[str, object]] = artifact["rows"]  # type: ignore[assignment]
    agg: dict[tuple[str, str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        key = (
            cast("str", r["variant"]),
            cast("str", r["model"]),
            cast("str", r["fixture"]),
            cast("str", r["kind"]),
        )
        agg[key][0] += cast("int", r["fp"])
        agg[key][1] += 1 if r["recall_hit"] else 0
        agg[key][2] += 1
    for (variant, model, fixture, kind), (fp, hits, reps) in agg.items():
        recall = f"{hits}/{reps}" if kind == "tp" else "—"
        lines.append(f"| {variant} | {model} | {fixture} | {kind} | {fp} | {recall} |")
    return "\n".join(lines) + "\n"
