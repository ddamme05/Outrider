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

Scoring goes through the STRUCTURAL grader (`grade()` / `ExpectedFinding`), not a finding-type
presence check — so a same-type finding at the wrong file/line counts as a recall MISS (it
matches no ground truth) and a duplicate/wrong-location same-type finding counts as a false
positive (`GradeResult.extra`). A type-only check would falsely certify "zero recall loss".

Metric, per variant, across both models and all reps:
- FALSE POSITIVES — `GradeResult.n_false_positives` summed over all fixtures (clean fixtures
  carry empty ground truth, so every finding there is an `extra`; true-positive fixtures
  count any finding that didn't structurally match the expected one).
- RECALL — a true-positive fixture's expected finding must be MATCHED (`not GradeResult.missed`).

A variant WINS only if it cuts total false positives vs the v7 control AND loses ZERO recall
(every expected finding matched, both models, every rep). This is the 5-rep RECONFIRM of the
3-rep winners (`clean-is-common` primary, `combined` backup), with the recall guard WIDENED to
the over-flag-prone types (`missing_input_validation` / `missing_error_handling`) — the subtle
recall a "don't manufacture" rule most threatens. A variant that clears cleanly here is the
`analyze-v8` candidate; if recall cracks or nothing cuts FPs, v7 stays the floor and the next
lever is GLM. Calls run with bounded per-variant concurrency (a Semaphore-capped `gather`) so
the larger matrix finishes in minutes, not ~20.

Test-tier: candidate rules are injected by APPENDING them to
`outrider.prompts.analyze.SYSTEM_PROMPT_STABLE_PREFIX` at runtime. Production
`analyze.VERSION` is never touched; promotion to v8 is a separate, evidence-gated step.
"""

from __future__ import annotations

import asyncio
import html
import json
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest import mock

import pytest

import outrider.prompts.analyze as analyze_prompt
from outrider.policy import FindingType, lookup_severity

from .grading import DEFAULT_LINE_WINDOW, ExpectedFinding, grade
from .model_comparison import run_analyze_under_model, state_from_eval_fixture
from .test_model_comparison import (
    _GROUND_TRUTH_BY_FIXTURE,
    _MISSING_ERROR_HANDLING_FIXTURE,
    _MISSING_INPUT_VALIDATION_FIXTURE,
    _N_PLUS_ONE_FIXTURE,
    _PATH_TRAVERSAL_FIXTURE,
    _PYGOAT_AUTH_FIXTURE,
    _PYGOAT_SQL_FIXTURE,
)

_REPS = 5  # reconfirm depth — recall loss can hide in a rare rep
_CONCURRENCY = 6  # in-flight analyze calls per variant; modest to stay under the API rate limit


def _gt(path: str, line: int, ft: FindingType) -> tuple[ExpectedFinding, ...]:
    """A one-finding ground truth; severity comes from policy (severity-set-by-policy)."""
    return (
        ExpectedFinding(
            file_path=path,
            line_start=line,
            line_end=line,
            finding_type=ft,
            severity=lookup_severity(ft),
        ),
    )


# (fixture filename, structural ground truth, kind). CLEAN fixtures carry empty ground
# truth (any finding is a false positive). TRUE-POSITIVE fixtures are the RECALL GUARD — a
# deliberately DIVERSE spread (injection / authz / ssrf / crypto / perf) so a rule that
# suppresses one class of real finding is caught. The three shared TPs reuse the canonical
# `_GROUND_TRUTH_BY_FIXTURE` registry (single source of truth with the scorecard).
_FIXTURES: tuple[tuple[str, tuple[ExpectedFinding, ...], str], ...] = (
    ("safe_refactor.json", (), "clean"),
    ("ssrf_fixed_host_safe.json", (), "clean"),
    ("safe_parameterized_query.json", (), "clean"),
    ("pygoat_sql_injection.json", _GROUND_TRUTH_BY_FIXTURE[_PYGOAT_SQL_FIXTURE], "tp"),
    ("pygoat_auth_bypass.json", _GROUND_TRUTH_BY_FIXTURE[_PYGOAT_AUTH_FIXTURE], "tp"),
    ("n_plus_one_query.json", _GROUND_TRUTH_BY_FIXTURE[_N_PLUS_ONE_FIXTURE], "tp"),
    ("ssrf_user_host.json", _gt("app/fetch.py", 7, FindingType.SSRF), "tp"),
    (
        "weak_password_hash_md5.json",
        _gt("accounts/auth.py", 5, FindingType.WEAK_PASSWORD_HASH),
        "tp",
    ),
    # Borderline recall guard: missing_input_validation + missing_error_handling are the EXACT
    # types the conservatism rule suppresses as FALSE positives, so guarding their REAL versions
    # is the critical test that "don't manufacture" did not become "miss findings".
    (
        "missing_input_validation.json",
        _GROUND_TRUTH_BY_FIXTURE[_MISSING_INPUT_VALIDATION_FIXTURE],
        "tp",
    ),
    (
        "missing_error_handling.json",
        _GROUND_TRUTH_BY_FIXTURE[_MISSING_ERROR_HANDLING_FIXTURE],
        "tp",
    ),
    ("path_traversal.json", _GROUND_TRUTH_BY_FIXTURE[_PATH_TRAVERSAL_FIXTURE], "tp"),
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
    ("clean-is-common", _RULE_CLEAN_IS_COMMON),  # 3-rep winner; primary v8 candidate
    ("combined", _RULE_COMBINED),  # backup — trades a clean FP for a tp-extra FP
)


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
    """OPT-IN real API spend — emits reports/probe/conservatism.{json,html}.

    Analyze-direct only (no run_review, no cost pass, no DB). For each variant × model ×
    fixture × rep, run the real analyze node with the candidate rule appended, then grade
    structurally for false positives + recall. Report-only: the assertion is only that the
    matrix COMPLETED; the verdict is the artifact + the printed decision, read by a human.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the conservatism probe")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.pricing import normalize_to_pricing_key  # noqa: PLC0415

    cfg = ModelConfig()
    baseline_model, candidate_model = cfg.analyze_model, "claude-haiku-4-5"
    # Guard the meaningless self-comparison (e.g. OUTRIDER_MODEL_ANALYZE_MODEL=Haiku) BEFORE
    # building the provider, so the artifact never claims two-model coverage while proving
    # nothing about Sonnet-vs-Haiku.
    if normalize_to_pricing_key(baseline_model) == normalize_to_pricing_key(candidate_model):
        pytest.fail(
            f"baseline ({baseline_model}) and candidate ({candidate_model}) normalize to the "
            "same model — the probe would prove nothing about Sonnet-vs-Haiku. Unset "
            "OUTRIDER_MODEL_ANALYZE_MODEL (or point it at Sonnet) for the probe."
        )
    models = (baseline_model, candidate_model)
    mock_dir = Path("tests/eval/fixtures/mock_github")
    live_prefix = analyze_prompt.SYSTEM_PROMPT_STABLE_PREFIX

    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )

    rows: list[dict[str, object]] = []
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def _cell(
        *,
        variant: str,
        model: str,
        fixture: str,
        ground_truth: tuple[ExpectedFinding, ...],
        kind: str,
        rep: int,
    ) -> dict[str, object]:
        # The semaphore caps in-flight analyze calls; the per-variant prompt monkeypatch
        # (below) is active around the whole gather, so every call renders the variant prompt.
        async with sem:
            state = state_from_eval_fixture(mock_dir / fixture)
            found = await run_analyze_under_model(state, provider=provider, model=model)
        g = grade(found, ground_truth, line_window=DEFAULT_LINE_WINDOW)
        return {
            "variant": variant,
            "model": model,
            "fixture": fixture,
            "kind": kind,
            "rep": rep,
            "fp": g.n_false_positives,
            "recall_hit": not g.missed,  # clean: vacuously True; tp: expected matched
        }

    try:
        for label, rule in _VARIANTS:
            prefix = live_prefix if rule is None else live_prefix + rule
            # Variants run sequentially (each needs a different prompt patch); WITHIN a
            # variant the cells run concurrently under the active patch — the gather completes
            # before the `with` restores the prompt, so no in-flight call sees the wrong text.
            with mock.patch.object(analyze_prompt, "SYSTEM_PROMPT_STABLE_PREFIX", prefix):
                cells = [
                    _cell(
                        variant=label, model=model, fixture=fx, ground_truth=gt, kind=kind, rep=rep
                    )
                    for model in models
                    for fx, gt, kind in _FIXTURES
                    for rep in range(_REPS)
                ]
                rows.extend(await asyncio.gather(*cells))
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
        f"WINNER(S): {winners} — fewer false positives than v7 with no recall loss across this "
        "5-rep reconfirm (borderline guards included). The primary winner is the analyze-v8 "
        "candidate, subject to the normal rollout (VERSION bump + contract sweep + "
        "security-test re-pin)."
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
        "clean_fixtures": [fx for fx, _gt_, kind in _FIXTURES if kind == "clean"],
        "tp_fixtures": {fx: gt[0].finding_type.value for fx, gt, kind in _FIXTURES if kind == "tp"},
        "rows": rows,
        "verdicts": verdicts,
        "decision": decision,
    }
    (out_dir / "conservatism.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "conservatism.html").write_text(_render_html(artifact), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nCONSERVATISM PROBE — REPORT ONLY: {out_dir}/conservatism.{{json,html}}\n{decision}"
    )
    # Report-only: assert only that the full matrix ran (a row per cell).
    assert len(rows) == len(_VARIANTS) * len(models) * len(_FIXTURES) * _REPS


def test_conservatism_probe_fixtures_and_rules_are_valid() -> None:
    """Non-paid guard (runs in the normal eval suite): every candidate rule appends without a
    `{`/`}` placeholder marker (the prefix invariant); every true-positive fixture carries a
    non-empty structural ground truth and every clean fixture an empty one. Fails before any
    paid run if a recall guard or a rule is malformed."""
    for _label, rule in _VARIANTS:
        if rule is None:
            continue
        assert "{" not in rule and "}" not in rule, "a rule must not add placeholder markers"
        assert rule.strip(), "a rule must be non-empty"
    for fx, gt, kind in _FIXTURES:
        if kind == "tp":
            assert gt, f"{fx} is a recall guard but has empty ground truth"
            assert all(isinstance(e, ExpectedFinding) for e in gt)
        else:
            assert gt == (), f"{fx} is a clean fixture but carries ground truth"
    fixtures = [fx for fx, _gt_, _kind in _FIXTURES]
    assert len(set(fixtures)) == len(fixtures)  # no duplicate fixtures


def test_render_html_escapes_and_self_certifies() -> None:
    """Non-paid: _render_html is only called in the gated paid test, so exercise it here — a
    self-contained, HTML-escaped doc that surfaces the decision and highlights a recall miss,
    so a render bug can't hide until a paid run."""
    artifact: dict[str, object] = {
        "generated_at": "2026-06-26T00:00:00Z",
        "analyze_base_version": "analyze-v7",
        "models": ["claude-sonnet-4-6", "claude-haiku-4-5"],
        "reps": 5,
        "clean_fixtures": ["<b>x</b>.json"],  # markup to prove escaping
        "tp_fixtures": {"pygoat_sql_injection.json": "sql_injection"},
        "rows": [
            {
                "variant": "clean-is-common",
                "model": "claude-haiku-4-5",
                "fixture": "missing_input_validation.json",
                "kind": "tp",
                "rep": 0,
                "fp": 0,
                "recall_hit": False,  # a recall MISS -> its row is highlighted
            },
        ],
        "verdicts": {
            "clean-is-common": {
                "clean_fp": 0,
                "tp_extra_fp": 0,
                "total_fp": 0,
                "recall_misses": 1,
                "wins": False,
            },
        },
        "decision": "NO WINNER — recall cracked",
    }
    out = _render_html(artifact)
    assert out.startswith("<!DOCTYPE html>") and out.rstrip().endswith("</html>")
    assert "analyze-v7" in out and "NO WINNER — recall cracked" in out
    assert "&lt;b&gt;x&lt;/b&gt;.json" in out  # markup escaped
    assert "<b>x</b>" not in out  # raw markup never reaches the doc
    assert 'class="miss"' in out  # the recall miss highlights its row


_HTML_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conservatism probe</title>
<style>
  body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    max-width: 1120px; margin: 2rem auto; padding: 0 1.25rem; line-height: 1.45; color: #1b1f24; }
  h1 { font-size: 1.5rem; margin: 0 0 .15rem; }
  h2 { font-size: 1.1rem; margin: 1.6rem 0 .5rem; border-bottom: 2px solid #e3e6ea; }
  .muted { color: #6a737d; font-size: .82rem; }
  .decision { margin: 1rem 0; padding: .8rem 1rem; border-radius: .5rem; background: #f6f8fa;
    border-left: 4px solid #24292f; font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: .82rem; margin-bottom: .4rem; }
  th, td { padding: .38rem .55rem; text-align: left; border-bottom: 1px solid #eceff1; }
  th { background: #24292f; color: #fff; font-weight: 600; white-space: nowrap; }
  td { white-space: nowrap; }
  tbody tr:nth-child(even) { background: #f6f8fa; }
  tr.miss td { background: #ffe3e3; }
  .badge { display: inline-block; padding: .04rem .45rem; border-radius: .7rem; font-weight: 700; }
  .badge.pass { background: #1a7f37; color: #fff; }
  .badge.fail { background: #6a737d; color: #fff; }
  code { background: #f3f4f6; padding: .05rem .25rem; border-radius: .25rem; }
</style>
</head>
<body>
"""
_HTML_TAIL = "</body>\n</html>\n"


def _render_html(artifact: dict[str, object]) -> str:
    """Self-contained HTML artifact (inline CSS, escaped cells) — the human-glance probe
    report. A true-positive RECALL MISS (the load-bearing failure) highlights its row red."""

    def esc(x: object) -> str:
        return html.escape(str(x))

    verdicts: dict[str, dict[str, object]] = artifact["verdicts"]  # type: ignore[assignment]
    rows: list[dict[str, object]] = artifact["rows"]  # type: ignore[assignment]

    vrows = ""
    for label, v in verdicts.items():
        badge = (
            '<span class="badge pass">WIN</span>'
            if v["wins"]
            else '<span class="badge fail">—</span>'
        )
        vrows += (
            f"<tr><td>{esc(label)}</td><td>{esc(v['clean_fp'])}</td>"
            f"<td>{esc(v['tp_extra_fp'])}</td><td>{esc(v['total_fp'])}</td>"
            f"<td>{esc(v['recall_misses'])}</td><td>{badge}</td></tr>\n"
        )

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
    brows = ""
    for (variant, model, fixture, kind), (fp, hits, reps) in agg.items():
        recall = f"{hits}/{reps}" if kind == "tp" else "—"
        cls = ' class="miss"' if (kind == "tp" and hits < reps) else ""  # a recall miss = failure
        brows += (
            f"<tr{cls}><td>{esc(variant)}</td><td>{esc(model)}</td><td>{esc(fixture)}</td>"
            f"<td>{esc(kind)}</td><td>{esc(fp)}</td><td>{esc(recall)}</td></tr>\n"
        )

    body = (
        "<h1>Broad-conservatism probe</h1>\n"
        '<p class="muted">'
        f"generated <code>{esc(artifact['generated_at'])}</code> · "
        f"base <code>{esc(artifact['analyze_base_version'])}</code> · "
        f"models {esc(artifact['models'])} · reps {esc(artifact['reps'])}<br>"
        f"clean (any finding = FP): {esc(artifact['clean_fixtures'])}<br>"
        f"recall guards: {esc(artifact['tp_fixtures'])}</p>\n"
        f'<div class="decision">{esc(artifact["decision"])}</div>\n'
        "<h2>Per-variant verdict (win = fewer total FP than v7 AND zero recall miss)</h2>\n"
        "<table><thead><tr><th>variant</th><th>clean FP</th><th>tp extra FP</th>"
        "<th>total FP</th><th>recall misses</th><th>wins</th></tr></thead>\n"
        f"<tbody>\n{vrows}</tbody></table>\n"
        "<h2>Per-(variant, model, fixture): FP / recall</h2>\n"
        "<table><thead><tr><th>variant</th><th>model</th><th>fixture</th><th>kind</th>"
        "<th>FP (sum)</th><th>recall (hits/reps)</th></tr></thead>\n"
        f"<tbody>\n{brows}</tbody></table>\n"
    )
    return _HTML_HEAD + body + _HTML_TAIL
