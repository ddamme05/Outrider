"""SSRF prompt-reachability probe (opt-in, analyze-direct, real-model spend).

The eval scorecard showed both Sonnet and Haiku over-flag a fixed-host fetch
(`requests.get("https://api.example.com/users/" + user_id)`) as `ssrf` under the
shipped `analyze-v6` prompt, and that v6 did NOT suppress it. This probe answers
ONE question before anyone spends effort on a v7 prompt: is the over-flag
PROMPT-REACHABLE, or is it a model bias to absorb via HITL?

It runs analyze-direct (NO full graph, NO scorecard cost pass) over two fixtures
under a v6 control plus three candidate ssrf phrasings, for both Sonnet and
Haiku, 5 reps each, and records the PER-REP outcome — so a variant that fixes the
FP in some-but-not-all reps reads as noise, not a clean fix.

Winner bar (strict, the operator's rule): a variant wins only if, across BOTH
models AND ALL reps, it produces ZERO fixed-host SSRF false positives AND keeps
real-SSRF recall at 1.0 (the fully caller-controlled `ssrf_user_host` fetch must
still be flagged). If no variant clears the bar, the over-flag is not
prompt-reachable — stop iterating and absorb it.

This is a TEST-TIER experiment. It injects variant ssrf text by monkeypatching
the `outrider.prompts.analyze.SYSTEM_PROMPT_STABLE_PREFIX` constant at runtime;
it NEVER edits production `analyze.VERSION` or the prompt source. A variant only
graduates to a real `analyze-v7` bump if it clears the bar here.
"""

from __future__ import annotations

import json
import os
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

# The ssrf-family finding types whose presence we count. On the clean fixed-host
# fixture ANY of these is a false positive; on the real-SSRF fixture either one
# counts as a recall hit (catching the SSRF, base or escalated).
_SSRF_FAMILY = {FindingType.SSRF, FindingType.SSRF_METADATA}

_REPS = 5

# Two fixtures, each a single changed file. `fixed_host` is CLEAN (any ssrf-family
# finding = FP — the over-flag we want to kill); `user_host` is a REAL SSRF (a
# ssrf-family finding = recall hit; absence = a miss we must not introduce).
_FP_FIXTURE = "ssrf_fixed_host_safe.json"
_TP_FIXTURE = "ssrf_user_host.json"

# --- variant ssrf bullets (replace ONLY the v6 ssrf carve-out at runtime) -----
# Each keeps the "STILL ssrf by ANY means" reach-the-authority clause (so the
# fully caller-controlled fetch still flags) and the ssrf_metadata escalation;
# each STRENGTHENS the fixed-host carve-out with an explicit "even unvalidated".
# No `{`/`}` markers (the prefix placeholder invariant forbids them).

_VARIANT_WORKED_NEGATIVE = """\
- `ssrf` — a server-side request whose DESTINATION (host, port, origin, or
  scheme — WHERE the request goes) is attacker-influenced. Judge by the
  destination, NOT by whether the value is validated.
  DO flag (user controls the host): `requests.get(request.GET["url"])`;
  `requests.get("http://" + user_host + "/x")`; a `?url=`/`?target=` proxy.
  Do NOT flag (host is a hardcoded literal the value cannot escape) — EVEN with
  no validation: `requests.get("https://api.example.com/users/" + user_id)`;
  `requests.get("https://api.example.com/search?q=" + term)`.
  It IS still ssrf when the value can reach the host/port/scheme by ANY means: a
  leading `//` or absolute URL, an `@`, a backslash, an encoded `%2F`/`%40`, a
  `urljoin`/`URL()` absolute value, a user-chosen port/scheme, or a host picked
  by a user-supplied key. When genuinely unsure whether the value can shift the
  destination host, flag ssrf. Check the metadata escalation BEFORE the safe
  case. Use `ssrf_metadata` INSTEAD when the reachable target can be a cloud
  metadata / link-local endpoint (`169.254.169.254`, `metadata.google.internal`)
  or an internal control plane.
"""

_VARIANT_DESTINATION_CONTROL = """\
- `ssrf` — flag ONLY when the user can influence the request's scheme, host, or
  port. A user value appended as a PATH segment or an ordinary query parameter
  of a hardcoded `scheme://host` literal is NOT ssrf — the destination is fixed
  and the value cannot escape it — and this holds EVEN WHEN the value is
  unvalidated (missing validation alone is never ssrf; e.g.
  `requests.get("https://api.example.com/users/" + user_id)` is not ssrf). It IS
  ssrf whenever the value can reach the host/port/scheme by ANY means: a leading
  `//` or absolute URL, an `@`, a backslash, an encoded `%2F`/`%40`, a
  `urljoin`/`URL()` absolute value, a user-chosen port/scheme, a host selected by
  a user-supplied key, or a fixed host that is itself a proxy/fetcher using the
  value as its target (`?url=`, `?target=`). When genuinely unsure whether the
  value can shift the host, flag ssrf. Check the metadata escalation BEFORE the
  safe case. Use `ssrf_metadata` INSTEAD when the reachable target can be a cloud
  metadata / link-local endpoint (`169.254.169.254`, `metadata.google.internal`)
  or an internal control plane.
"""

# Reconfirm candidate: the destination-control RULE + the worked-negative EXAMPLES
# (the two framings that each cleared the first probe), to see if the combination
# is the most robust general wording.
_VARIANT_COMBINED = """\
- `ssrf` — flag ONLY when the user can influence the request's scheme, host, or
  port. A user value appended as a PATH segment or an ordinary query parameter
  of a hardcoded `scheme://host` literal is NOT ssrf — the destination is fixed
  and the value cannot escape it — EVEN when the value is unvalidated.
  DO flag (user controls the host): `requests.get(request.GET["url"])`;
  `requests.get("http://" + user_host + "/x")`; a `?url=`/`?target=` proxy.
  Do NOT flag (host is a hardcoded literal):
  `requests.get("https://api.example.com/users/" + user_id)`;
  `requests.get("https://api.example.com/search?q=" + term)`.
  It IS still ssrf whenever the value can reach the host/port/scheme by ANY means:
  a leading `//` or absolute URL, an `@`, a backslash, an encoded `%2F`/`%40`, a
  `urljoin`/`URL()` absolute value, a user-chosen port/scheme, a host selected by
  a user-supplied key, or a fixed host that is itself a proxy/fetcher using the
  value as its target (`?url=`, `?target=`). When genuinely unsure whether the
  value can shift the host, flag ssrf. Check the metadata escalation BEFORE the
  safe case. Use `ssrf_metadata` INSTEAD when the reachable target can be a cloud
  metadata / link-local endpoint (`169.254.169.254`, `metadata.google.internal`)
  or an internal control plane.
"""

# (label, variant_block | None). None is the v6 control — the live prompt, no patch.
_VARIANTS: tuple[tuple[str, str | None], ...] = (
    ("v6-control", None),
    ("worked-negative", _VARIANT_WORKED_NEGATIVE),
    ("destination-control", _VARIANT_DESTINATION_CONTROL),
    ("combined", _VARIANT_COMBINED),
)


def _v6_ssrf_bounds(prefix: str) -> tuple[int, int]:
    """Locate the v6 `ssrf` bullet inside the live stable prefix by its bullet
    boundaries (start of the `ssrf` bullet → start of the next, `open_redirect`).
    Boundary-based, not text-based, so it survives whitespace drift; fails loud if
    the prompt structure changed (then the probe's variants need re-authoring)."""
    start = prefix.find("- `ssrf` —")
    end = prefix.find("- `open_redirect`", start + 1)
    if start == -1 or end == -1 or start >= end:
        raise AssertionError(
            "could not locate the v6 ssrf bullet boundaries in SYSTEM_PROMPT_STABLE_PREFIX — "
            "the analyze prompt drifted; re-author the probe variants against the new text"
        )
    block = prefix[start:end]
    if "ssrf_metadata" not in block or "169.254.169.254" not in block:
        raise AssertionError("extracted block does not look like the ssrf carve-out")
    return (start, end)


def _variant_prefix(live_prefix: str, variant_block: str) -> str:
    """Splice a variant ssrf bullet into the live stable prefix in place of v6."""
    start, end = _v6_ssrf_bounds(live_prefix)
    spliced = live_prefix[:start] + variant_block + live_prefix[end:]
    assert spliced != live_prefix, "variant splice did not change the prompt"
    assert "- `open_redirect`" in spliced, "variant splice mangled the following bullet"
    return spliced


def _ssrf_count(findings: Iterable[ReviewFinding]) -> int:
    return sum(1 for f in findings if f.finding_type in _SSRF_FAMILY)


class _NoOpExchangePersister:
    """No-op `LLMExchangePersister`: the provider is fail-closed on `persister=None`;
    this probe reads findings off the analyze return, so the exchange persist is
    discarded (no audit, no DB)."""

    async def persist(self, event: object, request: object, response: object) -> None:  # noqa: ARG002
        return None


@pytest.mark.skipif(
    os.environ.get("OUTRIDER_EVAL_REAL_MODELS") != "1",
    reason="SSRF probe spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 to run",
)
async def test_ssrf_prompt_reachability_probe() -> None:
    """OPT-IN real API spend — emits reports/probe/ssrf_reachability.{json,md}.

    Analyze-direct only (no run_review, no cost pass, no DB). For each variant ×
    model × fixture × rep, run the real analyze node with the variant's ssrf text
    monkeypatched in, count ssrf-family findings, and record the per-rep outcome.
    Report-only: the assertion is only that the matrix COMPLETED; the verdict is
    the artifact + the printed decision, read by a human.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY is required for the SSRF prompt-reachability probe")

    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.anthropic_provider import AnthropicProvider  # noqa: PLC0415
    from outrider.llm.config import ModelConfig  # noqa: PLC0415

    cfg = ModelConfig()
    models = (cfg.analyze_model, "claude-haiku-4-5")  # Sonnet baseline + Haiku candidate
    mock_dir = Path("tests/eval/fixtures/mock_github")
    live_prefix = analyze_prompt.SYSTEM_PROMPT_STABLE_PREFIX
    # Validate the v6 boundary + every variant splice BEFORE spending a token.
    _v6_ssrf_bounds(live_prefix)
    variant_prefixes = {
        label: (None if block is None else _variant_prefix(live_prefix, block))
        for label, block in _VARIANTS
    }

    provider = AnthropicProvider(
        api_key=SecretStr(api_key), model_config=cfg, persister=_NoOpExchangePersister()
    )

    rows: list[dict[str, object]] = []
    try:
        for label, _block in _VARIANTS:
            variant_prefix = variant_prefixes[label]
            patch = (
                mock.patch.object(analyze_prompt, "SYSTEM_PROMPT_STABLE_PREFIX", variant_prefix)
                if variant_prefix is not None
                else mock.patch.object(  # control: re-patch with the live value (a no-op swap)
                    analyze_prompt, "SYSTEM_PROMPT_STABLE_PREFIX", live_prefix
                )
            )
            with patch:
                for model in models:
                    for fixture, kind in ((_FP_FIXTURE, "fixed_host"), (_TP_FIXTURE, "user_host")):
                        for rep in range(_REPS):
                            state = state_from_eval_fixture(mock_dir / fixture)
                            findings = await run_analyze_under_model(
                                state, provider=provider, model=model
                            )
                            n = _ssrf_count(findings)
                            outcome = (
                                ("FP" if n else "clean")
                                if kind == "fixed_host"
                                else ("TP" if n else "MISS")
                            )
                            rows.append(
                                {
                                    "variant": label,
                                    "model": model,
                                    "fixture": kind,
                                    "rep": rep,
                                    "ssrf_findings": n,
                                    "outcome": outcome,
                                }
                            )
    finally:
        await provider.aclose()

    # --- per-variant verdict against the strict bar ---------------------------
    verdicts: dict[str, dict[str, object]] = {}
    for label, _block in _VARIANTS:
        v_rows = [r for r in rows if r["variant"] == label]
        fixed_n = [cast("int", r["ssrf_findings"]) for r in v_rows if r["fixture"] == "fixed_host"]
        user_n = [cast("int", r["ssrf_findings"]) for r in v_rows if r["fixture"] == "user_host"]
        fp_zero = all(n == 0 for n in fixed_n)  # 0 FP across both models, all reps
        recall_one = all(n >= 1 for n in user_n)  # recall 1.0 across both models, all reps
        verdicts[label] = {
            "fixed_host_fp_zero": fp_zero,
            "real_ssrf_recall_one": recall_one,
            "wins": bool(fp_zero and recall_one),
            "fixed_host_fp_reps": sum(1 for n in fixed_n if n > 0),
            "real_ssrf_miss_reps": sum(1 for n in user_n if n == 0),
        }

    winners = [label for label, v in verdicts.items() if v["wins"] and label != "v6-control"]
    decision = (
        f"WINNER(S): {winners} — candidate(s) for an analyze-v7 bump (clear the bar on this run; "
        "re-confirm before promoting)."
        if winners
        else "NO WINNER — the fixed-host SSRF over-flag is NOT prompt-reachable "
        "with these variants; treat it as model bias and absorb via HITL + the "
        "relative gate (v6 stays the floor)."
    )

    out_dir = Path("reports") / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "analyze_base_version": analyze_prompt.VERSION,
        "models": list(models),
        "fixtures": {"fixed_host": _FP_FIXTURE, "user_host": _TP_FIXTURE},
        "reps": _REPS,
        "rows": rows,
        "verdicts": verdicts,
        "decision": decision,
    }
    (out_dir / "ssrf_reachability.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "ssrf_reachability.md").write_text(_render_md(artifact), encoding="utf-8")

    print(  # noqa: T201 — operator artifact pointer
        f"\nSSRF PROBE — REPORT ONLY: wrote {out_dir}/ssrf_reachability.{{json,md}}\n{decision}"
    )
    # Report-only: assert only that the full matrix ran (a row per cell).
    assert len(rows) == len(_VARIANTS) * len(models) * 2 * _REPS


def test_probe_variants_splice_against_live_prompt() -> None:
    """Non-paid guard (runs in the normal eval suite): the boundary markers locate
    the v6 ssrf bullet in the LIVE prompt and every variant splices cleanly without
    introducing `{`/`}` placeholder markers. If the analyze prompt drifts so the
    markers miss, THIS fails — before anyone spends tokens on the gated probe."""
    prefix = analyze_prompt.SYSTEM_PROMPT_STABLE_PREFIX
    _v6_ssrf_bounds(prefix)  # raises (fails loud) if the boundaries are gone
    spliced_any = False
    for _label, block in _VARIANTS:
        if block is None:  # v6 control, no splice
            continue
        assert "{" not in block and "}" not in block, "variant must not add placeholder markers"
        spliced = _variant_prefix(prefix, block)
        assert spliced != prefix
        assert "ssrf_metadata" in spliced  # the variant kept the escalation clause
        assert "169.254.169.254" in spliced  # ...and the metadata indicator
        assert "- `open_redirect`" in spliced  # the following bullet survived
        spliced_any = True
    assert spliced_any  # at least one real variant exists


def _render_md(artifact: dict[str, object]) -> str:
    """Human-glance markdown: a per-rep table + the per-variant verdicts + decision."""
    lines = [
        "# SSRF prompt-reachability probe",
        "",
        f"- generated: `{artifact['generated_at']}`  ·  base: `{artifact['analyze_base_version']}`",
        f"- models: {artifact['models']}  ·  reps: {artifact['reps']}",
        f"- fixtures: fixed_host=`{_FP_FIXTURE}` (clean → any ssrf = FP), "
        f"user_host=`{_TP_FIXTURE}` (real ssrf → ssrf = recall)",
        "",
        "## Decision",
        "",
        f"**{artifact['decision']}**",
        "",
        "## Per-variant verdict (bar: 0 fixed-host FP AND recall 1.0, both models, all reps)",
        "",
        "| variant | wins | fixed-host FP reps | real-SSRF miss reps |",
        "|---|---|---|---|",
    ]
    verdicts: dict[str, dict[str, object]] = artifact["verdicts"]  # type: ignore[assignment]
    for label, v in verdicts.items():
        lines.append(
            f"| {label} | {'✅' if v['wins'] else '❌'} | "
            f"{v['fixed_host_fp_reps']} | {v['real_ssrf_miss_reps']} |"
        )
    lines += [
        "",
        "## Per-rep outcomes",
        "",
        "| variant | model | fixture | rep | ssrf findings | outcome |",
        "|---|---|---|---|---|---|",
    ]
    rows: list[dict[str, object]] = artifact["rows"]  # type: ignore[assignment]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['model']} | {r['fixture']} | {r['rep']} | "
            f"{r['ssrf_findings']} | {r['outcome']} |"
        )
    return "\n".join(lines) + "\n"
