"""Grounded, zero-spend cost-per-review measurement (COST_ANALYSIS.md P(-1)).

This is the repo's first cost-measurement harness. It drives the REAL 7-node graph
(`run_review`) over representative PR fixtures with a `CostProbe` attached, so every
LLM call's INPUT tokens are counted from the actually-rendered system+user prompt
(via the production `_estimate_tokens`) and priced through the production cost path
(`compute_cost_usd` -> `LLMCallEvent` -> the aggregate SUM). No Anthropic spend.

What it grounds and what it models:
  - INPUT tokens: REAL (the graph builds real prompts from the fixture's file content
    + scope context + system templates). Bounded by `_estimate_tokens` = UTF-8 bytes//3,
    which the docstring documents as a CONSERVATIVE UPPER bound (FUP-049 tracks a
    tokenizer-grade estimate), so the input cost is an over-estimate, not a tight figure.
  - OUTPUT tokens: MODELED. No real completion exists; the measured number uses the
    fixture's scripted-response size, and the report sweeps output to show sensitivity
    (output bills at ~5x input, so it moves the number materially).
  - CACHING: 0 hits. V1 analyze does not drive a stable cache prefix yet (lever
    unconsumed), so this reflects today's code, not the cached projection.
  - MODEL TIER: whatever the graph actually uses today (analyze -> Sonnet for every
    file, regardless of tier — tier-based routing is not wired). So this is the
    Sonnet-everywhere baseline the COST_ANALYSIS naive figure assumes.

Run it explicitly (it spins an ephemeral DB per fixture, so it is slow):

    docker compose up -d postgres-test
    set -a && source <(grep -E '^TEST_' .env) && set +a
    uv run pytest tests/eval/test_cost_measurement.py --is-eval -s

The `-s` shows the printed report; the test also asserts loose sanity bounds so a
gross cost regression (or a broken cost path) fails CI.
"""

from __future__ import annotations

from pathlib import Path

from outrider.agent.eval_driver import CostProbe, run_review
from outrider.agent.nodes.analyze import _estimate_tokens
from outrider.llm.pricing import compute_cost_usd

_FIXTURES = Path(__file__).parent / "fixtures" / "mock_github"

# Representative single-file Python PR fixtures. Each drives triage + analyze + synthesize.
_REPRESENTATIVE = [
    "pygoat_sql_injection.json",
    "n_plus_one_query.json",
    "missing_error_handling.json",
    "pygoat_auth_bypass.json",
    "safe_refactor.json",
]

# COST_ANALYSIS.md reference PR + target bands (2026-05-18).
_REFERENCE_FILES = 30
_BAND_NAIVE = 1.62
_BAND_DEFENSIBLE = 0.56
_BAND_TARGET = 0.21

# Output-token assumptions to sweep (per analyze call). COST_ANALYSIS models 2k->1k.
_OUTPUT_SWEEP = (500, 1000, 2000)


def _price(model: str, input_tokens: int, output_tokens: int) -> float:
    """Re-price one call analytically (no cache) through the production table."""
    return float(
        compute_cost_usd(
            model=model,
            input_tokens=input_tokens,
            cache_write_tokens=0,
            cache_read_tokens=0,
            output_tokens=output_tokens,
        )
    )


def _measure_one(fixture: str) -> list[dict]:
    """Drive one fixture with a cost probe; return its per-call metrics."""
    probe = CostProbe(token_estimator=_estimate_tokens)  # output=None -> from scripted text
    result = run_review(_FIXTURES / fixture, probe=probe)
    # Cross-check: the probe's per-call sum equals the review's aggregate snapshot.
    probe_total = sum(c["cost_usd"] for c in probe.calls)
    if result.review_metrics is not None:
        agg = result.review_metrics.total_cost_usd
        assert abs(probe_total - agg) < 1e-9, (
            f"{fixture}: probe sum {probe_total} != aggregate {agg} — the measurement "
            "diverged from the production rollup path"
        )
    return probe.calls


def test_cost_per_review_measurement() -> None:
    # Sync test on purpose: `run_review` calls `asyncio.run` internally, which would
    # raise inside a pytest-asyncio event loop. The eval scenarios call it the same way.
    runs = {f: _measure_one(f) for f in _REPRESENTATIVE}

    # Per-node unit costs, averaged across fixtures (each is 1 file => 1 analyze call).
    by_node: dict[str, list[dict]] = {}
    for calls in runs.values():
        for c in calls:
            by_node.setdefault(c["node_id"], []).append(c)

    def avg(node: str, key: str) -> float:
        rows = by_node.get(node, [])
        return sum(r[key] for r in rows) / len(rows) if rows else 0.0

    # Models the graph actually used (read, not assumed).
    model_of = {n: rows[0]["model"] for n, rows in by_node.items()}
    haiku = next((m for m in model_of.values() if "haiku" in m), None)

    print("\n" + "=" * 78)
    print("COST-PER-REVIEW MEASUREMENT (grounded input tokens, modeled output, zero spend)")
    print("=" * 78)
    print(f"Fixtures: {len(runs)} single-file Python PRs | reference = {_REFERENCE_FILES}-file PR")
    print(f"{'node':<12}{'model':<22}{'avg_in_tok':>11}{'avg_out_tok':>12}{'avg_cost$':>11}")
    print("-" * 78)
    for node in ("triage", "analyze", "trace", "synthesize"):
        if node not in by_node:
            continue
        print(
            f"{node:<12}{model_of[node]:<22}{avg(node, 'input_tokens'):>11.0f}"
            f"{avg(node, 'output_tokens'):>12.0f}{avg(node, 'cost_usd'):>11.4f}"
        )

    # Composed reference-PR cost: triage(1) + analyze(xN) + synthesize(1), measured output.
    def reference_cost(analyze_output: int | None) -> float:
        triage = avg("triage", "cost_usd")
        synth = avg("synthesize", "cost_usd")
        if analyze_output is None:
            analyze = avg("analyze", "cost_usd")
        else:
            analyze = _price(
                model_of["analyze"], int(avg("analyze", "input_tokens")), analyze_output
            )
        return triage + synth + analyze * _REFERENCE_FILES

    measured_ref = reference_cost(None)
    print("-" * 78)
    print(
        f"Composed {_REFERENCE_FILES}-file review (measured scripted-output): ${measured_ref:.4f}"
    )
    print("Output-token sensitivity (per analyze call):")
    for out in _OUTPUT_SWEEP:
        print(f"    analyze_output={out:>5} tok  ->  ${reference_cost(out):.4f} / review")

    # Cheap lever projection: flip synthesize -> Haiku (one-line config default change).
    if haiku is not None and "synthesize" in by_node:
        synth_sonnet = avg("synthesize", "cost_usd")
        synth_haiku = _price(
            haiku, int(avg("synthesize", "input_tokens")), int(avg("synthesize", "output_tokens"))
        )
        print(
            f"Lever (synthesize Sonnet->Haiku): ${synth_sonnet:.4f} -> ${synth_haiku:.4f} "
            f"(saves ${synth_sonnet - synth_haiku:.4f}/review)"
        )

    print("-" * 78)
    print(f"Bands: naive ${_BAND_NAIVE} | defensible ${_BAND_DEFENSIBLE} | target ${_BAND_TARGET}")
    print("Caveats: input = bytes//3 UPPER bound; output modeled; caching unconsumed;")
    print("         analyze = Sonnet for every file (tier routing not wired).")
    print("=" * 78)

    # Sanity assertions (regression guard — loose, not a tight cost SLA).
    assert by_node, "no LLM calls captured — the cost path is broken"
    assert avg("analyze", "cost_usd") > 0, "analyze cost is zero — pricing/token path broken"
    # A 30-file all-Sonnet review at a realistic 1k output must land in a sane band:
    # well under $100 (catches an order-of-magnitude pricing bug) and above the
    # fully-optimized target (this is the UN-optimized baseline).
    realistic = reference_cost(1000)
    assert _BAND_TARGET < realistic < 100.0, f"reference review cost {realistic} out of sane range"
