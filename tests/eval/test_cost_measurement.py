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
  - CACHING: 0 hits BY CHOICE in the baseline runs — the probe defaults to
    `model_cache=False`, so this measures the uncached cost shape. analyze-v4
    DOES drive a cross-file stable cache prefix (the cache-packing repartition);
    the cache-modeled proof lives in `test_cache_packing_cross_file_proof` below.
  - MODEL TIER: `standard_analyze_model` now defaults to Haiku (the eval-gated flip,
    DECISIONS.md#041), but these 5 representative fixtures are all DEEP-tier, so analyze
    routes to `analyze_model` (Sonnet) for every file regardless of the flip — this stays
    the all-Sonnet baseline the COST_ANALYSIS naive figure assumes. The flip's saving lands
    on STANDARD-tier files (a 3x same-token reduction, Sonnet $3/$15 vs Haiku $1/$5 per
    MTok); the analyze STANDARD->Haiku lever below projects it on this measured analyze call.

Run it explicitly (it spins an ephemeral DB per fixture, so it is slow):

    docker compose up -d postgres-test
    set -a && source <(grep -E '^TEST_' .env) && set +a
    uv run pytest tests/eval/test_cost_measurement.py --is-eval -s

The `-s` shows the printed report; the test also asserts loose sanity bounds so a
gross cost regression (or a broken cost path) fails CI.
"""

from __future__ import annotations

import math
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
    # `total_cost_usd` is Optional (None until synthesize rolls it up); skip the
    # check if absent. Use isclose, not a raw `< 1e-9`: both sums are over the same
    # per-call floats but in different order (Python sum vs SQL SUM), so a relative
    # tolerance is the robust comparison.
    probe_total = sum(c["cost_usd"] for c in probe.calls)
    if result.review_metrics is not None and result.review_metrics.total_cost_usd is not None:
        agg = result.review_metrics.total_cost_usd
        assert math.isclose(probe_total, agg, rel_tol=1e-9, abs_tol=1e-12), (
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

    # The composition (triage + analyze*N + synthesize) requires these nodes to have
    # run. Assert loud HERE so a size-gated or mis-scripted fixture fails with a clear
    # cause, not a downstream KeyError on `model_of["analyze"]` mid-report.
    for required in ("triage", "analyze", "synthesize"):
        assert required in by_node, (
            f"no {required!r} LLM calls captured across {list(runs)} — the cost "
            "composition needs fixtures that drive triage + analyze + synthesize"
        )

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

    # Precondition for the lever's division below (and the regression guard for a broken
    # pricing path) — asserted BEFORE the lever so a zero analyze cost fails with this clear
    # message rather than a cryptic ZeroDivisionError from `(1 - an_haiku / an_sonnet)`.
    assert avg("analyze", "cost_usd") > 0, "analyze cost is zero — pricing/token path broken"

    # Landed lever (DECISIONS#041): STANDARD-tier analyze -> Haiku. These fixtures are all
    # DEEP so they still bill at Sonnet; projected here on the measured analyze call to show
    # the per-STANDARD-file saving (3x same-token, the realized model-price portion).
    if haiku is not None and "analyze" in by_node:
        an_sonnet = avg("analyze", "cost_usd")
        an_haiku = _price(
            haiku, int(avg("analyze", "input_tokens")), int(avg("analyze", "output_tokens"))
        )
        print(
            f"Lever (STANDARD analyze Sonnet->Haiku, DECISIONS#041): ${an_sonnet:.4f} -> "
            f"${an_haiku:.4f} per STANDARD file ({(1 - an_haiku / an_sonnet) * 100:.0f}% cheaper)"
        )

    print("-" * 78)
    print(f"Bands: naive ${_BAND_NAIVE} | defensible ${_BAND_DEFENSIBLE} | target ${_BAND_TARGET}")
    print("Caveats: input = bytes//3 UPPER bound; output modeled; caching unconsumed;")
    print("         these fixtures are all DEEP, so analyze bills Sonnet regardless of the")
    print("         STANDARD->Haiku flip (#041), which saves on STANDARD-tier files.")
    print("=" * 78)

    # Sanity assertions (regression guard — loose, not a tight cost SLA). The analyze-cost
    # > 0 guard is asserted earlier (precondition for the lever division).
    assert by_node, "no LLM calls captured — the cost path is broken"
    # A 30-file all-Sonnet review at a realistic 1k output must land in a sane band:
    # well under $100 (catches an order-of-magnitude pricing bug) and above the
    # fully-optimized target (this is the UN-optimized baseline).
    realistic = reference_cost(1000)
    assert _BAND_TARGET < realistic < 100.0, f"reference review cost {realistic} out of sane range"


def test_cache_packing_cross_file_proof() -> None:
    """The analyze cache-packing proof (specs/2026-06-09-analyze-cache-packing.md).

    Drives a THREE-file STANDARD-tier review with the probe's deterministic
    cache model on (`model_cache=True`) over the REAL rendered analyze-v4
    prompts. Post-repartition contract, on Haiku — the tier with the
    STRICTER 4096-token min-cacheable floor:

      - analyze call 1 WRITES the stable prefix (cache_write > 0, read = 0);
      - analyze calls 2..N READ the byte-identical prefix (read == call 1's
        write, write = 0) — per-file content cannot vary the cache key;
      - the modeled write clears the model's floor on the real rendered
        prompt (below-floor would model the API's silent no-op as 0/0,
        failing the first assertion loudly).

    Pre-repartition packing (per-file context in system_prompt) makes every
    call a distinct first-occurrence: 3 writes, 0 reads — pinned at the
    unit level by `test_probe_cache_model_distinct_system_prompts_never_hit`.
    The printed summary quantifies the prefix saving vs the uncached
    repricing of the same calls.
    """
    from outrider.llm.pricing import min_cacheable_tokens

    probe = CostProbe(token_estimator=_estimate_tokens, model_cache=True)
    run_review(_FIXTURES / "cache_packing_three_files.json", probe=probe)

    analyze_calls = [c for c in probe.calls if c["node_id"] == "analyze"]
    assert len(analyze_calls) == 3, f"expected 3 analyze calls, got {len(analyze_calls)}"
    first, *rest = analyze_calls

    assert first["cache_write_tokens"] > 0, (
        "first analyze call modeled NO cache write — the rendered stable prefix "
        "is below the model's min-cacheable floor (the silent no-op the spec gates on)"
    )
    assert first["cache_read_tokens"] == 0
    assert first["cache_write_tokens"] >= min_cacheable_tokens(first["model"])
    for c in rest:
        assert c["cache_write_tokens"] == 0, (
            "a later analyze call re-WROTE the cache — the system prompt varied "
            "per file; the repartition's byte-identical-prefix property broke"
        )
        assert c["cache_read_tokens"] == first["cache_write_tokens"]

    # Quantify: same calls repriced with no cache (via the file's analytic
    # repricing helper) vs the cache-modeled cost.
    model = first["model"]
    uncached = sum(
        _price(
            c["model"],
            c["input_tokens"] + c["cache_read_tokens"] + c["cache_write_tokens"],
            c["output_tokens"],
        )
        for c in analyze_calls
    )
    cached = sum(c["cost_usd"] for c in analyze_calls)
    prefix_tokens = first["cache_write_tokens"]
    print("\n" + "=" * 78)
    print("Cache-packing proof (3-file STANDARD review, model_cache probe):")
    print(
        f"  model {model}; stable prefix ~{prefix_tokens} est tokens (floor "
        f"{min_cacheable_tokens(model)}); 1 write + {len(rest)} reads"
    )
    print(
        f"  analyze input cost: uncached ${uncached:.6f} -> cache-modeled ${cached:.6f} "
        f"({(1 - cached / uncached) * 100:.0f}% cheaper on these calls)"
    )
    print("=" * 78)
    assert cached < uncached, "cache-modeled cost must beat uncached repricing"
