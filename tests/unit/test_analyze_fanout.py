# Fan-out cutover pins per specs/2026-07-05-parallel-analyze.md (3b-2c-2).
"""The graph-split contracts: planner Command shape, zero-worker route,
pass-1 stays sequential, payload purity, the concurrency cap, and the
proxy-covers-real calibration gate.

These pin the SEAMS the cutover introduced; per-file pipeline behavior
is pinned by test_analyze_node.py through `run_analyze_pass`, and the
fold-vs-outcome parity by test_analyze_worker_wiring.py.
"""

# ruff: noqa: F811  — the imported deps fixture is intentionally shadowed by test params
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.types import Command

# Reuse the node harness: fixtures, builders, and the scripted deps.
from test_analyze_node import (  # noqa: F401  (deps is a fixture)
    _WORKER_DEP_KEYS,
    _build_changed_file,
    _build_pr_context,
    _build_review_state,
    _build_triage_result,
    _StubLLMProvider,
    analyze,
    analyze_file,
    deps,
    run_analyze_pass,
)

from outrider.agent.nodes.analyze import (
    _PROXY_RENDER_MARGIN_TOKENS,
    DEFAULT_REVIEW_BUDGET_TOKENS,
    AnalyzeWorkerPayload,
    _estimate_tokens,
)
from outrider.agent.nodes.analyze_budget import proxy_estimate_tokens
from outrider.prompts import analyze as analyze_prompt
from outrider.schemas.triage_result import ReviewTier


def _two_file_state() -> Any:
    files = (
        _build_changed_file(path="src/a.py"),
        _build_changed_file(path="src/b.py"),
    )
    return _build_review_state(
        pr_context=_build_pr_context(changed_files=files),
        triage_result=_build_triage_result(
            file_tiers={"src/a.py": ReviewTier.DEEP, "src/b.py": ReviewTier.STANDARD}
        ),
    )


# ---------------------------------------------------------------------------
# Planner Command shape.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pass_zero_returns_one_send_per_kept_file(deps: dict[str, Any]) -> None:
    """The planner emits Command(goto=[Send("analyze_file", payload)...])
    in worklist (tier-descending) order, with self-contained payloads
    carrying the pre-flight allocation, and writes the pass start anchor."""
    cmd = await analyze(_two_file_state(), **deps)

    assert isinstance(cmd, Command)
    assert cmd.update is not None
    assert cmd.update["analyze_pass_started_at"] is not None
    sends = cmd.goto
    assert isinstance(sends, list)
    assert [s.node for s in sends] == ["analyze_file", "analyze_file"]
    payloads = [s.arg for s in sends]
    assert all(isinstance(p, AnalyzeWorkerPayload) for p in payloads)
    # Tier-descending worklist order: DEEP before STANDARD.
    assert [p.changed_file.path for p in payloads] == ["src/a.py", "src/b.py"]
    assert [p.review_tier for p in payloads] == [ReviewTier.DEEP, ReviewTier.STANDARD]
    assert all(p.pass_index == 0 for p in payloads)
    # Funded under the default 200k budget: the allocation is the proxy
    # estimate, strictly positive.
    assert all(p.allocation_tokens > 0 for p in payloads)
    # No per-file work in the planner: no LLM call, no examination event.
    assert deps["provider"].calls == []
    assert deps["file_examination_sink"].events == []


@pytest.mark.asyncio
async def test_zero_worker_route_goes_to_aggregate_and_folds_empty_pass(
    deps: dict[str, Any],
) -> None:
    """No kept files → no Sends → goto names the aggregate directly, and
    the composed pass still yields one empty round + one completed event
    (today's empty-pass behavior, preserved across the split)."""
    state = _build_review_state(
        triage_result=_build_triage_result(file_tiers={"src/example.py": ReviewTier.SKIM}),
    )
    cmd = await analyze(state, **deps)
    assert cmd.goto == "analyze_aggregate"

    # The composed run (helper follows the same route) folds the empty pass.
    for sink in ("phase_event_sink", "analyze_event_sink"):
        deps[sink] = type(deps[sink])()  # fresh recorders for the composed run
    result = await run_analyze_pass(state, deps)
    (round_,) = result["analysis_rounds"]
    assert round_.findings == ()
    assert round_.files_examined == ()
    (completed,) = deps["analyze_event_sink"].completed
    assert completed.n_files_analyzed == 0
    assert completed.n_llm_calls == 0
    # Two keyed pairs on the zero-worker route: plan + aggregate (no
    # worker envelopes — there were no workers).
    keys = [e.phase_key for e in deps["phase_event_sink"].events]
    assert keys == ["plan#0", "plan#0", "aggregate#0", "aggregate#0"]
    markers = [e.marker for e in deps["phase_event_sink"].events]
    assert markers == ["start", "end"] * 2


@pytest.mark.asyncio
async def test_pass_one_stays_sequential_and_routes_to_synthesize(
    deps: dict[str, Any],
) -> None:
    """The trace re-entry pass NEVER fans out: with one round already in
    state and no trace work, analyze runs its sequential body end-to-end
    (round + completed event emitted from the node itself) and the
    Command routes to synthesize with no Sends."""
    pass_zero = await run_analyze_pass(_build_review_state(), deps)
    state_pass_one = _build_review_state().model_copy(
        update={"analysis_rounds": pass_zero["analysis_rounds"]}
    )
    cmd = await analyze(state_pass_one, **deps)
    assert isinstance(cmd, Command)
    assert cmd.goto == "synthesize"  # a string, never Sends
    assert cmd.update is not None
    (round_,) = cmd.update["analysis_rounds"]
    assert round_.pass_index == 1
    assert cmd.update["analyze_worker_outcomes"] == []
    # The pass-1 tail emitted its own completed event (the second one).
    assert len(deps["analyze_event_sink"].completed) == 2


# ---------------------------------------------------------------------------
# Payload purity + unfunded enforcement.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_payload_is_pure_data(deps: dict[str, Any]) -> None:
    """State discipline extends to the Send payload: it must survive a
    JSON round-trip unchanged (checkpoints serialize pending Sends)."""
    cmd = await analyze(_two_file_state(), **deps)
    assert isinstance(cmd.goto, list)
    payload = cmd.goto[0].arg
    round_tripped = AnalyzeWorkerPayload.model_validate_json(payload.model_dump_json())
    assert round_tripped == payload


@pytest.mark.asyncio
async def test_unfunded_allocation_skips_at_the_worker_gate(deps: dict[str, Any]) -> None:
    """WORKER-SIDE ALLOCATION ENFORCEMENT: an unfunded payload
    (allocation 0) reaches the real cost gate and skips
    COST_BUDGET_EXHAUSTED — the worker can never spend past its
    allocation, so N concurrent workers can never overshoot the pools."""
    from outrider.ast_facts.models import SkipReason

    cmd = await analyze(_two_file_state(), **deps)
    assert isinstance(cmd.goto, list)
    funded = cmd.goto[0].arg
    starved = funded.model_copy(update={"allocation_tokens": 0})
    worker_deps = {k: deps[k] for k in _WORKER_DEP_KEYS if k in deps}
    update = await analyze_file(starved, **worker_deps)
    (outcome,) = update["analyze_worker_outcomes"]
    assert outcome.parse_status == "skipped"
    assert outcome.skip_reason is SkipReason.COST_BUDGET_EXHAUSTED
    assert deps["provider"].calls == []  # the LLM never fired


# ---------------------------------------------------------------------------
# Concurrency cap.
# ---------------------------------------------------------------------------


class _InFlightTrackingProvider:
    """LLMProvider stub that records the maximum number of concurrent
    `complete` calls. The await inside forces real interleaving."""

    def __init__(self, response_text: str) -> None:
        self._text = response_text
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls: list[Any] = []

    async def aclose(self) -> None:
        return None

    async def complete(self, request: Any) -> Any:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.005)  # hold the slot so overlap is observable
        self.in_flight -= 1
        self.calls.append(request)
        stub = _StubLLMProvider(self._text)
        return await stub.complete(request)


@pytest.mark.asyncio
async def test_semaphore_bounds_in_flight_workers(deps: dict[str, Any]) -> None:
    """ANALYZE_MAX_CONCURRENCY: with 6 workers dispatched concurrently
    (as the Send superstep does) and a 2-permit semaphore closed into the
    worker, no more than 2 LLM calls are ever in flight. The unbounded
    control run proves the scenario CAN exceed 2, so the bound is doing
    the work (not the fixture)."""
    from test_analyze_node import _build_finding_proposal_json

    paths = [f"src/f{i}.py" for i in range(6)]
    files = tuple(_build_changed_file(path=p) for p in paths)
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=files),
        triage_result=_build_triage_result(file_tiers=dict.fromkeys(paths, ReviewTier.DEEP)),
    )
    cmd = await analyze(state, **deps)
    assert isinstance(cmd.goto, list) and len(cmd.goto) == 6  # noqa: PT018

    async def run_all(semaphore: asyncio.Semaphore | None) -> int:
        provider = _InFlightTrackingProvider(_build_finding_proposal_json())
        worker_deps = {k: deps[k] for k in _WORKER_DEP_KEYS if k in deps}
        worker_deps["provider"] = provider
        worker_deps["concurrency_semaphore"] = semaphore
        assert isinstance(cmd.goto, list)
        await asyncio.gather(*(analyze_file(s.arg, **worker_deps) for s in cmd.goto))
        return provider.max_in_flight

    unbounded_max = await run_all(None)
    assert unbounded_max > 2  # the control: overlap genuinely happens
    bounded_max = await run_all(asyncio.Semaphore(2))
    assert bounded_max <= 2


# ---------------------------------------------------------------------------
# Proxy-covers-real calibration (the regression gate for DUP_FACTOR).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_estimate_covers_real_rendered_estimate(deps: dict[str, Any]) -> None:
    """CALIBRATION GATE: for every funded fixture file, the planner's
    bytes-based proxy must be >= the worker's REAL rendered-prompt
    estimate (recorded on the outcome). A proxy under-estimate converts
    into spurious COST_BUDGET_EXHAUSTED skips — coverage loss, never
    overspend — and this property is what licenses tightening DUP_FACTOR.
    Not a proof (a fixture corpus cannot prove universality); a
    regression gate."""
    state = _two_file_state()
    result = await run_analyze_pass(state, deps)
    empty_parts = analyze_prompt.render(
        file_path="", scope_unit_context="", query_match_id_list="", diff_hunks="", pass_index=0
    )
    fixed_overhead = (
        _estimate_tokens(empty_parts.system_prompt)
        + _estimate_tokens(empty_parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
        + _PROXY_RENDER_MARGIN_TOKENS  # mirrors the planner's overhead exactly
    )
    by_path = {f.path: f for f in state.pr_context.changed_files}
    checked = 0
    for outcome in result["analyze_worker_outcomes"]:
        if outcome.source != "parser":
            continue
        cf = by_path[outcome.path]
        proxy = proxy_estimate_tokens(
            len((cf.content_head or "").encode("utf-8")),
            len((cf.patch or "").encode("utf-8")),
            fixed_overhead_tokens=fixed_overhead,
        )
        assert proxy >= outcome.estimated_tokens, (
            f"proxy under-covers real estimate for {outcome.path}: "
            f"{proxy} < {outcome.estimated_tokens}"
        )
        checked += 1
    assert checked == 2  # both files funded and LLM-run — the gate actually ran


@pytest.mark.asyncio
async def test_proxy_covers_real_on_overlap_dense_scope_context(deps: dict[str, Any]) -> None:
    """CALIBRATION, the adversarial shape: many small functions with dense
    same-file call links, all changed — the rendered scope context includes
    every unit PLUS same-file caller/callee excerpts, so content regions
    DUPLICATE into the prompt (the exact overlap DUP_FACTOR exists to
    absorb). The proxy must still cover the real rendered estimate."""
    n = 14
    lines: list[str] = []
    for i in range(n):
        callee = f"f{(i + 1) % n}"
        lines.append(f"def f{i}(x):")
        lines.append(f"    y = {callee}(x) if x > {i} else x  # link {i}")
        lines.append(f"    return y + {i}")
    body = "\n".join(lines) + "\n"
    n_lines = body.count("\n")
    patch = f"--- a/src/dense.py\n+++ b/src/dense.py\n@@ -0,0 +1,{n_lines} @@\n" + "".join(
        "+" + line + "\n" for line in body.splitlines()
    )
    cf = _build_changed_file(
        path="src/dense.py", content=body.encode(), patch=patch, content_base=""
    )
    state = _build_review_state(
        pr_context=_build_pr_context(changed_files=(cf,)),
        triage_result=_build_triage_result(file_tiers={"src/dense.py": ReviewTier.DEEP}),
    )
    result = await run_analyze_pass(state, deps)
    (outcome,) = result["analyze_worker_outcomes"]
    assert outcome.source == "parser"  # funded and LLM-run — the gate is live
    empty_parts = analyze_prompt.render(
        file_path="", scope_unit_context="", query_match_id_list="", diff_hunks="", pass_index=0
    )
    fixed_overhead = (
        _estimate_tokens(empty_parts.system_prompt)
        + _estimate_tokens(empty_parts.user_prompt)
        + analyze_prompt.MAX_TOKENS
        + _PROXY_RENDER_MARGIN_TOKENS
    )
    proxy = proxy_estimate_tokens(
        len(body.encode()), len(patch.encode()), fixed_overhead_tokens=fixed_overhead
    )
    assert proxy >= outcome.estimated_tokens, (
        f"proxy under-covers the overlap-dense file: {proxy} < {outcome.estimated_tokens} — "
        f"DUP_FACTOR/margin need re-calibration before trusting the planner on dense files"
    )


@pytest.mark.asyncio
async def test_composed_pass_matches_default_budget(deps: dict[str, Any]) -> None:
    """Sanity for the helper contract: composed pass over two files under
    the default budget examines both files, emits one round + one
    completed event, and the phase envelope closes start→end."""
    assert deps["total_review_budget_tokens"] == DEFAULT_REVIEW_BUDGET_TOKENS
    result = await run_analyze_pass(_two_file_state(), deps)
    (round_,) = result["analysis_rounds"]
    assert set(round_.files_examined) == {"src/a.py", "src/b.py"}
    # Four keyed pairs: plan + two workers + aggregate.
    keys = [e.phase_key for e in deps["phase_event_sink"].events]
    assert keys == [
        "plan#0",
        "plan#0",
        "file:src/a.py#0",
        "file:src/a.py#0",
        "file:src/b.py#0",
        "file:src/b.py#0",
        "aggregate#0",
        "aggregate#0",
    ]


# ---------------------------------------------------------------------------
# Phase-key stamping (increment 4): per-operation attribution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_operation_events_carry_the_worker_key(deps: dict[str, Any]) -> None:
    """Every per-file event the worker path emits carries the worker's
    `file:<path>#<pass>` key: the FileExaminationEvent (worker sink), the
    LLMRequest (propagated by providers onto LLMCallEvent — pinned here at
    the request, the provider pass-through has its own contract test),
    while the FindingEvent and AnalyzeCompletedEvent are AGGREGATE-keyed
    (admission and pass-level accounting are aggregate work)."""
    result = await run_analyze_pass(_build_review_state(), deps)
    assert result["analysis_rounds"][0].findings  # the scenario admitted work

    worker_key = "file:src/example.py#0"
    (fe_event,) = deps["file_examination_sink"].events
    assert fe_event.phase_key == worker_key
    (request,) = deps["provider"].calls
    assert request.phase_key == worker_key
    (finding_event,) = deps["analyze_event_sink"].findings
    assert finding_event.phase_key == "aggregate#0"
    (completed,) = deps["analyze_event_sink"].completed
    assert completed.phase_key == "aggregate#0"


@pytest.mark.asyncio
async def test_pass_one_events_stay_none_keyed(deps: dict[str, Any]) -> None:
    """The sequential pass-1 body emits the LEGACY shape: an un-keyed
    analyze-pass-1 envelope and None-keyed per-operation events — the
    replay hybrid's None-branch contract depends on sequential-era events
    never carrying keys."""
    pass_zero = await run_analyze_pass(_build_review_state(), deps)
    state_pass_one = _build_review_state().model_copy(
        update={"analysis_rounds": pass_zero["analysis_rounds"]}
    )
    n_phase_before = len(deps["phase_event_sink"].events)
    n_completed_before = len(deps["analyze_event_sink"].completed)
    await analyze(state_pass_one, **deps)

    pass_one_phases = deps["phase_event_sink"].events[n_phase_before:]
    assert [e.marker for e in pass_one_phases] == ["start", "end"]
    assert all(e.phase_key is None for e in pass_one_phases)
    pass_one_completed = deps["analyze_event_sink"].completed[n_completed_before:]
    assert all(e.phase_key is None for e in pass_one_completed)


@pytest.mark.asyncio
async def test_concurrency_gate_contends_per_loop_and_prunes_on_hit() -> None:
    """REAL contention on each loop (permits=1: the second holder must WAIT,
    which is what makes asyncio bind the semaphore to its loop) — plus the
    DETERMINISTIC prune-policy pin: after a secondary loop closes, a plain
    HIT on the still-live primary loop must evict it. Miss-only pruning
    (the reverted behavior) would keep the closed loop's semaphore forever
    when no new loop appears, so this assertion fails on that revert."""
    import threading

    from outrider.agent.nodes.analyze import AnalyzeConcurrencyGate

    gate = AnalyzeConcurrencyGate(1)

    async def contend() -> int:
        sem = gate.current()
        order: list[int] = []

        async def hold(tag: int) -> None:
            async with sem:  # tag 2 must WAIT on tag 1 → loop binding happens
                order.append(tag)
                await asyncio.sleep(0)

        await asyncio.gather(hold(1), hold(2))
        assert order == [1, 2]
        return id(sem)

    first_id = await contend()  # loop A (the running test loop)

    result: dict[str, int] = {}

    def run_on_fresh_loop() -> None:
        result["second_id"] = asyncio.run(contend())  # loop B; closes on return

    thread = threading.Thread(target=run_on_fresh_loop)
    thread.start()
    thread.join()
    assert result["second_id"] != first_id  # a fresh semaphore, no cross-loop rebind
    assert len(gate._by_loop) == 2  # closed B still present — nothing pruned it yet

    # THE PIN: a HIT on live loop A (no miss involved) prunes closed B.
    hit_id = id(gate.current())
    assert hit_id == first_id  # same live semaphore — this was a hit, not a miss
    assert len(gate._by_loop) == 1  # closed B evicted by per-call pruning


@pytest.mark.asyncio
async def test_concurrency_gate_survives_simultaneous_cross_thread_misses() -> None:
    """Cross-thread SMOKE: barrier-released threads miss simultaneously
    while closed-loop entries need pruning. `current()` is a pure-sync
    section, so under the GIL a removed lock may still pass this — the
    DETERMINISTIC lock guarantee is the probe-lock pin below; this test
    exists to exercise the real threading shape end-to-end (and fails
    loudly on gross corruption)."""
    import threading

    from outrider.agent.nodes.analyze import AnalyzeConcurrencyGate

    gate = AnalyzeConcurrencyGate(2)

    async def touch() -> None:
        async with gate.current():
            await asyncio.sleep(0)

    # Seed one already-closed entry so the very first barrier wave prunes.
    seed_thread = threading.Thread(target=lambda: asyncio.run(touch()))
    seed_thread.start()
    seed_thread.join()

    n_threads = 4
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def hammer() -> None:
        try:
            barrier.wait(timeout=10)
            for _ in range(25):
                asyncio.run(touch())  # every run: fresh loop → miss + prune pressure
        except BaseException as exc:  # noqa: BLE001 — the failure IS the assertion
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []  # no KeyError double-del, no corrupted map
    await touch()  # a final call on the live test loop prunes all closed loops
    assert len(gate._by_loop) == 1  # only this loop remains


@pytest.mark.asyncio
async def test_concurrency_gate_full_critical_section_under_its_lock() -> None:
    """DETERMINISTIC lock pin, self-verifying coverage: every map operation
    the critical section performs — create (set), lookup (get), prune scan
    (iter), AND eviction (del) — must run while `self._lock` is held, and
    each must actually be EXERCISED by this test (the ops_seen assertion),
    so no operation can be vacuously guarded. The eviction arm needs a
    closed-loop entry: loop B is seeded through the guarded map in its own
    thread, closes, and the test loop's subsequent MISS prunes it — a
    deletion hoisted outside the lock fails on that call, no race
    required."""
    import threading

    from outrider.agent.nodes.analyze import AnalyzeConcurrencyGate

    gate = AnalyzeConcurrencyGate(2)

    class _ProbeLock:
        def __init__(self) -> None:
            self.acquisitions = 0
            self.held = False

        def __enter__(self) -> _ProbeLock:
            self.acquisitions += 1
            self.held = True
            return self

        def __exit__(self, *exc: object) -> None:
            self.held = False

    class _GuardedMap(dict):  # type: ignore[type-arg]
        """Rejects any access — reads, writes, iteration, deletion —
        outside the probe lock, and records which ops ran so the test can
        prove each guard was exercised, not just present."""

        def __init__(self, probe: _ProbeLock) -> None:
            super().__init__()
            self._probe = probe
            self.ops_seen: set[str] = set()

        def _check(self, op: str) -> None:
            assert self._probe.held, f"gate map {op} outside its lock"
            self.ops_seen.add(op)

        def __setitem__(self, key: object, value: object) -> None:
            self._check("set")
            super().__setitem__(key, value)

        def __delitem__(self, key: object) -> None:
            self._check("del")
            super().__delitem__(key)

        def __getitem__(self, key: object) -> object:
            self._check("get")
            return super().__getitem__(key)

        def get(self, key: object, default: object = None) -> object:
            self._check("get")
            return super().get(key, default)

        def __iter__(self) -> object:
            self._check("iter")
            return iter(list(super().keys()))

    probe = _ProbeLock()
    guarded = _GuardedMap(probe)
    gate._lock = probe  # type: ignore[assignment]
    gate._by_loop = guarded  # type: ignore[assignment]

    # Seed loop B THROUGH the guarded map (exercises set under the lock),
    # then let it close — the eviction target.
    seed_thread = threading.Thread(target=lambda: asyncio.run(_touch_gate(gate)))
    seed_thread.start()
    seed_thread.join()
    assert len(guarded) == 1  # closed B is present, awaiting eviction

    # The test loop's MISS: get (lookup) + iter (prune scan) + del (evict
    # closed B) + set (create) — the full critical section, in one call.
    semaphore = gate.current()
    assert gate.current() is semaphore  # and a plain HIT for good measure
    assert probe.acquisitions == 3  # seed + miss + hit, each under the lock
    assert len(guarded) == 1  # B evicted, only the live test loop remains
    # Coverage is self-verifying: every guarded operation actually ran.
    assert guarded.ops_seen == {"set", "get", "iter", "del"}


async def _touch_gate(gate: object) -> None:
    async with gate.current():  # type: ignore[attr-defined]
        await asyncio.sleep(0)
