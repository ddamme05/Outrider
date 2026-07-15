"""Frozen per-provider, across-prompt-version baseline harness for the analyze EXEMPLARS
prompt-optimization arc (`specs/2026-07-14-analyze-exemplars-prompt-optimization.md`).

DETERMINISTIC CORE only — offline, unit-tested, no models: the frozen-artifact schema, the
N-run majority aggregator, the per-provider ε=0 pre/post comparator that encodes the
pre-registered acceptance contract, and the pure HTML renderers for the human-readable reports.
The PAID runner that drives the real providers lives in `test_exemplar_baseline.py` behind
`OUTRIDER_EVAL_REAL_MODELS=1` and feeds `aggregate()` here.

Two artifact tiers, deliberately different: the JSON under `BASELINE_DIR` is TRACKED, immutable
EVIDENCE (create-once); the HTML under `REPORT_DIR` is a gitignored derived VIEW, freely
re-renderable from that JSON. A view is not evidence, so re-rendering it cannot rewrite history.

Pre-registration contract (spec step 1), enforced by `aggregate()` at freeze time and
re-checked by `compare()`:

- exactly N=3 clean reps, a >=2/3 majority per (fixture, dimension);
- the EXACT acceptance set {Claude DEEP, Claude STANDARD, Fireworks GLM}; Baseten is
  supporting (never gating);
- the runs are provably COMPARABLE: identical fixture identities + SEMANTIC content digests
  (source + ground-truth types + safe/unsafe, via `fixture_content_digest`), per-type totals,
  resolved model identity, host/profile-contract identity;
- a preregistered candidate MUST change BOTH the analyze VERSION (`prompt_version`) and the prompt
  content (`prompt_digest`) — reusing either fails closed (it is not a real prompt change);
- provider-correct INPUT-SIDE tokens are persisted per rep + aggregated, split by class
  (input / cache_read / cache_write), as the cost evidence the shrink is measured by. All three
  classes are required: the prefix is CACHED on Claude (`#042`), so it lands in cache_read and is
  NET of `input_tokens` — an input-only measure would show ~zero saving for Claude. Recorded and
  reported (`token_delta` / `cost_objective`), never gated by `compare()`.
- structured-output RAW COUNTS (accepted / rejected / void) are persisted per fixture and per
  provider (schema v3, FUP-219): raw counts, never a derived rate, so a later yield-metric
  definition change costs nothing. Recorded, never gated — yield is evidence, like tokens.
- per-fixture EXTRA-FINDINGS raw counts on recall fixtures (mc-2): sorted per-rep values +
  total, gated no-increase on BOTH total and max — over-emission conservatism, not correctness
  adjudication (extras are unadjudicated; a genuine improvement is admitted by amendment).
- the fixture suite carries a versioned identity (`fixture_suite`), gated and woven into freeze
  labels (`{prompt_version}+{suite}`), so suite changes get new immutable artifacts without
  prompt-VERSION abuse.
- the artifact self-records its producing harness (`harness_digest`, FUP-238), so provenance is
  readable from the artifact instead of reconstructed from the git DAG. Provenance and
  comparability are SEPARATE contracts: `harness_digest` is informational (surfaced by
  `provenance_notes()`, never gated), while `measurement_contract` is the blocking semantics
  identity `compare()` and `preflight_comparability()` enforce.

Accept iff, per (acceptance-provider, finding_type), recall does not decrease and the
false-positive count does not increase (ε = 0).

Narrow by design (per the arc decision): serves THIS optimization's fixed fixtures + fixed provider
set only — NOT a generalized multi-provider evaluation framework.
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# v3 adds `harness_digest` (FUP-238) + per-fixture/per-provider `structured_output` raw counts
# (FUP-219). v2 artifacts stay readable via `read_baseline`'s deterministic in-memory upgrade —
# the frozen on-disk evidence is never rewritten.
SCHEMA_VERSION = 3
_READABLE_SCHEMA_VERSIONS = frozenset({2, SCHEMA_VERSION})

# The BLOCKING comparability identity for measurement SEMANTICS — deliberately separate from
# `harness_digest` (informational provenance, surfaced not gated: any source edit rotates a code
# digest, which would deadlock the immutable baseline). This string covers: the >=2/3 N-rep
# majority rule, detection semantics (`not grade.missed` / `n_false_positives > 0` incl.
# `grading.py`'s match criteria + line window — code OUTSIDE the source-digest file list), the
# recall/FP cell model, the extras evidence + gate (below), the acceptance-set semantics, and
# the sequential single-file rep protocol. Any change to those semantics MUST rotate this string
# or ship a reviewed compatibility mapping in `_upgrade_v2`-style code; shape-only schema
# changes must NOT rotate it.
#
# Rotation history:
# - exemplar-mc-1 — the frozen analyze-v10 collection: majority recall + majority FP counts,
#   no extras evidence. `_upgrade_v2` declares v2 artifacts mc-1 (a LITERAL fill, deliberately
#   not this constant — the declaration must not rotate with it).
# - exemplar-mc-2 — adds per-fixture extra-findings raw counts on recall fixtures and the
#   candidate-total<=baseline-total AND candidate-max<=baseline-max no-increase gate
#   (spec 2026-07-15-exemplar-coverage-fixture-suite-v2, second-review statistics).
MEASUREMENT_CONTRACT = "exemplar-mc-2"

# Versioned fixture-suite identity, independent of the prompt VERSION: freeze labels are
# "{prompt_version}+{fixture_suite}", so a suite change gets a new immutable artifact without
# bumping the prompt VERSION (the prompt didn't change) and without touching older evidence.
# suite-v1 = the frozen 20-fixture set (16 recall + 4 safe); `_upgrade_v2` declares it on v2
# artifacts as a literal, same non-rotating rule as the mc-1 fill above.
FIXTURE_SUITE_VERSION = "suite-v2"

REQUIRED_REPS = 3  # the pre-registration pins exactly three clean reps
BASELINE_DIR = Path(__file__).parent / "baselines" / "analyze-exemplars"
# Derived HTML views (gitignored, like `reports/scorecard/`) — NOT evidence, so re-renderable.
REPORT_DIR = Path("reports") / "exemplar-baseline"

RECALL = "recall"
PRECISION = "precision"

ACCEPTANCE = "acceptance"
SUPPORTING = "supporting"
_ROLES = frozenset({ACCEPTANCE, SUPPORTING})

# Stable LOGICAL provider keys (not the volatile resolved model id, which is stored as provenance).
CLAUDE_DEEP = "claude-deep"
CLAUDE_STANDARD = "claude-standard"
FIREWORKS_GLM = "fireworks-glm"
BASETEN_GLM = "baseten-glm"
EXPECTED_ACCEPTANCE = frozenset({CLAUDE_DEEP, CLAUDE_STANDARD, FIREWORKS_GLM})
EXPECTED_SUPPORTING = frozenset({BASETEN_GLM})
_EXPECTED_PROVIDERS = EXPECTED_ACCEPTANCE | EXPECTED_SUPPORTING


class TokenUsage(NamedTuple):
    """Provider-correct INPUT-side token accounting for one fixture/rep, summed over its calls.

    All THREE classes must be carried, because Anthropic reports them separately and
    `LLMResponse.input_tokens` is NET of the other two (`anthropic_provider` maps
    `usage.input_tokens` / `cache_read_input_tokens` / `cache_creation_input_tokens`). The analyze
    stable prefix IS cached on Claude (`DECISIONS.md#042`), so the prefix lands in `cache_read` (or
    `cache_write` on the first call) and NOT in `input_tokens` — measuring the shrink by
    `input_tokens` alone would show ~zero saving for both Claude tiers even when the shrink works.
    GLM/Fireworks is the mirror case: it realizes ~no prefix cache, so the prefix sits in
    `input_tokens`.

    The classes are DISJOINT on every host, so `total` is uniform and needs no per-provider
    arithmetic: the raw wire shapes differ (OpenAI-compatible `prompt_tokens` INCLUDES its cached
    subset) but `llm/host_profiles.read_usage` normalizes that at the wrapper boundary
    (`vendor-payloads-normalized-at-boundary`, trust-boundaries §5 sub-rule 6), subtracting
    cached so `input + cache_read == prompt_tokens` and setting `cache_write=0` where the host
    has no such class. Re-deriving a "total" from raw semantics here would double-count on one
    path and undercount on the other; the per-provider §8a mode is recorded on
    `ProviderMeta.token_accounting` for auditability, not for the reader to apply. Components are
    persisted so a dollar model can weight the classes (cache_read ~0.1x, cache_write ~1.25x)
    later without another paid run.
    """

    input_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


class Observation(NamedTuple):
    """One scenario result for one provider in one rep.

    RECALL: `detected` = the expected finding was matched. PRECISION (safe-code): `detected` =
    the model produced a finding (a false positive). `finding_type` is the expected type for
    recall fixtures, "" for safe. `tokens` is the provider-correct input-side usage summed across
    the analyze pass's LLM call(s) for this fixture/rep — the cost evidence the shrink is measured
    by; `None` when the host reported no usage (telemetry-absent is NOT zero — it is dropped from
    the aggregate, not counted as 0). `n_rejected` is the structured-output yield signal: the
    count of rejected model outputs in this rep's analyze pass (0 or 1 — the runner asserts
    single-file fixtures, so each rep is exactly one structured-output attempt). `n_extra` is
    the over-emission signal on RECALL fixtures only (`len(GradeResult.extra)`): findings that
    matched no expected finding. Deliberately unadjudicated — see the extras gate in `compare()`.
    On PRECISION fixtures it must stay 0 (safe-code emissions are measured as false positives).
    """

    provider: str
    fixture: str
    dimension: str  # RECALL | PRECISION
    finding_type: str
    detected: bool
    tokens: TokenUsage | None = None
    n_rejected: int = 0
    n_extra: int = 0


class ProviderMeta(NamedTuple):
    role: str  # ACCEPTANCE | SUPPORTING
    model: str  # resolved model id (provenance; gated for equality)
    profile_contract: (
        str  # #056 profile_contract_digest, or the model for the Anthropic path (gated)
    )
    # The §8a `TokenAccounting` mode the wrapper normalized this host's usage under. Recorded so the
    # artifact is self-describing (and gated, since a mode flip would make two runs incomparable).
    # It does NOT change how `input_side_tokens` is read: see `TokenUsage` — the wrapper normalizes
    # every host to the same disjoint representation, so the class sum is uniform across providers.
    token_accounting: str


class RunMeta(NamedTuple):
    n_reps: int
    prompt_version: str  # analyze VERSION (e.g. "analyze-v10") — MUST bump baseline->candidate
    prompt_digest: str  # sha256 of the analyze prompt CONTENT — MUST differ baseline->candidate
    # fixture -> canonical SEMANTIC digest via fixture_content_digest() (gated for equality);
    # commits to source + ground-truth types + safe/unsafe, not just the source bytes.
    fixture_digests: dict[str, str]
    providers: dict[str, ProviderMeta]
    # sha256 over the harness source (harness_source_digest()) so the artifact states which code
    # produced it (FUP-238). Defaulted for construction ergonomics; aggregate() rejects "".
    harness_digest: str = ""
    # Blocking measurement-semantics identity (see MEASUREMENT_CONTRACT). Gated by compare() and
    # preflight_comparability(); rotate on semantic change, never on shape-only change.
    measurement_contract: str = MEASUREMENT_CONTRACT
    # Versioned suite identity (see FIXTURE_SUITE_VERSION). Gated; also woven into freeze labels.
    fixture_suite: str = FIXTURE_SUITE_VERSION


@dataclass
class _Cell:
    dimension: str
    finding_type: str
    detected: int = 0
    seen: int = 0
    rejected: int = 0  # structured-output rejections summed over reps (yield evidence)
    extra_values: list[int] = field(default_factory=list)  # per-rep extras (recall fixtures)
    token_values: list[TokenUsage] = field(default_factory=list)  # per-rep usage (None dropped)


# The harness's own source, in fixed order, hashed into `harness_digest` (FUP-238).
_HARNESS_FILES = ("exemplar_baseline.py", "test_exemplar_baseline.py")


def harness_source_digest(filenames: tuple[str, ...] = _HARNESS_FILES) -> str:
    """sha256 over the named `tests/eval/` files' bytes, each prefixed by its name + NUL separators.

    Reproducible from git blobs at any commit (same recipe over `git show <sha>:<path>`), which is
    how a reader verifies WHICH code produced a frozen artifact without trusting the git DAG story.
    Defaults to this harness's two files; other instruments (e.g. `glm_yield.py`) pass their own.
    """
    h = hashlib.sha256()
    for name in filenames:
        h.update(name.encode("utf-8") + b"\0")
        h.update((Path(__file__).parent / name).read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def majority_threshold(n_reps: int) -> int:
    """Minimum detected-rep count for a >=2/3 majority (ceil): N=3 -> 2, N=5 -> 4, N=1 -> 1."""
    if n_reps < 1:
        raise ValueError("n_reps must be >= 1")
    return math.ceil(2 * n_reps / 3)


def _is_majority(detected_reps: int, n_reps: int) -> bool:
    return detected_reps >= majority_threshold(n_reps)


def _empty_token_rollup() -> dict[str, object]:
    return {
        "expected": 0,
        "observed": 0,
        "missing": 0,
        "total": 0,
        "by_class": {"input": 0, "cache_read": 0, "cache_write": 0},
    }


def fixture_content_digest(*, source: str, expected_types: Iterable[str], is_safe: bool) -> str:
    """Canonical SEMANTIC digest of a fixture's full contract, for `RunMeta.fixture_digests`.

    Commits to (source, sorted ground-truth finding types, safe/unsafe classification) — NOT just
    the source bytes. A sha over source alone would let the expected labels drift under a stable
    fixture identity (relabel a positive, reclassify a safe fixture) and still pass `compare()`'s
    fixture-digest equality gate. The paid runner MUST populate `fixture_digests` via this helper.
    """
    payload = json.dumps(
        {"source": source, "expected_types": sorted(expected_types), "is_safe": bool(is_safe)},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def aggregate(observations: list[Observation], meta: RunMeta) -> dict[str, object]:
    """Fold per-rep observations into a frozen, pre-registration-checked baseline dict.

    Enforces the freeze-time contract (raises on any breach, so an unclean/underspecified run is
    never frozen): exactly `REQUIRED_REPS` reps per (provider, fixture); `meta.providers` covers
    EXACTLY the observed providers with valid roles; ACCEPTANCE == EXPECTED_ACCEPTANCE and
    SUPPORTING ⊆ EXPECTED_SUPPORTING (no unexpected providers); and `meta.fixture_digests` covers
    EXACTLY the observed fixtures. Stores provenance (role/model/profile_contract per provider,
    fixture digests, prompt_version + prompt_digest) so `compare()` can prove the two runs are
    comparable, plus per-fixture and per-provider `input_side_tokens` (expected/observed/missing +
    total + a by_class split of input / cache_read / cache_write + the sorted per-rep totals) as the
    cost evidence — recorded, never gated.
    """
    if meta.n_reps != REQUIRED_REPS:
        raise ValueError(
            f"n_reps must be exactly {REQUIRED_REPS} (pre-registration); got {meta.n_reps}"
        )
    if not meta.harness_digest:
        raise ValueError(
            "meta.harness_digest is empty — a run that cannot state its producing harness must "
            "not be frozen (populate via harness_source_digest(); FUP-238)"
        )
    if not meta.measurement_contract:
        raise ValueError(
            "meta.measurement_contract is empty — a run must state the measurement semantics it "
            "was collected under (see MEASUREMENT_CONTRACT)"
        )
    if not meta.fixture_suite:
        raise ValueError(
            "meta.fixture_suite is empty — a run must name its fixture-suite identity "
            "(see FIXTURE_SUITE_VERSION)"
        )
    n_reps = meta.n_reps
    seen_providers = {o.provider for o in observations}
    if set(meta.providers) != seen_providers:
        raise ValueError(
            f"meta.providers must cover exactly the observed providers: "
            f"meta={sorted(meta.providers)} observed={sorted(seen_providers)}"
        )
    for prov, pmeta in meta.providers.items():
        if pmeta.role not in _ROLES:
            raise ValueError(f"provider {prov!r} has invalid role {pmeta.role!r}")
        if prov not in _EXPECTED_PROVIDERS:
            raise ValueError(
                f"unexpected provider {prov!r}; expected one of {sorted(_EXPECTED_PROVIDERS)}"
            )
    acceptance = {p for p, m in meta.providers.items() if m.role == ACCEPTANCE}
    if acceptance != EXPECTED_ACCEPTANCE:
        raise ValueError(
            f"acceptance set must equal {sorted(EXPECTED_ACCEPTANCE)}; got {sorted(acceptance)}"
        )
    supporting = {p for p, m in meta.providers.items() if m.role == SUPPORTING}
    if not supporting <= EXPECTED_SUPPORTING:
        raise ValueError(
            f"unexpected supporting providers: {sorted(supporting - EXPECTED_SUPPORTING)}"
        )
    seen_fixtures = {o.fixture for o in observations}
    if set(meta.fixture_digests) != seen_fixtures:
        raise ValueError(
            f"fixture_digests must cover exactly the observed fixtures: "
            f"digests={sorted(meta.fixture_digests)} observed={sorted(seen_fixtures)}"
        )

    acc: dict[tuple[str, str], _Cell] = {}
    for o in observations:
        key = (o.provider, o.fixture)
        cell = acc.setdefault(key, _Cell(o.dimension, o.finding_type))
        if cell.dimension != o.dimension or cell.finding_type != o.finding_type:
            raise ValueError(
                f"inconsistent dimension/finding_type for {key}: "
                f"{cell.dimension}/{cell.finding_type} vs {o.dimension}/{o.finding_type}"
            )
        cell.seen += 1
        cell.detected += 1 if o.detected else 0
        # Single-file fixtures = one structured-output attempt per rep, so the per-rep rejection
        # count is 0 or 1. A value outside that range means a multi-file fixture reached the cell
        # model — fail loud rather than mislabel the yield accounting.
        if o.n_rejected not in (0, 1):
            raise ValueError(
                f"{key}: n_rejected={o.n_rejected} — the cell model records one structured-output "
                "attempt per rep (single-file fixtures); extend the schema before aggregating "
                "multi-attempt reps"
            )
        cell.rejected += o.n_rejected
        if o.n_extra < 0:
            raise ValueError(f"{key}: n_extra={o.n_extra} must be >= 0")
        if o.dimension == PRECISION and o.n_extra:
            raise ValueError(
                f"{key}: n_extra={o.n_extra} on a safe fixture — safe-code emissions are "
                "measured as false positives, never as extras"
            )
        if o.dimension == RECALL:
            cell.extra_values.append(o.n_extra)
        if o.tokens is not None:
            cell.token_values.append(o.tokens)

    for provider in meta.providers:
        prov_fixtures = {fx for (p, fx) in acc if p == provider}
        if prov_fixtures != seen_fixtures:
            raise ValueError(
                f"{provider} covers fixtures {sorted(prov_fixtures)}, expected the full "
                f"set {sorted(seen_fixtures)} — every provider must run every fixture"
            )

    providers: dict[str, dict[str, object]] = {}
    for (provider, fixture), cell in sorted(acc.items()):
        if cell.seen != n_reps:
            raise ValueError(
                f"{provider}/{fixture} has {cell.seen} reps, expected exactly {n_reps} "
                "(a missing/extra rep means the run was not clean — do not freeze it)"
            )
        pmeta = meta.providers[provider]
        p = providers.setdefault(
            provider,
            {
                "role": pmeta.role,
                "model": pmeta.model,
                "profile_contract": pmeta.profile_contract,
                "token_accounting": pmeta.token_accounting,
                "recall_by_type": {},
                "fp_count": 0,
                "per_fixture": {},
                "input_side_tokens": _empty_token_rollup(),
                "structured_output": {"attempts": 0, "accepted": 0, "rejected": 0, "void": 0},
            },
        )
        recall_by_type: dict[str, dict[str, int]] = p["recall_by_type"]  # type: ignore[assignment]
        per_fixture: dict[str, dict[str, object]] = p["per_fixture"]  # type: ignore[assignment]
        maj = _is_majority(cell.detected, n_reps)
        usage = sorted(cell.token_values, key=lambda u: u.total)  # rep order untracked
        # `expected` vs `observed` keeps a telemetry GAP distinguishable from a measured value all
        # the way through the artifact — `total` alone would read as a saving when reps went
        # unreported. `token_delta()` refuses to price a provider with any `missing`.
        per_fixture[fixture] = {
            "dimension": cell.dimension,
            "finding_type": cell.finding_type,
            "detected_reps": cell.detected,
            "n_reps": n_reps,
            "majority_detected": maj,
            # Over-emission evidence (recall fixtures only; mc-2): deterministic integer counts —
            # sorted per-rep extras + their total. Gated total+max no-increase in compare();
            # UNADJUDICATED (an extra may be a correct secondary defect), so evidence + a
            # conservatism gate, never a correctness verdict. None on safe fixtures (their
            # emissions are the FP dimension).
            "extra_findings": (
                {"values": sorted(cell.extra_values), "total": sum(cell.extra_values)}
                if cell.dimension == RECALL
                else None
            ),
            # RAW yield counts (FUP-219), never a derived rate: one single-file structured-output
            # attempt per rep. `void` is 0 by construction — a provider error propagates and aborts
            # the sequential attempt before any artifact exists (no per-call containment), so an
            # artifact that exists observed no voided calls. The slot is persisted so a runner
            # change can populate it without another schema bump.
            "structured_output": {
                "attempts": n_reps,
                "accepted": n_reps - cell.rejected,
                "rejected": cell.rejected,
                "void": 0,
            },
            "input_side_tokens": {
                "expected": n_reps,
                "observed": len(usage),
                "missing": n_reps - len(usage),
                "total": sum(u.total for u in usage),
                "by_class": {
                    "input": sum(u.input_tokens for u in usage),
                    "cache_read": sum(u.cache_read_tokens for u in usage),
                    "cache_write": sum(u.cache_write_tokens for u in usage),
                },
                "values": [u.total for u in usage],
            },
        }
        prov_so: dict[str, int] = p["structured_output"]  # type: ignore[assignment]
        prov_so["attempts"] += n_reps
        prov_so["accepted"] += n_reps - cell.rejected
        prov_so["rejected"] += cell.rejected
        prov_tokens: dict[str, object] = p["input_side_tokens"]  # type: ignore[assignment]
        prov_tokens["expected"] = int(prov_tokens["expected"]) + n_reps  # type: ignore[arg-type]
        prov_tokens["observed"] = int(prov_tokens["observed"]) + len(usage)  # type: ignore[arg-type]
        prov_tokens["missing"] = int(prov_tokens["missing"]) + (n_reps - len(usage))  # type: ignore[arg-type]
        prov_tokens["total"] = int(prov_tokens["total"]) + sum(u.total for u in usage)  # type: ignore[arg-type]
        by_class: dict[str, int] = prov_tokens["by_class"]  # type: ignore[assignment]
        by_class["input"] += sum(u.input_tokens for u in usage)
        by_class["cache_read"] += sum(u.cache_read_tokens for u in usage)
        by_class["cache_write"] += sum(u.cache_write_tokens for u in usage)
        if cell.dimension == RECALL:
            t = recall_by_type.setdefault(cell.finding_type, {"passed": 0, "total": 0})
            t["total"] += 1
            if maj:
                t["passed"] += 1
        elif cell.dimension == PRECISION:
            if maj:  # a safe fixture that produced a finding by majority = a false positive
                p["fp_count"] = int(p["fp_count"]) + 1  # type: ignore[arg-type]
        else:
            raise ValueError(f"unknown dimension {cell.dimension!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement_contract": meta.measurement_contract,
        "fixture_suite": meta.fixture_suite,
        "n_reps": n_reps,
        "majority_threshold": majority_threshold(n_reps),
        "prompt_version": meta.prompt_version,
        "prompt_digest": meta.prompt_digest,
        "harness_digest": meta.harness_digest,
        "fixture_digests": dict(meta.fixture_digests),
        "providers": providers,
    }


class Regression(NamedTuple):
    provider: str  # "" for run-level integrity
    kind: str
    detail: str


def compare(baseline: dict, candidate: dict) -> dict[str, object]:
    """Apply the role-aware ε=0 gate, after proving the two runs are COMPARABLE.

    Run-level integrity (always gating): equal `schema_version`, `measurement_contract`,
    `fixture_suite`, `n_reps`, `majority_threshold`, and identical `fixture_digests`; the
    acceptance set on both equals EXPECTED_ACCEPTANCE. Per recall fixture, extras must not
    increase on total OR max (the mc-2 over-emission gate). Both the
    analyze `prompt_version` AND the `prompt_digest` MUST DIFFER — a preregistered candidate that
    reuses either the VERSION or the exact prompt content is not a real prompt change and FAILS
    CLOSED (else ε=0 could "pass" without any prompt change under test).
    Per provider: model / profile_contract / role must match; the per_fixture set and each type's
    `total` must match (no silently-skipped fixtures); then recall no-decrease + FP no-increase.
    ACCEPTANCE providers veto; SUPPORTING providers are advisory. A candidate provider not in the
    baseline is a gating integrity failure. Token counts and structured-output yield counts are
    NOT gated (evidence, not acceptance criteria); `harness_digest` is provenance, surfaced by
    `provenance_notes()` and never gated — the baseline is immutable, so gating it would deadlock
    every future comparison after any harness edit.

    Returns {"passed", "regressions" (gating), "advisories" (non-gating),
    "providers": {p: {"role", "ok": bool|None}}}.
    """
    regressions: list[Regression] = []
    advisories: list[Regression] = []
    per_provider: dict[str, dict[str, object]] = {}
    passed = True

    # --- run-level integrity (always gating) ---
    # measurement_contract is the BLOCKING semantics identity: two artifacts collected under
    # different aggregation/grading/majority/acceptance semantics must never ε=0-compare, no
    # matter how their shapes line up. fixture_suite is the suite identity (fixture_digests
    # equality is the content gate; the suite label catches misuse at the naming layer).
    # (harness_digest stays non-gated — see provenance_notes.)
    for field_name in (
        "schema_version",
        "measurement_contract",
        "fixture_suite",
        "n_reps",
        "majority_threshold",
    ):
        b, c = baseline.get(field_name), candidate.get(field_name)
        if b != c:
            regressions.append(Regression("", "integrity", f"{field_name} mismatch: {b} vs {c}"))
            passed = False
    if baseline.get("fixture_digests") != candidate.get("fixture_digests"):
        regressions.append(
            Regression("", "integrity", "fixture_digests mismatch (fixture set/content differs)")
        )
        passed = False
    base_accept = {p for p, m in baseline["providers"].items() if m.get("role") == ACCEPTANCE}
    cand_accept = {p for p, m in candidate["providers"].items() if m.get("role") == ACCEPTANCE}
    if base_accept != EXPECTED_ACCEPTANCE or cand_accept != EXPECTED_ACCEPTANCE:
        regressions.append(
            Regression(
                "",
                "integrity",
                f"acceptance set != {sorted(EXPECTED_ACCEPTANCE)} "
                f"(baseline={sorted(base_accept)} candidate={sorted(cand_accept)})",
            )
        )
        passed = False
    # A preregistered candidate MUST bump BOTH the VERSION and the prompt content — reusing either
    # means no real prompt change is under test, so fail closed rather than passing ε=0 vacuously.
    for field_name, human in (
        ("prompt_version", "analyze VERSION"),
        ("prompt_digest", "prompt content"),
    ):
        if baseline.get(field_name) == candidate.get(field_name):
            regressions.append(
                Regression(
                    "",
                    "integrity",
                    f"{human} unchanged ({field_name} identical) — a preregistered candidate must "
                    "change both the VERSION and the prompt content",
                )
            )
            passed = False
    extra = set(candidate["providers"]) - set(baseline["providers"])
    if extra:
        regressions.append(
            Regression("", "integrity", f"unexpected candidate providers: {sorted(extra)}")
        )
        passed = False

    # --- per-provider comparison ---
    for provider, base_p in sorted(baseline["providers"].items()):
        role = base_p.get("role", ACCEPTANCE)
        gating = role == ACCEPTANCE
        bucket = regressions if gating else advisories
        cand_p = candidate["providers"].get(provider)
        if cand_p is None:
            bucket.append(
                Regression(provider, "missing", f"{role} provider missing from candidate")
            )
            per_provider[provider] = {"role": role, "ok": False if gating else None}
            if gating:
                passed = False
            continue
        integrity_detail = _provider_integrity(base_p, cand_p)
        if integrity_detail:
            bucket.append(Regression(provider, "integrity", integrity_detail))
            per_provider[provider] = {"role": role, "ok": False}
            if gating:
                passed = False
            continue
        ok = True
        for ftype in sorted(base_p["recall_by_type"]):
            b = base_p["recall_by_type"][ftype]
            c = cand_p["recall_by_type"][ftype]
            if c["passed"] < b["passed"]:
                ok = False
                bucket.append(
                    Regression(
                        provider,
                        "recall",
                        f"{ftype}: recall {c['passed']}/{b['total']} < "
                        f"baseline {b['passed']}/{b['total']}",
                    )
                )
        if cand_p["fp_count"] > base_p["fp_count"]:
            ok = False
            bucket.append(
                Regression(
                    provider,
                    "false_positive",
                    f"false positives {cand_p['fp_count']} > baseline {base_p['fp_count']}",
                )
            )
        # Extras gate (mc-2): per recall fixture, candidate TOTAL <= baseline total AND
        # candidate MAX <= baseline max. Total alone misses (1,1,1)->(0,0,3) — equal mass,
        # tripled worst rep; max closes it. Conservatism over unadjudicated emissions: an
        # increase is unproven behavior change, blocked pending human adjudication (which may
        # accept a genuine improvement via explicit amendment). Skipped when either side lacks
        # the evidence (pre-mc-2 dict) — the measurement_contract gate has already failed then.
        for fixture in sorted(base_p["per_fixture"]):
            b_extra = base_p["per_fixture"][fixture].get("extra_findings")
            c_extra = cand_p["per_fixture"].get(fixture, {}).get("extra_findings")
            if b_extra is None or c_extra is None:
                continue
            b_values, c_values = b_extra["values"], c_extra["values"]
            b_max = b_values[-1] if b_values else 0
            c_max = c_values[-1] if c_values else 0
            if c_extra["total"] > b_extra["total"] or c_max > b_max:
                ok = False
                bucket.append(
                    Regression(
                        provider,
                        "extras",
                        f"{fixture}: extra findings total {c_extra['total']}/max {c_max} exceed "
                        f"baseline total {b_extra['total']}/max {b_max}",
                    )
                )
        per_provider[provider] = {"role": role, "ok": ok}
        if gating and not ok:
            passed = False

    return {
        "passed": passed,
        "regressions": [r._asdict() for r in regressions],
        "advisories": [r._asdict() for r in advisories],
        "providers": per_provider,
    }


def token_delta(baseline: dict, candidate: dict) -> dict[str, object]:
    """Price the shrink's INPUT-token saving per provider — but only on complete paired coverage.

    The spec's cost claim is an OBSERVED token saving, so a telemetry gap must never read as one:
    if either side has any `missing` reps, or the two sides expected a different number of calls,
    that provider's cost evidence is `inconclusive` with a reason rather than a number. Complete
    coverage yields `measured` with per-call means (total/expected) and `delta_per_call`
    (candidate - baseline; NEGATIVE = the shrink saved tokens).

    Cost evidence only — never a gate. `compare()` decides accept/reject; this only prices it.
    """
    out: dict[str, object] = {}
    for provider, base_p in sorted(baseline["providers"].items()):
        cand_p = candidate["providers"].get(provider)
        if cand_p is None:
            out[provider] = {"status": "inconclusive", "reason": "provider missing from candidate"}
            continue
        b_t = base_p.get("input_side_tokens") or {}
        c_t = cand_p.get("input_side_tokens") or {}
        b_exp, c_exp = b_t.get("expected", 0), c_t.get("expected", 0)
        b_missing, c_missing = b_t.get("missing", 0), c_t.get("missing", 0)
        if not b_exp or not c_exp:
            out[provider] = {"status": "inconclusive", "reason": "no calls expected on a side"}
            continue
        if b_exp != c_exp:
            out[provider] = {
                "status": "inconclusive",
                "reason": f"call-count mismatch: baseline expected {b_exp}, candidate {c_exp}",
            }
            continue
        if b_missing or c_missing:
            out[provider] = {
                "status": "inconclusive",
                "reason": (
                    f"incomplete token telemetry: {b_missing}/{b_exp} baseline reps and "
                    f"{c_missing}/{c_exp} candidate reps reported no usage"
                ),
            }
            continue
        b_mean = b_t["total"] / b_exp
        c_mean = c_t["total"] / c_exp
        out[provider] = {
            "status": "measured",
            "expected_calls": b_exp,
            "baseline_total": b_t["total"],
            "candidate_total": c_t["total"],
            "baseline_mean_per_call": b_mean,
            "candidate_mean_per_call": c_mean,
            "delta_per_call": c_mean - b_mean,  # negative = saving
            # Which class the saving actually came from — on Claude the shrunk prefix shows up in
            # cache_read/cache_write, on Fireworks in input (it realizes ~no prefix cache).
            "baseline_by_class": b_t.get("by_class", {}),
            "candidate_by_class": c_t.get("by_class", {}),
        }
    return out


def preflight_comparability(baseline: dict, meta: RunMeta) -> list[str]:
    """Static reasons a PLANNED run could not be compared to `baseline` — computed with NO spend.

    Every field checked here is derivable before the paid loop: the schema/N contract is constants,
    the fixture + ground-truth digests come from files on disk, and the provider set / roles / model
    ids / profile-contract digests / token-accounting modes come from config + host profiles. So
    static drift fails for free instead of surfacing as an integrity regression after the paid
    loop. This is exactly the static half of `compare()`; the quality and cost halves
    necessarily need the observations.
    """
    reasons: list[str] = []
    if baseline.get("schema_version") != SCHEMA_VERSION:
        reasons.append(
            f"baseline schema_version {baseline.get('schema_version')} != {SCHEMA_VERSION}"
        )
    if baseline.get("n_reps") != meta.n_reps:
        reasons.append(f"n_reps {baseline.get('n_reps')} != planned {meta.n_reps}")
    if meta.n_reps != REQUIRED_REPS:
        reasons.append(f"planned n_reps {meta.n_reps} != required {REQUIRED_REPS}")
    if baseline.get("measurement_contract") != meta.measurement_contract:
        reasons.append(
            f"measurement_contract {baseline.get('measurement_contract')!r} != planned "
            f"{meta.measurement_contract!r} — the measurement semantics moved; rotate "
            "deliberately or ship a reviewed compatibility mapping"
        )
    if baseline.get("fixture_suite") != meta.fixture_suite:
        reasons.append(
            f"fixture_suite {baseline.get('fixture_suite')!r} != planned "
            f"{meta.fixture_suite!r} — the suite identity moved; freeze a new baseline for the "
            "new suite (never overwrite the old one)"
        )
    if baseline.get("fixture_digests") != dict(meta.fixture_digests):
        reasons.append("fixture_digests differ (fixture set, source, or ground-truth labels moved)")
    if baseline.get("prompt_version") == meta.prompt_version:
        reasons.append(
            f"prompt_version {meta.prompt_version!r} unchanged — bump the analyze VERSION"
        )
    if baseline.get("prompt_digest") == meta.prompt_digest:
        reasons.append("prompt_digest unchanged — the prompt content is identical")
    base_providers = set(baseline.get("providers", {}))
    planned = set(meta.providers)
    if base_providers != planned:
        reasons.append(f"provider set {sorted(planned)} != baseline {sorted(base_providers)}")
        return reasons
    for provider in sorted(planned):
        base_p = baseline["providers"][provider]
        want = meta.providers[provider]
        for field_name, planned_value in (
            ("role", want.role),
            ("model", want.model),
            ("profile_contract", want.profile_contract),
            ("token_accounting", want.token_accounting),
        ):
            if base_p.get(field_name) != planned_value:
                reasons.append(
                    f"{provider}.{field_name}: planned {planned_value!r} != baseline "
                    f"{base_p.get(field_name)!r}"
                )
    return reasons


def cost_objective(baseline: dict, candidate: dict) -> dict[str, object]:
    """Disposition of the SHRINK's COST objective — independent of the ε=0 quality gate.

    `proven` ONLY when every ACCEPTANCE provider has complete paired coverage AND a measured
    per-call REDUCTION (`delta_per_call < 0`). Any acceptance provider with incomplete telemetry
    makes the whole objective `inconclusive` — never a saving. Complete coverage with no reduction
    is `not_met`. Baseten (SUPPORTING) is advisory and never decides the objective.

    Pinned deliberately: `compare()` passing means quality did not regress, which says NOTHING about
    whether the shrink actually saved tokens. The two verdicts are reported separately and a passing
    quality gate must never be read as proving the cost objective.
    """
    delta = token_delta(baseline, candidate)
    acceptance = sorted(p for p, m in baseline["providers"].items() if m.get("role") == ACCEPTANCE)
    inconclusive = [p for p in acceptance if delta.get(p, {}).get("status") != "measured"]
    if inconclusive:
        return {
            "status": "inconclusive",
            "reason": f"incomplete cost evidence for {', '.join(inconclusive)}",
            "per_provider": delta,
        }
    no_saving = [p for p in acceptance if delta[p]["delta_per_call"] >= 0]
    if no_saving:
        return {
            "status": "not_met",
            "reason": f"no measured per-call reduction for {', '.join(no_saving)}",
            "per_provider": delta,
        }
    return {
        "status": "proven",
        "reason": (
            f"every acceptance provider measured a per-call reduction ({', '.join(acceptance)})"
        ),
        "per_provider": delta,
    }


def _provider_integrity(base_p: dict, cand_p: dict) -> str:
    """Return a non-empty reason if base/candidate providers are not COMPARABLE (identity or domain
    mismatch — no vacuous zero-defaulting), else ""."""
    for field_name in ("model", "profile_contract", "role", "token_accounting"):
        if base_p.get(field_name) != cand_p.get(field_name):
            return (
                f"{field_name} mismatch: {base_p.get(field_name)!r} vs {cand_p.get(field_name)!r}"
            )
    if set(base_p["recall_by_type"]) != set(cand_p["recall_by_type"]):
        return (
            f"finding-type domain mismatch: missing="
            f"{sorted(set(base_p['recall_by_type']) - set(cand_p['recall_by_type']))} "
            f"extra={sorted(set(cand_p['recall_by_type']) - set(base_p['recall_by_type']))}"
        )
    for ftype, b in base_p["recall_by_type"].items():
        c_total = cand_p["recall_by_type"][ftype]["total"]
        if b["total"] != c_total:
            return f"{ftype}: total {c_total} != baseline {b['total']}"
    if set(base_p["per_fixture"]) != set(cand_p["per_fixture"]):
        return "per_fixture set mismatch (a fixture was skipped or added)"
    return ""


_CSS = """
body{font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:2rem;max-width:1100px;
color:#1a1a1a;background:#fff}
h1{font-size:1.4rem;margin:0 0 .25rem}h2{font-size:1.05rem;margin:2rem 0 .5rem}
.sub{color:#666;margin:0 0 1.5rem}
table{border-collapse:collapse;width:100%;margin:.5rem 0 1rem;font-variant-numeric:tabular-nums}
th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}
th{background:#f6f6f6;font-weight:600}
tr:nth-child(even) td{background:#fafafa}
.badge{display:inline-block;padding:.1rem .5rem;border-radius:3px;font-weight:600;font-size:.85em}
.ok{background:#d7f0d7;color:#0a5c0a}.bad{background:#f8d7d7;color:#8a1010}
.warn{background:#fdf0d0;color:#7a5200}.mut{background:#eee;color:#555}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem;margin:0}
dt{color:#666}dd{margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
li{margin:.2rem 0}
@media(prefers-color-scheme:dark){body{background:#161616;color:#e6e6e6}
th{background:#242424}tr:nth-child(even) td{background:#1c1c1c}th,td{border-color:#333}
.sub,dt{color:#999}.ok{background:#12401a;color:#8fe39b}.bad{background:#4a1414;color:#ffb0b0}
.warn{background:#463200;color:#ffd479}.mut{background:#2a2a2a;color:#aaa}}
"""


def _esc(value: object) -> str:
    return html.escape(str(value))


def _badge(text: object, kind: str) -> str:
    """`kind` is a fixed internal class name; `text` is escaped, so no content can inject markup."""
    return f'<span class="badge {kind}">{_esc(text)}</span>'


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Rows may contain pre-built badge markup from `_badge` (whose text is already escaped); all
    other cell content must be escaped by the caller via `_esc`."""
    out = ["<table><thead><tr>", *(f"<th>{_esc(h)}</th>" for h in headers), "</tr></thead><tbody>"]
    out += ["<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows]
    out.append("</tbody></table>")
    return "".join(out)


def _page(title: str, body: str) -> str:
    return (
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>{body}</body></html>"
    )


def _recall_totals(provider_data: dict) -> tuple[int, int]:
    by_type = provider_data.get("recall_by_type", {})
    return (
        sum(t["passed"] for t in by_type.values()),
        sum(t["total"] for t in by_type.values()),
    )


def render_run_html(data: dict, *, title: str) -> str:
    """Human-readable view of ONE frozen run (baseline or candidate attempt). Pure + deterministic.

    A rendering of the JSON evidence, never a substitute for it: everything shown here is read back
    out of the artifact, so the report can be regenerated at any time and cannot drift from the
    evidence it describes.
    """
    validity = run_validity(data)
    providers: dict = data.get("providers", {})
    head = (
        f"<h1>{_esc(title)}</h1><p class=sub>"
        + (
            _badge("VALID EVIDENCE", "ok")
            if validity["valid"]
            else _badge("VOID — failed to measure", "warn")
        )
        + f" {_esc(validity['reason'])}</p>"
        + "<dl>"
        + f"<dt>prompt version</dt><dd>{_esc(data.get('prompt_version'))}</dd>"
        + f"<dt>prompt digest</dt><dd>{_esc(str(data.get('prompt_digest'))[:16])}…</dd>"
        + f"<dt>reps</dt><dd>{_esc(data.get('n_reps'))} "
        + f"(majority ≥ {_esc(data.get('majority_threshold'))})</dd>"
        + f"<dt>fixtures</dt><dd>{_esc(len(data.get('fixture_digests', {})))}</dd>"
        + f"<dt>schema</dt><dd>v{_esc(data.get('schema_version'))}</dd>"
        + "</dl>"
    )

    rows = []
    for name, p in sorted(providers.items()):
        passed, total = _recall_totals(p)
        tokens = p.get("input_side_tokens", {})
        so = p.get("structured_output")
        gating = p.get("role") == ACCEPTANCE
        rows.append(
            [
                _esc(name),
                _badge(p.get("role"), "mut" if not gating else "ok"),
                _esc(p.get("model")),
                _esc(f"{passed}/{total}"),
                _esc(p.get("fp_count")),
                # None = unrecorded under schema v2 — distinct from a measured zero-rejection run.
                (
                    _esc(f"{so['accepted']}/{so['attempts']}")
                    if so
                    else _badge("unrecorded (v2)", "mut")
                ),
                _esc(f"{tokens.get('total', 0):,}"),
                (
                    _badge(f"{tokens.get('missing', 0)} missing", "warn")
                    if tokens.get("missing")
                    else _esc("complete")
                ),
            ]
        )
    summary = "<h2>Providers</h2>" + _table(
        [
            "provider",
            "role",
            "model",
            "recall",
            "false positives",
            "yield (accepted/attempts)",
            "input-side tokens",
            "telemetry",
        ],
        rows,
    )

    ftypes = sorted({t for p in providers.values() for t in p.get("recall_by_type", {})})
    names = sorted(providers)
    recall_rows = [
        [
            _esc(ft),
            *(
                _esc(
                    f"{providers[n]['recall_by_type'][ft]['passed']}"
                    f"/{providers[n]['recall_by_type'][ft]['total']}"
                    if ft in providers[n].get("recall_by_type", {})
                    else "—"
                )
                for n in names
            ),
        ]
        for ft in ftypes
    ]
    recall = "<h2>Recall by finding type</h2>" + _table(["finding type", *names], recall_rows)

    token_rows = []
    for name in names:
        t = providers[name].get("input_side_tokens", {})
        by_class = t.get("by_class", {})
        token_rows.append(
            [
                _esc(name),
                _esc(providers[name].get("token_accounting")),
                _esc(f"{t.get('observed', 0)}/{t.get('expected', 0)}"),
                _esc(f"{t.get('total', 0):,}"),
                _esc(f"{by_class.get('input', 0):,}"),
                _esc(f"{by_class.get('cache_read', 0):,}"),
                _esc(f"{by_class.get('cache_write', 0):,}"),
            ]
        )
    tokens_tbl = (
        "<h2>Input-side token evidence</h2>"
        "<p class=sub>The classes are disjoint (normalized at the wrapper boundary), so "
        "total = input + cache_read + cache_write. On Claude the cached prefix sits in "
        "cache_read; on Fireworks it sits in input.</p>"
        + _table(
            [
                "provider",
                "§8a accounting",
                "observed/expected",
                "total",
                "input",
                "cache read",
                "cache write",
            ],
            token_rows,
        )
    )
    return _page(title, head + summary + recall + tokens_tbl)


def render_comparison_html(baseline: dict, candidate: dict, *, title: str) -> str:
    """The GATE view: the two independent verdicts side by side, plus the cost evidence.

    Quality and cost are rendered as separate verdicts because they ARE separate: a green ε=0 gate
    proves no regression, never a saving. Acceptance needs both.
    """
    verdict = compare(baseline, candidate)
    cost = cost_objective(baseline, candidate)
    quality_ok = bool(verdict["passed"])
    proven = cost["status"] == "proven"
    accepted = quality_ok and proven

    head = (
        f"<h1>{_esc(title)}</h1><p class=sub>"
        + _badge("ACCEPTED" if accepted else "NOT ACCEPTED", "ok" if accepted else "bad")
        + " — acceptance requires BOTH verdicts below.</p><dl>"
        + f"<dt>baseline prompt</dt><dd>{_esc(baseline.get('prompt_version'))} "
        + f"({_esc(str(baseline.get('prompt_digest'))[:12])}…)</dd>"
        + f"<dt>candidate prompt</dt><dd>{_esc(candidate.get('prompt_version'))} "
        + f"({_esc(str(candidate.get('prompt_digest'))[:12])}…)</dd>"
        + "</dl>"
    )

    quality = (
        "<h2>Quality gate (ε = 0)</h2><p>"
        + _badge("PASS" if quality_ok else "FAIL", "ok" if quality_ok else "bad")
        + " recall must not decrease and false positives must not increase, per acceptance "
        "provider per finding type.</p>"
    )
    regressions: list = verdict["regressions"]  # type: ignore[assignment]
    advisories: list = verdict["advisories"]  # type: ignore[assignment]
    if regressions:
        quality += (
            "<ul>"
            + "".join(
                f"<li>{_badge(r['kind'], 'bad')} {_esc(r['provider'] or 'run')}: "
                f"{_esc(r['detail'])}</li>"
                for r in regressions
            )
            + "</ul>"
        )
    if advisories:
        quality += (
            "<h2>Advisories (non-gating)</h2><ul>"
            + "".join(
                f"<li>{_badge(a['kind'], 'mut')} {_esc(a['provider'] or 'run')}: "
                f"{_esc(a['detail'])}</li>"
                for a in advisories
            )
            + "</ul>"
        )

    cost_kind = {"proven": "ok", "not_met": "bad", "inconclusive": "warn"}[str(cost["status"])]
    cost_html = (
        "<h2>Cost objective</h2><p>"
        + _badge(str(cost["status"]).upper(), cost_kind)
        + f" {_esc(cost['reason'])}</p>"
    )
    per_provider: dict = cost["per_provider"]  # type: ignore[assignment]
    rows = []
    for name in sorted(per_provider):
        d = per_provider[name]
        if d.get("status") != "measured":
            rows.append(
                [_esc(name), _badge("inconclusive", "warn"), _esc(d.get("reason", "")), "—", "—"]
            )
            continue
        delta = d["delta_per_call"]
        rows.append(
            [
                _esc(name),
                _badge("measured", "ok"),
                _esc(f"{d['baseline_mean_per_call']:,.1f}"),
                _esc(f"{d['candidate_mean_per_call']:,.1f}"),
                _badge(f"{delta:+,.1f}", "ok" if delta < 0 else "bad"),
            ]
        )
    cost_html += _table(
        ["provider", "status", "baseline tokens/call", "candidate tokens/call", "delta"], rows
    )
    return _page(title, head + quality + cost_html)


def write_report(html_text: str, *, label: str) -> Path:
    """Write a derived HTML VIEW to the gitignored reports dir.

    Deliberately OVERWRITABLE, unlike `write_baseline`: a report renders immutable evidence, it is
    not evidence, so re-rendering it cannot rewrite history. Keeping views out of the tracked
    evidence dir is exactly what lets them stay freely regenerable without touching create-once.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / f"{label}.html"
    path.write_text(html_text, encoding="utf-8")
    return path


def write_baseline(data: dict[str, object], *, label: str) -> Path:
    """Persist a run to the TRACKED baselines dir (committable evidence, unlike gitignored
    `reports/`). `label` names the prompt version (the `analyze` VERSION string).

    IMMUTABLE — there is no overwrite path, for either a baseline or a candidate. A frozen baseline
    is the preregistered bar, and a candidate is acceptance evidence: if either could be rewritten,
    a run could be repeated until a majority landed and the earlier attempts erased, which is
    exactly the cherry-picking the pre-registration forbids. Creation is EXCLUSIVE (`open(..., "x")`
    i.e. O_EXCL), not exists()-then-write, which is not create-once under concurrent freezes.
    Use `write_attempt()` to record a repeated candidate run without destroying its predecessors.
    """
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{label}.json"
    try:
        with path.open("x", encoding="utf-8") as fh:  # atomic: fails if it already exists
            fh.write(json.dumps(data, indent=2, sort_keys=True))
    except FileExistsError as exc:
        raise FileExistsError(
            f"{label!r} already exists at {path} — refusing to overwrite committed evidence "
            "(use write_attempt() to record another attempt, or delete it deliberately)"
        ) from exc
    return path


def write_attempt(data: dict[str, object], *, label_prefix: str) -> Path:
    """Record a candidate run as the next NUMBERED attempt, preserving every earlier one.

    Every completed attempt survives (`<prefix>-attempt-1.json`, `-attempt-2.json`, ...), so a
    re-run cannot quietly replace a worse result — the full history of attempts is the evidence that
    the reported one wasn't cherry-picked. Race-safe: the ordinal is claimed via `write_baseline`'s
    exclusive creation, so a loser simply advances to the next free ordinal.

    Preserving attempts is necessary but NOT sufficient to prevent cherry-picking — see
    `authoritative_attempt()` for the decision rule that makes acceptance monotonic.
    """
    for n in itertools.count(1):
        try:
            return write_baseline(data, label=f"{label_prefix}-attempt-{n}")
        except FileExistsError:
            continue
    raise AssertionError("unreachable: itertools.count is infinite")  # pragma: no cover


def run_validity(data: dict) -> dict[str, object]:
    """Is this run VALID EVIDENCE, or did its execution simply not complete?

    VALID = every ACCEPTANCE provider reported COMPLETE token telemetry. Validity is deliberately
    NOT about the result: an unfavorable-but-complete run ("the shrink held quality but saved
    nothing") IS a valid answer and stands. An INVALID run is one that failed to MEASURE — a
    transient telemetry omission is not evidence about the prompt, so it is void and may be re-run.

    This is the spec's rep rule at the attempt level: an errored rep is voided and re-run, an
    unfavorable rep is not. Without the distinction, one dropped usage payload would permanently
    decide the experiment.
    """
    providers = data.get("providers", {})
    present_acceptance = {p for p, m in providers.items() if m.get("role") == ACCEPTANCE}
    if present_acceptance != EXPECTED_ACCEPTANCE:
        # A dict without the full acceptance set (partial/hand-made artifact) must never read as
        # valid evidence — an empty providers map would otherwise pass vacuously and could become
        # the authoritative attempt for its prompt identity.
        return {
            "valid": False,
            "reason": (
                f"acceptance providers {sorted(present_acceptance)} != "
                f"{sorted(EXPECTED_ACCEPTANCE)} — not a complete run"
            ),
        }
    incomplete = []
    for provider, m in sorted(providers.items()):
        if m.get("role") != ACCEPTANCE:
            continue  # SUPPORTING telemetry never decides anything, incl. validity
        tokens = m.get("input_side_tokens") or {}
        if not tokens.get("expected", 0) or tokens.get("missing", 0):
            incomplete.append(provider)
    if incomplete:
        return {"valid": False, "reason": f"incomplete token telemetry for {', '.join(incomplete)}"}
    return {"valid": True, "reason": "every acceptance provider reported complete telemetry"}


def authoritative_attempt(label_prefix: str) -> Path | None:
    """The attempt that DECIDES this prompt identity — the first VALID one — or None if undecided.

    Pre-registered rule: **the first VALID attempt wins, permanently.** Keeping every attempt on
    disk preserves evidence but does not, on its own, stop cherry-picking — without a decision rule
    a failed attempt-1 could be re-run until an attempt-N came back green, and the green one would
    be the one reported. First-valid-wins makes acceptance monotonic: a later attempt is recorded
    but can never promote a failure to a pass.

    INVALID attempts (see `run_validity`) are preserved and SKIPPED, not authoritative: failing to
    measure is not a result, so it must not decide the experiment. That is the ONLY re-run path.

    A valid attempt is authoritative PERMANENTLY. There is deliberately no delete-and-re-decide
    escape hatch: these artifacts are untracked until committed, so deleting one leaves no trace in
    exactly the window between the paid run and the commit — "delete it deliberately" would be a
    silent cherry-pick, not an auditable act. Re-deciding requires a NEW prompt identity (bump the
    VERSION and change the content) or an explicit amendment to the pre-registration.
    """
    for n in itertools.count(1):
        path = BASELINE_DIR / f"{label_prefix}-attempt-{n}.json"
        if not path.exists():
            return None  # no valid attempt yet — this identity is still undecided
        data = json.loads(path.read_text(encoding="utf-8"))
        if run_validity(data)["valid"]:
            return path
    raise AssertionError("unreachable: itertools.count is infinite")  # pragma: no cover


def _upgrade_v2(data: dict[str, object]) -> dict[str, object]:
    """Deterministic IN-MEMORY v2 → v3 upgrade; the frozen on-disk artifact is never rewritten.

    v3 only ADDS fields, so the upgrade fills them with `None` = UNRECORDED-under-v2 — distinct
    from a measured zero, same discipline as token telemetry-absence. Every field `compare()`
    gates is identical across v2/v3, which is what keeps a v2 baseline comparable to a v3
    candidate without touching the evidence (FUP-238 route c).

    The `measurement_contract` and `fixture_suite` fills are NOT unknown-markers: they are the
    reviewed compatibility DECLARATIONS that v2 artifacts were collected under exemplar-mc-1
    semantics over the suite-v1 fixture set. Both are LITERALS on purpose — the constants have
    since rotated (mc-2 / suite-v2), and following them would falsely re-declare old evidence.
    The mc-1 → mc-2 rotation ships no compatibility mapping: the v2 baseline legitimately stops
    comparing (superseded as the bar by the suite-v2 freeze, not migrated).
    """
    data["schema_version"] = SCHEMA_VERSION
    data.setdefault("measurement_contract", "exemplar-mc-1")
    data.setdefault("fixture_suite", "suite-v1")
    data.setdefault("harness_digest", None)
    for p in data.get("providers", {}).values():  # type: ignore[union-attr]
        p.setdefault("structured_output", None)
        for fx in p.get("per_fixture", {}).values():
            fx.setdefault("structured_output", None)
            fx.setdefault("extra_findings", None)
    return data


def read_baseline(label: str) -> dict[str, object]:
    raw = (BASELINE_DIR / f"{label}.json").read_text(encoding="utf-8")
    data: dict[str, object] = json.loads(raw)
    if data.get("schema_version") not in _READABLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"baseline {label!r} has schema_version {data.get('schema_version')}, "
            f"expected one of {sorted(_READABLE_SCHEMA_VERSIONS)} — refusing to compare under "
            "the wrong contract"
        )
    if data.get("schema_version") != SCHEMA_VERSION:
        return _upgrade_v2(data)
    return data


def provenance_notes(baseline: dict, meta: RunMeta) -> list[str]:
    """Non-blocking provenance surface (FUP-238): does the planned harness match the baseline's?

    Deliberately NOT part of `preflight_comparability` and never gating: the frozen baseline is
    immutable, so a blocking digest check would permanently deadlock every future gate after any
    harness edit. A mismatch is a fact the reader weighs (did the collection semantics change?),
    not an integrity failure — the gated comparability fields have their own equality checks.
    """
    base_digest = baseline.get("harness_digest")
    if base_digest is None:
        return [
            "baseline does not record its producing harness (v2 artifact) — provenance rests on "
            "the git DAG / a committed sidecar, not the artifact"
        ]
    if base_digest != meta.harness_digest:
        return [
            f"harness digest differs from the baseline's ({str(base_digest)[:12]}… vs planned "
            f"{meta.harness_digest[:12]}…) — the harness changed between runs; weigh whether the "
            "collection semantics moved"
        ]
    return []
