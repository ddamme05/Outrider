# See specs/2026-05-23-trace-node.md M5 + M7 + M8.
"""Trace node unit tests — load-bearing contracts only.

Covers:
  - `TraceJoinIntegrityError` raises on duplicate proposal_hash across
    findings in `state.analysis_rounds` (M5 last-resort guard).
  - `_candidate_paths_for(import_string)` constructs module + package
    paths deterministically (the single module→path mapping rule per
    M8), with `_tier_paths_for` delegating suffix-strip ladder levels
    to it and `_resolve_via_probes` enforcing the bucket-level ladder:
    shallowest-level-wins, symbol verification on stripped-level hits,
    pass-level probe memo + budget (FUP-209).
  - Bucket dropping for already-traced findings (M1 + #025 point 5
    within-graph re-entry idempotency).

DB-touching integration tests (Phase 1 probes + Phase 2 fetch end-to-
end with mock GitHub) are deferred to a follow-up integration test
file; the unit tests pin the producer-deterministic invariants the
spec calls out explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from outrider.agent.nodes import trace as trace_module
from outrider.agent.nodes.trace import (
    MAX_PROBE_FETCHES_PER_PASS,
    TraceJoinIntegrityError,
    _aggregate_candidate_reasons,
    _bucket_candidates_by_finding,
    _build_proposal_hash_join,
    _candidate_paths_for,
    _dedupe_by_import_string,
    _ProbeBudget,
    _ProbeOutcome,
    _resolve_via_probes,
    _symbol_in_content,
    _tier_paths_for,
)
from outrider.audit.events import compute_finding_content_hash
from outrider.policy import EvidenceTier, FindingSeverity, FindingType
from outrider.policy.canonical import compute_candidate_id, compute_round_id
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.schemas import (
    AnalysisRound,
    ReviewDimension,
    ReviewFinding,
    ReviewState,
    TraceCandidate,
)
from outrider.schemas.pr_context import PRContext

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _build_finding(
    *, proposal_hash: str | None = None, file_path: str = "src/foo.py"
) -> ReviewFinding:
    """Build a ReviewFinding fixture; defaults to a fresh proposal_hash.
    `file_path` is parameterized so callers can vary it across siblings
    in one round (the AnalysisRound validator rejects duplicate
    content_hashes; varying file_path produces distinct hashes).

    The default `proposal_hash` is a fresh hex-64 string (two `uuid4().hex`
    concatenated, 32+32 chars). Two default `_build_finding()` calls
    therefore produce distinct hashes — required so AnalysisRound's
    `_enforce_findings_proposal_hash_unique` validator AND trace's
    `TraceJoinIntegrityError` guard don't false-fire on tests that
    happen to use two default findings in one round."""
    return ReviewFinding(
        finding_id=uuid4(),
        review_id=uuid4(),
        installation_id=12345,
        finding_type=FindingType.SQL_INJECTION,
        dimension=ReviewDimension.SECURITY,
        severity=FindingSeverity.CRITICAL,
        file_path=file_path,
        line_start=10,
        line_end=12,
        title="SQL injection",
        description="raw concat",
        evidence=f"concat at {file_path}:11",
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=file_path,
            line_start=10,
            line_end=12,
            finding_type=FindingType.SQL_INJECTION,
        ),
        proposal_hash=(proposal_hash if proposal_hash is not None else (uuid4().hex + uuid4().hex)),
    )


def _build_round(findings: tuple[ReviewFinding, ...], *, pass_index: int = 0) -> AnalysisRound:
    """Build an AnalysisRound with a canonical round_id derived from content."""
    now = datetime.now(UTC)
    files_examined = tuple(sorted({f.file_path for f in findings})) or ("src/foo.py",)
    return AnalysisRound(
        round_id=compute_round_id(
            pass_index=pass_index,
            files_examined=files_examined,
            files_skipped=(),
            finding_content_hashes=tuple(f.content_hash for f in findings),
        ),
        pass_index=pass_index,
        findings=findings,
        files_examined=files_examined,
        files_skipped=(),
        started_at=now,
        ended_at=now,
    )


def _build_state(rounds: tuple[AnalysisRound, ...]) -> ReviewState:
    return ReviewState(
        review_id=uuid4(),
        pr_context=PRContext(
            installation_id=1,
            owner="o",
            repo="r",
            pr_number=1,
            pr_title="x",
            head_sha="a" * 40,
            base_sha="b" * 40,
            author="dev",
            total_additions=5,
            total_deletions=2,
            changed_files=(),
        ),
        received_at=datetime.now(UTC),
        analysis_rounds=list(rounds),
    )


def _build_candidate(
    *,
    source_proposal_hash: str,
    import_string: str = "pkg.mod",
) -> TraceCandidate:
    reason = "x"
    return TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=source_proposal_hash,
            import_string=import_string,
            reason=reason,
        ),
        source_proposal_hash=source_proposal_hash,
        reason=reason,
        import_string=import_string,
    )


# ---------------------------------------------------------------------------
# M5: TraceJoinIntegrityError raises on duplicate proposal_hash.
# ---------------------------------------------------------------------------


def test_join_integrity_error_raises_on_duplicate_proposal_hash_across_rounds() -> None:
    """Two findings in two different rounds sharing the same proposal_hash —
    `_build_proposal_hash_join` raises `TraceJoinIntegrityError` with both
    finding_ids. M5's last-resort guard: the analyze-side
    `AnalysisRound._enforce_findings_proposal_hash_unique` validator
    catches within-round collisions; this guard catches cross-round
    collisions that would only arise from a `compute_proposal_hash`
    recipe drift (or a producer bypassing the validator).

    Within-round collisions are already rejected by the AnalysisRound
    validator before reaching trace, so this test uses two SEPARATE
    rounds to exercise the cross-round path.
    """
    shared_hash = "c" * 64
    finding_a = _build_finding(proposal_hash=shared_hash, file_path="src/foo.py")
    finding_b = _build_finding(proposal_hash=shared_hash, file_path="src/bar.py")
    round_1 = _build_round((finding_a,), pass_index=0)
    round_2 = _build_round((finding_b,), pass_index=1)
    state = _build_state((round_1, round_2))

    with pytest.raises(TraceJoinIntegrityError) as exc_info:
        _build_proposal_hash_join(state)

    assert exc_info.value.proposal_hash == shared_hash
    assert exc_info.value.first_finding_id == finding_a.finding_id
    assert exc_info.value.second_finding_id == finding_b.finding_id


def test_join_lookup_succeeds_on_distinct_proposal_hashes() -> None:
    """Distinct hashes across findings → join succeeds with one entry per."""
    finding_a = _build_finding(proposal_hash="e" * 64, file_path="src/alpha.py")
    finding_b = _build_finding(proposal_hash="f" * 64, file_path="src/beta.py")
    state = _build_state((_build_round((finding_a, finding_b)),))

    join = _build_proposal_hash_join(state)
    assert join == {
        "e" * 64: finding_a.finding_id,
        "f" * 64: finding_b.finding_id,
    }


# ---------------------------------------------------------------------------
# Bucket-build: unjoinable candidates drop silently.
# ---------------------------------------------------------------------------


def test_bucket_drops_candidates_whose_proposal_hash_has_no_finding() -> None:
    """Candidate whose source_proposal_hash isn't in the join is dropped
    (logged at DEBUG, not raised). Other candidates land in their
    proper bucket. The join contract is the producer-side responsibility;
    trace consumes state defensively."""
    finding = _build_finding(proposal_hash="1" * 64)
    candidate_in_join = _build_candidate(source_proposal_hash="1" * 64)
    candidate_unjoinable = _build_candidate(
        source_proposal_hash="2" * 64,
        import_string="pkg.unjoined",
    )
    join = {"1" * 64: finding.finding_id}

    buckets = _bucket_candidates_by_finding(
        (candidate_in_join, candidate_unjoinable),
        join,
    )
    assert set(buckets.keys()) == {finding.finding_id}
    assert buckets[finding.finding_id] == [candidate_in_join]


# ---------------------------------------------------------------------------
# M8: probe-path construction is deterministic.
# ---------------------------------------------------------------------------


def test_candidate_paths_for_emits_module_and_package_forms() -> None:
    """`foo.bar` → exactly two module-form candidate paths:
    `foo/bar.py` and `foo/bar/__init__.py`. Pinned per M8 — this is THE
    module→path mapping rule; the ladder's deeper levels reach it via
    `_tier_paths_for` (FUP-209)."""
    assert _candidate_paths_for("foo.bar") == ("foo/bar.py", "foo/bar/__init__.py")
    assert _candidate_paths_for("single") == ("single.py", "single/__init__.py")
    assert _candidate_paths_for("a.b.c") == ("a/b/c.py", "a/b/c/__init__.py")


def test_tier_paths_delegate_to_candidate_paths() -> None:
    """Every ladder level is exactly `_candidate_paths_for` of the
    stripped prefix — the mapping rule must not fork (a divergence
    would let tier-k silently probe differently-shaped paths than
    tier-0). Stripping the whole string yields () (never probed)."""
    for import_string in ("a.b.c.d", "svc.queries.run_query"):
        parts = import_string.split(".")
        assert _tier_paths_for(import_string, 0) == _candidate_paths_for(import_string)
        for strip_level in range(1, len(parts)):
            assert _tier_paths_for(import_string, strip_level) == _candidate_paths_for(
                ".".join(parts[:-strip_level])
            )
    assert _tier_paths_for("single", 1) == ()
    assert _tier_paths_for("a.b", 2) == ()


def test_symbol_in_content_admits_binding_contexts() -> None:
    """Symbol verification admits defining contexts only: def / async
    def / class, module-level bindings and annotations, import lines,
    and parenthesized multi-line from-import continuations."""
    assert _symbol_in_content("run_query", b"def run_query(): ...\n")
    assert _symbol_in_content("run_query", b"async def run_query(): ...\n")
    assert _symbol_in_content("UserService", b"class UserService:\n    ...\n")
    assert _symbol_in_content("run_query", b"from .queries import run_query\n")
    assert _symbol_in_content("run_query", b"import run_query\n")
    multiline_import = b"from .queries import (\n    run_query,\n    other,\n)\n"
    assert _symbol_in_content("run_query", multiline_import)
    assert _symbol_in_content("DEFAULT_TIMEOUT", b"DEFAULT_TIMEOUT = 30\n")
    assert _symbol_in_content("session", b"session: Session = make_session()\n")


def test_symbol_in_content_rejects_incidental_uses() -> None:
    """FUP-209 review F1: incidental appearances are NOT bindings — a
    bare word-boundary match falsely resolved candidates with common
    trailing names (get/data/id) to modules that never define them.
    Attribute access, comments, string literals, substrings, absent
    names, equality comparisons, and non-UTF-8 content all reject."""
    assert not _symbol_in_content("get", b"session.get(url)\n")
    assert not _symbol_in_content("data", b"# process data\n")
    assert not _symbol_in_content("id", b'x = {"id": 1}\n')
    assert not _symbol_in_content("ghost", b"")
    assert not _symbol_in_content("query", b"def run_query(): ...\n")
    assert not _symbol_in_content("flag", b"if flag == other:\n    ...\n")
    assert not _symbol_in_content("x", "é".encode("utf-16"))


# ---------------------------------------------------------------------------
# FUP-209: `_resolve_via_probes` suffix-strip ladder. Real models emit
# symbol-form candidates (`module.function`, `module.Class.method`); the
# ladder resolves them to the defining module. Bucket-level barriers
# (shallowest level wins) + symbol verification + pass-level memo/budget
# came out of the FUP-209 review round.
# ---------------------------------------------------------------------------


def _fake_fetch_for(
    real_files: dict[str, bytes],
) -> tuple[Callable[..., Awaitable[bytes | None]], list[str]]:
    """Build a fetch_file_content_at stand-in over a path→bytes repo
    snapshot. Returns (fake, probed-paths log). Unknown paths return
    None — the 404-equivalent probe negative."""
    probed: list[str] = []

    async def fake_fetch(*_args: object, path: str, **_kwargs: object) -> bytes | None:
        probed.append(path)
        return real_files.get(path)

    return fake_fetch, probed


async def _probe_outcome_for(
    monkeypatch: pytest.MonkeyPatch,
    *,
    import_strings: tuple[str, ...],
    real_files: dict[str, bytes],
    probe_memo: dict[str, bytes | None] | None = None,
    budget: int = MAX_PROBE_FETCHES_PER_PASS,
) -> tuple[_ProbeOutcome, list[str]]:
    """Run `_resolve_via_probes` for one bucket against a fake repo
    snapshot; return (outcome, probed-paths log)."""
    fake_fetch, probed = _fake_fetch_for(real_files)
    monkeypatch.setattr(trace_module, "fetch_file_content_at", fake_fetch)
    candidates = tuple(
        _build_candidate(source_proposal_hash="1" * 64, import_string=import_string)
        for import_string in import_strings
    )
    outcome = await _resolve_via_probes(
        candidates=candidates,
        gh_client=object(),  # type: ignore[arg-type]  # opaque pass-through to the fake
        owner="o",
        repo="r",
        head_sha="a" * 40,
        probe_memo=probe_memo if probe_memo is not None else {},
        probe_budget=_ProbeBudget(remaining=budget),
    )
    return outcome, probed


async def test_symbol_form_candidate_resolves_via_parent_module_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 core fix: `svc.queries.run_query` (symbol-form, as real
    models emit) misses both level-0 module paths, falls back one strip
    level, and resolves `svc/queries.py` (which names the symbol)."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query",),
        real_files={"svc/queries.py": b"def run_query(): ...\n"},
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "svc/queries.py"
    assert outcome.resolved_candidate_paths == ("svc/queries.py",)
    # Level 0 probed first (both module-form paths), level 1 after.
    assert probed == [
        "svc/queries/run_query.py",
        "svc/queries/run_query/__init__.py",
        "svc/queries.py",
        "svc/queries/__init__.py",
    ]


async def test_symbol_form_candidate_resolves_parent_package_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symbol defined in (or re-exported from) a package's `__init__.py`:
    `svc.queries.run_query` with `svc/queries/` a package resolves to
    `svc/queries/__init__.py` at strip level 1."""
    outcome, _ = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query",),
        real_files={"svc/queries/__init__.py": b"def run_query(): ...\n"},
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "svc/queries/__init__.py"


async def test_method_form_candidate_resolves_two_strip_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`app.services.UserService.authenticate` (method on a class — a
    real emission shape) needs strip level 2 to reach `app/services.py`;
    the level-2 hit verifies against the FIRST stripped component
    (`UserService`), the immediate child of the kept prefix."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("app.services.UserService.authenticate",),
        real_files={"app/services.py": b"class UserService:\n    def authenticate(self): ...\n"},
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "app/services.py"
    assert len(probed) == 6  # 2 paths per level x 3 levels


async def test_module_form_candidate_never_probes_parent_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression pin against flat (non-laddered) parent probing: a
    module-form candidate `app.models` in a normal package layout
    (`app/__init__.py` exists) must resolve to `app/models.py` — NOT go
    ambiguous against the parent `__init__.py` — and must not pay the
    deeper-level probes at all."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("app.models",),
        real_files={
            "app/models.py": b"class QueryBuilder: ...\n",
            "app/__init__.py": b"",
        },
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "app/models.py"
    assert probed == ["app/models.py", "app/models/__init__.py"]


async def test_sibling_fallback_cannot_demote_module_form_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 review (bucket pooling regression): the ladder level is a
    BUCKET-level barrier. `app.models` resolves at level 0, so the
    hallucinated sibling `app.ghost` never reaches its fallback levels —
    the parent `__init__.py` cannot pool into the aggregate and flip a
    clean resolution to ambiguous."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("app.models", "app.ghost"),
        real_files={
            "app/models.py": b"class QueryBuilder: ...\n",
            "app/__init__.py": b"",
        },
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "app/models.py"
    # Level 0 of BOTH candidates probed; no deeper level for either.
    assert probed == [
        "app/models.py",
        "app/models/__init__.py",
        "app/ghost.py",
        "app/ghost/__init__.py",
    ]


async def test_hallucinated_module_under_real_package_stays_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 review (false parent resolution): existence alone would
    resolve any hallucinated `pkg.ghost` to `pkg/__init__.py` — which
    exists in essentially every package. Symbol verification rejects
    the level-1 hit (the fetched parent must actually name the stripped
    component), keeping hallucinated and PR-deleted module candidates
    unresolved, exactly as before the fallback existed."""
    outcome, _ = await _probe_outcome_for(
        monkeypatch,
        import_strings=("pkg.ghost",),
        real_files={
            "pkg/__init__.py": b"",
            "pkg/real_module.py": b"def real(): ...\n",
        },
    )
    assert outcome.resolution_status == "unresolved"
    assert outcome.target_file is None
    assert outcome.resolved_candidate_paths == ()


async def test_full_string_module_wins_over_symbol_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BOTH readings exist (`svc/queries/run_query.py` is a real
    module AND `svc/queries.py` is real), level precedence resolves the
    full-string module — mirroring Python's own import resolution —
    rather than reporting ambiguous. Deliberate FUP-209 choice; the
    parent paths are never probed."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query",),
        real_files={
            "svc/queries/run_query.py": b"def run_query(): ...\n",
            "svc/queries.py": b"def run_query(): ...\n",
        },
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "svc/queries/run_query.py"
    assert probed == [
        "svc/queries/run_query.py",
        "svc/queries/run_query/__init__.py",
    ]


async def test_tier_one_module_and_package_both_real_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguity within a level is still a real outcome: both `x/y.py`
    and `x/y/__init__.py` real → ambiguous, no target_file."""
    outcome, _ = await _probe_outcome_for(
        monkeypatch,
        import_strings=("x.y",),
        real_files={"x/y.py": b"", "x/y/__init__.py": b""},
    )
    assert outcome.resolution_status == "ambiguous"
    assert outcome.target_file is None
    assert set(outcome.resolved_candidate_paths) == {"x/y.py", "x/y/__init__.py"}


async def test_two_symbol_candidates_on_distinct_parents_are_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live FUP-209 shape: two symbol-form candidates for ONE
    finding, resolving to two different (verified) parent modules, is
    genuinely ambiguous under M8's single-target-per-finding contract —
    pinned so the outcome is deliberate, not accidental."""
    outcome, _ = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query", "svc.utils.normalize_owner"),
        real_files={
            "svc/queries.py": b"def run_query(): ...\n",
            "svc/utils.py": b"def normalize_owner(): ...\n",
        },
    )
    assert outcome.resolution_status == "ambiguous"
    assert outcome.target_file is None
    assert set(outcome.resolved_candidate_paths) == {"svc/queries.py", "svc/utils.py"}


async def test_shared_parent_probes_once_and_resolves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 review (probe memo): sibling symbol-form candidates on
    the SAME module fetch the shared parent paths once (no byte-identical
    duplicate GitHub round-trips), and the deduped verified hit is a
    single real path → resolved, not ambiguous."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query", "svc.queries.other_func"),
        real_files={"svc/queries.py": b"def run_query(): ...\ndef other_func(): ...\n"},
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "svc/queries.py"
    assert len(probed) == len(set(probed))  # no duplicate fetches
    assert len(probed) == 6  # 4 level-0 paths + 2 shared level-1 paths


async def test_probe_budget_exhaustion_resolves_unresolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 review (hostile fetch bound): once the pass budget is
    consumed, no further probe fetches are issued and unfunded paths
    count as not-real — the candidate lands unresolved instead of
    burning GitHub rate limit."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query",),
        real_files={"svc/queries.py": b"def run_query(): ...\n"},
        budget=2,
    )
    assert outcome.resolution_status == "unresolved"
    assert len(probed) == 2  # only level 0 funded; deeper levels unfunded


async def test_seeded_memo_path_resolves_without_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FUP-209 review (in-PR seeding): a parent path whose head content
    is already in the memo (seeded from `pr_context.changed_files`)
    resolves without a GitHub round-trip for that path."""
    outcome, probed = await _probe_outcome_for(
        monkeypatch,
        import_strings=("svc.queries.run_query",),
        real_files={},
        probe_memo={"svc/queries.py": b"def run_query(): ...\n"},
    )
    assert outcome.resolution_status == "resolved"
    assert outcome.target_file == "svc/queries.py"
    assert "svc/queries.py" not in probed  # served from the memo
    assert probed == [
        "svc/queries/run_query.py",
        "svc/queries/run_query/__init__.py",
        "svc/queries/__init__.py",
    ]


# ---------------------------------------------------------------------------
# Round-N+1 regression: H1 — `_dedupe_by_import_string` keeps the audit
# event's `proposed_import_strings` set-semantic invariant under benign
# LLM behavior (same import_string, different reasons).
# ---------------------------------------------------------------------------


def test_dedupe_by_import_string_collapses_same_import_different_reason() -> None:
    """Round-N+1 H1 regression test: two TraceCandidates with the same
    `import_string` but different `reason` have distinct `candidate_id`s
    (content-hash over `(source_proposal_hash, import_string, reason)`)
    and both survive `state.trace_candidates`'s reducer. Without the
    dedup helper, `TraceDecisionEvent.proposed_import_strings`'s
    `_enforce_proposed_import_strings_unique` validator would raise mid-
    emit-loop on this benign LLM behavior, breaking the M7 audit-first
    contract. The dedup is order-stable (first occurrence wins).
    """
    proposal_hash = "1" * 64
    first_reason_candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="middleware.auth",
            reason="first reasoning",
        ),
        source_proposal_hash=proposal_hash,
        reason="first reasoning",
        import_string="middleware.auth",
    )
    second_reason_candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="middleware.auth",
            reason="alternative reasoning",
        ),
        source_proposal_hash=proposal_hash,
        reason="alternative reasoning",
        import_string="middleware.auth",
    )
    distinct_candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="handlers.login",
            reason="x",
        ),
        source_proposal_hash=proposal_hash,
        reason="x",
        import_string="handlers.login",
    )
    # Pre-condition: same import_string, different reason → distinct
    # candidate_ids (the bug the dedup defends against).
    assert first_reason_candidate.candidate_id != second_reason_candidate.candidate_id
    assert first_reason_candidate.import_string == second_reason_candidate.import_string

    deduped = _dedupe_by_import_string(
        (first_reason_candidate, second_reason_candidate, distinct_candidate)
    )

    # First occurrence wins → first_reason_candidate's `reason` survives;
    # second_reason_candidate is dropped; distinct_candidate kept.
    assert len(deduped) == 2
    assert deduped[0] is first_reason_candidate
    assert deduped[1] is distinct_candidate
    # The audit-event invariant: extracting import_strings yields a set
    # with no duplicates (what the validator enforces).
    import_strings = tuple(c.import_string for c in deduped)
    assert len(import_strings) == len(set(import_strings))


def test_dedupe_by_import_string_preserves_single_candidate() -> None:
    """Trivial case: one candidate → unchanged tuple."""
    proposal_hash = "2" * 64
    candidate = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="pkg.mod",
            reason="x",
        ),
        source_proposal_hash=proposal_hash,
        reason="x",
        import_string="pkg.mod",
    )
    assert _dedupe_by_import_string((candidate,)) == (candidate,)


def test_dedupe_by_import_string_empty_input() -> None:
    """Empty input → empty tuple. Defensive: trace's pre-condition is a
    non-empty bucket, but the helper is total."""
    assert _dedupe_by_import_string(()) == ()


# ---------------------------------------------------------------------------
# Coverage gap from cross-file consistency audit: _aggregate_candidate_reasons
# ---------------------------------------------------------------------------


def test_aggregate_candidate_reasons_concatenates_with_separator() -> None:
    """Per the audit-row contract: aggregated reason carries
    `<import_string>: <reason>` per candidate, joined with ` | `."""
    proposal_hash = "3" * 64
    c1 = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="pkg.alpha",
            reason="first",
        ),
        source_proposal_hash=proposal_hash,
        reason="first",
        import_string="pkg.alpha",
    )
    c2 = TraceCandidate(
        candidate_id=compute_candidate_id(
            source_proposal_hash=proposal_hash,
            import_string="pkg.beta",
            reason="second",
        ),
        source_proposal_hash=proposal_hash,
        reason="second",
        import_string="pkg.beta",
    )

    aggregated = _aggregate_candidate_reasons((c1, c2))
    assert aggregated == "pkg.alpha: first | pkg.beta: second"


def test_aggregate_candidate_reasons_truncates_to_500_chars() -> None:
    """Aggregated reason that exceeds 500 chars truncates to 497 + ellipsis
    (matching the schema's max_length=500 cap on TraceDecisionEvent.reason).
    The truncation is lossy and biased toward early candidates — the
    architectural lens flagged this as the structured-tuple FUP."""
    proposal_hash = "4" * 64
    # Construct candidates whose aggregate exceeds 500 chars.
    candidates = tuple(
        TraceCandidate(
            candidate_id=compute_candidate_id(
                source_proposal_hash=proposal_hash,
                import_string=f"pkg.mod{i}",
                reason="x" * 100,
            ),
            source_proposal_hash=proposal_hash,
            reason="x" * 100,
            import_string=f"pkg.mod{i}",
        )
        for i in range(10)
    )

    aggregated = _aggregate_candidate_reasons(candidates)
    # Contract: respect the schema's 500-char cap on
    # TraceDecisionEvent.reason. Don't pin the truncation marker shape
    # ("..." today; a future fix may use a different marker per FUP-075's
    # structured-field migration) — only that truncation occurred and
    # the cap holds. Single-candidate aggregation provides a known-shorter
    # baseline to prove truncation HAPPENED.
    assert len(aggregated) <= 500
    assert aggregated != _aggregate_candidate_reasons(candidates[:1])
