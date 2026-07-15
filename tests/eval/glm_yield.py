"""Two-host GLM structured-output yield collector — deterministic core (FUP-219).

A SEPARATE instrument from the exemplar ε=0 baseline (`exemplar_baseline.py`): it measures
structured-output yield for the two GLM hosts (Fireworks strict constrained-decoding vs Baseten
soft-fenced) over the same 20 exemplar fixtures, persisting RAW accepted/rejected/void counts —
never a derived rate, so a later yield-metric definition change costs nothing. It deliberately
records NO detection data: recall/FP authority stays with the frozen analyze-exemplars baseline
under its pinned pooled-expectation estimand, and keeping detection out of this artifact is what
prevents the two collection protocols from being silently pooled.

Artifact tier: TRACKED, immutable, create-once evidence under `tests/eval/baselines/glm-yield/`
(same discipline as the exemplar baselines — no overwrite path, re-decide via a new prompt
identity). The PAID collector loop lives in `test_glm_yield.py` behind two explicit opt-ins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from .exemplar_baseline import BASETEN_GLM, FIREWORKS_GLM, harness_source_digest

if TYPE_CHECKING:
    from collections.abc import Sequence

YIELD_SCHEMA_VERSION = 1

# The BLOCKING measurement-semantics identity for THIS instrument (separate from the exemplar
# baseline's MEASUREMENT_CONTRACT — different experiment, different semantics). It covers:
# - one analyze pass per rep over a SINGLE-file fixture (asserted by the collector);
# - ACCEPTED = the pass completed with zero AnalyzeResponseRejectedEvents for the file;
# - REJECTED = the pass completed with exactly one (n_rejected > 1 is impossible single-file and
#   fails loud);
# - VOID = the pass raised LLMProviderError, recorded as the exception TYPE name (type-only, no
#   message content) and excluded from the yield denominator; any other exception is a harness
#   bug and propagates;
# - conservation: accepted + rejected + void == n_reps per (host, fixture) — every planned call
#   accounted for, no silent re-runs;
# - yield = accepted / (accepted + rejected), recomputed by readers from the raw counts.
# Any change to these semantics rotates this string or ships a reviewed compatibility mapping.
YIELD_CONTRACT = "glm-yield-mc-1"

YIELD_REPS = 3
YIELD_DIR = Path(__file__).parent / "baselines" / "glm-yield"
_YIELD_SOURCE_FILES = ("glm_yield.py", "test_glm_yield.py")

ACCEPTED_OUTCOME = "accepted"
REJECTED_OUTCOME = "rejected"
VOID_OUTCOME = "void"
_OUTCOMES = frozenset({ACCEPTED_OUTCOME, REJECTED_OUTCOME, VOID_OUTCOME})

EXPECTED_HOSTS = frozenset({FIREWORKS_GLM, BASETEN_GLM})


def yield_source_digest() -> str:
    """Digest of THIS instrument's producing source, for the artifact's `harness_digest`."""
    return harness_source_digest(_YIELD_SOURCE_FILES)


class YieldCall(NamedTuple):
    """One paid call's outcome. `void_reason` is the exception type name, non-empty iff void."""

    host: str  # logical key: fireworks-glm | baseten-glm
    fixture: str
    outcome: str  # ACCEPTED_OUTCOME | REJECTED_OUTCOME | VOID_OUTCOME
    void_reason: str = ""


class YieldHostMeta(NamedTuple):
    model: str  # resolved model id (provenance)
    profile_contract: str  # #056 profile_contract_digest


class YieldRunMeta(NamedTuple):
    n_reps: int
    prompt_version: str  # analyze VERSION the calls ran under
    prompt_digest: str  # sha256 of the analyze prompt content
    # fixture -> semantic digest via exemplar_baseline.fixture_content_digest — the SAME recipe as
    # the frozen baseline, so the two artifacts' fixture identities cross-reference exactly.
    fixture_digests: dict[str, str]
    hosts: dict[str, YieldHostMeta]
    harness_digest: str = ""  # yield_source_digest(); aggregate_yield rejects ""
    yield_contract: str = YIELD_CONTRACT


def aggregate_yield(calls: Sequence[YieldCall], meta: YieldRunMeta) -> dict[str, object]:
    """Fold per-call outcomes into the immutable yield artifact, enforcing conservation.

    Raises on any breach so an unclean collection is never persisted: exactly the two expected
    hosts, every host over every fixture, `accepted + rejected + void == n_reps` per (host,
    fixture), outcome values in the enum, void reasons present iff void, and non-empty
    harness/contract identities. Raw counts only — no derived rate is persisted.
    """
    if meta.n_reps != YIELD_REPS:
        raise ValueError(f"n_reps must be exactly {YIELD_REPS}; got {meta.n_reps}")
    if not meta.harness_digest:
        raise ValueError("meta.harness_digest is empty — populate via yield_source_digest()")
    if not meta.yield_contract:
        raise ValueError("meta.yield_contract is empty — see YIELD_CONTRACT")
    if set(meta.hosts) != EXPECTED_HOSTS:
        raise ValueError(
            f"hosts must be exactly {sorted(EXPECTED_HOSTS)} (the two-host delta is the point); "
            f"got {sorted(meta.hosts)}"
        )
    seen_hosts = {c.host for c in calls}
    if seen_hosts != EXPECTED_HOSTS:
        raise ValueError(f"observed hosts {sorted(seen_hosts)} != {sorted(EXPECTED_HOSTS)}")
    seen_fixtures = {c.fixture for c in calls}
    if set(meta.fixture_digests) != seen_fixtures:
        raise ValueError(
            "fixture_digests must cover exactly the observed fixtures: "
            f"digests={sorted(meta.fixture_digests)} observed={sorted(seen_fixtures)}"
        )

    cells: dict[tuple[str, str], dict[str, object]] = {}
    for c in calls:
        if c.outcome not in _OUTCOMES:
            raise ValueError(f"unknown outcome {c.outcome!r} for {c.host}/{c.fixture}")
        if (c.outcome == VOID_OUTCOME) != bool(c.void_reason):
            raise ValueError(
                f"{c.host}/{c.fixture}: void_reason must be present iff outcome is void "
                f"(outcome={c.outcome!r}, void_reason={c.void_reason!r})"
            )
        cell = cells.setdefault(
            (c.host, c.fixture),
            {"accepted": 0, "rejected": 0, "void": 0, "void_reasons": []},
        )
        cell[c.outcome] = int(cell[c.outcome]) + 1  # type: ignore[call-overload]
        if c.void_reason:
            cell["void_reasons"].append(c.void_reason)  # type: ignore[union-attr]

    hosts: dict[str, dict[str, object]] = {}
    for host in sorted(meta.hosts):
        host_fixtures = {fx for (h, fx) in cells if h == host}
        if host_fixtures != seen_fixtures:
            raise ValueError(
                f"{host} covers {len(host_fixtures)} fixtures, expected all "
                f"{len(seen_fixtures)} — every host must run every fixture"
            )
        hmeta = meta.hosts[host]
        totals = {"attempts_planned": 0, "accepted": 0, "rejected": 0, "void": 0}
        per_fixture: dict[str, dict[str, object]] = {}
        for fx in sorted(seen_fixtures):
            cell = cells[(host, fx)]
            total = int(cell["accepted"]) + int(cell["rejected"]) + int(cell["void"])  # type: ignore[arg-type]
            if total != meta.n_reps:
                raise ValueError(
                    f"{host}/{fx}: conservation violated — accepted+rejected+void = {total}, "
                    f"expected exactly {meta.n_reps} (a missing/extra call means the collection "
                    "was not clean; do not persist it)"
                )
            per_fixture[fx] = {
                "accepted": cell["accepted"],
                "rejected": cell["rejected"],
                "void": cell["void"],
                "void_reasons": sorted(cell["void_reasons"]),  # type: ignore[arg-type]
            }
            totals["attempts_planned"] += meta.n_reps
            for k in ("accepted", "rejected", "void"):
                totals[k] += int(cell[k])  # type: ignore[arg-type]
        hosts[host] = {
            "model": hmeta.model,
            "profile_contract": hmeta.profile_contract,
            "per_fixture": per_fixture,
            "totals": totals,
        }

    return {
        "schema_version": YIELD_SCHEMA_VERSION,
        "yield_contract": meta.yield_contract,
        "n_reps": meta.n_reps,
        "prompt_version": meta.prompt_version,
        "prompt_digest": meta.prompt_digest,
        "harness_digest": meta.harness_digest,
        "fixture_digests": dict(meta.fixture_digests),
        "hosts": hosts,
    }


def yield_rate(host_block: dict) -> float | None:
    """`accepted / (accepted + rejected)` from a host's totals — the ONE reader-side derivation,
    here so consumers don't re-derive it inconsistently. None when no valid attempts (all void)."""
    totals = host_block["totals"]
    valid = int(totals["accepted"]) + int(totals["rejected"])
    if not valid:
        return None
    return int(totals["accepted"]) / valid


def write_yield_artifact(data: dict[str, object], *, label: str) -> Path:
    """Persist a collection to the tracked yield-evidence dir. IMMUTABLE, create-once (`O_EXCL`) —
    same no-overwrite discipline as `exemplar_baseline.write_baseline`: re-deciding a yield read
    requires a new prompt identity, never replacing recorded evidence."""
    YIELD_DIR.mkdir(parents=True, exist_ok=True)
    path = YIELD_DIR / f"{label}.json"
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(data, indent=2, sort_keys=True))
    except FileExistsError as exc:
        raise FileExistsError(
            f"{label!r} already exists at {path} — refusing to overwrite committed evidence; "
            "re-decide via a new prompt identity"
        ) from exc
    return path


def read_yield_artifact(label: str) -> dict[str, object]:
    raw = (YIELD_DIR / f"{label}.json").read_text(encoding="utf-8")
    data: dict[str, object] = json.loads(raw)
    if data.get("schema_version") != YIELD_SCHEMA_VERSION:
        raise ValueError(
            f"yield artifact {label!r} has schema_version {data.get('schema_version')}, "
            f"expected {YIELD_SCHEMA_VERSION}"
        )
    return data
