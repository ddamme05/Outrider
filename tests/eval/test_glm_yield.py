"""Offline tests for the GLM yield collector's deterministic core (`glm_yield.py`) PLUS the
opt-in PAID collection loop (FUP-219's remaining measurement).

The offline tests pin the collector's contract: conservation (accepted + rejected + void ==
n_reps per host/fixture), the exact two-host domain, void-reason discipline (exception TYPE
only, present iff void), create-once persistence, and the wiring guard that drives the real
collection loop — all three outcome paths — with no spend.

Run offline: uv run pytest tests/eval/test_glm_yield.py --is-eval -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .exemplar_baseline import BASETEN_GLM, FIREWORKS_GLM, fixture_content_digest
from .glm_yield import (
    ACCEPTED_OUTCOME,
    EXPECTED_HOSTS,
    REJECTED_OUTCOME,
    VOID_OUTCOME,
    YIELD_CONTRACT,
    YIELD_DIR,
    YIELD_REPS,
    YieldCall,
    YieldHostMeta,
    YieldRunMeta,
    aggregate_yield,
    read_yield_artifact,
    write_yield_artifact,
    yield_rate,
    yield_source_digest,
)

_FX_A = "fx_a"
_FX_B = "fx_b"
_DIGESTS = {_FX_A: "d-a", _FX_B: "d-b"}


def _meta(**overrides) -> YieldRunMeta:
    base = YieldRunMeta(
        n_reps=YIELD_REPS,
        prompt_version="analyze-v10",
        prompt_digest="dig",
        fixture_digests=dict(_DIGESTS),
        hosts={
            FIREWORKS_GLM: YieldHostMeta("fw-model", "fw-contract"),
            BASETEN_GLM: YieldHostMeta("bt-model", "bt-contract"),
        },
        harness_digest="h-digest",
    )
    return base._replace(**overrides)


def _calls(
    fw: tuple[int, int, int] = (3, 0, 0), bt: tuple[int, int, int] = (3, 0, 0)
) -> list[YieldCall]:
    """Per host: (accepted, rejected, void) call counts on BOTH fixtures."""
    out: list[YieldCall] = []
    for host, (acc, rej, void) in ((FIREWORKS_GLM, fw), (BASETEN_GLM, bt)):
        for fx in (_FX_A, _FX_B):
            out += [YieldCall(host, fx, ACCEPTED_OUTCOME)] * acc
            out += [YieldCall(host, fx, REJECTED_OUTCOME)] * rej
            out += [YieldCall(host, fx, VOID_OUTCOME, "LLMTimeoutError")] * void
    return out


# --- aggregation contract -------------------------------------------------------------------


def test_aggregate_yield_records_raw_counts_and_reasons() -> None:
    data = aggregate_yield(_calls(fw=(1, 1, 1), bt=(3, 0, 0)), _meta())
    assert data["schema_version"] == 1
    assert data["yield_contract"] == YIELD_CONTRACT
    assert data["harness_digest"] == "h-digest"
    fw = data["hosts"][FIREWORKS_GLM]
    assert fw["per_fixture"][_FX_A] == {
        "accepted": 1,
        "rejected": 1,
        "void": 1,
        "void_reasons": ["LLMTimeoutError"],
    }
    assert fw["totals"] == {"attempts_planned": 6, "accepted": 2, "rejected": 2, "void": 2}
    bt = data["hosts"][BASETEN_GLM]
    assert bt["totals"] == {"attempts_planned": 6, "accepted": 6, "rejected": 0, "void": 0}


def test_aggregate_yield_enforces_conservation() -> None:
    calls = _calls()
    calls.pop()  # one call missing -> accepted+rejected+void != n_reps on that cell
    with pytest.raises(ValueError, match="conservation violated"):
        aggregate_yield(calls, _meta())


def test_aggregate_yield_requires_exactly_the_two_hosts() -> None:
    meta = _meta(hosts={FIREWORKS_GLM: YieldHostMeta("m", "c")})
    with pytest.raises(ValueError, match="hosts must be exactly"):
        aggregate_yield(_calls(), meta)
    fw_only = [c for c in _calls() if c.host == FIREWORKS_GLM]
    with pytest.raises(ValueError, match="observed hosts"):
        aggregate_yield(fw_only, _meta())


def test_aggregate_yield_void_reason_discipline() -> None:
    bad_void = [*_calls()]
    bad_void[0] = bad_void[0]._replace(outcome=VOID_OUTCOME)  # void without a reason
    with pytest.raises(ValueError, match="void_reason must be present iff"):
        aggregate_yield(bad_void, _meta())
    bad_reason = [*_calls()]
    bad_reason[0] = bad_reason[0]._replace(void_reason="LLMTimeoutError")  # reason without void
    with pytest.raises(ValueError, match="void_reason must be present iff"):
        aggregate_yield(bad_reason, _meta())


def test_aggregate_yield_requires_identities() -> None:
    with pytest.raises(ValueError, match="harness_digest"):
        aggregate_yield(_calls(), _meta(harness_digest=""))
    with pytest.raises(ValueError, match="yield_contract"):
        aggregate_yield(_calls(), _meta(yield_contract=""))
    with pytest.raises(ValueError, match="fixture_digests"):
        aggregate_yield(_calls(), _meta(fixture_digests={_FX_A: "d-a"}))


def test_yield_rate_derives_from_raw_counts_and_excludes_void() -> None:
    data = aggregate_yield(_calls(fw=(1, 1, 1), bt=(0, 0, 3)), _meta())
    assert yield_rate(data["hosts"][FIREWORKS_GLM]) == 0.5  # 2 accepted / 4 valid attempts
    assert yield_rate(data["hosts"][BASETEN_GLM]) is None  # all void -> no valid attempts


# --- persistence ----------------------------------------------------------------------------


def test_yield_artifact_round_trips_and_is_create_once(tmp_path, monkeypatch) -> None:
    from . import glm_yield as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "YIELD_DIR", tmp_path)
    data = aggregate_yield(_calls(), _meta())
    path = write_yield_artifact(data, label="analyze-v10-glm-yield")
    assert path.name == "analyze-v10-glm-yield.json"
    assert read_yield_artifact("analyze-v10-glm-yield") == data
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_yield_artifact(data, label="analyze-v10-glm-yield")


def test_read_yield_artifact_rejects_wrong_schema(tmp_path, monkeypatch) -> None:
    from . import glm_yield as mod  # noqa: PLC0415

    monkeypatch.setattr(mod, "YIELD_DIR", tmp_path)
    stale = aggregate_yield(_calls(), _meta())
    stale["schema_version"] = 999
    (tmp_path / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        read_yield_artifact("stale")


def test_yield_source_digest_is_stable_sha256() -> None:
    d = yield_source_digest()
    assert d == yield_source_digest()
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)


# --- PAID collector -------------------------------------------------------------------------
_REAL_MODELS = os.environ.get("OUTRIDER_EVAL_REAL_MODELS") == "1"
_REAL_SKIP = "spends API tokens; set OUTRIDER_EVAL_REAL_MODELS=1 and run under `op run` to execute"


class _NullPersister:
    """Satisfies the provider persister seam; the yield collector records no token telemetry."""

    async def persist(self, event: object, request: object, response: object) -> None:
        return None


def _yield_fixture_digests() -> dict[str, str]:
    """SAME semantic-digest recipe as the frozen baseline, so fixture identities cross-reference."""
    from .test_model_comparison import (  # noqa: PLC0415
        _GROUND_TRUTH_BY_FIXTURE,
        _SAFE_CODE_FIXTURES,
    )

    digests: dict[str, str] = {}
    for fx, gt in _GROUND_TRUTH_BY_FIXTURE.items():
        types = sorted({ef.finding_type.value for ef in gt})
        digests[fx] = fixture_content_digest(
            source=Path(fx).read_text(encoding="utf-8"), expected_types=types, is_safe=False
        )
    for fx in _SAFE_CODE_FIXTURES:
        digests[fx] = fixture_content_digest(
            source=Path(fx).read_text(encoding="utf-8"), expected_types=[], is_safe=True
        )
    return digests


def _build_yield_context() -> tuple[list[tuple], YieldRunMeta]:
    """The two GLM providers + full YieldRunMeta, built with NO spend. Skips without keys."""
    from pydantic import SecretStr  # noqa: PLC0415

    from outrider.llm.config import ModelConfig  # noqa: PLC0415
    from outrider.llm.host_profiles import BASETEN_PROFILE, FIREWORKS_PROFILE  # noqa: PLC0415
    from outrider.llm.openai_compatible_provider import (  # noqa: PLC0415
        GLM_MODEL_ID,
        GLMProvider,
        OpenAICompatibleProvider,
    )

    from .test_exemplar_baseline import _prompt_identity  # noqa: PLC0415

    keys = {
        "FIREWORKS_API_KEY": os.environ.get("FIREWORKS_API_KEY"),
        "BASETEN_API_KEY": os.environ.get("BASETEN_API_KEY"),
    }
    for name, val in keys.items():
        if not val or val.startswith("op://"):
            pytest.skip(f"{name} (resolved, not an op:// ref) required; run under `op run`")

    fw_model = ModelConfig.for_host("fireworks").analyze_model
    specs = [
        (
            FIREWORKS_GLM,
            OpenAICompatibleProvider(
                api_key=SecretStr(keys["FIREWORKS_API_KEY"]),
                profile=FIREWORKS_PROFILE,
                persister=_NullPersister(),
                models=(fw_model,),
            ),
            fw_model,
        ),
        (
            BASETEN_GLM,
            GLMProvider(api_key=SecretStr(keys["BASETEN_API_KEY"]), persister=_NullPersister()),
            GLM_MODEL_ID,
        ),
    ]
    version, digest = _prompt_identity()
    meta = YieldRunMeta(
        n_reps=YIELD_REPS,
        prompt_version=version,
        prompt_digest=digest,
        fixture_digests=_yield_fixture_digests(),
        hosts={
            FIREWORKS_GLM: YieldHostMeta(fw_model, FIREWORKS_PROFILE.profile_contract_digest),
            BASETEN_GLM: YieldHostMeta(GLM_MODEL_ID, BASETEN_PROFILE.profile_contract_digest),
        },
        harness_digest=yield_source_digest(),
    )
    return specs, meta


async def _collect_yield_calls(specs: list[tuple]) -> list[YieldCall]:
    """THE PAID LOOP: 2 hosts x YIELD_REPS x 20 fixtures, fully sequential.

    Sequential on purpose: no token attribution here (so FUP-239's snapshot-slice constraint
    doesn't bind), but sequential keeps provider behavior comparable to the frozen baseline's
    collection and stays rate-limit-benign. Per call: `LLMProviderError` -> VOID recorded as the
    exception TYPE name and the loop continues (voids are visible in the artifact, excluded from
    the yield denominator); any other exception is a harness bug and propagates.
    """
    from outrider.llm.base import LLMProviderError  # noqa: PLC0415

    from .model_comparison import run_analyze_under_model, state_from_eval_fixture  # noqa: PLC0415
    from .test_model_comparison import (  # noqa: PLC0415
        _GROUND_TRUTH_BY_FIXTURE,
        _SAFE_CODE_FIXTURES,
    )

    fixtures = [*_GROUND_TRUTH_BY_FIXTURE, *_SAFE_CODE_FIXTURES]
    calls: list[YieldCall] = []
    for key, provider, model in specs:
        for _rep in range(YIELD_REPS):
            for fx in fixtures:
                state = state_from_eval_fixture(fx)
                n_files = len(state.pr_context.changed_files)
                if n_files != 1:
                    raise AssertionError(
                        f"{fx} has {n_files} changed files; the yield contract counts one "
                        "structured-output attempt per rep (single-file fixtures only)"
                    )
                try:
                    _findings, n_rejected = await run_analyze_under_model(
                        state, provider=provider, model=model
                    )
                except LLMProviderError as exc:
                    calls.append(YieldCall(key, fx, VOID_OUTCOME, type(exc).__name__))
                    continue
                if n_rejected not in (0, 1):
                    raise AssertionError(
                        f"{fx}: n_rejected={n_rejected} on a single-file fixture — the yield "
                        "contract's conservation equation no longer holds; extend the schema"
                    )
                calls.append(
                    YieldCall(key, fx, REJECTED_OUTCOME if n_rejected else ACCEPTED_OUTCOME)
                )
    return calls


@pytest.mark.skipif(not _REAL_MODELS, reason=_REAL_SKIP)
@pytest.mark.asyncio
async def test_collect_glm_yield() -> None:
    """PAID collection (FUP-219): the two-GLM-host structured-output yield read.

    Doubly opt-in like the baseline freeze: this spends ~120 calls AND writes the tracked
    evidence tree. One collection per prompt identity — the create-once preflight fails for
    free if this identity is already measured.
    """
    if os.environ.get("OUTRIDER_COLLECT_GLM_YIELD") != "1":
        pytest.skip("set OUTRIDER_COLLECT_GLM_YIELD=1 to collect (spends + writes tracked tree)")
    specs, meta = _build_yield_context()
    label = f"{meta.prompt_version}-glm-yield"
    if (YIELD_DIR / f"{label}.json").exists():
        pytest.fail(
            f"a yield artifact for {label!r} already exists — one collection per prompt "
            "identity; to measure again, bump the analyze VERSION"
        )
    calls = await _collect_yield_calls(specs)
    data = aggregate_yield(calls, meta)
    path = write_yield_artifact(data, label=label)
    for host in sorted(EXPECTED_HOSTS):
        block = data["hosts"][host]
        rate = yield_rate(block)
        print(f"\n{host}: yield={rate if rate is None else f'{rate:.4f}'} totals={block['totals']}")
    print(f"artifact: {path}")
    assert read_yield_artifact(label) == data


# --- collector WIRING guard: the whole loop, all three outcome paths, NO spend ----------------
@pytest.mark.asyncio
async def test_yield_collector_wiring_covers_all_outcomes(monkeypatch) -> None:
    from outrider.llm.base import LLMTimeoutError  # noqa: PLC0415

    from . import model_comparison as mc  # noqa: PLC0415
    from .test_model_comparison import _SAFE_CODE_FIXTURES  # noqa: PLC0415

    # Keyed by the fixture's single changed-file path (fixture-unique): one fixture always
    # rejects, one always raises (-> VOID via the real except path), the rest accept. A uniform
    # zero would pass even if the collector discarded outcomes — same guard shape as the
    # exemplar runner's rejection injection.
    rejected_fx = "tests/eval/fixtures/mock_github/cmd_injection_eval_indirect.json"
    rejected_path = "app/calc.py"
    void_fx = next(iter(_SAFE_CODE_FIXTURES))
    void_path = json.loads(Path(void_fx).read_text(encoding="utf-8"))["files"][0]["path"]
    assert void_path != rejected_path

    async def _fake_run(state, *, provider, model):  # noqa: ANN001, ANN202, ARG001
        path = state.pr_context.changed_files[0].path
        if path == void_path:
            raise LLMTimeoutError("synthetic timeout")
        return (), 1 if path == rejected_path else 0

    monkeypatch.setattr(mc, "run_analyze_under_model", _fake_run)

    fake_provider = object()
    specs = [(FIREWORKS_GLM, fake_provider, "m-fw"), (BASETEN_GLM, fake_provider, "m-bt")]
    calls = await _collect_yield_calls(specs)
    from .test_model_comparison import _GROUND_TRUTH_BY_FIXTURE as _GT  # noqa: PLC0415

    n_fx = len(_GT) + len(_SAFE_CODE_FIXTURES)
    assert len(calls) == 2 * YIELD_REPS * n_fx  # every planned call accounted for, incl. voids

    meta = _build_wiring_meta()
    data = aggregate_yield(calls, meta)
    for host in EXPECTED_HOSTS:
        per_fixture = data["hosts"][host]["per_fixture"]
        assert per_fixture[rejected_fx] == {
            "accepted": 0,
            "rejected": 3,
            "void": 0,
            "void_reasons": [],
        }
        assert per_fixture[void_fx] == {
            "accepted": 0,
            "rejected": 0,
            "void": 3,
            "void_reasons": ["LLMTimeoutError"] * 3,
        }
        assert data["hosts"][host]["totals"] == {
            "attempts_planned": YIELD_REPS * n_fx,
            "accepted": YIELD_REPS * (n_fx - 2),
            "rejected": 3,
            "void": 3,
        }


def _build_wiring_meta() -> YieldRunMeta:
    """Real fixture digests + prompt identity (both free), synthetic host identities."""
    from .test_exemplar_baseline import _prompt_identity  # noqa: PLC0415

    version, digest = _prompt_identity()
    return YieldRunMeta(
        n_reps=YIELD_REPS,
        prompt_version=version,
        prompt_digest=digest,
        fixture_digests=_yield_fixture_digests(),
        hosts={
            FIREWORKS_GLM: YieldHostMeta("m-fw", "c-fw"),
            BASETEN_GLM: YieldHostMeta("m-bt", "c-bt"),
        },
        harness_digest=yield_source_digest(),
    )
