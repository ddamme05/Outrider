"""Patch admission instrument for the openai host (openai-native-host spec).

Frozen pass predicate (spec "Gates before any production-shaped use"): a valid
constrained patch that APPLIES — parses and survives the production
`apply_patch_batch` rules — touching ONLY the expected target span (exact
`original_line` echo; anchor drift is dropped), no out-of-scope edits
(single-line, markdown/diff-marker/Trojan-free via
`is_safe_suggestion_replacement`), and zero rejected responses (the batch
parsed). The graders here ARE the production pipeline — the instrument never
re-implements a rule the node enforces.

ONE canonical paid path (per the review clarification): the paid row lives in
the wire probe (`spikes/openai/probe.py`, row `gpt-5.6-luna:patch`). This file
never spends: it proves the graders can FAIL via scripted negative twins
(free, runs in the normal eval gate) and grades the captured probe fixture
OFFLINE — skipped until the capture exists, HARD-asserted once it does (a
red gate is the correct signal; the spec's miss-rule is a Terra swap plus an
instrument rerun, never a softened predicate).

The scenario is duplicated in the probe DELIBERATELY (spikes/ is not
importable from tests). Drift fails loud: a probe that sends a different
target line yields fixtures whose `original_line` echo cannot match this
file's grader.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

import pytest
from pydantic import ValidationError

from outrider.agent.nodes.patch_generation import (
    PatchSuggestionBatch,
    apply_patch_batch,
    generate_patches,
    select_eligible_findings,
)
from outrider.audit.events import compute_finding_content_hash
from outrider.llm.parsing import strip_outer_json_fence
from outrider.policy import EvidenceTier, FindingType, lookup_severity
from outrider.policy.dimensions import lookup_dimension
from outrider.policy.severity import ACTIVE_POLICY_VERSION
from outrider.prompts import patch as patch_prompt
from outrider.schemas import ReviewFinding
from outrider.schemas.pr_context import ChangedFile, PRContext
from outrider.schemas.review_state import ReviewState

from .test_model_comparison import _ScriptedProvider
from .test_openai_scorecard import _PROBE_MANIFEST, verified_capture_fixture

# --- The deterministic admission scenario (duplicated in the probe) ---------
# yaml.load -> yaml.safe_load: a real HIGH finding with an unambiguous,
# genuinely single-line fix that needs no new import — the cleanest possible
# "applies, touches only the target span" instance.
_FINDING_ID = UUID("00000000-0000-0000-0000-0000000000f1")
_REVIEW_ID = UUID("00000000-0000-0000-0000-0000000000e1")
_FILE_PATH = "app/config_loader.py"
_TARGET_LINE_NO = 6
_TARGET_LINE = "    return yaml.load(stream)"
_EXPECTED_REPLACEMENT = "    return yaml.safe_load(stream)"
_LOADER_CONTENT = (
    "import yaml\n"
    "\n"
    "\n"
    "def load_config(stream):\n"
    '    """Parse the operator-provided config stream."""\n'
    "    return yaml.load(stream)\n"
)

# The scenario-specific SEMANTIC predicate: `apply_patch_batch` proves a
# patch is safe and anchored, not that it FIXES anything — a wrong-but-safe
# line (yaml.dump) or a composite that ALSO invokes an unsafe loader
# (`safe_load(stream) and unsafe_load(stream)`) survives it, and substring
# regexes proved gameable against exactly that composite. The fixture has ONE
# unambiguous fix, so remediation is a CLOSED set of reviewed replacements —
# extended consciously by a human, never pattern-matched.
_REVIEWED_FIXES = frozenset({_EXPECTED_REPLACEMENT})


def _is_remediation(replacement: str) -> bool:
    return replacement in _REVIEWED_FIXES


# Models whose captures this instrument grades when present in the verified
# manifest (Terra appears only when a swap capture declares it).
_NODE_MODELS = ("gpt-5.6-luna", "gpt-5.6-terra")


def _admission_finding() -> ReviewFinding:
    return ReviewFinding(
        finding_id=_FINDING_ID,
        review_id=_REVIEW_ID,
        installation_id=42,
        finding_type=FindingType.UNSAFE_DESERIALIZATION,
        severity=lookup_severity(FindingType.UNSAFE_DESERIALIZATION),
        file_path=_FILE_PATH,
        line_start=_TARGET_LINE_NO,
        line_end=_TARGET_LINE_NO,
        title="yaml.load without a safe loader deserializes arbitrary objects",
        description=(
            "load_config parses an operator-provided stream with yaml.load and no "
            "Loader argument; a crafted document instantiates arbitrary Python objects."
        ),
        evidence=_TARGET_LINE,
        dimension=lookup_dimension(FindingType.UNSAFE_DESERIALIZATION),
        evidence_tier=EvidenceTier.JUDGED,
        policy_version=ACTIVE_POLICY_VERSION,
        content_hash=compute_finding_content_hash(
            file_path=_FILE_PATH,
            line_start=_TARGET_LINE_NO,
            line_end=_TARGET_LINE_NO,
            finding_type=FindingType.UNSAFE_DESERIALIZATION,
        ),
        proposal_hash="a" * 64,
    )


def _target_lines() -> dict[UUID, str]:
    return {_FINDING_ID: _TARGET_LINE}


def _batch_json(original: str, replacement: str | None) -> str:
    item: dict[str, object] = {
        "finding_id": str(_FINDING_ID),
        "original_line": original,
        "replacement_line": replacement,
        "reason": None if replacement is not None else "not fixable in one line",
    }
    return json.dumps({"items": [item]})


def _classify_batch(raw: str) -> str:
    """'rejected' vs 'parsed' — the "zero rejected responses" half of the
    predicate. A parsed batch with a null replacement is a MODEL VERDICT
    (counted, not a rejection); only schema/JSON failure is a rejection."""
    try:
        PatchSuggestionBatch.model_validate_json(strip_outer_json_fence(raw))
    except (ValidationError, ValueError):
        return "rejected"
    return "parsed"


def _state() -> ReviewState:
    changed = ChangedFile(
        path=_FILE_PATH,
        status="added",  # head-only content per the §7.2 status invariants
        additions=6,
        deletions=0,
        patch=None,
        content_base=None,
        content_head=_LOADER_CONTENT,
        previous_path=None,
        language="python",
    )
    return ReviewState(
        review_id=_REVIEW_ID,
        received_at=datetime.now(UTC),
        pr_context=PRContext(
            installation_id=99999,
            owner="o",
            repo="r",
            pr_number=1,
            base_sha="a" * 40,
            head_sha="b" * 40,
            pr_title="t",
            pr_body=None,
            author="a",
            total_additions=1,
            total_deletions=0,
            changed_files=(changed,),
        ),
        is_eval=True,
    )


@pytest.mark.asyncio
async def test_scripted_full_path_sets_suggested_fix() -> None:
    """Harness proof through the REAL node path: `generate_patches` drives
    eligibility, target-line extraction, the real prompt render, the provider
    call, and the fail-closed parser — a conforming scripted response ends as
    `suggested_fix` on the finding, and the request the node built matches the
    production patch-call shape (node_id, template version, free-form JSON)."""
    provider = _ScriptedProvider(_batch_json(_TARGET_LINE, _EXPECTED_REPLACEMENT))
    result = await generate_patches(
        (_admission_finding(),),
        _state(),
        provider=provider,  # type: ignore[arg-type]
        patch_model="gpt-5.6-luna",
        patches_enabled=True,
        max_suggestions=4,
    )
    assert result[0].suggested_fix == _EXPECTED_REPLACEMENT
    (request,) = provider.calls
    assert request.node_id == "synthesize"  # patch cost rolls into synthesize
    assert request.prompt_template_version == patch_prompt.VERSION
    assert request.response_schema_json is None  # prompt-described JSON, no schema wire
    assert request.model == "gpt-5.6-luna"
    assert _TARGET_LINE in request.user_prompt  # the fenced target line rode the wire


def test_scenario_is_eligible_and_anchored() -> None:
    """The admission scenario must actually clear the production gates it
    claims to exercise — HIGH single-line eligibility and an extractable
    target line — else the instrument grades a vacuous path."""
    finding = _admission_finding()
    assert select_eligible_findings((finding,), max_suggestions=4) == (finding,)
    lines = _LOADER_CONTENT.split("\n")
    assert lines[_TARGET_LINE_NO - 1] == _TARGET_LINE


def test_grader_negative_twins() -> None:
    """The graders can FAIL — one mutant per predicate dimension, each
    differing from the passing control in exactly the graded property
    (revert-the-fold per variant)."""
    finding = _admission_finding()
    eligible = (finding,)
    targets = _target_lines()

    # Passing control: the exact positive the twins mutate.
    good = apply_patch_batch(_batch_json(_TARGET_LINE, _EXPECTED_REPLACEMENT), eligible, targets)
    assert good == {_FINDING_ID: _EXPECTED_REPLACEMENT}

    # Anchor drift: the model "fixed" a different line than the target span.
    drifted = apply_patch_batch(
        _batch_json("    return yaml.dump(stream)", _EXPECTED_REPLACEMENT), eligible, targets
    )
    assert drifted == {}

    # Out-of-scope edit: a multi-line replacement escapes the target span.
    multiline = apply_patch_batch(
        _batch_json(_TARGET_LINE, "    import shlex\n    return yaml.safe_load(stream)"),
        eligible,
        targets,
    )
    assert multiline == {}

    # Markdown smuggling: fenced/backticked content is not plain code.
    fenced = apply_patch_batch(
        _batch_json(_TARGET_LINE, "```python\nreturn yaml.safe_load(stream)\n```"),
        eligible,
        targets,
    )
    assert fenced == {}

    # No-op: echoing the original back is not a fix.
    noop = apply_patch_batch(_batch_json(_TARGET_LINE, _TARGET_LINE), eligible, targets)
    assert noop == {}

    # WRONG-BUT-SAFE: yaml.dump survives every structural rule (echoed
    # anchor, single line, safe, changed) — proving apply_patch_batch alone
    # cannot certify remediation — and the SEMANTIC predicate catches it.
    wrong_but_safe = "    return yaml.dump(stream)"
    survived = apply_patch_batch(_batch_json(_TARGET_LINE, wrong_but_safe), eligible, targets)
    assert survived == {_FINDING_ID: wrong_but_safe}  # structural layer admits it...
    assert not _is_remediation(wrong_but_safe)  # ...the semantic layer refuses
    assert _is_remediation(_EXPECTED_REPLACEMENT)

    # COMPOSITE-UNSAFE: the exact regex-defeating line from review — contains
    # the safe call yet still invokes an unsafe loader. The structural layer
    # admits it; only closed-set equality refuses it.
    composite = "    return yaml.safe_load(stream) and yaml.unsafe_load(stream)"
    survived = apply_patch_batch(_batch_json(_TARGET_LINE, composite), eligible, targets)
    assert survived == {_FINDING_ID: composite}
    assert not _is_remediation(composite)
    # Other near-miss shapes fail closed-set equality too.
    assert not _is_remediation("    return yaml.safe_load(stream) or yaml.load(stream)")
    assert not _is_remediation("    return  yaml.safe_load(stream)")  # not the reviewed bytes

    # Unknown finding: a patch for a finding we never asked about is dropped.
    stray = json.dumps(
        {
            "items": [
                {
                    "finding_id": "00000000-0000-0000-0000-0000000000ff",
                    "original_line": _TARGET_LINE,
                    "replacement_line": _EXPECTED_REPLACEMENT,
                    "reason": None,
                }
            ]
        }
    )
    assert apply_patch_batch(stray, eligible, targets) == {}


def test_rejected_vs_parsed_classification() -> None:
    """'Zero rejected responses' must be measurable: schema/JSON failure is a
    rejection; a parsed batch with a null replacement is a model verdict, not
    a rejection — conflating them would let a broken wire read as caution."""
    assert _classify_batch("the model produced prose, not JSON") == "rejected"
    assert _classify_batch('{"items": [{"finding_id": "not-a-uuid"}]}') == "rejected"
    assert _classify_batch(_batch_json(_TARGET_LINE, None)) == "parsed"
    assert _classify_batch("```json\n" + _batch_json(_TARGET_LINE, None) + "\n```") == "parsed"


def test_captured_paid_fixture_passes_frozen_predicate() -> None:
    """Grade every node-capable model's captured patch row OFFLINE, resolved
    through the VERIFIED manifest (versions, hashes, row verdicts,
    adjudication — a stale fixture left by a failed rerun cannot grade).
    Skips until a capture exists; once it does, a miss FAILS — the spec's
    rule is a Terra swap + rerun, never a softened gate. The predicate is
    structural AND semantic: survives apply_patch_batch AND actually
    remediates the scenario."""
    if not _PROBE_MANIFEST.exists():
        pytest.skip(
            "paid probe capture absent — run the wire probe first "
            "(op run --env-file=.env -- uv run python spikes/openai/probe.py)"
        )
    graded = []
    for model in _NODE_MODELS:
        data = verified_capture_fixture(f"{model}:patch")
        if data is None:
            continue  # model not in the declared capture matrix
        message = json.loads(data.decode("utf-8"))["choices"][0]["message"]
        assert not message.get("refusal"), f"{model} patch row returned a refusal"
        text = str(message.get("content") or "")
        assert _classify_batch(text) == "parsed", (
            f"{model} patch response REJECTED (schema/JSON failure) — the "
            "zero-rejected-responses predicate fails"
        )
        patches = apply_patch_batch(text, (_admission_finding(),), _target_lines())
        replacement = patches.get(_FINDING_ID)
        assert replacement, (
            f"{model}: no valid, anchored, single-line patch survived the production "
            "rules — the applies/target-span/no-out-of-scope predicate fails"
        )
        assert _is_remediation(replacement), (
            f"{model} produced a structurally safe but NON-REMEDIATING line "
            f"{replacement!r} — the scenario's fix is yaml.safe_load(stream)"
        )
        graded.append(model)
        print(  # noqa: T201 — operator verdict line
            f"\n[patch admission: PASS — {model} produced {replacement!r} "
            f"for {_FILE_PATH}:{_TARGET_LINE_NO}]"
        )
    assert graded, "verified manifest carried no node-model patch rows"
