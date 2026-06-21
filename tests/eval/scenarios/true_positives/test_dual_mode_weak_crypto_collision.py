"""Dual-mode collision eval scenario: OBSERVED + JUDGED weak_crypto.

The FUP-193 step-1 analogue of the command_injection collision: with the
new `weak_crypto_broken_cipher` OBSERVED query (DECISIONS.md#053 type +
the weak-crypto OBSERVED-queries spec), a `DES.new(key)` construction now
fires OBSERVED `weak_crypto`, and the model may also flag `weak_crypto`
JUDGED on the same line.

The content-hash dedup (`analyze.py`, prefer-first) collapses the two:
`content_hash` is keyed on (file, line_start, line_end, finding_type) —
NOT evidence_tier — so the same-type OBSERVED finding whose hash collides
with the already-admitted JUDGED one is dropped. Exactly one `weak_crypto`
survives, with the policy-set HIGH severity.

Tier-agnostic: prefer-first means the JUDGED finding survives today, but
prefer-OBSERVED is a deferred change (FUP-193 step 4), and this scenario
must hold under either dedup policy. The companion
`test_observed_producer_alone_flags_weak_crypto` is the non-vacuity
control — with the model proposing nothing, the OBSERVED producer alone
flags `weak_crypto`, proving the collision is a real dedup, not a producer
that silently never fired.
"""

from outrider.policy import EvidenceTier, FindingSeverity, FindingType, lookup_severity

_COLLISION_FIXTURE = "tests/eval/fixtures/mock_github/dual_mode_weak_crypto_collision.json"
_OBSERVED_ONLY_FIXTURE = "tests/eval/fixtures/mock_github/dual_mode_weak_crypto_observed_only.json"


def test_dual_mode_collision_yields_single_weak_crypto() -> None:
    """Model JUDGED + producer OBSERVED at the same line -> exactly one
    weak_crypto survives, HIGH. Tier-agnostic (prefer-first today,
    prefer-OBSERVED deferred)."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_COLLISION_FIXTURE)
    wc = [f for f in findings if f.finding_type == FindingType.WEAK_CRYPTO]
    assert len(wc) == 1, f"dual-mode collision must collapse to one finding, got {len(wc)}"
    finding = wc[0]
    assert finding.severity == lookup_severity(FindingType.WEAK_CRYPTO)
    assert finding.severity == FindingSeverity.HIGH
    assert finding.line_start == 5 and finding.line_end == 5


def test_observed_producer_alone_flags_weak_crypto() -> None:
    """Non-vacuity control: with the model proposing nothing, the OBSERVED
    producer alone flags weak_crypto OBSERVED at the same line — so the
    collision test above is a real dedup, not a no-op producer."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_OBSERVED_ONLY_FIXTURE)
    wc = [f for f in findings if f.finding_type == FindingType.WEAK_CRYPTO]
    assert len(wc) == 1, f"OBSERVED producer must flag weak_crypto alone, got {len(wc)}"
    finding = wc[0]
    assert finding.evidence_tier == EvidenceTier.OBSERVED
    assert finding.query_match_id == "python.weak_crypto_broken_cipher"
    assert finding.severity == FindingSeverity.HIGH
    assert finding.line_start == 5 and finding.line_end == 5
