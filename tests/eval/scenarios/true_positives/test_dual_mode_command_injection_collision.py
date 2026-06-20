"""Dual-mode collision eval scenario: OBSERVED + JUDGED command_injection.

Under the dual-mode taxonomy (DECISIONS.md#053) a security FindingType may
be emitted by BOTH the deterministic OBSERVED producer (a `.scm` query
fired) AND the model as a JUDGED contextual call. `os.system` with an
untrusted argument is exactly that case: `command_injection_os_system.scm`
fires OBSERVED, and the model also flags `command_injection`.

The existing analyze content-hash dedup (`analyze.py`, prefer-first)
collapses the two: `content_hash` is keyed on (file, line_start, line_end,
finding_type) — NOT evidence_tier — so the OBSERVED finding whose hash
collides with the already-admitted JUDGED one is dropped. Exactly one
`command_injection` survives, with the policy-set CRITICAL severity.

The scenario does NOT assert which tier wins: prefer-first means the JUDGED
finding (admitted before the OBSERVED producer runs) survives today, but
prefer-OBSERVED is an explicitly deferred change (handoff Finding 4), and
this scenario must hold under either dedup policy. The companion
`test_observed_producer_alone_flags_command_injection` is the non-vacuity
control: with the model proposing nothing, the OBSERVED producer alone
still flags `command_injection`, proving the collision above is a real
dedup, not a producer that silently never fired.

Driver-backed via `run_review` against the two collision fixtures.
"""

from outrider.policy import EvidenceTier, FindingSeverity, FindingType, lookup_severity

_COLLISION_FIXTURE = "tests/eval/fixtures/mock_github/dual_mode_command_injection_collision.json"
_OBSERVED_ONLY_FIXTURE = (
    "tests/eval/fixtures/mock_github/dual_mode_command_injection_observed_only.json"
)


def test_dual_mode_collision_yields_single_command_injection() -> None:
    """Model JUDGED + producer OBSERVED at the same line -> exactly one
    command_injection survives, CRITICAL. Tier-agnostic (prefer-first today,
    prefer-OBSERVED deferred)."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_COLLISION_FIXTURE)
    ci = [f for f in findings if f.finding_type == FindingType.COMMAND_INJECTION]
    assert len(ci) == 1, f"dual-mode collision must collapse to one finding, got {len(ci)}"
    finding = ci[0]
    assert finding.severity == lookup_severity(FindingType.COMMAND_INJECTION)
    assert finding.severity == FindingSeverity.CRITICAL
    assert finding.line_start == 5 and finding.line_end == 5


def test_observed_producer_alone_flags_command_injection() -> None:
    """Non-vacuity control: with the model proposing nothing, the OBSERVED
    producer alone flags command_injection OBSERVED at the same line — so the
    collision test above is a real dedup, not a no-op producer."""
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review(_OBSERVED_ONLY_FIXTURE)
    ci = [f for f in findings if f.finding_type == FindingType.COMMAND_INJECTION]
    assert len(ci) == 1, f"OBSERVED producer must flag command_injection alone, got {len(ci)}"
    finding = ci[0]
    assert finding.evidence_tier == EvidenceTier.OBSERVED
    assert finding.query_match_id is not None
    assert finding.severity == FindingSeverity.CRITICAL
    assert finding.line_start == 5 and finding.line_end == 5
