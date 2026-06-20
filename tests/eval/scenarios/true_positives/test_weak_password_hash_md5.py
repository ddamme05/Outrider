"""True-positive eval scenario: MD5 password hashing -> weak_password_hash.

The flagship case for the dual-mode security taxonomy (DECISIONS.md#053).
The PR #8 coverage smoke showed MD5-for-passwords force-mapped to
`deprecated_api` -> INFO, badly understating a real vuln. With the 1.2.0
taxonomy the model can name `weak_password_hash` directly, and the
deterministic policy assigns CRITICAL (`severity-set-by-policy`).

Dual-mode: `weak_password_hash` is a JUDGED finding here — it is a
contextual call (a query can't tell a password hash from a cache-key
hash), so the model carries the proof via `evidence_tier="judged"` with
no structural artifact. No OBSERVED .scm producer exists for the
contextual 1.2.0 types (spec non-goal), so nothing else fires on this
file.

Driver-backed: drives the real graph via `run_review` against
`mock_github/weak_password_hash_md5.json`. Severity is asserted via
`lookup_severity` (set by policy, not the model).
"""

from outrider.policy import EvidenceTier, FindingSeverity, FindingType, lookup_severity


def test_md5_password_hash_detected_as_critical_weak_password_hash() -> None:
    """Agent produces WEAK_PASSWORD_HASH + JUDGED + CRITICAL policy severity.

    This is the smoke's argument made concrete: the same code that used to
    land as `deprecated_api`/INFO now lands as `weak_password_hash`/CRITICAL.
    """
    from outrider.agent import run_review  # type: ignore[import-not-found]

    findings = run_review("tests/eval/fixtures/mock_github/weak_password_hash_md5.json")
    matches = [f for f in findings if f.finding_type == FindingType.WEAK_PASSWORD_HASH]
    assert len(matches) == 1, f"expected exactly one weak_password_hash finding, got {len(matches)}"
    finding = matches[0]
    # JUDGED — the contextual call carries no structural artifact.
    assert finding.evidence_tier == EvidenceTier.JUDGED
    assert finding.query_match_id is None
    # Severity is policy-set, and the policy rates this CRITICAL.
    assert finding.severity == lookup_severity(FindingType.WEAK_PASSWORD_HASH)
    assert finding.severity == FindingSeverity.CRITICAL
    # The smoke regression: it must NOT degrade to deprecated_api/INFO.
    assert finding.finding_type != FindingType.DEPRECATED_API
