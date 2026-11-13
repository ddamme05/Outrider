"""Check the password-hashing finding type and policy severity for an MD5 example.

Drive the graph with a scripted contextual finding. No structural query proves
that the hash is used for passwords; the evidence here is model judgment."""

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
