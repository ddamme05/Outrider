"""enforce_proof_boundary admits JUDGED with neither artifact.

Backs ``evidence-tier-schema-enforced``. JUDGED makes no structural claim, so it
admits with any artifact combination. OBSERVED and INFERRED requirements are
covered in their dedicated proof-boundary modules.
"""

from outrider.policy import EvidenceTier, enforce_proof_boundary


def test_judged_with_neither_artifact_admits() -> None:
    """JUDGED + None query_match_id + None trace_path is admitted."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=None,
    )


def test_judged_with_query_match_id_admits() -> None:
    """JUDGED admits even with a query_match_id (extra info, not a violation)."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id="py.security.placeholder",
        trace_path=None,
    )


def test_judged_with_trace_path_admits() -> None:
    """JUDGED admits even with a trace_path (extra info, not a violation)."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id=None,
        trace_path=["module", "function"],
    )


def test_judged_with_both_artifacts_admits() -> None:
    """JUDGED admits regardless of which artifacts are present."""
    enforce_proof_boundary(
        evidence_tier=EvidenceTier.JUDGED,
        query_match_id="py.something",
        trace_path=["a", "b"],
    )
