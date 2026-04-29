"""enforce_proof_boundary admits JUDGED with neither artifact.

Backs ``evidence-tier-schema-enforced``. JUDGED is the explicit
no-structural-claim tier — a model interpretation that doesn't ride on
a tree-sitter query or an ast_facts traversal. The validator admits
JUDGED unconditionally; downstream confidence is lower-by-construction
because the tier itself signals that lower confidence.

The complement to test_proof_boundary_{observed,inferred}: those test
the tiers that REQUIRE artifacts; this tests the tier that DOES NOT.
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
