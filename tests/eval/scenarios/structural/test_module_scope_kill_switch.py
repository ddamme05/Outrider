"""Structural eval scenario: the module-scope admission arm on the TLS kill switch.

Per specs/2026-07-04-module-scope-admission-arm.md: the kill switch's canonical
real-world form — `process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0"` at module top
level, an inert parse fixture here (SYNTHETIC coverage; the vendored Juice Shop
corpus carries no upstream kill-switch file) — historically got ZERO review: a
module-only diff skipped at `NO_CHANGED_SCOPE_UNITS` before the OBSERVED producer
ran, and the containment gate dropped module-level matches otherwise.

LLM-free, exercising the real surfaces end-to-end:
- `coordinates.added_line_byte_ranges` — the diff anchor (the same ranges the
  degraded JUDGED admission gates against).
- `analyze_observed.has_module_level_eligible_match` — the routing pre-check,
  delegating to the producer's own admission chain.
- `agent/nodes/degradation.decide_degradation` — the module-only diff degrades
  (`module_level_observed_match`, parse_status truthfully `clean`) instead of
  skipping; parse-error precedence keeps the error branch first.
- `run_observed_matches` + `produce_observed_findings` — the module-level
  OBSERVED finding, proof anchored on the changed lines.
"""

from __future__ import annotations

# Heavy imports (parse dispatch lazy-loads tree_sitter) live in the test bodies.

SOURCE = 'process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\nmodule.exports = {};\n'

# Hunks-only patch (GitHub wire shape) ADDING line 1 — the kill switch — above
# the pre-existing module.exports line.
PATCH = '@@ -1,1 +1,2 @@\n+process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n module.exports = {};\n'

# The reverse diff: the kill switch is PRE-EXISTING (unchanged) and the patch
# adds only the harmless second line — no diff anchor, no admission.
PATCH_UNCHANGED_KILL_SWITCH = (
    '@@ -1,1 +1,2 @@\n process.env.NODE_TLS_REJECT_UNAUTHORIZED = "0";\n+module.exports = {};\n'
)


def _observe(patch: str):
    from unittest.mock import MagicMock

    from outrider.agent.nodes.analyze_observed import (
        has_module_level_eligible_match,
        run_observed_matches,
    )
    from outrider.agent.nodes.degradation import decide_degradation
    from outrider.ast_facts.registry import parse_source
    from outrider.coordinates import added_line_byte_ranges, lookup_patched_file

    parsed = parse_source(SOURCE.encode(), "src/index.js", MagicMock())
    assert parsed.parser_outcome == "clean"
    assert not parsed.error_lines
    patched_file = lookup_patched_file(patch, "src/index.js")
    assert patched_file is not None
    ranges = added_line_byte_ranges(patched_file, SOURCE)
    candidate = has_module_level_eligible_match(
        file_path="src/index.js",
        head_content=SOURCE,
        all_scope_units=parsed.scope_units,
        added_line_ranges=ranges,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
    )
    decision = decide_degradation(parsed, patched_file, module_level_observed_candidate=candidate)
    matches = run_observed_matches(
        file_path="src/index.js",
        head_content=SOURCE,
        included_scope_units=decision.included_scope_units,
        import_refs=parsed.imports,
        lexical_bindings=parsed.lexical_bindings,
        all_scope_units=parsed.scope_units,
        added_line_ranges=ranges,
    )
    return decision, matches


def test_module_only_kill_switch_diff_degrades_and_produces_observed() -> None:
    """The veto this arm closes: the module-only kill-switch diff routes to the
    degraded review (clean parse status — a routing choice, not a parse
    defect) and the producer emits the OBSERVED finding on the changed line."""
    from uuid import UUID

    from outrider.agent.nodes.analyze_observed import produce_observed_findings
    from outrider.policy.findings import EvidenceTier
    from outrider.policy.severity import ACTIVE_POLICY_VERSION

    decision, matches = _observe(PATCH)
    assert decision.mode == "degraded"
    assert decision.degradation_reason == "module_level_observed_match"
    assert decision.parse_status == "clean"

    (finding,) = produce_observed_findings(
        matches,
        file_path="src/index.js",
        review_id=UUID(int=1),
        installation_id=1,
        active_policy_version=ACTIVE_POLICY_VERSION,
    )
    assert finding.query_match_id == "javascript.tls_env_verify_disabled"
    assert finding.evidence_tier is EvidenceTier.OBSERVED
    assert finding.line_start == 1


def test_unchanged_kill_switch_still_skips() -> None:
    """The diff anchors the proof: a PRE-EXISTING kill switch with only a
    harmless added line stays a NO_CHANGED_SCOPE_UNITS skip — module-level
    matches in unchanged code are never admitted."""
    from outrider.ast_facts.models import SkipReason

    decision, matches = _observe(PATCH_UNCHANGED_KILL_SWITCH)
    assert decision.mode == "skip"
    assert decision.skip_reason is SkipReason.NO_CHANGED_SCOPE_UNITS
    assert matches == ()
