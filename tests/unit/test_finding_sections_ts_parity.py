"""Parity guard: the dashboard's humanized finding labels mirror the Python presentation layer.

`dashboard/src/lib/findingSections.ts` is the TypeScript mirror of
`src/outrider/presentation/finding_sections.py` (PR A). Two invariants:

1. The five SHARED maps (SEVERITY_LABEL, TYPE_LABEL, TIER_PHRASE, DEST_LABEL, DIMENSION_LABEL) match
   the Python maps byte-for-byte — a drift means GitHub/Slack and the dashboard would humanize the
   same finding differently.
2. The three dashboard-only maps (ELIGIBILITY_PHRASE, ELIGIBILITY_REASON_PHRASE, HITL_OUTCOME_LABEL)
   are TOTAL over their backing enums — a new enum member without a TS label would render as a raw
   slug (the frontend's *Phrase() helpers fall back to the raw value, so it wouldn't crash; this
   test is the fail-closed guard that catches the gap the Python `_assert_total` catches at import).

This fails if either invariant breaks — the fix is to edit findingSections.ts to match. Pure: reads
the checked-in .ts as text (no node, no build).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from outrider.audit.events import PublishEligibility, PublishEligibilityReason
from outrider.presentation.finding_sections import (
    DEST_LABEL,
    DIMENSION_LABEL,
    SEVERITY_LABEL,
    TIER_PHRASE,
    TYPE_LABEL,
)
from outrider.schemas.hitl import PerFindingOutcome

if TYPE_CHECKING:
    from enum import Enum

_TS = Path(__file__).resolve().parents[2] / "dashboard" / "src" / "lib" / "findingSections.ts"
_ENTRY = re.compile(r'(\w+):\s*"([^"]*)"')


def _ts_map(name: str) -> dict[str, str]:
    """Extract the `key: "value"` pairs from a `export const <name>: Record<…> = {…};` block."""
    src = _TS.read_text(encoding="utf-8")
    m = re.search(rf"export const {name}: Record<string, string> = \{{(.*?)\n\}};", src, re.DOTALL)
    if m is None:
        raise AssertionError(f"{name} not found in {_TS.name}")
    return dict(_ENTRY.findall(m.group(1)))


def _py_by_wire(mapping: dict[Enum, str]) -> dict[str, str]:
    return {member.value: label for member, label in mapping.items()}


def test_shared_maps_match_python_byte_for_byte() -> None:
    for name, py in [
        ("SEVERITY_LABEL", SEVERITY_LABEL),
        ("TYPE_LABEL", TYPE_LABEL),
        ("TIER_PHRASE", TIER_PHRASE),
        ("DEST_LABEL", DEST_LABEL),
        ("DIMENSION_LABEL", DIMENSION_LABEL),
    ]:
        assert _ts_map(name) == _py_by_wire(py), (
            f"{name} in dashboard/src/lib/findingSections.ts drifted from "
            f"presentation/finding_sections.py — reconcile the TS mirror."
        )


def test_dashboard_only_maps_are_total_over_enums() -> None:
    for name, enum in [
        ("ELIGIBILITY_PHRASE", PublishEligibility),
        ("ELIGIBILITY_REASON_PHRASE", PublishEligibilityReason),
        ("HITL_OUTCOME_LABEL", PerFindingOutcome),
    ]:
        missing = {m.value for m in enum} - set(_ts_map(name))
        assert not missing, (
            f"{name} in findingSections.ts is missing labels for {sorted(missing)} — "
            f"every {enum.__name__} member needs one."
        )
