"""ReviewPhaseEvent: marker Literal + V1.5 forward-compat phase_key.

Backs `phase-events-bound-work`. The marker Literal constrains the
deterministic vocabulary; phase_key is nullable so V1 events emit None
and V1.5's parallel-analyze workers can populate it without a schema
migration.
"""

from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

from outrider.audit.events import ReviewPhaseEvent


def _build_event(**overrides: Any) -> ReviewPhaseEvent:
    fields: dict[str, Any] = {
        "review_id": uuid4(),
        "phase_id": "p1",
        "node_id": "analyze",
        "marker": "start",
    }
    fields.update(overrides)
    return ReviewPhaseEvent(**fields)


def test_review_phase_event_marker_admits_start_and_end() -> None:
    """Both canonical marker values admit."""
    event_start = _build_event(marker="start")
    event_end = _build_event(marker="end")
    assert event_start.marker == "start"
    assert event_end.marker == "end"


def test_review_phase_event_marker_rejects_other_values() -> None:
    """Anything outside the start/end Literal raises."""
    with pytest.raises(ValidationError):
        _build_event(marker="middle")
    with pytest.raises(ValidationError):
        _build_event(marker="complete")


def test_review_phase_event_phase_key_defaults_to_none() -> None:
    """V1 callers omit phase_key; V1.5 parallel workers populate it."""
    event = _build_event()
    assert event.phase_key is None

    keyed = _build_event(phase_key="src/foo.py")
    assert keyed.phase_key == "src/foo.py"
