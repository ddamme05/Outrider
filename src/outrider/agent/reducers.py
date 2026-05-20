# See specs/2026-05-19-analyze-foundation.md §2.
"""LangGraph reducer factory: dedup-keyed list append.

Per `reducers-dedup-not-concat`: plain `operator.add` (the LangGraph
default for `list[...]`) double-accumulates under checkpoint replay
because the framework can re-apply state deltas during rehydration.
Dedup-keyed merge is idempotent — items already present (by key) are
skipped on re-emission.

Consumers:
- `ReviewState.analysis_rounds` — keyed on `AnalysisRound.round_id`.
- `ReviewState.trace_candidates` — keyed on `TraceCandidate.candidate_id`.

The factory is generic so future state fields with content-derived
identity hashes (sister-spec finding proposal lists, etc.) can reuse the
same idempotent-merge contract without copying the dedup logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable


def append_with_dedup_by[T](
    key_fn: Callable[[T], Hashable],
) -> Callable[[list[T], list[T]], list[T]]:
    """Return a LangGraph reducer that appends incoming items, deduped by `key_fn`.

    Replay re-application is idempotent: items already present (by key)
    are skipped. Plain concat would double-accumulate under LangGraph
    checkpoint replay; this factory is the durable fix.

    Key extraction collisions (two items with same key but different
    content) resolve to first-seen: the existing entry is preserved and
    the incoming one is dropped. Callers ensure key uniqueness via
    content-derived hashes (e.g., `AnalysisRound.round_id`,
    `TraceCandidate.candidate_id`) so this collision case represents
    legitimate re-emission of the same logical item, not divergent state.
    """

    def reducer(existing: list[T], incoming: list[T]) -> list[T]:
        seen: set[Hashable] = {key_fn(item) for item in existing}
        merged = list(existing)
        for item in incoming:
            k = key_fn(item)
            if k not in seen:
                seen.add(k)
                merged.append(item)
        return merged

    return reducer
