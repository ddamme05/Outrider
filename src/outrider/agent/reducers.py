# See specs/2026-05-19-analyze-foundation.md §2 and DECISIONS.md#017.
"""LangGraph reducer factory: dedup-keyed list append.

Per `reducers-dedup-not-concat`: plain `operator.add` (the LangGraph
default for `list[...]`) double-accumulates under checkpoint replay
because the framework can re-apply state deltas during rehydration.
Dedup-keyed merge is idempotent — items already present (by key) are
skipped on re-emission.

Consumers:
- `ReviewState.analysis_rounds` — keyed on `AnalysisRound.round_id`; one
  round per analyze PASS, never per worker (see DECISIONS.md#063).
- `ReviewState.trace_candidates` — keyed on `TraceCandidate.candidate_id`.

The factory is generic so future state fields with content-derived
identity hashes (sister-spec finding proposal lists, etc.) can reuse the
same idempotent-merge contract without copying the dedup logic.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from pydantic import BaseModel


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


def semantic_digest(
    model: BaseModel,
    *,
    exclude_fields: frozenset[str],
) -> str:
    """Canonical SHA-256 over a model's semantic content (DECISIONS.md#063).

    Recipe: `model_dump(mode="json")`, then every field named in
    `exclude_fields` is removed RECURSIVELY (a worker outcome nests
    findings, and each finding carries its own generated `finding_id`),
    then compact sorted-key JSON, UTF-8, SHA-256 hex. Exclusion is
    by-name-anywhere-in-the-tree by design: inclusion-by-default is the
    fail-safe direction (a future field digests automatically; a false
    divergence is loud, a silently ignored field is not), and the caller's
    exclusion list must therefore name ONLY generated identities whose
    names are never semantic at any depth (`finding_id`-class UUIDs,
    timestamps). The worker-outcome model pins its list against its
    generated fields one-for-one.
    """

    def strip(value: object) -> object:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in exclude_fields}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    stripped = strip(model.model_dump(mode="json"))
    payload = json.dumps(stripped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SlotDivergenceError(RuntimeError):
    """Same slot, different semantic content — state would fork from audit.

    A `(file, pass)` slot key is positional, not content-derived, so the
    generic first-wins dedup cannot apply (DECISIONS.md#063 amendment): an
    LLM-backed worker re-executed into the same slot can legitimately
    produce a DIFFERENT outcome, and silently keeping the first would
    leave state disagreeing with the audit stream's record of the retry.
    A review that forks state from audit must abort, not publish.
    """

    def __init__(self, slot: Hashable) -> None:
        self.slot = slot
        super().__init__(
            f"divergent content for worker-outcome slot {slot!r}: same slot, "
            f"different semantic digest — state would fork from audit; aborting"
        )


def append_with_slot_guard[T](
    slot_fn: Callable[[T], Hashable],
    digest_fn: Callable[[T], str],
) -> Callable[[list[T], list[T]], list[T]]:
    """Reducer for POSITIONAL slot keys (DECISIONS.md#063 amendment).

    Same slot + identical semantic digest → idempotent no-op (checkpoint
    replay re-application); same slot + divergent digest → raise
    `SlotDivergenceError` (never first-wins); new slot → append. Use
    `append_with_dedup_by` instead whenever the key is content-derived.
    """

    def reducer(existing: list[T], incoming: list[T]) -> list[T]:
        seen: dict[Hashable, str] = {slot_fn(item): digest_fn(item) for item in existing}
        merged = list(existing)
        for item in incoming:
            slot = slot_fn(item)
            digest = digest_fn(item)
            if slot not in seen:
                seen[slot] = digest
                merged.append(item)
            elif seen[slot] != digest:
                raise SlotDivergenceError(slot)
            # equal digest: replay re-application — no-op.
        return merged

    return reducer
