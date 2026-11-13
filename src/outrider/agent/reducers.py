"""Merge saved lists without duplicating work during replay.

append_with_dedup_by keeps the first item for each identity
(reducers-dedup-not-concat). append_with_slot_guard also rejects conflicting
results for the same worker slot instead of silently choosing one.

See specs/2026-05-19-analyze-foundation.md and DECISIONS.md#017, #063.
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
    """Append unseen keys, keeping existing entries and incoming order.

    Replaying the same keys is idempotent. Positional worker slots use
    append_with_slot_guard to detect divergent content on retries."""

    def reducer(existing: list[T], incoming: list[T]) -> list[T]:
        seen: set[Hashable] = {key_fn(item) for item in existing}
        merged = list(existing)
        for item in incoming:
            key = key_fn(item)
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged

    return reducer


def semantic_digest(
    model: BaseModel,
    *,
    exclude_paths: frozenset[str],
) -> str:
    """Canonical SHA-256 over a model's semantic content (DECISIONS.md#063).

    Recipe: `model_dump(mode="json")`, then every EXPLICIT PATH in
    `exclude_paths` is removed, then compact sorted-key JSON, UTF-8,
    SHA-256 hex. Paths are dot-separated with `[]` for list traversal —
    `"findings.[].finding_id"` strips the generated identity inside each
    nested finding while a future top-level SEMANTIC field that happens to
    be named `finding_id` still digests (per #063's amendment, exclusions
    are positional, never name-anywhere — a name-based strip would
    silently ignore it). Inclusion-by-default stays the fail-safe
    direction: a future field digests automatically, and the
    worker-outcome model pins its path list against its generated fields
    one-for-one.
    """
    parsed: tuple[tuple[str, ...], ...] = tuple(
        tuple(path.split(".")) for path in sorted(exclude_paths)
    )

    def strip(value: object, paths: tuple[tuple[str, ...], ...]) -> object:
        if not paths:
            return value
        if isinstance(value, dict):
            out: dict[str, object] = {}
            for k, v in value.items():
                if any(p == (k,) for p in paths):
                    continue
                tails = tuple(p[1:] for p in paths if len(p) > 1 and p[0] == k)
                out[k] = strip(v, tails)
            return out
        if isinstance(value, list):
            tails = tuple(p[1:] for p in paths if len(p) > 1 and p[0] == "[]")
            stripped_items: list[object] = []
            for list_item in value:
                stripped_items.append(strip(list_item, tails))
            return stripped_items
        return value

    stripped = strip(model.model_dump(mode="json"), parsed)
    # Encoding settings MATCH `policy.canonical.canonicalize_for_hash`
    # (sorted keys, compact separators, ensure_ascii=False + UTF-8,
    # allow_nan=False) so the two recipes cannot drift on the dimensions
    # they share. The canonical primitive itself is NOT reusable here by
    # design: it BANS floats (cross-implementation identity hashes must
    # not depend on float serialization), while a model dump legitimately
    # carries computed floats (`ReviewFinding.confidence`) — safe for THIS
    # digest because it is merge-time-ephemeral, never persisted, and only
    # ever compared against a digest from the same process's encoder.
    payload = json.dumps(
        stripped, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
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
        # Validate existing slots even when incoming is empty: divergent
        # duplicates must raise; identical replay duplicates keep the first.
        # Hash only collisions. LangGraph folds fan-out writes one at a time,
        # so hashing every existing item on every fold would cost quadratic work
        # in the normal case where slots are distinct.
        seen: dict[Hashable, T] = {}
        merged: list[T] = []

        def check_collision(slot: Hashable, occupant: T, item: T) -> None:
            if digest_fn(occupant) != digest_fn(item):
                raise SlotDivergenceError(slot)
            # equal digest: replay re-application — no-op.

        for item in existing:
            slot = slot_fn(item)
            occupant = seen.get(slot)
            if occupant is None:
                seen[slot] = item
                merged.append(item)
            else:
                check_collision(slot, occupant, item)
        for item in incoming:
            slot = slot_fn(item)
            occupant = seen.get(slot)
            if occupant is None:
                seen[slot] = item
                merged.append(item)
            else:
                check_collision(slot, occupant, item)
        return merged

    return reducer
