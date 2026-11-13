"""Typed failures for source, patch and GitHub location checks.

Callers use the error type to decide how to handle a finding; they must not
infer the failure from its message. Translation helpers normalize their
failures to CoordinateError subclasses.
"""

from __future__ import annotations

from enum import StrEnum


class CoordinateErrorKind(StrEnum):
    """Structured discriminator for `CoordinateError` raise classes.

    Total over the raise-site inventory in `coordinates/translator.py`,
    `coordinates/diff_parser.py`, and `coordinates/spans.py`. New raise
    sites MUST add a value here AND set `kind=` at construction; the
    required-keyword-only constraint on `CoordinateError.__init__` enforces
    this at runtime.

    **Append-only discipline.** The `.value` strings of members below get
    persisted into `audit_events.payload.coordinate_error_kind` and round-trip
    on replay. Renaming or removing a member breaks every historical row
    that carried the old value — `PublishRoutingEvent`'s membership validator
    rejects the unknown string and replay fails. Treat the value set as
    append-only: add new members freely; never rename or remove an existing
    one. Same discipline as `FindingType` for the same reason. If a member
    genuinely needs to be retired, the path is "introduce successor + leave
    predecessor in place + emitters move to successor", not in-place edit.
    """

    # Raised by `translator.tree_sitter_to_github` when the span landed
    # outside any hunk's reviewable range. Publish routing branches this
    # to PublishDestination.REVIEW_BODY.
    UNCHANGED_REGION = "unchanged_region"

    # Raised by `translator.tree_sitter_to_github` /
    # `source_line_to_github` / `_line_to_byte_offset` for: byte_start out
    # of bounds, byte_end out of bounds, byte_end < byte_start, line_end
    # past EOF.
    BYTE_OFFSET_INVALID = "byte_offset_invalid"

    # Raised by `translator._find_patched_file` / `diff_parser.file_in_patch`
    # / `diff_parser.lookup_patched_file` when `unidiff.PatchSet(...)`
    # raised UnidiffParseError; wrapped for the CoordinateError contract.
    MALFORMED_PATCH = "malformed_patch"

    # Raised by `translator._find_patched_file` / `diff_parser.file_in_patch`
    # / `diff_parser.lookup_patched_file` when the patch contains duplicate
    # entries for the queried normalized path.
    DUPLICATE_FILE_ENTRY = "duplicate_file_entry"

    # Raised by `translator._find_patched_file` when file_path is absent
    # from the patch entirely. NOTE: distinct from the `ChangedFile`
    # registry-miss case that publish short-circuits BEFORE calling
    # coordinates; FILE_NOT_IN_PATCH fires when registry says "yes" but
    # coordinates says "no" (patch-text disagreement). Publish routing
    # surfaces both via reason=non_diffed_file.
    FILE_NOT_IN_PATCH = "file_not_in_patch"

    # Raised by `diff_parser.diff_line_to_scope` when called with
    # diff_line < 1. NOT reachable from publish (publish doesn't call
    # diff_line_to_scope) but enumerated for totality so the enum stays
    # a complete taxonomy.
    INVALID_DIFF_LINE = "invalid_diff_line"

    # Raised by `diff_parser.validate_diff_path` — umbrella for the
    # eight sub-rules (empty, backslash, shell metachars, trojan source,
    # Windows drive, absolute, .. traversal, .git internal). The
    # sub-discrimination lives in the human-readable .args[0] message
    # for audit-stream regex queries; the publish-routing decision treats
    # all sub-cases identically (DASHBOARD_ONLY). The message is NOT
    # serialized into PublishRoutingEvent payload to avoid leaking the
    # validate_diff_path rule set as an enumeration oracle to anyone with
    # audit-log read access.
    PATH_VALIDATION_FAILED = "path_validation_failed"

    # Raised by `spans.*` helpers when the caller passed negative
    # byte_length / max_lines / max_chars. These are programmer-error
    # preconditions, structurally different from runtime data-validation
    # kinds above. Distinct enum value so the publish-side branch can
    # treat them as "publisher bug, surface as DASHBOARD_ONLY +
    # AnomalyEvent" rather than silently grouping with BYTE_OFFSET_INVALID
    # (which is data-driven, not caller-driven).
    ARGUMENT_VALIDATION_FAILED = "argument_validation_failed"

    # _resolve_inline_location found the file in the diff registry but it has
    # no head content (for example, a deletion). Keep this distinct from a
    # non-diffed file for replay; route as PublishRoutingReason.COORDINATE_ERROR.
    HEAD_CONTENT_UNAVAILABLE = "head_content_unavailable"


class CoordinateError(Exception):
    """Raised when coordinate translation cannot produce a reviewable result.

    Single failure mode for the coordinates module per docs/spec.md §5.6.
    Catchable as `Exception`; the specific intermediate base in the MRO is
    an implementation detail.

    The `kind` keyword-only argument is REQUIRED at construction. Forgetting
    `kind=` raises `TypeError` at the raise site, before the exception
    propagates — that's the structural enforcement that backs the AST-walk
    totality test in `tests/unit/test_coordinates_error_kind_totality.py`.
    """

    def __init__(self, message: str, *, kind: CoordinateErrorKind) -> None:
        # isinstance assert catches the case where a caller mistypes
        # the kind argument (e.g., passes the string "unchanged_region"
        # rather than the enum member). Pydantic-style validation isn't
        # available on Exception subclasses, so this is the runtime fence.
        if not isinstance(kind, CoordinateErrorKind):
            raise TypeError(
                f"CoordinateError kind must be a CoordinateErrorKind member, "
                f"got {type(kind).__name__}={kind!r}"
            )
        super().__init__(message)
        self.kind: CoordinateErrorKind = kind
