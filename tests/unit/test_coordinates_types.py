"""Unit tests for coordinates' boundary types: GitHubCommentLocation + CoordinateError.

Covers the type-level validators only — Field(ge=1), paired-fields rule for
multi-line comments, range rule, ConfigDict(extra='forbid'), Literal
constraints. Per docs/spec.md §7.2 (GitHubCommentLocation canonical shape)
and §5.6 (CoordinateError as the translator's single failure mode).

Per-function tests (test_coordinates_translator.py, test_coordinates_diff_parser.py)
land in subsequent commits as the function bodies are implemented.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from outrider.coordinates import CoordinateError, GitHubCommentLocation

# ----------------------------------------------------------------------------
# GitHubCommentLocation — minimum valid construction
# ----------------------------------------------------------------------------


def test_github_comment_location_minimum_valid_single_line() -> None:
    """Smallest valid construction: single-line comment at line 1, side=RIGHT."""
    loc = GitHubCommentLocation(file_path="src/outrider/foo.py", line=1, side="RIGHT")
    assert loc.file_path == "src/outrider/foo.py"
    assert loc.line == 1
    assert loc.side == "RIGHT"
    assert loc.start_line is None
    assert loc.start_side is None


def test_github_comment_location_left_side_constructible() -> None:
    """side='LEFT' is valid even though V1 translator only produces 'RIGHT'.

    The model preserves the canonical Literal['LEFT', 'RIGHT']; V1's single-side
    output is a translator-policy commitment, not a model-level constraint.
    """
    loc = GitHubCommentLocation(file_path="x.py", line=1, side="LEFT")
    assert loc.side == "LEFT"


# ----------------------------------------------------------------------------
# Field constraints — Field(ge=1) on `line` and `start_line`
# ----------------------------------------------------------------------------


def test_github_comment_location_line_must_be_positive() -> None:
    """line=0 fails the Field(ge=1) validator (1-indexed per canonical §7.2)."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(file_path="x.py", line=0, side="RIGHT")


def test_github_comment_location_line_negative_rejected() -> None:
    """Negative line numbers rejected."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(file_path="x.py", line=-5, side="RIGHT")


def test_github_comment_location_start_line_must_be_positive_when_set() -> None:
    """start_line=0 fails the Field(ge=1) validator."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=5,
            side="RIGHT",
            start_line=0,
            start_side="RIGHT",
        )


# ----------------------------------------------------------------------------
# Multi-line pairing rule: (start_line is None) == (start_side is None)
# ----------------------------------------------------------------------------


def test_github_comment_location_start_fields_paired_both_none() -> None:
    """start_line=None + start_side=None — V1 single-line case."""
    loc = GitHubCommentLocation(
        file_path="x.py",
        line=5,
        side="RIGHT",
        start_line=None,
        start_side=None,
    )
    assert loc.start_line is None and loc.start_side is None


def test_github_comment_location_start_fields_paired_both_set() -> None:
    """start_line + start_side both set — valid multi-line."""
    loc = GitHubCommentLocation(
        file_path="x.py",
        line=10,
        side="RIGHT",
        start_line=5,
        start_side="RIGHT",
    )
    assert loc.start_line == 5 and loc.start_side == "RIGHT"


def test_github_comment_location_start_line_set_without_start_side_rejected() -> None:
    """start_line set but start_side None — paired-fields rule violated."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=10,
            side="RIGHT",
            start_line=5,
            start_side=None,
        )


def test_github_comment_location_start_side_set_without_start_line_rejected() -> None:
    """start_side set but start_line None — paired-fields rule violated."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=10,
            side="RIGHT",
            start_line=None,
            start_side="LEFT",
        )


# ----------------------------------------------------------------------------
# Multi-line range rule: start_line <= line when both set
# ----------------------------------------------------------------------------


def test_github_comment_location_start_line_must_be_le_line() -> None:
    """start_line > line — range rule violated."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=5,
            side="RIGHT",
            start_line=10,
            start_side="RIGHT",
        )


def test_github_comment_location_start_line_equals_line_allowed() -> None:
    """start_line == line — boundary case for the <= rule, allowed."""
    loc = GitHubCommentLocation(
        file_path="x.py",
        line=5,
        side="RIGHT",
        start_line=5,
        start_side="RIGHT",
    )
    assert loc.start_line == 5 and loc.line == 5


def test_github_comment_location_start_line_strictly_less_than_line_allowed() -> None:
    """Typical multi-line: start_line < line."""
    loc = GitHubCommentLocation(
        file_path="x.py",
        line=10,
        side="RIGHT",
        start_line=3,
        start_side="RIGHT",
    )
    assert loc.start_line == 3 and loc.line == 10


# ----------------------------------------------------------------------------
# ConfigDict(extra="forbid") + Literal constraints
# ----------------------------------------------------------------------------


def test_github_comment_location_extra_fields_rejected() -> None:
    """ConfigDict(extra='forbid') per docs/conventions.md (silent extras drift schemas)."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=1,
            side="RIGHT",
            unknown_field="x",  # type: ignore[call-arg]
        )


def test_github_comment_location_side_constrained_to_left_or_right() -> None:
    """side must be Literal['LEFT', 'RIGHT'] — no other values."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(file_path="x.py", line=1, side="MIDDLE")  # type: ignore[arg-type]


def test_github_comment_location_start_side_constrained_to_left_right_or_none() -> None:
    """start_side must be Literal['LEFT', 'RIGHT'] | None."""
    with pytest.raises(ValidationError):
        GitHubCommentLocation(
            file_path="x.py",
            line=5,
            side="RIGHT",
            start_line=3,
            start_side="MIDDLE",  # type: ignore[arg-type]
        )


# ----------------------------------------------------------------------------
# CoordinateError — class hierarchy + raise/catch behavior
# ----------------------------------------------------------------------------


def test_coordinate_error_is_exception_subclass() -> None:
    """CoordinateError is a typed exception — catchable as Exception.

    Public contract: callers can `except CoordinateError` (and the broader
    `except Exception`). The specific intermediate base in the MRO is an
    implementation detail and intentionally not asserted here.
    """
    assert issubclass(CoordinateError, Exception)


def test_coordinate_error_can_be_raised_and_caught() -> None:
    """Basic raise/catch behavior with a message."""
    with pytest.raises(CoordinateError, match="test message"):
        raise CoordinateError("test message")


# ----------------------------------------------------------------------------
# Serialization round-trip — replay/audit safety
# ----------------------------------------------------------------------------


def test_github_comment_location_round_trip_single_line() -> None:
    """model_dump() → model_validate() round-trips for the V1 single-line shape.

    Locks in JSON-safety as part of the type contract. A future custom
    validator that mutated state during construction would break this.
    """
    loc = GitHubCommentLocation(file_path="src/foo.py", line=42, side="RIGHT")
    assert GitHubCommentLocation.model_validate(loc.model_dump()) == loc


def test_github_comment_location_round_trip_multi_line() -> None:
    """model_dump() → model_validate() round-trips for the multi-line shape.

    Even though V1 translator output is single-line only, the model accepts
    multi-line construction; round-trip must hold for both shapes so future
    multi-line support doesn't introduce a serialization regression.
    """
    loc = GitHubCommentLocation(
        file_path="src/foo.py",
        line=42,
        side="RIGHT",
        start_line=38,
        start_side="RIGHT",
    )
    assert GitHubCommentLocation.model_validate(loc.model_dump()) == loc
