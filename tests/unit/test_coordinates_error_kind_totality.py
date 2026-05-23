# See specs/2026-05-21-publish-node.md Q1.
# This test is the structural defense that every `raise CoordinateError(...)`
# in coordinates/ carries a typed `kind`. Pairs with the required-kwarg
# discipline on `CoordinateError.__init__` (which catches bare
# `raise CoordinateError("msg")` at runtime via TypeError) — this AST walk
# catches the rarer aliased-class-raise patterns the kwarg check might miss.
"""Totality: every `raise CoordinateError(...)` in coordinates/ carries `kind=`.

Mechanism: AST-walk `coordinates/translator.py` + `coordinates/diff_parser.py`,
find every `ast.Raise` whose exc is a `CoordinateError(...)` Call, and assert
the call has a `kind=` keyword argument drawn from the `CoordinateErrorKind`
enum.

This test backstops the structural enforcement at construction time
(`CoordinateError.__init__` requires `kind` as a keyword-only arg). Without
the structural enforcement, this test would be the only defense; with it,
this test catches the corner patterns the runtime check might miss
(aliased-class raises, factory helpers that wrap construction).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from outrider.coordinates.errors import CoordinateErrorKind

COORDINATES_FILES: tuple[Path, ...] = (
    Path("src/outrider/coordinates/translator.py"),
    Path("src/outrider/coordinates/diff_parser.py"),
    Path("src/outrider/coordinates/spans.py"),
)


def _project_root() -> Path:
    # Walk up from this file until we hit pyproject.toml — robust to
    # being invoked from any pytest cwd.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("could not locate project root from test file")


def _find_coordinate_error_raises(source: str) -> list[tuple[int, ast.Call | None]]:
    """Return (lineno, Call-node-or-None) for every `raise CoordinateError(...)`.

    Returns the Call node when the raise has an exception-construction
    argument; None when the raise is bare (`raise CoordinateError`).
    """
    tree = ast.parse(source)
    found: list[tuple[int, ast.Call | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise):
            continue
        exc = node.exc
        if exc is None:
            # `raise` with no exc is `raise <current>` inside `except` —
            # not a fresh construction; skip.
            continue
        # `raise CoordinateError(...)` — Call form.
        if (
            isinstance(exc, ast.Call)
            and isinstance(exc.func, ast.Name)
            and exc.func.id == "CoordinateError"
        ):
            found.append((node.lineno, exc))
            continue
        # `raise CoordinateError` — bare class form (no parens).
        if isinstance(exc, ast.Name) and exc.id == "CoordinateError":
            found.append((node.lineno, None))
            continue
    return found


def _extract_kind_keyword(call: ast.Call) -> str | None:
    """Return the value-name passed as `kind=`, or None if missing."""
    for kw in call.keywords:
        if kw.arg != "kind":
            continue
        # CoordinateErrorKind.UNCHANGED_REGION → AST shape:
        # Attribute(value=Name("CoordinateErrorKind"), attr="UNCHANGED_REGION")
        if (
            isinstance(kw.value, ast.Attribute)
            and isinstance(kw.value.value, ast.Name)
            and kw.value.value.id == "CoordinateErrorKind"
        ):
            return kw.value.attr
        # Other shapes — e.g., a passthrough variable — return a sentinel
        # naming what was used, so the test failure is informative.
        return ast.unparse(kw.value)
    return None


_VALID_KIND_NAMES = {member.name for member in CoordinateErrorKind}


@pytest.mark.parametrize("relpath", [str(p) for p in COORDINATES_FILES])
def test_every_coordinate_error_raise_carries_a_typed_kind(relpath: str) -> None:
    """Every `raise CoordinateError(...)` in this file passes `kind=<member>`.

    Forbidden patterns:
    - `raise CoordinateError("msg")` (no kind kwarg)
    - `raise CoordinateError` (bare class)
    - `raise CoordinateError("msg", kind=some_local_var)` (kind value not
      drawn from `CoordinateErrorKind` enum member directly)
    """
    path = _project_root() / relpath
    source = path.read_text(encoding="utf-8")
    raises = _find_coordinate_error_raises(source)
    assert raises, (
        f"no `raise CoordinateError(...)` sites found in {relpath} — file inventory may have moved"
    )

    errors: list[str] = []
    for lineno, call in raises:
        if call is None:
            errors.append(
                f"{relpath}:{lineno} — bare `raise CoordinateError` "
                f"(no construction); kind cannot be set"
            )
            continue
        kind_value = _extract_kind_keyword(call)
        if kind_value is None:
            errors.append(f"{relpath}:{lineno} — missing `kind=` keyword argument")
            continue
        if kind_value not in _VALID_KIND_NAMES:
            errors.append(
                f"{relpath}:{lineno} — `kind=` value {kind_value!r} is not a "
                f"CoordinateErrorKind member name (valid: {sorted(_VALID_KIND_NAMES)})"
            )
    assert not errors, "CoordinateError totality violations:\n" + "\n".join(errors)


def test_coordinate_error_kind_enum_is_total_over_documented_raise_classes() -> None:
    """The seven enum members map to the seven raise-site categories
    enumerated in the publish-node spec's raise-site inventory.

    If a new kind needs to be added, this test fails and the spec section
    "Q1 / `CoordinateErrorKind` totality" needs the new value AND the
    raise-site inventory comment updated.
    """
    expected_names = {
        "UNCHANGED_REGION",
        "BYTE_OFFSET_INVALID",
        "MALFORMED_PATCH",
        "DUPLICATE_FILE_ENTRY",
        "FILE_NOT_IN_PATCH",
        "INVALID_DIFF_LINE",
        "PATH_VALIDATION_FAILED",
        "ARGUMENT_VALIDATION_FAILED",
        # Publish node's _resolve_inline_location raises this kind when
        # the finding's file_path is in the ChangedFile registry but
        # head_content=None (e.g., status="removed"). Distinct from
        # FILE_NOT_IN_PATCH per the audit-stream replay-distinction
        # requirement.
        "HEAD_CONTENT_UNAVAILABLE",
    }
    actual_names = {member.name for member in CoordinateErrorKind}
    assert actual_names == expected_names, (
        f"CoordinateErrorKind drift detected. Expected {expected_names}, "
        f"got {actual_names}. If adding a new kind, update this test AND "
        f"the publish-node spec's raise-site inventory."
    )


def test_coordinate_error_init_requires_kind_keyword() -> None:
    """Constructing CoordinateError without `kind=` raises TypeError.

    This is the structural enforcement requested by the publish-node spec in
    lieu of (or in addition to) the AST walk. Forgetting `kind=` at any
    raise site fails fast at the raise — before propagation.
    """
    from outrider.coordinates.errors import CoordinateError

    with pytest.raises(TypeError, match=r"missing 1 required keyword-only argument: 'kind'"):
        CoordinateError("msg")  # type: ignore[call-arg]


def test_coordinate_error_init_rejects_non_enum_kind() -> None:
    """Constructing CoordinateError with a non-enum kind raises TypeError
    with a helpful message.

    Defends against typos that pass a string ("unchanged_region") rather
    than the enum member (CoordinateErrorKind.UNCHANGED_REGION).
    """
    from outrider.coordinates.errors import CoordinateError

    with pytest.raises(TypeError, match=r"must be a CoordinateErrorKind member"):
        CoordinateError("msg", kind="unchanged_region")  # type: ignore[arg-type]


def test_coordinate_error_construction_with_valid_kind_works() -> None:
    """Happy-path construction sets .kind to the passed enum member and
    keeps the canonical message in .args[0]."""
    from outrider.coordinates.errors import CoordinateError

    exc = CoordinateError("test message", kind=CoordinateErrorKind.UNCHANGED_REGION)
    assert exc.kind is CoordinateErrorKind.UNCHANGED_REGION
    assert exc.args == ("test message",)
    assert str(exc) == "test message"
