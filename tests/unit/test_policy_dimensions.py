# See specs/2026-05-19-analyze-foundation.md §6.
"""`FINDING_TYPE_TO_DIMENSION` mapping + module-load lockstep tests.

Pins:
- Every `FindingType` has a dimension entry; conversely no extras.
- Lockstep with `SEVERITY_POLICY` (same key set).
- `MappingProxyType` blocks runtime mutation.
- `verify_lockstep()` raises when the three sets drift (simulated
  via temporary set difference).
- `lookup_dimension` returns the canonical mapping.
"""

from __future__ import annotations

import pytest

from outrider.policy import SEVERITY_POLICY, FindingType
from outrider.policy.dimensions import (
    FINDING_TYPE_TO_DIMENSION,
    lookup_dimension,
    verify_lockstep,
)
from outrider.schemas import ReviewDimension


def test_finding_type_to_dimension_has_every_finding_type() -> None:
    """Every FindingType enum value has a dimension entry — no fallthrough."""
    missing = set(FindingType) - set(FINDING_TYPE_TO_DIMENSION)
    assert missing == set(), f"FindingType members without dimension: {missing}"


def test_finding_type_to_dimension_has_no_extras() -> None:
    """No dimension keys that aren't FindingType members."""
    extras = set(FINDING_TYPE_TO_DIMENSION) - set(FindingType)
    assert extras == set(), f"dimension keys that aren't FindingType: {extras}"


def test_finding_type_to_dimension_lockstep_with_severity_policy() -> None:
    """Spec §6 invariant: SEVERITY_POLICY and FINDING_TYPE_TO_DIMENSION
    have identical key sets."""
    assert set(SEVERITY_POLICY) == set(FINDING_TYPE_TO_DIMENSION)


def test_finding_type_to_dimension_values_are_review_dimensions() -> None:
    """Every value must be a ReviewDimension enum member."""
    for ftype, dim in FINDING_TYPE_TO_DIMENSION.items():
        assert isinstance(dim, ReviewDimension), (
            f"{ftype.value} maps to {dim!r} which is not a ReviewDimension"
        )


def test_finding_type_to_dimension_is_read_only() -> None:
    """`MappingProxyType` blocks runtime mutation — same defense-in-depth
    posture as `SEVERITY_POLICY`."""
    with pytest.raises(TypeError):
        FINDING_TYPE_TO_DIMENSION[FindingType.SQL_INJECTION] = ReviewDimension.PERFORMANCE  # type: ignore[index]


def test_lookup_dimension_returns_mapped_value() -> None:
    """`lookup_dimension(ftype)` returns the canonical mapping."""
    assert lookup_dimension(FindingType.SQL_INJECTION) == ReviewDimension.SECURITY
    assert lookup_dimension(FindingType.UNUSED_IMPORT) == ReviewDimension.CODE_QUALITY


def test_verify_lockstep_passes_in_canonical_state() -> None:
    """`verify_lockstep()` is a no-op when the three sets match."""
    verify_lockstep()  # raises AssertionError on drift; canonical state should pass


def test_verify_lockstep_raises_on_simulated_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """If `FINDING_TYPE_TO_DIMENSION` is replaced with an incomplete mapping,
    the lockstep guard fires.

    Verifies the assertion ACTUALLY fails-loud — without this, a future
    refactor that loosens the guard (or rewrites the comparison) could
    silently pass when drift is introduced.
    """
    from types import MappingProxyType

    import outrider.policy.dimensions as dim_mod

    # Replace with a mapping missing one key.
    short_dict = dict(FINDING_TYPE_TO_DIMENSION)
    short_dict.pop(FindingType.SQL_INJECTION)
    monkeypatch.setattr(
        dim_mod,
        "FINDING_TYPE_TO_DIMENSION",
        MappingProxyType(short_dict),
    )
    with pytest.raises(AssertionError, match="Policy lockstep violation"):
        verify_lockstep()


def test_module_load_lockstep_runs_at_import() -> None:
    """`outrider.policy.dimensions` is force-imported from `outrider/__init__.py`
    so the lockstep guard fires at app startup / test collection — even
    when no analyze code is on the import path.

    Runs in a SUBPROCESS so the assertion isn't self-fulfilling: the
    parent test process has already imported `outrider.policy.dimensions`
    at file-load (line 18) just by setting up the rest of the test
    suite, so an in-process check would pass regardless of whether
    `outrider/__init__.py` does anything. The subprocess starts with a
    clean `sys.modules`, imports only `outrider`, and prints the
    membership of `outrider.policy.dimensions`.
    """
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603 — argv list, no shell, fixed args
        [
            sys.executable,
            "-c",
            ("import sys; import outrider; print('outrider.policy.dimensions' in sys.modules)"),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == "True", (
        "outrider.policy.dimensions must be force-imported by outrider/__init__.py "
        "so the lockstep guard runs at app startup, not just at first analyze import. "
        f"subprocess stdout={result.stdout!r} stderr={result.stderr!r}"
    )
