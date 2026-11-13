"""Relative reference candidates stay anchored to the importing source path."""

import pytest

from outrider.coordinates import CoordinateError, python_relative_import_candidate_paths


@pytest.mark.parametrize(
    ("importer", "module", "level", "expected"),
    [
        ("src/pkg/service.py", "helpers", 1, ("src/pkg/helpers.py", "src/pkg/helpers/__init__.py")),
        (
            "pkg/sub/__init__.py",
            "shared.codec",
            2,
            ("pkg/shared/codec.py", "pkg/shared/codec/__init__.py"),
        ),
    ],
)
def test_reference_paths_preserve_the_explicit_relative_anchor(importer, module, level, expected):
    assert (
        python_relative_import_candidate_paths(importer_path=importer, module=module, level=level)
        == expected
    )


@pytest.mark.parametrize(
    ("importer", "module", "level"),
    [
        ("pkg/main.py", "../outside", 1),
        ("pkg/main.py", "helper", 2),
        ("pkg/main.py", "", 1),
        ("pkg/main.py", "helper", 0),
    ],
)
def test_unsupported_or_escaping_imports_have_no_reference_candidates(importer, module, level):
    with pytest.raises(CoordinateError):
        python_relative_import_candidate_paths(importer_path=importer, module=module, level=level)
