"""Structural eval scenario: import statement parsing.

Per spec §11.2: `import x`, `from x import y`, `from .x import y`,
`from x import *` all produce well-formed `ImportRef` objects.

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

import pytest

pytestmark = pytest.mark.skip(reason="requires ast_facts")

SOURCE = """\
import os
import sqlalchemy as sa
from typing import Any, Optional
from .helpers import normalize
from collections.abc import Sequence
from outrider.policy import *
"""

EXPECTED_IMPORTS = (
    {"module": "os", "names": (), "is_relative": False},
    {"module": "sqlalchemy", "names": (), "alias": "sa", "is_relative": False},
    {"module": "typing", "names": ("Any", "Optional"), "is_relative": False},
    {"module": ".helpers", "names": ("normalize",), "is_relative": True},
    {"module": "collections.abc", "names": ("Sequence",), "is_relative": False},
    {"module": "outrider.policy", "names": ("*",), "is_relative": False},
)


def test_import_forms_all_parse_to_well_formed_imports() -> None:
    """Six import forms in SOURCE produce six ImportRef objects with the expected shapes."""
    from outrider.ast_facts import extract_imports  # type: ignore[import-not-found]

    imports = extract_imports(SOURCE)
    assert len(imports) == len(EXPECTED_IMPORTS)
    for actual, expected in zip(imports, EXPECTED_IMPORTS, strict=True):
        assert actual.module == expected["module"]
        assert tuple(actual.names) == expected["names"]
        assert actual.is_relative == expected["is_relative"]
        if "alias" in expected:
            assert actual.alias == expected["alias"]
