"""Structural eval scenario: import statement parsing.

Per spec §11.2: `import x`, `from x import y`, `from .x import y`,
`from x import *` all produce well-formed `ImportRef` objects.
"""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python

SOURCE = """\
import os
import sqlalchemy as sa
from typing import Any, Optional
from .helpers import normalize
from collections.abc import Sequence
from outrider.policy import *
"""

# (import_kind, module, names) per canonical §5.4 ImportRef.
# - For `import x` / `import x as y`: import_kind="direct"; the
#   adapter's V1 convention puts the alias (or nothing) in `names`.
# - For `from x import y, z`: import_kind="from"; names is the tuple
#   of imported names.
# - For `from .x import y`: import_kind="relative".
# - For `from x import *`: import_kind="star"; names is () (the star
#   itself is not a name).
EXPECTED_IMPORTS = (
    {"import_kind": "direct", "module": "os", "names": ()},
    {"import_kind": "direct", "module": "sqlalchemy", "names": ("sa",)},
    {"import_kind": "from", "module": "typing", "names": ("Any", "Optional")},
    {"import_kind": "relative", "module": ".helpers", "names": ("normalize",)},
    {
        "import_kind": "from",
        "module": "collections.abc",
        "names": ("Sequence",),
    },
    {"import_kind": "star", "module": "outrider.policy", "names": ()},
)


def test_import_forms_all_parse_to_well_formed_imports() -> None:
    """Six import forms in SOURCE produce six ImportRef objects with the expected shapes."""
    result = parse_python(SOURCE.encode(), "test.py", MagicMock())
    assert len(result.imports) == len(EXPECTED_IMPORTS)
    for actual, expected in zip(result.imports, EXPECTED_IMPORTS, strict=True):
        assert actual.import_kind == expected["import_kind"]
        assert actual.module == expected["module"]
        assert tuple(actual.names) == expected["names"]
