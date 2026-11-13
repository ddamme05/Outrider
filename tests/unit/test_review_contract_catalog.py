"""Check the generated catalog linked from the public review contract."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_CATALOG = Path("docs/review-contract.json")


def test_catalog_generator_creates_missing_output_directory(tmp_path: Path) -> None:
    shutil.copytree(_ROOT / "src", tmp_path / "src", ignore=shutil.ignore_patterns("__pycache__"))
    generator = tmp_path / "scripts/gen_review_contract.py"
    generator.parent.mkdir()
    shutil.copyfile(_ROOT / "scripts/gen_review_contract.py", generator)
    result = subprocess.run(  # noqa: S603
        [sys.executable, str(generator)],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(tmp_path / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / _CATALOG).read_bytes() == (_ROOT / _CATALOG).read_bytes()
