"""Generate a Markdown snapshot of the repository structure."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


DEFAULT_EXCLUDED_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "htmlcov",
        "wheels",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root to scan. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("struct.md"),
        help="Markdown file to write. Defaults to struct.md.",
    )
    return parser.parse_args()


def should_exclude(path: Path) -> bool:
    return path.name in DEFAULT_EXCLUDED_NAMES or path.name.endswith(".egg-info")


def sorted_children(path: Path) -> list[Path]:
    children = [child for child in path.iterdir() if not should_exclude(child)]
    return sorted(children, key=lambda child: (not child.is_dir(), child.name.lower()))


def render_tree(root: Path) -> Iterable[str]:
    yield f"{root.name}/"
    yield from render_children(root, prefix="")


def render_children(path: Path, prefix: str) -> Iterable[str]:
    children = sorted_children(path)
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        branch = "`-- " if is_last else "|-- "
        suffix = "/" if child.is_dir() else ""

        yield f"{prefix}{branch}{child.name}{suffix}"

        if child.is_dir():
            extension = "    " if is_last else "|   "
            yield from render_children(child, prefix=f"{prefix}{extension}")


def markdown_for(root: Path) -> str:
    tree = "\n".join(render_tree(root))
    return f"# Repository Structure\n\n```text\n{tree}\n```\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output
    if not output.is_absolute():
        output = root / output

    output.write_text(markdown_for(root), encoding="utf-8")
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    main()
