"""Collect docs directory contents into one Markdown file."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path("docs"),
        help="Docs directory to collect. Defaults to docs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("contents.md"),
        help="Markdown file to write. Defaults to contents.md.",
    )
    return parser.parse_args()


def resolve_from_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def docs_files(docs_dir: Path) -> list[Path]:
    return sorted(path for path in docs_dir.rglob("*") if path.is_file())


def fence_for(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def markdown_for(root: Path, docs_dir: Path) -> str:
    lines = ["# Docs Contents", ""]

    for path in docs_files(docs_dir):
        relative_path = path.relative_to(root)
        contents = path.read_text(encoding="utf-8")
        fence = fence_for(contents)
        lines.extend(
            [
                f"## {relative_path.as_posix()}",
                "",
                f"{fence}markdown",
                contents.rstrip(),
                fence,
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    docs_dir = resolve_from_root(root, args.docs_dir).resolve()
    output = resolve_from_root(root, args.output)

    if not docs_dir.is_dir():
        raise SystemExit(f"Docs directory does not exist: {docs_dir}")

    output.write_text(markdown_for(root, docs_dir), encoding="utf-8")
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    main()
