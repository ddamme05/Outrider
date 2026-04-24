"""Generate a Markdown snapshot of repository structure and file contents."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


DEFAULT_EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "__pycache__",
        "build",
        "dist",
        "env",
        "htmlcov",
        "node_modules",
        "venv",
        "wheels",
    }
)

DEFAULT_EXCLUDED_FILE_NAMES: frozenset[str] = frozenset(
    {
        ".codex",
        ".env",
        ".env.local",
        ".env.private",
        "contents.md",
        "snapshot.md",
        "struct.md",
        "uv.lock",
    }
)

LANGUAGES_BY_SUFFIX: dict[str, str] = {
    ".css": "css",
    ".env": "dotenv",
    ".gitignore": "gitignore",
    ".html": "html",
    ".js": "javascript",
    ".json": "json",
    ".lock": "toml",
    ".md": "markdown",
    ".py": "python",
    ".rst": "rst",
    ".sh": "bash",
    ".sql": "sql",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".yaml": "yaml",
    ".yml": "yaml",
}


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
        default=Path("snapshot.md"),
        help="Markdown file to write. Defaults to snapshot.md.",
    )
    return parser.parse_args()


def resolve_from_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def should_exclude_dir(path: Path) -> bool:
    return (
        path.name.startswith(".")
        or path.name in DEFAULT_EXCLUDED_DIR_NAMES
        or path.name.endswith(".egg-info")
    )


def should_exclude_file(path: Path) -> bool:
    return path.name in DEFAULT_EXCLUDED_FILE_NAMES


def sorted_children(path: Path) -> list[Path]:
    children = [
        child
        for child in path.iterdir()
        if not (
            child.is_dir()
            and should_exclude_dir(child)
            or child.is_file()
            and should_exclude_file(child)
        )
    ]
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


def codebase_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for child in sorted_children(root):
        if child.is_dir():
            files.extend(codebase_files(child))
        elif child.is_file():
            files.append(child)
    return files


def language_for(path: Path) -> str:
    if path.name == ".gitignore":
        return "gitignore"
    if path.name == ".env.example":
        return "dotenv"
    return LANGUAGES_BY_SUFFIX.get(path.suffix.lower(), "text")


def fence_for(content: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def render_file(root: Path, path: Path) -> list[str]:
    relative_path = path.relative_to(root).as_posix()
    try:
        contents = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return [
            f"## {relative_path}",
            "",
            "_Skipped: file is not valid UTF-8 text._",
            "",
        ]

    fence = fence_for(contents)
    language = language_for(path)
    return [
        f"## {relative_path}",
        "",
        f"{fence}{language}",
        contents.rstrip(),
        fence,
        "",
    ]


def markdown_for(root: Path) -> str:
    lines = [
        "# Codebase Snapshot",
        "",
        "## File Structure",
        "",
        "```text",
        *render_tree(root),
        "```",
        "",
        "## File Contents",
        "",
    ]

    for path in codebase_files(root):
        lines.extend(render_file(root, path))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = resolve_from_root(root, args.output).resolve()

    if not root.is_dir():
        raise SystemExit(f"Repository root does not exist: {root}")

    output.write_text(markdown_for(root), encoding="utf-8")
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    main()
