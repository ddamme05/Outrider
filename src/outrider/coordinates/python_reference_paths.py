"""Build bounded repository candidates for explicit Python relative imports."""

from pathlib import PurePosixPath

from outrider.coordinates.diff_parser import (
    is_valid_python_relative_import_string,
    validate_diff_path,
)
from outrider.coordinates.errors import CoordinateError, CoordinateErrorKind


def python_relative_import_candidate_paths(
    *, importer_path: str, module: str, level: int
) -> tuple[str, str]:
    """Return module/package candidates without probing or inferring import roots.

    These are repository read candidates, not proof of Python runtime resolution.
    The caller must check both and reject ambiguous module/package matches.
    """
    safe = validate_diff_path(importer_path)
    importer = PurePosixPath(safe)
    parts = module.split(".")
    if (
        safe != importer_path
        or importer.suffix != ".py"
        or type(level) is not int
        or level < 1
        or not all(part.isidentifier() for part in parts)
    ):
        raise CoordinateError(
            "reference requires a canonical Python relative module import",
            kind=CoordinateErrorKind.ARGUMENT_VALIDATION_FAILED,
        )
    package = importer.parent.parts
    if level > len(package):
        raise CoordinateError(
            "relative module reference leaves the repository package path",
            kind=CoordinateErrorKind.PATH_VALIDATION_FAILED,
        )
    stem = PurePosixPath(*package[: len(package) - level + 1], *parts)
    return (
        validate_diff_path(f"{stem}.py"),
        validate_diff_path(f"{stem}/__init__.py"),
    )


def python_relative_trace_candidate_paths(
    import_string: str, importer_path: str, strip_level: int
) -> tuple[str, ...]:
    """Anchor a relative module or symbol prefix to its Python importing file.

    Invalid or escaping references have no candidates. The trace consumer owns
    suffix-strip limits, existence probes, symbol verification and ambiguity.
    """
    if type(strip_level) is not int or strip_level < 0:
        return ()
    try:
        canonical = is_valid_python_relative_import_string(import_string)
        module = canonical.lstrip(".")
        level = len(canonical) - len(module)
        parts = module.split(".")
        if strip_level >= len(parts):
            return ()
        prefix = ".".join(parts[: len(parts) - strip_level])
        return python_relative_import_candidate_paths(
            importer_path=importer_path, module=prefix, level=level
        )
    except (ValueError, CoordinateError):
        return ()
