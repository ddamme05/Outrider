"""Security battery for the JS/TS relative-specifier coordinates surfaces.

Covers `is_valid_relative_specifier`, `is_valid_trace_import_string`,
`relative_specifier_candidate_paths`, and `resolve_specifier_candidate_paths`
per `DECISIONS.md#024` (Amended 2026-07-03) and
`specs/2026-07-03-js-ts-trace-resolver.md`. Negative properties are scripted
as explicit attacks asserting the rejected outcome (raise or empty return),
never as absence-on-benign-input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outrider.coordinates import (
    is_valid_relative_specifier,
    is_valid_trace_import_string,
    relative_specifier_candidate_paths,
    resolve_specifier_candidate_paths,
    validate_diff_path,
)

INDEX_NAMES = ("index.js", "index.ts")


class TestIsValidRelativeSpecifier:
    """Shape-only validator: accepts the two relative forms, rejects the rest."""

    @pytest.mark.parametrize(
        "specifier",
        [
            "./db",
            "./middleware/validate",
            "../db",
            "../../shared/utils",
            "../../../deep/chain",
            ".",
            "..",
            "./with-dash",
            "./with.dot/seg.ment",
            "./__proto__",
        ],
    )
    def test_valid_forms_returned_unchanged(self, specifier: str) -> None:
        assert is_valid_relative_specifier(specifier) == specifier

    def test_nfc_composition_normalizes_decomposed_unicode(self) -> None:
        # 'e' + COMBINING ACUTE (U+0301) composes to U+00E9 under NFC.
        decomposed = "./café"
        composed = "./café"
        assert is_valid_relative_specifier(decomposed) == composed

    def test_zwj_and_zwnj_deliberately_admitted(self) -> None:
        # Parity with `is_valid_import_string`'s documented trade-off:
        # ZWNJ (U+200C) / ZWJ (U+200D) serve legitimate non-Latin scripts.
        assert is_valid_relative_specifier("./fo‌o") == "./fo‌o"
        assert is_valid_relative_specifier("./fo‍o") == "./fo‍o"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            is_valid_relative_specifier("")

    @pytest.mark.parametrize(
        "specifier",
        ["express", "lodash/merge", "@app/utils", "src/db"],
    )
    def test_bare_specifier_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="must begin with"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize("specifier", ["/etc/passwd", "/db", "//host/share"])
    def test_absolute_path_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="must begin with"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize("specifier", [".foo", "..foo", "...", "...x"])
    def test_dot_prefixed_non_relative_forms_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="must begin with"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize("specifier", [".\\db", "./a\\b", "..\\escape"])
    def test_backslash_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="backslash"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize(
        "metachar",
        [";", "&", "|", "`", "$", "(", ")", "<", ">", "*", "?", "~", "[", "]", "{", "}", "'", '"'],
    )
    def test_shell_metacharacters_rejected(self, metachar: str) -> None:
        with pytest.raises(ValueError, match="shell metacharacters"):
            is_valid_relative_specifier(f"./evil{metachar}x")

    @pytest.mark.parametrize("control", ["\n", "\r", "\x00"])
    def test_control_characters_rejected(self, control: str) -> None:
        with pytest.raises(ValueError, match="shell metacharacters"):
            is_valid_relative_specifier(f"./evil{control}x")

    @pytest.mark.parametrize(
        "trojan",
        ["‮", "​", "‎", "‏", "⁦", "⁩", "﻿"],
    )
    def test_trojan_source_characters_rejected(self, trojan: str) -> None:
        with pytest.raises(ValueError, match="bidi-override or invisible-format"):
            is_valid_relative_specifier(f"./evil{trojan}x")

    @pytest.mark.parametrize("specifier", [".//x", "./x//y", "./x/", "..//x", "../x/"])
    def test_empty_segment_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="empty path segment"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize("specifier", ["./a/./b", "../."])
    def test_interior_single_dot_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="interior '.' segment"):
            is_valid_relative_specifier(specifier)

    @pytest.mark.parametrize("specifier", ["./a/../b", "./..", "../a/../b", "./a/.."])
    def test_interior_double_dot_rejected(self, specifier: str) -> None:
        with pytest.raises(ValueError, match="interior '..' segment"):
            is_valid_relative_specifier(specifier)


class TestIsValidTraceImportString:
    """Leading-dot dispatcher: the two forms partition the value space."""

    def test_module_form_dispatches_to_import_string_rules(self) -> None:
        assert is_valid_trace_import_string("svc.db") == "svc.db"

    def test_specifier_form_dispatches_to_relative_rules(self) -> None:
        assert is_valid_trace_import_string("../db") == "../db"

    def test_leading_dot_python_relative_import_rejected(self) -> None:
        # `.foo` is a Python relative-import spelling, NOT a valid JS
        # specifier and NOT a valid dotted module string. The dispatcher
        # routes it to the specifier validator, which rejects the shape.
        with pytest.raises(ValueError, match="must begin with"):
            is_valid_trace_import_string(".foo")

    def test_slash_bearing_module_form_rejected(self) -> None:
        with pytest.raises(ValueError, match="path separators"):
            is_valid_trace_import_string("svc/db")

    def test_empty_rejected_via_module_form(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            is_valid_trace_import_string("")

    def test_no_value_admits_under_both_forms(self) -> None:
        # Partition property: a value the specifier form accepts must be
        # rejected by the module form (leading dot), and vice versa
        # (module form has no leading dot, so specifier form rejects it).
        from outrider.coordinates import is_valid_import_string

        with pytest.raises(ValueError):
            is_valid_import_string("./db")
        with pytest.raises(ValueError):
            is_valid_relative_specifier("svc.db")


class TestRelativeSpecifierCandidatePaths:
    """Construction surface: contained join + pragmatic-six fan-out."""

    def test_parent_specifier_fans_out_pragmatic_six_in_order(self) -> None:
        assert relative_specifier_candidate_paths("../db", "src/routes/user.js") == (
            "src/db.js",
            "src/db.jsx",
            "src/db.ts",
            "src/db.tsx",
            "src/db/index.js",
            "src/db/index.ts",
        )

    def test_same_dir_specifier(self) -> None:
        assert relative_specifier_candidate_paths("./db", "src/routes/user.js") == (
            "src/routes/db.js",
            "src/routes/db.jsx",
            "src/routes/db.ts",
            "src/routes/db.tsx",
            "src/routes/db/index.js",
            "src/routes/db/index.ts",
        )

    def test_nested_subdir_specifier(self) -> None:
        result = relative_specifier_candidate_paths("./middleware/validate", "src/app.js")
        assert result[0] == "src/middleware/validate.js"
        assert result[-1] == "src/middleware/validate/index.ts"
        assert len(result) == 6

    def test_root_level_importing_file(self) -> None:
        result = relative_specifier_candidate_paths("./db", "app.js")
        assert result == ("db.js", "db.jsx", "db.ts", "db.tsx", "db/index.js", "db/index.ts")

    def test_dot_from_root_file_yields_root_index_forms_only(self) -> None:
        assert relative_specifier_candidate_paths(".", "app.js") == INDEX_NAMES

    def test_parent_landing_at_root_yields_root_index_forms_only(self) -> None:
        assert relative_specifier_candidate_paths("..", "src/a.js") == INDEX_NAMES

    def test_dot_from_nested_file_targets_its_directory(self) -> None:
        # `.` from src/a.js is the src directory: uniform fan-out includes
        # the file forms (src.js is a probe miss, not a hazard) + index forms.
        result = relative_specifier_candidate_paths(".", "src/a.js")
        assert result == ("src.js", "src.jsx", "src.ts", "src.tsx", "src/index.js", "src/index.ts")

    def test_every_candidate_passes_validate_diff_path(self) -> None:
        for candidate in relative_specifier_candidate_paths("../db", "src/routes/user.js"):
            assert validate_diff_path(candidate) == candidate

    # --- attacks: every rejection asserts the explicit empty outcome ---

    @pytest.mark.parametrize(
        ("specifier", "importing"),
        [
            ("..", "app.js"),  # parent of repo root from a root-level file
            ("../..", "src/a.js"),  # second `..` escapes
            ("../../evil", "src/a.js"),
            ("../../../../../../etc/passwd", "src/routes/user.js"),
        ],
    )
    def test_repo_root_escape_returns_empty(self, specifier: str, importing: str) -> None:
        assert relative_specifier_candidate_paths(specifier, importing) == ()

    @pytest.mark.parametrize(
        "specifier",
        [
            "express",  # bare
            "/etc/passwd",  # absolute
            "./a/../b",  # interior `..`
            "./evil;x",  # shell metachar
            "./evil\x00x",  # NUL
            "./evil‮x",  # Trojan Source RLO
            ".\\db",  # backslash
            "",  # empty
        ],
    )
    def test_invalid_specifier_returns_empty(self, specifier: str) -> None:
        assert relative_specifier_candidate_paths(specifier, "src/a.js") == ()

    @pytest.mark.parametrize(
        "importing",
        [
            "../outside.js",  # traversal in importing path
            "/abs/file.js",  # absolute importing path
            "src/evil;x.js",  # metachars in importing path
            ".git/config",  # git-internal importing path
            "",  # empty importing path
        ],
    )
    def test_invalid_importing_path_returns_empty(self, importing: str) -> None:
        assert relative_specifier_candidate_paths("./db", importing) == ()


class TestResolveSpecifierCandidatePaths:
    """Filesystem twin: same construction, root-aware symlink-safe walk."""

    def test_returns_repo_relative_paths_without_requiring_existence(self, tmp_path: Path) -> None:
        result = resolve_specifier_candidate_paths("../db", "src/routes/user.js", tmp_path)
        assert result == [
            Path("src/db.js"),
            Path("src/db.jsx"),
            Path("src/db.ts"),
            Path("src/db.tsx"),
            Path("src/db/index.js"),
            Path("src/db/index.ts"),
        ]
        assert all(not p.is_absolute() for p in result)

    def test_symlinked_import_root_returns_empty(self, tmp_path: Path) -> None:
        real_root = tmp_path / "real"
        real_root.mkdir()
        linked_root = tmp_path / "linked"
        linked_root.symlink_to(real_root)
        assert resolve_specifier_candidate_paths("./db", "src/a.js", linked_root) == []

    def test_final_component_symlink_omits_that_candidate(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        outside = tmp_path.parent / "outside.js"
        outside.touch()
        (tmp_path / "src" / "db.js").symlink_to(outside)
        result = resolve_specifier_candidate_paths("./db", "src/a.js", tmp_path)
        assert Path("src/db.js") not in result
        assert Path("src/db.ts") in result

    def test_ancestor_directory_symlink_omits_descendant_candidates(self, tmp_path: Path) -> None:
        (tmp_path / "src").mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (tmp_path / "src" / "db").symlink_to(elsewhere)
        result = resolve_specifier_candidate_paths("./db", "src/a.js", tmp_path)
        assert Path("src/db/index.js") not in result
        assert Path("src/db/index.ts") not in result
        assert Path("src/db.js") in result  # sibling file form unaffected

    def test_escape_specifier_returns_empty(self, tmp_path: Path) -> None:
        assert resolve_specifier_candidate_paths("../../evil", "src/a.js", tmp_path) == []

    def test_invalid_specifier_returns_empty(self, tmp_path: Path) -> None:
        assert resolve_specifier_candidate_paths("express", "src/a.js", tmp_path) == []


class TestIsRelativeSpecifierForm:
    """The exported two-form discriminator — one partition rule."""

    def test_leading_dot_forms_are_specifier(self) -> None:
        from outrider.coordinates import is_relative_specifier_form

        for value in (".", "..", "./db", "../db", ".foo"):
            assert is_relative_specifier_form(value) is True

    def test_dotless_forms_are_module(self) -> None:
        from outrider.coordinates import is_relative_specifier_form

        for value in ("svc.db", "express", "src/db", ""):
            assert is_relative_specifier_form(value) is False


class TestValidateDiffPathLengthCap:
    """Attack: a constructed probe path compounding two near-cap fields
    (importing path + specifier) must reject at string validation — the
    schema/audit path fields cap at 1024, and an over-long path that
    probe-resolved would otherwise abort the trace pass at event
    construction instead of degrading to unresolved."""

    def test_path_at_cap_admits(self) -> None:
        path = "a/" * 511 + "bb"  # 1024 chars exactly
        assert len(path) == 1024
        assert validate_diff_path(path) == path

    def test_path_over_cap_rejected(self) -> None:
        import pytest as _pytest

        from outrider.coordinates import CoordinateError

        path = "a/" * 512 + "b"  # 1025 chars
        assert len(path) == 1025
        with _pytest.raises(CoordinateError, match="exceeds 1024"):
            validate_diff_path(path)

    def test_compounded_specifier_candidates_over_cap_drop(self) -> None:
        """Near-cap importing path + long specifier: every constructed
        candidate exceeds the cap and drops inside the construction
        surface — the attack yields zero probe paths, not an event-time
        ValidationError."""
        importing = "d/" * 505 + "x.js"  # 1014 chars, valid
        assert validate_diff_path(importing) == importing
        specifier = "./" + "s" * 300
        assert relative_specifier_candidate_paths(specifier, importing) == ()
