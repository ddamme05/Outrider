"""PRContext + ChangedFile: construction-time validation, round-trip, frozen guards.

Per spec §7.2 + docs/conventions.md "Pydantic models use ConfigDict(extra='forbid')".
Both models are frozen=True (round-trip through LangGraph state JSON; immutability
prevents mid-graph mutation by any node).
"""

import pytest
from pydantic import ValidationError

from outrider.schemas import ChangedFile, PRContext


def _minimal_changed_file(**overrides: object) -> ChangedFile:
    base = dict(
        path="src/foo.py",
        status="modified",
        additions=3,
        deletions=1,
        patch="@@ -1,3 +1,5 @@\n a\n b\n c\n+d\n+e",
    )
    base.update(overrides)
    return ChangedFile(**base)  # type: ignore[arg-type]


def _minimal_pr_context(**overrides: object) -> PRContext:
    base = dict(
        installation_id=12345,
        owner="acme",
        repo="widget",
        pr_number=42,
        pr_title="Add the thing",
        pr_body="Adds a thing that does the thing.",
        base_sha="a" * 40,
        head_sha="b" * 40,
        author="alice",
        changed_files=[_minimal_changed_file()],
        total_additions=3,
        total_deletions=1,
    )
    base.update(overrides)
    return PRContext(**base)  # type: ignore[arg-type]


# ChangedFile -----------------------------------------------------------------


def test_changed_file_minimal_construction_succeeds() -> None:
    cf = _minimal_changed_file()
    assert cf.path == "src/foo.py"
    assert cf.status == "modified"
    assert cf.content_base is None
    assert cf.content_head is None
    assert cf.language is None


def test_changed_file_optional_fields_accept_strings() -> None:
    cf = _minimal_changed_file(
        content_base="old content",
        content_head="new content",
        language="python",
    )
    assert cf.content_base == "old content"
    assert cf.content_head == "new content"
    assert cf.language == "python"


def test_changed_file_status_rejects_invalid_literal() -> None:
    with pytest.raises(ValidationError):
        _minimal_changed_file(status="created")  # not in the four canonical values


def test_changed_file_status_accepts_each_of_four_canonical_values() -> None:
    for status in ("added", "modified", "removed", "renamed"):
        cf = _minimal_changed_file(status=status)
        assert cf.status == status


def test_changed_file_additions_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_changed_file(additions=-1)


def test_changed_file_deletions_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_changed_file(deletions=-1)


def test_changed_file_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ChangedFile(  # type: ignore[call-arg]
            path="src/foo.py",
            status="modified",
            additions=0,
            deletions=0,
            patch="",
            unknown_field="oops",
        )


def test_changed_file_is_frozen() -> None:
    cf = _minimal_changed_file()
    with pytest.raises(ValidationError):
        cf.path = "src/bar.py"  # type: ignore[misc]


def test_changed_file_round_trip_json() -> None:
    cf = _minimal_changed_file(content_base="old", content_head="new", language="python")
    rehydrated = ChangedFile.model_validate_json(cf.model_dump_json())
    assert rehydrated == cf


# PRContext -------------------------------------------------------------------


def test_pr_context_minimal_construction_succeeds() -> None:
    ctx = _minimal_pr_context()
    assert ctx.owner == "acme"
    assert ctx.pr_number == 42
    assert ctx.installation_id == 12345
    assert len(ctx.changed_files) == 1


def test_pr_context_installation_id_rejects_zero() -> None:
    """GitHub App installation IDs are positive integers (Field(ge=1))."""
    with pytest.raises(ValidationError):
        _minimal_pr_context(installation_id=0)


def test_pr_context_installation_id_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_pr_context(installation_id=-1)


def test_pr_context_installation_id_required() -> None:
    """installation_id has no default; omitting it raises ValidationError. The
    canonical-shape gap (spec §15.2 used state.pr_context.installation_id but
    canonical §7.2 didn't define it) was closed 2026-05-08; this test pins the
    required-no-default behavior so the field can't silently regress to optional."""
    with pytest.raises(ValidationError):
        PRContext(  # type: ignore[call-arg]
            owner="acme",
            repo="widget",
            pr_number=1,
            pr_title="t",
            pr_body="b",
            base_sha="a" * 40,
            head_sha="b" * 40,
            author="alice",
            changed_files=[],
            total_additions=0,
            total_deletions=0,
            # installation_id intentionally omitted
        )


def test_pr_context_pr_number_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        _minimal_pr_context(pr_number=0)


def test_pr_context_pr_number_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_pr_context(pr_number=-1)


def test_pr_context_total_additions_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_pr_context(total_additions=-1)


def test_pr_context_total_deletions_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        _minimal_pr_context(total_deletions=-1)


def test_pr_context_extra_forbid() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PRContext(  # type: ignore[call-arg]
            installation_id=12345,
            owner="acme",
            repo="widget",
            pr_number=1,
            pr_title="t",
            pr_body="b",
            base_sha="a" * 40,
            head_sha="b" * 40,
            author="alice",
            changed_files=[],
            total_additions=0,
            total_deletions=0,
            unknown_field="oops",
        )


def test_pr_context_is_frozen() -> None:
    ctx = _minimal_pr_context()
    with pytest.raises(ValidationError):
        ctx.pr_title = "different"  # type: ignore[misc]


def test_pr_context_round_trip_json() -> None:
    ctx = _minimal_pr_context()
    rehydrated = PRContext.model_validate_json(ctx.model_dump_json())
    assert rehydrated == ctx


def test_pr_context_round_trip_preserves_changed_files() -> None:
    """LangGraph checkpoint round-trips through Postgres JSON; nested ChangedFile
    objects must rehydrate as ChangedFile instances, not dicts."""
    ctx = _minimal_pr_context(
        changed_files=[
            _minimal_changed_file(path="src/a.py", status="modified"),
            _minimal_changed_file(path="src/b.py", status="added", deletions=0),
        ]
    )
    rehydrated = PRContext.model_validate_json(ctx.model_dump_json())
    assert len(rehydrated.changed_files) == 2
    assert all(isinstance(f, ChangedFile) for f in rehydrated.changed_files)
    assert rehydrated.changed_files[0].path == "src/a.py"
    assert rehydrated.changed_files[1].status == "added"


def test_pr_context_empty_changed_files_admits() -> None:
    """Empty PR (no file changes) is valid at the schema level; size-cap policy
    (separate spec) decides whether to skip the review entirely."""
    ctx = _minimal_pr_context(changed_files=[], total_additions=0, total_deletions=0)
    assert ctx.changed_files == ()


def test_pr_context_changed_files_is_tuple_not_list() -> None:
    """frozen=True is faux-immutable over .append() on a list field; spec §7.2
    was amended 2026-05-08 to use tuple[ChangedFile, ...] for true immutability.
    Same precedent as HITLDecision.decisions (spec §7.4 line 290)."""
    ctx = _minimal_pr_context()
    assert isinstance(ctx.changed_files, tuple)


def test_pr_context_changed_files_rejects_in_place_append() -> None:
    """Tuple has no .append(); a node attempting to mutate the changed-files list
    on a state-carried PRContext now raises AttributeError instead of silently
    succeeding (the protection frozen=True alone fails to deliver)."""
    ctx = _minimal_pr_context()
    with pytest.raises(AttributeError):
        ctx.changed_files.append(_minimal_changed_file(path="src/sneaky.py"))  # type: ignore[attr-defined]


def test_pr_context_dict_round_trip() -> None:
    """LangGraph reducer merges receive partial-update dicts; model_dump() →
    model_validate() must preserve all nested structure exactly. Distinct from
    the JSON round-trip — reducers don't serialize through JSON."""
    ctx = _minimal_pr_context()
    rehydrated = PRContext.model_validate(ctx.model_dump())
    assert rehydrated == ctx
    assert isinstance(rehydrated.changed_files, tuple)
    assert all(isinstance(f, ChangedFile) for f in rehydrated.changed_files)
