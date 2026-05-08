"""PRContext + ChangedFile: construction-time validation, round-trip, frozen guards.

Per spec §7.2 + docs/conventions.md "Pydantic models use ConfigDict(extra='forbid')".
Both models are frozen=True (round-trip through LangGraph state JSON; immutability
prevents mid-graph mutation by any node).
"""

import pytest
from pydantic import ValidationError

from outrider.schemas import ChangedFile, PRContext


def _minimal_changed_file(**overrides: object) -> ChangedFile:
    """Default fixture: status='modified' with both content sides populated.

    Per Round 14 / DECISIONS.md#020 status↔content invariants, ChangedFile
    instances are constructed by intake AFTER fetching base/head content,
    so the default fixture must satisfy the post-intake contract. Tests
    that need a different status pass overrides for status + content fields.
    """
    base = dict(
        path="src/foo.py",
        status="modified",
        additions=3,
        deletions=1,
        patch="@@ -1,3 +1,5 @@\n a\n b\n c\n+d\n+e",
        content_base="old\n",
        content_head="new\n",
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
    """Per Round 14, the default fixture has status='modified' with both
    content_base and content_head populated (post-intake contract per
    DECISIONS.md#020)."""
    cf = _minimal_changed_file()
    assert cf.path == "src/foo.py"
    assert cf.status == "modified"
    assert cf.content_base == "old\n"
    assert cf.content_head == "new\n"
    assert cf.previous_path is None
    assert cf.language is None


def test_changed_file_language_field_accepts_string() -> None:
    cf = _minimal_changed_file(language="python")
    assert cf.language == "python"


def test_changed_file_status_rejects_invalid_literal() -> None:
    with pytest.raises(ValidationError):
        _minimal_changed_file(status="created")  # not in the four canonical values


def test_changed_file_status_accepts_each_of_four_canonical_values() -> None:
    """Per Round 14 + Round 15 + Round 16 invariants, each status needs aligned
    content fields AND aligned counts AND (for `renamed`) aligned previous_path
    distinct from path. This test pins that each canonical status admits when
    the corresponding shape is provided."""
    cf_added = _minimal_changed_file(
        path="new.py",
        status="added",
        deletions=0,  # added: no deletions per Round-15 count invariant
        content_base=None,
        content_head="new content",
    )
    assert cf_added.status == "added"

    cf_modified = _minimal_changed_file(
        status="modified"
    )  # default has both content + nonzero counts
    assert cf_modified.status == "modified"

    cf_removed = _minimal_changed_file(
        path="old.py",
        status="removed",
        additions=0,  # removed: no additions per Round-15 count invariant
        content_base="old content",
        content_head=None,
    )
    assert cf_removed.status == "removed"

    cf_renamed = _minimal_changed_file(
        path="new_path.py", status="renamed", previous_path="old_path.py"
    )
    assert cf_renamed.status == "renamed"
    assert cf_renamed.previous_path == "old_path.py"


def test_changed_file_added_requires_content_head_and_no_content_base() -> None:
    """status='added' must have content_head set and content_base None."""
    with pytest.raises(ValidationError, match="status='added' requires content_head"):
        _minimal_changed_file(status="added", deletions=0, content_base=None, content_head=None)
    with pytest.raises(ValidationError, match="status='added' requires content_base to be None"):
        _minimal_changed_file(
            status="added", deletions=0, content_base="should not be set", content_head="new"
        )


def test_changed_file_added_rejects_nonzero_deletions() -> None:
    """Round-15 count invariant: status='added' requires deletions=0
    (an added file has no pre-existing content to delete from)."""
    with pytest.raises(ValidationError, match="status='added' requires deletions=0"):
        _minimal_changed_file(status="added", deletions=1, content_base=None, content_head="new")


def test_changed_file_removed_requires_content_base_and_no_content_head() -> None:
    """status='removed' must have content_base set and content_head None."""
    with pytest.raises(ValidationError, match="status='removed' requires content_base"):
        _minimal_changed_file(status="removed", additions=0, content_base=None, content_head=None)
    with pytest.raises(ValidationError, match="status='removed' requires content_head to be None"):
        _minimal_changed_file(
            status="removed",
            additions=0,
            content_base="old",
            content_head="should not be set",
        )


def test_changed_file_removed_rejects_nonzero_additions() -> None:
    """Round-15 count invariant: status='removed' requires additions=0
    (a removed file has nothing being added)."""
    with pytest.raises(ValidationError, match="status='removed' requires additions=0"):
        _minimal_changed_file(status="removed", additions=3, content_base="old", content_head=None)


def test_changed_file_modified_requires_both_content_sides() -> None:
    """status='modified' must have BOTH content_base and content_head set."""
    with pytest.raises(ValidationError, match="status='modified' requires both"):
        _minimal_changed_file(status="modified", content_base=None, content_head="new")
    with pytest.raises(ValidationError, match="status='modified' requires both"):
        _minimal_changed_file(status="modified", content_base="old", content_head=None)
    with pytest.raises(ValidationError, match="status='modified' requires both"):
        _minimal_changed_file(status="modified", content_base=None, content_head=None)


def test_changed_file_renamed_requires_both_content_sides_and_previous_path() -> None:
    """status='renamed' must have both content sides set AND previous_path set."""
    # Missing previous_path
    with pytest.raises(ValidationError, match="status='renamed' requires previous_path"):
        _minimal_changed_file(status="renamed", previous_path=None)
    # Missing content_base
    with pytest.raises(ValidationError, match="status='renamed' requires both"):
        _minimal_changed_file(
            status="renamed",
            content_base=None,
            content_head="new",
            previous_path="old.py",
        )


def test_changed_file_non_renamed_must_not_carry_previous_path() -> None:
    """previous_path is renamed-status-specific; other statuses must not carry it."""
    with pytest.raises(ValidationError, match="must not carry previous_path"):
        _minimal_changed_file(status="modified", previous_path="something.py")
    with pytest.raises(ValidationError, match="must not carry previous_path"):
        _minimal_changed_file(
            status="added",
            deletions=0,  # Round-15 count invariant
            content_base=None,
            content_head="new",
            previous_path="should-not-be-here.py",
        )


def test_changed_file_renamed_with_full_shape_admits() -> None:
    """Happy path: renamed file with both content sides + previous_path set."""
    cf = ChangedFile(
        path="new_name.py",
        status="renamed",
        additions=2,
        deletions=2,
        patch="@@ -1 +1 @@\n-old\n+new\n",
        content_base="old\n",
        content_head="new\n",
        previous_path="old_name.py",
    )
    assert cf.previous_path == "old_name.py"
    assert cf.path == "new_name.py"


def test_changed_file_renamed_rejects_same_old_and_new_path() -> None:
    """Round-16 rename path-shape invariant: status='renamed' requires
    previous_path != path. A same-path 'rename' is GitHub-impossible —
    that would be status='modified' or no change. Admitting it would
    let intake silently fan out base+head fetches against the same
    path for a non-rename, hiding upstream-buggy data."""
    with pytest.raises(ValidationError, match="status='renamed' requires previous_path != path"):
        _minimal_changed_file(
            path="same.py",
            status="renamed",
            previous_path="same.py",
        )


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


def test_pr_context_installation_id_admits_synthetic_eval_values() -> None:
    """Round 7 reversal: installation_id is plain int (no Field(ge=1)) per
    the eval-isolation convention. Eval factories use synthetic non-colliding
    IDs (including negatives like -1 to signal 'not a real installation').
    Production webhook validation enforces real GitHub IDs at the input
    boundary; this shared schema supports both contexts. This test pins the
    Round 7 reversal so a future re-tightening can't ship without the
    accompanying eval-factory migration."""
    for synthetic in (-1, 0, -999_999):
        ctx = _minimal_pr_context(installation_id=synthetic)
        assert ctx.installation_id == synthetic


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
            _minimal_changed_file(
                path="src/b.py",
                status="added",
                deletions=0,
                content_base=None,
                content_head="new content",
            ),
        ]
    )
    rehydrated = PRContext.model_validate_json(ctx.model_dump_json())
    assert len(rehydrated.changed_files) == 2
    assert all(isinstance(f, ChangedFile) for f in rehydrated.changed_files)
    assert rehydrated.changed_files[0].path == "src/a.py"
    assert rehydrated.changed_files[1].status == "added"


def test_pr_context_empty_changed_files_admits() -> None:
    """changed_files=() is the NORMAL webhook seed shape per DECISIONS.md#020:
    GitHub pull_request webhook payloads do not include the per-file list, so
    every webhook seed has changed_files=() until intake fetches the file list.
    Real PRs reach intake with changed_files=() AND nonzero totals (the
    payload's pull_request.additions / pull_request.deletions). This test
    pins schema-level admittance of the seed shape; the size-cap policy gate
    (separate spec) is what decides whether a review is skipped, NOT the
    schema."""
    ctx = _minimal_pr_context(changed_files=[], total_additions=0, total_deletions=0)
    assert ctx.changed_files == ()


def test_pr_context_seed_shape_with_nonzero_totals_admits() -> None:
    """The realistic webhook-seed scenario per DECISIONS.md#020: changed_files
    is empty (intake hasn't fetched yet) but total_additions / total_deletions
    are nonzero (read directly from the webhook payload's pull_request.additions
    / pull_request.deletions). The schema must admit this shape — it's what
    every real PR looks like at graph start."""
    ctx = _minimal_pr_context(
        changed_files=[],
        total_additions=152,
        total_deletions=37,
    )
    assert ctx.changed_files == ()
    assert ctx.total_additions == 152
    assert ctx.total_deletions == 37


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
