"""Captured dependency hints retain guards and report missing evidence."""

from outrider.ast_facts.candidate_context import candidate_dependency_context
from outrider.ast_facts.review_context import build_called_function_index
from outrider.coordinates import COORDINATES_IMPORT_PATH_RESOLVER


def _select(sources, *, cap=2000, claim_range=(2, 2)):
    index = build_called_function_index(
        sources, import_path_resolver=COORDINATES_IMPORT_PATH_RESOLVER
    )
    return candidate_dependency_context(
        index,
        file_path="pkg/view.py",
        line_start=claim_range[0],
        line_end=claim_range[1],
        max_source_bytes=cap,
    )


def test_attribute_read_finds_guarded_assignment_and_possible_constructor() -> None:
    sources = {
        "pkg/view.py": "def view():\n    return service.gateway.provider_names\n",
        "pkg/core.py": (
            "from .gateway import Gateway\n"
            "def configure(app):\n"
            "    if app.enabled:\n"
            "        app.gateway = Gateway(app)\n"
        ),
        "pkg/gateway.py": (
            "class Gateway:\n"
            "    def __init__(self, app):\n"
            "        if app is None:\n"
            "            raise ValueError('app required')\n"
            "        self.provider_names = []\n"
        ),
    }
    selected = _select(sources)
    by_path = {item.file_path: item.excerpt for item in selected.references}
    assert by_path["pkg/core.py"].text == (
        "    if app.enabled:\n        app.gateway = Gateway(app)"
    )
    assert by_path["pkg/core.py"].line_start == 3
    assert by_path["pkg/gateway.py"].text.startswith("    def __init__(self, app):\n")
    assert "raise ValueError" in by_path["pkg/gateway.py"].text
    assert all(
        "receiver unproven" in item.excerpt.name or "possible constructor" in item.excerpt.name
        for item in selected.references
    )


def test_oversized_control_is_omitted_instead_of_hiding_the_guard() -> None:
    selected = _select(
        {
            "pkg/view.py": "def view():\n    return service.gateway.ready\n",
            "pkg/core.py": (
                "def configure(app):\n"
                "    if app.enabled:\n"
                "        app.gateway = Gateway(app)\n"
                f"        note = {'é' * 100!r}\n"
            ),
        },
        cap=200,
    )
    assert not selected.references
    assert any("assignment context exceeds cap" in reason for reason in selected.omissions)


def test_ambiguous_constructor_is_reported_without_selecting_an_arbitrary_class() -> None:
    selected = _select(
        {
            "pkg/view.py": "def view():\n    return service.gateway.ready\n",
            "pkg/core.py": "def configure(app):\n    app.gateway = Gateway()\n",
            "pkg/a.py": "class Gateway:\n    def __init__(self):\n        self.ready = True\n",
            "pkg/b.py": "class Gateway:\n    def __init__(self):\n        self.ready = False\n",
        }
    )
    assert all("possible constructor" not in item.excerpt.name for item in selected.references)
    assert "constructor Gateway: ambiguous captured name" in selected.omissions


def test_missing_attribute_constraints_are_explicit_and_do_not_invent_a_reference() -> None:
    selected = _select({"pkg/view.py": "def view():\n    return user.email == current.email\n"})
    assert selected.references == ()
    assert selected.omissions == ("attribute email: no captured assignment found",)


def test_reference_limit_keeps_selection_bounded_and_reports_omissions() -> None:
    sources = {
        "pkg/view.py": "def view():\n    return service.gateway.ready\n",
        **{
            f"pkg/config{number}.py": f"def configure(app):\n    app.gateway = value{number}\n"
            for number in range(6)
        },
    }
    selected = _select(sources)
    assert len(selected.references) == 4
    assert "further attribute references omitted (reference limit)" in selected.omissions


def test_ineligible_defaults_do_not_invent_an_implementation() -> None:
    source = (
        "from unavailable import imported\n"
        "def helper(value):\n"
        "    return value\n"
        "@decorate\n"
        "def decorated(value):\n"
        "    return value\n"
        "def run(value, external=imported, computed=factory(), replaced=helper, "
        "wrapped=decorated, unused=helper):\n"
        "    replaced = replacement\n"
        "    external(value)\n"
        "    computed(value)\n"
        "    replaced(value)\n"
        "    return wrapped(value)\n"
    )
    selected = _select({"pkg/view.py": source}, claim_range=(9, 12))
    assert selected.references == ()
    assert selected.omissions == (
        "default callable external: no unambiguous same-file definition",
        "default callable computed: no unambiguous same-file definition",
        "default callable replaced: parameter is rebound",
        "default callable wrapped: decorated implementation omitted",
    )


def test_oversized_default_implementation_is_omitted_whole() -> None:
    source = (
        "def helper(value):\n"
        f"    note = {'é' * 100!r}\n"
        "    return value\n"
        "def run(value, callback=helper):\n"
        "    return callback(value)\n"
    )
    selected = _select({"pkg/view.py": source}, cap=80, claim_range=(5, 5))
    assert selected.references == ()
    assert selected.omissions == ("default callable callback: complete implementation exceeds cap",)
