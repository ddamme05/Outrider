"""Structural eval scenario: Flask + FastAPI route decorators extracted.

Per spec §11.2: `@app.route` (Flask) and `@router.get` (FastAPI) decorators
are extracted on the `ScopeUnit` for the decorated function.

V1: scaffolded; assertion runs at `ast_facts/` flip time.
"""

from unittest.mock import MagicMock

from outrider.ast_facts import parse_python

FLASK_SOURCE = """\
from flask import Flask
app = Flask(__name__)

@app.route("/users/<int:user_id>")
def get_user(user_id):
    return {"id": user_id}
"""

FASTAPI_SOURCE = """\
from fastapi import APIRouter
router = APIRouter()

@router.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"id": item_id}
"""

EXPECTED_FLASK_DECORATORS = ('app.route("/users/<int:user_id>")',)
EXPECTED_FASTAPI_DECORATORS = ('router.get("/items/{item_id}")',)


def test_flask_route_decorator_extracted() -> None:
    """Flask @app.route decorator captured on the get_user ScopeUnit."""
    result = parse_python(FLASK_SOURCE.encode(), "test.py", MagicMock())
    get_user = next(s for s in result.scope_units if s.name == "get_user")
    assert tuple(get_user.decorators) == EXPECTED_FLASK_DECORATORS


def test_fastapi_route_decorator_extracted() -> None:
    """FastAPI @router.get decorator captured on the read_item ScopeUnit."""
    result = parse_python(FASTAPI_SOURCE.encode(), "test.py", MagicMock())
    read_item = next(s for s in result.scope_units if s.name == "read_item")
    assert tuple(read_item.decorators) == EXPECTED_FASTAPI_DECORATORS
