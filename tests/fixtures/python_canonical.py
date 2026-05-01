"""Canonical Python source for ast_facts/ tests.

This file is a TEST DATA FIXTURE, not a Python module to be imported.
Tests read its bytes via `Path(...).read_bytes()` and feed it through
`parse_python(...)` or the per-method PythonAdapter API.

Covers the constructs the unit and integration tests assert against:
top-level functions; class methods; decorated functions (call-form
produces a CallSite, bare-name does not); nested functions
(qualified_name `outer.inner`, no `<locals>`); async functions and
async methods; multiline signatures; the four import shapes; inline
calls inside scopes.
"""

import os
from pathlib import Path
from .helpers import format_name
from collections import *


def hello(name: str) -> str:
    """Top-level function."""
    greeting = format_name(name)
    return greeting


def outer():
    """Top-level function with a nested function inside."""

    def inner():
        return 42

    return inner()


class Greeter:
    """Top-level class with a method and an async method."""

    def greet(self, name: str) -> str:
        return hello(name)

    async def greet_async(self, name: str) -> str:
        return hello(name)


@property
def bare_decorator_func():
    """Bare-name decorator should NOT produce a CallSite."""
    return 1


@route("/api/foo")
def call_form_decorator_func():
    """Call-form decorator SHOULD produce a CallSite for `route`."""
    return 2


def multiline_signature(
    first: str,
    second: int,
    third: bool = True,
) -> tuple[str, int, bool]:
    """Multiline signature spans multiple lines."""
    return (first, second, third)


async def async_top_level(arg):
    """Async top-level function."""
    return await arg


class Outer:
    class Inner:
        def method(self):
            return self
