# Fixture: Python 3.13 features that might stress the grammar.
# Must parse with zero ERROR/MISSING nodes under tree-sitter-python 0.25.0.

from typing import Any


# PEP 695: generic type alias.
type Vector[T] = list[T]


# PEP 695: generic function.
def first[T](items: list[T]) -> T | None:
    return items[0] if items else None


# PEP 695: generic class with a bound.
class Box[T: (int, str)]:
    def __init__(self, value: T) -> None:
        self.value = value

    def get(self) -> T:
        return self.value


# Structural pattern matching (PEP 634).
def describe(point: tuple[int, int] | tuple[int, int, int]) -> str:
    match point:
        case (0, 0):
            return "origin"
        case (x, 0):
            return f"x-axis at {x}"
        case (0, y):
            return f"y-axis at {y}"
        case (x, y):
            return f"({x}, {y})"
        case (x, y, z):
            return f"({x}, {y}, {z})"
        case _:
            return "unknown"


# Walrus + async + decorated method.
class Repo:
    @staticmethod
    async def latest(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        if (n := len(entries)) == 0:
            return None
        return entries[n - 1]


# f-string with nested quotes and expression (3.12+ relaxed rules).
def fmt(name: str) -> str:
    return f"Hello, {name.upper() if name else 'world'}!"
