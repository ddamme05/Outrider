# Fixture: a syntax error *inside* a function body.
# Simulates "parse error inside a changed region" per spec §5.5.
# The broken function is broken; the untouched function is fine.


def good_function(x: int) -> int:
    return x + 1


def bad_function(y: int) -> int:
    # Dangling operator — tree-sitter should emit ERROR/MISSING inside
    # this scope but not inside good_function.
    return y +


def also_good():
    return 42
