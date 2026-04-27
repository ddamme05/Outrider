# Fixture: nested scopes and decorator stacks.
# Exercises qualified-name derivation and decorated_definition parent shape.

import functools


def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def retry(times: int):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    continue
            raise RuntimeError("exhausted")

        return wrapper

    return decorator


class Outer:
    class Inner:
        @staticmethod
        @log
        def greet(name: str) -> str:
            def _clean(s: str) -> str:
                return s.strip()

            return _clean(name)

        # a class-body comment, not inside any method
        @retry(times=3)
        def fetch(self, url: str) -> bytes:
            raise NotImplementedError


def top_level():
    def nested_one():
        def nested_two():
            return 42

        return nested_two()

    return nested_one()
