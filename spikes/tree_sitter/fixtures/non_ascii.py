# Fixture: non-ASCII identifiers, emoji docstrings, multi-byte UTF-8.
# Exercises byte/point duality — start_byte/end_byte must stay consistent with
# text slicing via source.encode("utf-8") per python-source-analysis.md.

# Greek letter identifier.
def α(x: int) -> int:
    """Return x squared. 🧮"""
    return x * x


# Cyrillic identifier inside a class.
class Привет:
    """Клас приветствий. 👋"""

    def say(self, name: str) -> str:
        # Non-ASCII in string literal and comment: «bonjour — здравствуйте».
        return f"Привет, {name}! 🌍"


# Emoji in a docstring body — each emoji is 4 UTF-8 bytes.
def rocket() -> str:
    """🚀🚀🚀 launch."""
    return "🚀"
