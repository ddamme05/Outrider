"""Research tools kept separate from the application.

The package name lets tests and type checkers import individual experiments
consistently. Production code does not use these modules, and the src/ mypy
check does not cover them.
"""
