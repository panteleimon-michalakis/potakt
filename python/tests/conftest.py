"""Tests never depend on colour, terminal-width or CI environment variables.

Typer's ``rich_utils`` reads several of these when it is imported (``GITHUB_ACTIONS``,
``FORCE_COLOR`` and ``PY_COLORS`` force Rich terminal output with ANSI codes), so they are removed
here, before any test module imports Typer. Subprocess tests copy ``os.environ`` and inherit the
clean environment. ``CI`` stays: Hypothesis uses it to load its ``ci`` profile.
"""

import os
import sys

_ENVIRONMENT_VARIABLES = (
    "GITHUB_ACTIONS",
    "FORCE_COLOR",
    "PY_COLORS",
    "CLICOLOR_FORCE",
    "NO_COLOR",
    "COLUMNS",
    "TERMINAL_WIDTH",
    "TYPER_USE_RICH",
    "_TYPER_FORCE_DISABLE_TERMINAL",
)

assert "typer.rich_utils" not in sys.modules, (
    "typer.rich_utils was imported before tests/conftest.py could reset the environment"
)
for _name in _ENVIRONMENT_VARIABLES:
    os.environ.pop(_name, None)
