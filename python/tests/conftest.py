"""Tests never depend on colour, terminal-width or CI environment variables.

Typer's ``rich_utils`` reads several of these when it is imported (``GITHUB_ACTIONS``,
``FORCE_COLOR`` and ``PY_COLORS`` force Rich terminal output with ANSI codes), so they are removed
here, before any test module imports Typer. Subprocess tests copy ``os.environ`` and inherit the
clean environment. ``CI`` stays: Hypothesis uses it to load its ``ci`` profile.

The experiment API keys are removed too, so no test and no ``potakt`` subprocess started by a test
can ever make a paid model call with the owner's credentials. Their values are never read.

Every test also runs with the socket entry points replaced (``no_network``, TASK-005c2 E6).
"""

import os
import socket
import sys
from collections.abc import Iterator

import pytest

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
    # Experiment credentials (CLAUDE.md golden rule 3): removed, never read.
    "POTAKT_ANTHROPIC_API_KEY",
    "POTAKT_GEMINI_API_KEY",
)

assert "typer.rich_utils" not in sys.modules, (
    "typer.rich_utils was imported before tests/conftest.py could reset the environment"
)
for _name in _ENVIRONMENT_VARIABLES:
    os.environ.pop(_name, None)


def _no_network(*args: object, **kwargs: object) -> object:
    raise RuntimeError("tests never use the network")


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """TASK-005c2 E6: no test may open a connection or resolve a name, not even on loopback.

    The socket entry points raise instead, so a test that would reach a model API (or any
    other host) fails at once. ``potakt`` subprocesses started by tests are not covered; they
    run without credentials (above) and no product module imports a network library
    (``test_no_network.py``).
    """
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    yield
