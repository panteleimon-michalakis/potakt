"""No test uses the network and no product module can reach it yet (TASK-005c2 E6).

``tests/conftest.py`` replaces the socket entry points for every test; these tests prove that the
guard is in place and that no module under ``potakt`` imports a network library.
"""

import ast
import socket
import urllib.request
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "potakt"
NETWORK_MODULES = frozenset({"socket", "ssl", "http", "urllib"})
GUARD_MESSAGE = "tests never use the network"


def test_urlopen_is_stopped_by_the_guard() -> None:
    with pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        urllib.request.urlopen("https://example.com")


def test_a_loopback_connection_is_stopped_too() -> None:
    with socket.socket() as connection, pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        connection.connect(("127.0.0.1", 9))


@pytest.mark.parametrize(
    "call",
    [
        lambda: socket.create_connection(("127.0.0.1", 9)),
        lambda: socket.getaddrinfo("localhost", 9),
        lambda: socket.socket().connect_ex(("127.0.0.1", 9)),
    ],
    ids=["create_connection", "getaddrinfo", "connect_ex"],
)
def test_every_socket_entry_point_is_guarded(call: object) -> None:
    assert callable(call)
    with pytest.raises(RuntimeError, match=GUARD_MESSAGE):
        call()


def network_imports(path: Path) -> list[str]:
    """The network modules a source file imports (``import x.y`` and ``from x.y import z``)."""
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        else:
            continue
        found += [name for name in names if name.split(".")[0] in NETWORK_MODULES]
    return found


def test_no_product_module_imports_a_network_library() -> None:
    offenders = {
        path.relative_to(SOURCE).as_posix(): imports
        for path in sorted(SOURCE.rglob("*.py"))
        if (imports := network_imports(path))
    }

    assert offenders == {}, (
        "no module under potakt may import socket, ssl, http or urllib; TASK-005c3 will allow "
        "exactly one module, potakt.evaluation.transport"
    )


def test_the_boundary_check_sees_every_import_form(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os, socket\nimport http.client\nfrom urllib import request\n"
        "from ssl import create_default_context\nfrom . import http\nimport httpx\n",
        encoding="utf-8",
    )

    assert network_imports(sample) == ["socket", "http.client", "urllib", "ssl"]
