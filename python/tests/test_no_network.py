"""No test uses the network, and only the transport can reach it (TASK-005c2 E6, TASK-005c3 F1).

``tests/conftest.py`` replaces the socket entry points for every test; these tests prove that the
guard is in place and that no module under ``potakt`` imports a network library except
``potakt.evaluation.transport``.
"""

import ast
import socket
import urllib.request
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "potakt"
NETWORK_MODULES = frozenset({"socket", "ssl", "http", "urllib"})
TRANSPORT = "evaluation/transport.py"  # the only module allowed to import them (TASK-005c3 F1)
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


def network_importers(source: Path) -> dict[str, list[str]]:
    """The modules under ``source`` that import a network library, except the transport."""
    return {
        relative: imports
        for path in sorted(source.rglob("*.py"))
        if (relative := path.relative_to(source).as_posix()) != TRANSPORT
        and (imports := network_imports(path))
    }


def test_only_the_transport_imports_a_network_library() -> None:
    assert network_importers(SOURCE) == {}, (
        "no module under potakt may import socket, ssl, http or urllib except "
        "potakt.evaluation.transport (TASK-005c3 F1)"
    )


def test_the_transport_is_the_module_that_imports_them() -> None:
    assert network_imports(SOURCE / TRANSPORT) != []


@pytest.mark.parametrize(
    "relative", ["evaluation/runner.py", "cli.py", "evaluation/sub/transport.py", "transport.py"]
)
def test_any_other_module_importing_a_network_library_fails(tmp_path: Path, relative: str) -> None:
    (tmp_path / TRANSPORT).parent.mkdir(parents=True)
    (tmp_path / TRANSPORT).write_text("import urllib.request\n", encoding="utf-8")
    offender = tmp_path / relative
    offender.parent.mkdir(parents=True, exist_ok=True)
    offender.write_text("import http.client\n", encoding="utf-8")

    assert network_importers(tmp_path) == {relative: ["http.client"]}


def test_the_boundary_check_sees_every_import_form(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import os, socket\nimport http.client\nfrom urllib import request\n"
        "from ssl import create_default_context\nfrom . import http\nimport httpx\n",
        encoding="utf-8",
    )

    assert network_imports(sample) == ["socket", "http.client", "urllib", "ssl"]
