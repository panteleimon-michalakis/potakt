"""No test module reads a file when it is imported (TASK-005e4 decision 7).

pytest-xdist imports every test module in every worker, so a read at import runs once per
worker, and a missing file stops the whole module from being collected instead of failing only
the tests that need it. Test data is read inside a test, or by a function a test calls (cached
with ``functools.cache`` where it is read often). Directory listings that feed ``parametrize``
are not reads and stay.

The check is an AST walk over the statements that run at import: module-level code, class
bodies, and the decorators and argument defaults of every function and class defined there. It
looks for direct calls of ``read_text``, ``read_bytes`` or ``open``.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent
READS = frozenset({"read_text", "read_bytes", "open"})


def _is_read(call: ast.Call) -> bool:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr in READS
    return isinstance(function, ast.Name) and function.id in READS


def _import_time(node: ast.AST) -> list[ast.AST]:
    """The child nodes of ``node`` whose code runs when ``node`` runs at import."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        arguments = node.args
        return [
            *node.decorator_list,
            *arguments.defaults,
            *(default for default in arguments.kw_defaults if default is not None),
        ]
    if isinstance(node, ast.Lambda):
        return [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
    if isinstance(node, ast.ClassDef):
        return [*node.decorator_list, *node.bases, *node.keywords, *node.body]
    return list(ast.iter_child_nodes(node))


def import_time_reads(path: Path) -> list[int]:
    """The lines of ``path`` where a statement run at import calls a read directly."""
    lines: list[int] = []
    pending: list[ast.AST] = [ast.parse(path.read_text(encoding="utf-8"))]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Call) and _is_read(node):
            lines.append(node.lineno)
        pending += _import_time(node)
    return sorted(lines)


def test_no_test_module_reads_a_file_at_import() -> None:
    found = [
        f"{path.name}:{line}"
        for path in sorted(TESTS.glob("*.py"))
        for line in import_time_reads(path)
    ]

    assert found == []


def test_the_check_finds_reads_at_import_and_not_in_a_function_body(tmp_path: Path) -> None:
    module = tmp_path / "sample.py"
    module.write_text(
        "from pathlib import Path\n"
        'DATA = Path("x").read_text(encoding="utf-8")\n'
        "\n"
        'def uses_a_default(raw: bytes = Path("y").read_bytes()) -> bytes:\n'
        "    return raw\n"
        "\n"
        "def reads_when_called() -> str:\n"
        '    return Path("z").read_text(encoding="utf-8")\n',
        encoding="utf-8",
    )

    assert import_time_reads(module) == [2, 4]
