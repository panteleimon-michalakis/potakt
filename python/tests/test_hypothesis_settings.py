"""Every property test sets ``deadline=None``: Windows CI runners are slow (TASK-004b F9).

On CI (the ``CI`` variable is set) Hypothesis loads its ``ci`` profile, which already has
``deadline=None`` and ``derandomize=True``; the explicit setting makes local runs behave the same.
"""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def _name(node: ast.expr) -> str | None:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id
    return target.attr if isinstance(target, ast.Attribute) else None


def _sets_no_deadline(decorator: ast.expr) -> bool:
    return (
        isinstance(decorator, ast.Call)
        and _name(decorator) == "settings"
        and any(
            keyword.arg == "deadline"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is None
            for keyword in decorator.keywords
        )
    )


def test_every_property_test_sets_no_deadline() -> None:
    missing = [
        f"{source.name}::{node.name}"
        for source in sorted(TESTS.glob("test_*.py"))
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
        and any(_name(decorator) == "given" for decorator in node.decorator_list)
        and not any(_sets_no_deadline(decorator) for decorator in node.decorator_list)
    ]

    assert missing == []
