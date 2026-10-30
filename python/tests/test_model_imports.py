"""``potakt.model`` stays framework-independent: Pydantic, stdlib and tzdata only (DESIGN §6)."""

import ast
import sys
from pathlib import Path

import potakt.model

MODEL_DIR = Path(potakt.model.__file__).parent
ALLOWED_THIRD_PARTY = {"pydantic", "pydantic_core", "tzdata"}


def _imported_top_level_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def test_model_package_imports_only_pydantic_stdlib_and_tzdata() -> None:
    sources = sorted(MODEL_DIR.glob("*.py"))
    assert sources, "no sources found"

    offending = {
        f"{source.name}: {module}"
        for source in sources
        for module in _imported_top_level_modules(source)
        if module not in sys.stdlib_module_names
        and module not in ALLOWED_THIRD_PARTY
        and module != "potakt"
    }

    assert offending == set()


def test_model_package_does_not_import_other_potakt_packages() -> None:
    for source in MODEL_DIR.glob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and node.module
                and node.module.startswith("potakt")
            ):
                assert node.module.startswith("potakt.model"), (source.name, node.module)
