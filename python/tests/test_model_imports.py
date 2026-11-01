"""Import boundaries of the core packages (DESIGN §6).

``potakt.model`` stays framework-independent: Pydantic, stdlib and tzdata only.
``potakt.checker`` trusts nothing but the model: ``potakt.model`` and the standard library only.
``potakt.evaluation`` (ADR-0007) builds on both, never the other way round, and stays free of
LLM SDKs and the CLI: ``potakt.model``, ``potakt.checker``, Pydantic and the standard library.
"""

import ast
import sys
from pathlib import Path

import potakt.checker
import potakt.evaluation
import potakt.model

MODEL_DIR = Path(potakt.model.__file__).parent
CHECKER_DIR = Path(potakt.checker.__file__).parent
EVALUATION_DIR = Path(potakt.evaluation.__file__).parent
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


def _imported_potakt_modules(source: Path) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names if alias.name.startswith("potakt"))
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{source.name}: use absolute imports"
            if node.module and node.module.startswith("potakt"):
                modules.add(node.module)
    return modules


def _within(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


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
        for module in _imported_potakt_modules(source):
            assert _within(module, "potakt.model"), (source.name, module)


def test_checker_package_imports_only_the_stdlib_and_potakt_model() -> None:
    sources = sorted(CHECKER_DIR.glob("*.py"))
    assert sources, "no sources found"

    third_party = {
        f"{source.name}: {module}"
        for source in sources
        for module in _imported_top_level_modules(source)
        if module not in sys.stdlib_module_names and module != "potakt"
    }
    other_potakt = {
        f"{source.name}: {module}"
        for source in sources
        for module in _imported_potakt_modules(source)
        if not (_within(module, "potakt.model") or _within(module, "potakt.checker"))
    }

    assert third_party == set()
    assert other_potakt == set()


def test_evaluation_package_imports_only_model_checker_pydantic_and_the_stdlib() -> None:
    sources = sorted(EVALUATION_DIR.glob("*.py"))
    assert sources, "no sources found"

    third_party = {
        f"{source.name}: {module}"
        for source in sources
        for module in _imported_top_level_modules(source)
        if module not in sys.stdlib_module_names
        and module not in {"pydantic", "pydantic_core"}
        and module != "potakt"
    }
    other_potakt = {
        f"{source.name}: {module}"
        for source in sources
        for module in _imported_potakt_modules(source)
        if not any(
            _within(module, package)
            for package in ("potakt.model", "potakt.checker", "potakt.evaluation")
        )
    }

    assert third_party == set()
    assert other_potakt == set()


def test_model_and_checker_never_import_the_evaluation_package() -> None:
    importers = {
        f"{source.parent.name}/{source.name}: {module}"
        for directory in (MODEL_DIR, CHECKER_DIR)
        for source in sorted(directory.glob("*.py"))
        for module in _imported_potakt_modules(source)
        if _within(module, "potakt.evaluation")
    }

    assert importers == set()
