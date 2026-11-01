"""Builders for evaluation tests: small scenario directories in a temporary folder."""

import copy
import json
from pathlib import Path
from typing import Any

from helpers import RESOURCES_CSV, TASKS_CSV, minimal_dm

REQUEST = "Please staff the two shifts in resources.csv and tasks.csv.\n"

_SCENARIO: dict[str, Any] = {
    "id": "01-test",
    "title": "Test scenario",
    "language": "en",
    "domain": "test",
    "difficulty": "easy",
    "style": "email",
    "csv_files": ["resources.csv", "tasks.csv"],
    "reference_score": {"hard": 0, "soft": 0},
    "disruption": {"resource_id": "r1", "start": "2026-11-02T00:00", "end": "2026-11-03T00:00"},
    "unsupported": [],
}

_SOLUTION: dict[str, Any] = {
    "schema_version": "0.1",
    "model_id": "dm-test",
    "model_version": 1,
    "assignments": [
        {"task_id": "t1", "resource_id": "r1"},
        {"task_id": "t1", "resource_id": "r2"},
        {"task_id": "t2", "resource_id": "r3"},
    ],
}


def scenario_metadata() -> dict[str, Any]:
    """A fresh, valid ``scenario.json`` document for the directory ``01-test``."""
    return copy.deepcopy(_SCENARIO)


def write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")


def write_scenario_dir(
    parent: Path,
    name: str = "01-test",
    metadata: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    solution: dict[str, Any] | None = None,
    request: str = REQUEST,
) -> Path:
    """A complete, valid scenario directory (unless an argument breaks it)."""
    directory = parent / name
    directory.mkdir()
    (directory / "request.md").write_text(request, encoding="utf-8")
    (directory / "resources.csv").write_text(RESOURCES_CSV, encoding="utf-8")
    (directory / "tasks.csv").write_text(TASKS_CSV, encoding="utf-8")
    write_json(directory / "scenario.json", metadata or scenario_metadata())
    write_json(directory / "truth.json", truth or minimal_dm())
    write_json(directory / "solution.json", solution or copy.deepcopy(_SOLUTION))
    return directory
