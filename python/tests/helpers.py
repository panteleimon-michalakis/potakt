"""Shared builders for Decision Model tests."""

import copy
import json
from pathlib import Path
from typing import Any

from potakt.model import DecisionModelError, ValidationIssue, parse_decision_model

_MINIMAL_DM: dict[str, Any] = {
    "schema_version": "0.2",
    "id": "dm-test",
    "version": 1,
    "status": "draft",
    "title": "Test model",
    "language": "en",
    "archetype": "assignment",
    "timezone": "Europe/Athens",
    "horizon": {"start": "2026-11-02", "end": "2026-11-08"},
    "data": {
        "resources": {
            "source": "resources.csv",
            "id": "id",
            "attributes": {
                "skills": {"column": "skills", "type": "list", "separator": ";"},
                "contract": {"column": "contract", "type": "str"},
            },
        },
        "tasks": {
            "source": "tasks.csv",
            "id": "id",
            "start": "start",
            "end": "end",
            "demand": "demand",
            "attributes": {
                "type": {"column": "type", "type": "str"},
                "required_skill": {"column": "required_skill", "type": "str"},
            },
        },
    },
    "constraints": [
        {
            "id": "coverage",
            "template": "coverage",
            "hardness": "hard",
            "rationale": "Every task is staffed with its demand.",
            "provenance": {"kind": "default"},
        }
    ],
}


def minimal_dm() -> dict[str, Any]:
    """Return a fresh, structurally valid Decision Model document."""
    return copy.deepcopy(_MINIMAL_DM)


def constraint(template: str, **fields: Any) -> dict[str, Any]:
    """Build a constraint document with sensible defaults for the common fields."""
    doc: dict[str, Any] = {
        "id": fields.pop("id", template),
        "template": template,
        "rationale": "test rule",
        "provenance": {"kind": "default"},
    }
    doc.update(fields)
    return doc


def structural_issues(doc: dict[str, Any]) -> list[ValidationIssue]:
    """Parse ``doc`` and return the reported issues (empty if it is valid)."""
    try:
        parse_decision_model(json.dumps(doc))
    except DecisionModelError as error:
        return list(error.issues)
    return []


def paths(issues: list[ValidationIssue]) -> list[str]:
    return [issue.path for issue in issues]


RESOURCES_CSV = """id,name,skills,contract
r1,Ana,opening;closing,full_time
r2,Ben,,part_time
r3,Χαρά,closing ; als,full_time
"""

TASKS_CSV = """id,start,end,demand,type,required_skill
t1,2026-11-02T07:00,2026-11-02T15:00,2,πρωί,
t2,2026-11-02T23:00,2026-11-03T07:00,1,νύχτα,als
"""


def write_scenario(
    directory: Path,
    doc: dict[str, Any] | None = None,
    resources: str = RESOURCES_CSV,
    tasks: str = TASKS_CSV,
    **extra_files: str,
) -> Path:
    model_file = directory / "model.json"
    model_file.write_text(json.dumps(doc or minimal_dm(), ensure_ascii=False), encoding="utf-8")
    (directory / "resources.csv").write_text(resources, encoding="utf-8")
    (directory / "tasks.csv").write_text(tasks, encoding="utf-8")
    for name, content in extra_files.items():
        (directory / name.replace("_csv", ".csv")).write_text(content, encoding="utf-8")
    return model_file
