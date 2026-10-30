"""Shared builders for Decision Model tests."""

import copy
import json
from typing import Any

from potakt.model import DecisionModelError, ValidationIssue, parse_decision_model

_MINIMAL_DM: dict[str, Any] = {
    "schema_version": "0.1",
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
