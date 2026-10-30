"""Loading Decision Models from JSON (ADR-0004)."""

import json
from pathlib import Path

from pydantic import ValidationError

from potakt.model.decision_model import DecisionModel
from potakt.model.validation import DecisionModelError, ValidationIssue, issues_from_pydantic

ROOT = "/"


def parse_decision_model(text: str) -> DecisionModel:
    """Parse and structurally validate DM JSON text; raise ``DecisionModelError`` on issues."""
    try:
        document: object = json.loads(text)
    except json.JSONDecodeError as error:
        message = f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        raise DecisionModelError([ValidationIssue(ROOT, message)]) from None
    try:
        return DecisionModel.model_validate_json(text)
    except ValidationError as error:
        raise DecisionModelError(issues_from_pydantic(error, document)) from None


def load_decision_model(path: Path) -> DecisionModel:
    """Read a DM JSON file (UTF-8, BOM tolerated) and structurally validate it."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise DecisionModelError([ValidationIssue(ROOT, f"cannot read {path}: {error}")]) from None
    return parse_decision_model(text)
