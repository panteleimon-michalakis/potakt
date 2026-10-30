"""JSON Schema of the Decision Model, generated from the Pydantic models.

The rendered schema is committed at ``spec/decision-model.schema.json``; a test guards drift.
"""

import json
from typing import Any

from potakt.model.decision_model import DecisionModel

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def decision_model_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of DM v0 in validation mode."""
    return {"$schema": JSON_SCHEMA_DIALECT, **DecisionModel.model_json_schema()}


def render_json_schema() -> str:
    """The schema as committed: two-space indented UTF-8 JSON with a trailing newline."""
    return json.dumps(decision_model_json_schema(), indent=2, ensure_ascii=False) + "\n"
