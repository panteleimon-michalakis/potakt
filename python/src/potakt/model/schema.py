"""JSON Schemas of the Decision Model and of solutions, generated from the Pydantic models.

The rendered schemas are committed at ``spec/decision-model.schema.json`` and
``spec/solution.schema.json``; tests guard drift.
"""

import json
from typing import Any

from potakt.model.decision_model import DecisionModel
from potakt.model.solution import Solution

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def decision_model_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of DM v0 in validation mode."""
    return {"$schema": JSON_SCHEMA_DIALECT, **DecisionModel.model_json_schema()}


def render_json_schema() -> str:
    """The schema as committed: two-space indented UTF-8 JSON with a trailing newline."""
    return json.dumps(decision_model_json_schema(), indent=2, ensure_ascii=False) + "\n"


def solution_json_schema() -> dict[str, Any]:
    """The JSON Schema (draft 2020-12) of a solution in validation mode."""
    return {"$schema": JSON_SCHEMA_DIALECT, **Solution.model_json_schema()}


def render_solution_json_schema() -> str:
    """The solution schema as committed, formatted like the DM schema."""
    return json.dumps(solution_json_schema(), indent=2, ensure_ascii=False) + "\n"
