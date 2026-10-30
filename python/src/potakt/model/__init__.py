"""Decision Model v0 (ADR-0004): types, validation and loading.

Framework-independent by design: depends only on Pydantic, the standard library and tzdata.
"""

from potakt.model.catalog import (
    Apart,
    Balance,
    Constraint,
    Coverage,
    MaxConsecutiveDays,
    MaxPerPeriod,
    MinPerPeriod,
    MinRest,
    NoOverlap,
    Preference,
    RequiresAttribute,
    Stability,
    Together,
    Unavailable,
)
from potakt.model.common import Condition, Hardness, Provenance, Scope
from potakt.model.data import (
    AttributeBinding,
    DataBindings,
    ProblemInstance,
    Resource,
    ResourceBinding,
    Task,
    TaskBinding,
    TimeWindow,
    WindowsBinding,
)
from potakt.model.decision_model import Assumption, DecisionModel, Horizon, Vocabulary
from potakt.model.loader import load_decision_model, load_instance, parse_decision_model
from potakt.model.schema import decision_model_json_schema, render_json_schema
from potakt.model.validation import DecisionModelError, ValidationIssue, validate_data

__all__ = [
    "Apart",
    "Assumption",
    "AttributeBinding",
    "Balance",
    "Condition",
    "Constraint",
    "Coverage",
    "DataBindings",
    "DecisionModel",
    "DecisionModelError",
    "Hardness",
    "Horizon",
    "MaxConsecutiveDays",
    "MaxPerPeriod",
    "MinPerPeriod",
    "MinRest",
    "NoOverlap",
    "Preference",
    "ProblemInstance",
    "Provenance",
    "RequiresAttribute",
    "Resource",
    "ResourceBinding",
    "Scope",
    "Stability",
    "Task",
    "TaskBinding",
    "TimeWindow",
    "Together",
    "Unavailable",
    "ValidationIssue",
    "Vocabulary",
    "WindowsBinding",
    "decision_model_json_schema",
    "load_decision_model",
    "load_instance",
    "parse_decision_model",
    "render_json_schema",
    "validate_data",
]
