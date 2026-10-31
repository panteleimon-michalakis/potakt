"""Shared types of the Decision Model: base model, enumerations, provenance and scopes."""

import math
import re
from collections.abc import Sequence
from typing import Literal, NoReturn, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError, ValidationError

Hardness = Literal["hard", "soft"]
ConstraintStatus = Literal["proposed", "confirmed"]
ProvenanceKind = Literal["utterance", "data", "assumption", "default"]
ConditionOp = Literal["eq", "ne", "in", "not_in", "contains", "gte", "lte"]
Measure = Literal["count", "hours"]
Period = Literal["day", "week", "horizon"]

Scalar = str | int | float | bool
AttributeValue = Scalar | list[str]
Loc = tuple[str | int, ...]

DERIVED_TASK_ATTRIBUTES = frozenset({"weekday", "start_hour", "duration_hours", "date"})
RESERVED_ATTRIBUTE_NAMES = frozenset({"id", "start", "end", "demand"}) | DERIVED_TASK_ATTRIBUTES
ATTRIBUTE_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
CUSTOM_ERROR_TYPE = "decision_model"
SOFT_RULE_NEEDS_WEIGHT = "a soft rule needs a weight (integer ≥ 1)"
WEIGHT_ONLY_ON_SOFT_RULES = "weight is only allowed on soft rules"


class StrictModel(BaseModel):
    """Base for all DM types: strict types, no unknown fields, finite numbers (ADR-0004)."""

    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


def custom_error(message: str) -> PydanticCustomError:
    """A validation error whose message is taken literally (no template expansion)."""
    return PydanticCustomError(CUSTOM_ERROR_TYPE, "{message}", {"message": message})


def raise_errors(title: str, errors: Sequence[tuple[Loc, str]]) -> NoReturn:
    """Raise one ``ValidationError`` carrying several located errors.

    Locations are relative to the model being validated; Pydantic prefixes them when the model
    is nested, so cross-field checks can point at the exact offending field.
    """
    details = [
        InitErrorDetails(type=custom_error(message), loc=loc, input=None) for loc, message in errors
    ]
    raise ValidationError.from_exception_data(title, details)


def check_whole_minutes(hours: float) -> float:
    """Reject hour values that are not a whole number of minutes (e.g. 7.501)."""
    minutes = hours * 60
    if not math.isclose(minutes, round(minutes), abs_tol=1e-6):
        raise custom_error(f"{hours} hours is not a whole number of minutes")
    return hours


class Provenance(StrictModel):
    """Where a rule came from: a user utterance, a data column, an assumption or a default."""

    kind: ProvenanceKind
    turn_id: str | None = Field(default=None, min_length=1)
    quote: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_required_details(self) -> Self:
        if self.kind == "utterance" and self.quote is None:
            raise_errors("Provenance", [(("quote",), "quote is required when kind is 'utterance'")])
        if self.kind == "data" and self.source is None:
            raise_errors("Provenance", [(("source",), "source is required when kind is 'data'")])
        return self


class Condition(StrictModel):
    """One filter ``{attribute, op, value}``; the conditions of a scope are combined with AND."""

    attribute: str = Field(min_length=1)
    op: ConditionOp
    value: AttributeValue

    @model_validator(mode="after")
    def _check_value_shape(self) -> Self:
        value = self.value
        problem: str | None = None
        if self.op in ("in", "not_in"):
            if not isinstance(value, list):
                problem = "a list"
        elif self.op == "contains":
            if isinstance(value, list):
                problem = "a single value"
        elif self.op in ("gte", "lte") and (isinstance(value, bool | list)):
            problem = "a number or a string"
        if problem is not None:
            raise_errors("Condition", [(("value",), f"value must be {problem} for op {self.op!r}")])
        return self


class Scope(StrictModel):
    """Which resources and tasks a rule applies to; an empty list means all."""

    resources: list[Condition] = Field(default_factory=list[Condition])
    tasks: list[Condition] = Field(default_factory=list[Condition])
