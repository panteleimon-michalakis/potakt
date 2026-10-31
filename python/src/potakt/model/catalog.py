"""Constraint catalog of the assignment archetype (ADR-0004, amended by ADR-0006).

Each template is a Pydantic model; ``Constraint`` is their discriminated union on ``template``.
Exact scoring semantics live in the checker (TASK-003); here: parameters and allowed hardness.
"""

import math
from typing import Annotated, Literal, Self, get_args

from pydantic import (
    AfterValidator,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from potakt.model.common import (
    MAX_DAYS,
    MAX_HOURS,
    MAX_WEIGHT,
    SOFT_RULE_NEEDS_WEIGHT,
    WEIGHT_ONLY_ON_SOFT_RULES,
    ConstraintStatus,
    Hardness,
    Measure,
    Period,
    Provenance,
    Scope,
    StrictModel,
    above_limit_maximum,
    check_whole_minutes,
    custom_error,
    is_whole,
    raise_errors,
)
from potakt.model.data import TimeWindow, WindowsBinding

Handler = ValidatorFunctionWrapHandler


def _check_distinct_pair(resource_ids: list[str]) -> list[str]:
    if len(set(resource_ids)) != len(resource_ids):
        raise custom_error("the two resource ids must be different")
    return resource_ids


# Bounds on each member (not the union) so the JSON Schema shows them (ADR-0006 §6).
Hours = Annotated[
    Annotated[int, Field(gt=0, le=MAX_HOURS)] | Annotated[float, Field(gt=0, le=MAX_HOURS)],
    AfterValidator(check_whole_minutes),
]
Weight = Annotated[int, Field(ge=1, le=MAX_WEIGHT)]
ResourceId = Annotated[str, Field(min_length=1)]
ResourcePair = Annotated[
    list[ResourceId], Field(min_length=2, max_length=2), AfterValidator(_check_distinct_pair)
]


# --- common fields ---------------------------------------------------------------------------


class _ConstraintBase(StrictModel):
    id: str = Field(min_length=1)
    scope: Scope = Field(default_factory=Scope)
    rationale: str = Field(min_length=1)
    provenance: Provenance
    status: ConstraintStatus = "proposed"


def _only(
    model: type[_ConstraintBase], hardness: Hardness, value: object, handler: Handler
) -> object:
    """Validate a one-hardness template's ``hardness``; name the template if it is wrong.

    A wrap validator keeps the ``Literal`` type, so the JSON Schema still says ``const``.
    """
    try:
        return handler(value)
    except ValidationError:
        template = get_args(model.model_fields["template"].annotation)[0]
        raise custom_error(f"template {template!r} allows only {hardness} rules") from None


class _HardOnly(_ConstraintBase):
    hardness: Literal["hard"]

    @field_validator("hardness", mode="wrap")
    @classmethod
    def _check_hardness(cls, value: object, handler: Handler) -> object:
        return _only(cls, "hard", value, handler)


class _SoftOnly(_ConstraintBase):
    hardness: Literal["soft"]
    weight: Weight

    @field_validator("hardness", mode="wrap")
    @classmethod
    def _check_hardness(cls, value: object, handler: Handler) -> object:
        return _only(cls, "soft", value, handler)


class _HardOrSoft(_ConstraintBase):
    hardness: Hardness
    weight: Weight | None = None

    @model_validator(mode="after")
    def _check_weight(self) -> Self:
        if self.hardness == "soft" and self.weight is None:
            raise_errors("Constraint", [(("weight",), SOFT_RULE_NEEDS_WEIGHT)])
        if self.hardness == "hard" and self.weight is not None:
            raise_errors("Constraint", [(("weight",), WEIGHT_ONLY_ON_SOFT_RULES)])
        return self


# --- parameters ------------------------------------------------------------------------------


class NoParams(StrictModel):
    """The template takes no parameters."""


class RequiresAttributeParams(StrictModel):
    task_attribute: str = Field(min_length=1)
    resource_attribute: str = Field(min_length=1)
    match: Literal["contains", "equals"]


class MinRestParams(StrictModel):
    hours: Hours


class AttributeRef(StrictModel):
    """A per-resource value: each resource's value of the named attribute (ADR-0006 §5)."""

    attribute: str = Field(min_length=1)


def _limit_kind(value: object) -> str:
    return "per_resource" if isinstance(value, dict | AttributeRef) else "literal"


# Tagged by shape so an invalid limit reports one alternative's errors, not every member's.
Limit = Annotated[
    Annotated[
        Annotated[int, Field(ge=0)] | Annotated[float, Field(ge=0)],
        Tag("literal"),
    ]
    | Annotated[AttributeRef, Tag("per_resource")],
    Discriminator(_limit_kind),
]


LIMIT_SHAPE = 'limit must be a number (at least 0) or {"attribute": "<resource attribute>"}'


class PeriodLimitParams(StrictModel):
    measure: Measure
    period: Period
    limit: Limit

    @field_validator("limit", mode="wrap")
    @classmethod
    def _check_limit_shape(cls, value: object, handler: Handler) -> object:
        """A limit of the wrong shape gets one message naming both forms (helps LLM repair); a
        number out of range keeps its bound message. The wrap keeps the JSON Schema unchanged."""
        try:
            return handler(value)
        except ValidationError:
            finite = isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
            if finite and not isinstance(value, bool):
                raise
            raise custom_error(LIMIT_SHAPE) from None

    @model_validator(mode="after")
    def _check_limit_unit(self) -> Self:
        if isinstance(self.limit, AttributeRef):
            return self  # each resource's value is checked against the data (validate_data)
        # The maximum depends on ``measure``, so it is checked here, before any arithmetic, and
        # the JSON Schema shows only the minimum (ADR-0006 §6).
        above = above_limit_maximum(self.limit, self.measure)
        if above is not None:
            raise_errors("PeriodLimitParams", [(("limit",), f"limit {above}")])
        if self.measure == "count" and not is_whole(self.limit):
            message = "limit must be a whole number when measure is 'count'"
            raise_errors("PeriodLimitParams", [(("limit",), message)])
        if self.measure == "hours":
            try:
                check_whole_minutes(self.limit)
            except ValueError as error:
                raise_errors("PeriodLimitParams", [(("limit",), str(error))])
        return self


class MaxConsecutiveDaysParams(StrictModel):
    days: int = Field(ge=1, le=MAX_DAYS)


class UnavailableParams(StrictModel):
    windows: list[TimeWindow] = Field(default_factory=list[TimeWindow])
    source: WindowsBinding | None = None

    @model_validator(mode="after")
    def _check_not_empty(self) -> Self:
        if not self.windows and self.source is None:
            raise_errors(
                "UnavailableParams", [((), "at least one of 'windows' or 'source' is required")]
            )
        return self


class PreferenceParams(StrictModel):
    kind: Literal["prefer", "avoid"]


class BalanceParams(StrictModel):
    measure: Measure


class ResourcePairParams(StrictModel):
    resource_ids: ResourcePair


class BaselineAssignment(StrictModel):
    task_id: str = Field(min_length=1)
    resource_id: ResourceId


class StabilityParams(StrictModel):
    baseline: list[BaselineAssignment]


# --- templates -------------------------------------------------------------------------------


class Coverage(_HardOnly):
    """Each in-scope task gets exactly ``demand`` resources (structural, hard only)."""

    template: Literal["coverage"]
    params: NoParams = Field(default_factory=NoParams)


class RequiresAttribute(_HardOrSoft):
    """A resource may work a task only if its attribute matches the task's requirement.

    Tasks without the task attribute, or with an empty list, are unrestricted.
    """

    template: Literal["requires_attribute"]
    params: RequiresAttributeParams


class NoOverlap(_HardOrSoft):
    """A resource cannot work two time-overlapping tasks."""

    template: Literal["no_overlap"]
    params: NoParams = Field(default_factory=NoParams)


class MinRest(_HardOrSoft):
    """The gap between consecutive tasks of a resource is at least ``hours``."""

    template: Literal["min_rest"]
    params: MinRestParams


class MaxPerPeriod(_HardOrSoft):
    """Upper bound of a measure (count or hours) per resource per period."""

    template: Literal["max_per_period"]
    params: PeriodLimitParams


class MinPerPeriod(_HardOrSoft):
    """Lower bound of a measure (count or hours) per resource per period."""

    template: Literal["min_per_period"]
    params: PeriodLimitParams


class MaxConsecutiveDays(_HardOrSoft):
    """A resource works at most ``days`` consecutive days."""

    template: Literal["max_consecutive_days"]
    params: MaxConsecutiveDaysParams


class Unavailable(_HardOrSoft):
    """A resource cannot work tasks that overlap its unavailability windows."""

    template: Literal["unavailable"]
    params: UnavailableParams


class Preference(_SoftOnly):
    """Scoped resources prefer or avoid scoped tasks (soft only)."""

    template: Literal["preference"]
    params: PreferenceParams


class Balance(_SoftOnly):
    """Minimize max - min of a measure over scoped resources, counting scoped tasks (soft only)."""

    template: Literal["balance"]
    params: BalanceParams


class Together(_SoftOnly):
    """The two resources should work the same in-scope tasks (soft only)."""

    template: Literal["together"]
    params: ResourcePairParams


class Apart(_HardOrSoft):
    """The two resources should not work the same task."""

    template: Literal["apart"]
    params: ResourcePairParams


class Stability(_SoftOnly):
    """Penalty per baseline assignment that changes (soft only)."""

    template: Literal["stability"]
    params: StabilityParams


Constraint = Annotated[
    Coverage
    | RequiresAttribute
    | NoOverlap
    | MinRest
    | MaxPerPeriod
    | MinPerPeriod
    | MaxConsecutiveDays
    | Unavailable
    | Preference
    | Balance
    | Together
    | Apart
    | Stability,
    Field(discriminator="template"),
]
