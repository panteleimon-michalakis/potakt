"""Constraint catalog of the assignment archetype (ADR-0004, amended by ADR-0006).

Each template is a Pydantic model; ``Constraint`` is their discriminated union on ``template``.
Exact scoring semantics live in the checker (TASK-003); here: parameters and allowed hardness.
"""

import math
from typing import Annotated, Literal, Self, get_args

from pydantic import (
    AfterValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
)

from potakt.model.common import (
    MAX_COUNT,
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
    # TASK-005e1: "weight iff soft" (``_check_weight``) as schema keywords; inherited by every
    # hard-or-soft template (ADR-0008 D2).
    model_config = ConfigDict(
        json_schema_extra={
            "if": {"properties": {"hardness": {"const": "soft"}}},
            "then": {"required": ["weight"], "properties": {"weight": {"type": "integer"}}},
            "else": {"properties": {"weight": {"type": "null"}}},
        }
    )

    hardness: Hardness
    weight: Weight | None = Field(
        default=None,
        description=(
            "Required (an integer from 1 to 10000) when `hardness` is `soft`; must be omitted or "
            "null when `hardness` is `hard`."
        ),
    )

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
    __doc__ = (
        "`task_attribute` and `resource_attribute` name attributes bound in "
        "`data.tasks.attributes` and `data.resources.attributes`."
    )

    task_attribute: str = Field(min_length=1)
    resource_attribute: str = Field(min_length=1)
    match: Literal["contains", "equals"]


class MinRestParams(StrictModel):
    hours: Hours = Field(
        description="Hours, more than 0 and in whole minutes, e.g. `11` or `7.5` (not `7.501`)."
    )


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
    # TASK-005e1: the maximum of a literal limit depends on ``measure`` (``above_limit_maximum``);
    # ``maximum`` ignores the ``{"attribute": …}`` form, so these keywords are exact.
    model_config = ConfigDict(
        json_schema_extra={
            "allOf": [
                {
                    "if": {"properties": {"measure": {"const": "count"}}},
                    "then": {"properties": {"limit": {"maximum": MAX_COUNT}}},
                },
                {
                    "if": {"properties": {"measure": {"const": "hours"}}},
                    "then": {"properties": {"limit": {"maximum": MAX_HOURS}}},
                },
            ]
        }
    )

    measure: Measure
    period: Period
    limit: Limit = Field(
        description=(
            'A number, or `{"attribute": \N{HORIZONTAL ELLIPSIS}}` naming a resource attribute '
            "bound as `int` or `float` that every in-scope resource has. Limits are at least 0; "
            "with `measure` `count` whole numbers up to 100000, with `measure` `hours` up to "
            "8784 in whole minutes."
        )
    )

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
    """At least one of `windows` (non-empty) or `source`."""

    # TASK-005e1: ``_check_not_empty`` as a schema keyword.
    model_config = ConfigDict(
        json_schema_extra={
            "anyOf": [
                {"required": ["windows"], "properties": {"windows": {"minItems": 1}}},
                {"required": ["source"], "properties": {"source": {"type": "object"}}},
            ]
        }
    )

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
    resource_ids: ResourcePair = Field(
        description="Two different resource ids from the data, written exactly as there."
    )


class BaselineAssignment(StrictModel):
    task_id: str = Field(min_length=1)
    resource_id: ResourceId


class StabilityParams(StrictModel):
    baseline: list[BaselineAssignment] = Field(
        description=(
            "Pairs of a task id and a resource id from the data, written exactly as there; each "
            "pair at most once."
        )
    )


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
