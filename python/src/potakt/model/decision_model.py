"""The Decision Model root and its structural (cross-field) validation (ADR-0004)."""

from datetime import date
from functools import cache
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo, available_timezones

from pydantic import AfterValidator, Field, model_validator

from potakt.model.catalog import Constraint, Coverage, Stability
from potakt.model.common import Loc, StrictModel, custom_error, raise_errors
from potakt.model.data import DataBindings


@cache
def _known_timezones() -> frozenset[str]:
    return frozenset(available_timezones())


def _check_timezone(value: str) -> str:
    # Membership in the bundled database is exact and portable; ZoneInfo(key) alone would accept
    # case variants on case-insensitive file systems (Windows) and fail elsewhere.
    if value not in _known_timezones():
        raise custom_error(f"unknown IANA timezone {value!r}")
    return value


Timezone = Annotated[str, AfterValidator(_check_timezone)]


MAX_HORIZON_DAYS = 366


class Horizon(StrictModel):
    """The planning period: ``start`` and ``end`` dates, both inclusive, at most 366 days."""

    start: date
    end: date

    @model_validator(mode="after")
    def _check_order_and_length(self) -> Self:
        if self.end < self.start:
            message = f"horizon end {self.end} is before its start {self.start}"
            raise_errors("Horizon", [(("end",), message)])
        days = (self.end - self.start).days + 1
        if days > MAX_HORIZON_DAYS:
            message = f"horizon is {days} days long; at most {MAX_HORIZON_DAYS} days are allowed"
            raise_errors("Horizon", [((), message)])
        return self


class Vocabulary(StrictModel):
    """Domain words for display only, e.g. ``{"resource": "employee", "task": "shift"}``."""

    resource: str = Field(min_length=1)
    task: str = Field(min_length=1)


class Assumption(StrictModel):
    """Something the agent assumed and still needs to confirm."""

    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    status: Literal["proposed", "confirmed"]


class DecisionModel(StrictModel):
    """A typed, solver-agnostic description of an assignment decision problem (DM 0.2)."""

    schema_version: Literal["0.2"]
    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    status: Literal["draft", "confirmed", "approved"]
    title: str = Field(min_length=1)
    language: Literal["en", "el"]
    archetype: Literal["assignment"]
    vocabulary: Vocabulary | None = None
    timezone: Timezone
    horizon: Horizon
    data: DataBindings
    constraints: list[Constraint] = Field(min_length=1)
    assumptions: list[Assumption] = Field(default_factory=list[Assumption])
    objective: Literal["lexicographic_hard_then_soft"] = "lexicographic_hard_then_soft"

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @model_validator(mode="after")
    def _check_cross_field_rules(self) -> Self:
        errors: list[tuple[Loc, str]] = []
        if self.parent_version is not None and self.parent_version >= self.version:
            message = (
                f"parent_version ({self.parent_version}) must be lower than "
                f"version ({self.version})"
            )
            errors.append((("parent_version",), message))
        errors += _duplicate_ids("constraint", "constraints", [c.id for c in self.constraints])
        errors += _duplicate_ids("assumption", "assumptions", [a.id for a in self.assumptions])
        errors += _coverage_errors(self.constraints)
        errors += _duplicate_baseline_pairs(self.constraints)
        if errors:
            raise_errors("DecisionModel", errors)
        return self


def _duplicate_ids(kind: str, field: str, ids: list[str]) -> list[tuple[Loc, str]]:
    first_index: dict[str, int] = {}
    errors: list[tuple[Loc, str]] = []
    for index, item_id in enumerate(ids):
        if item_id in first_index:
            message = (
                f"duplicate {kind} id {item_id!r} (first used at /{field}/{first_index[item_id]})"
            )
            errors.append(((field, index, "id"), message))
        else:
            first_index[item_id] = index
    return errors


def _coverage_errors(constraints: list[Constraint]) -> list[tuple[Loc, str]]:
    positions = [i for i, rule in enumerate(constraints) if isinstance(rule, Coverage)]
    if not positions:
        return [(("constraints",), "exactly one 'coverage' constraint is required, found none")]
    first = positions[0]
    message = (
        f"exactly one 'coverage' constraint is allowed; another one is at /constraints/{first}"
    )
    return [(("constraints", index), message) for index in positions[1:]]


def _duplicate_baseline_pairs(constraints: list[Constraint]) -> list[tuple[Loc, str]]:
    """A ``(task_id, resource_id)`` pair may appear only once in a baseline (ADR-0006 §4)."""
    errors: list[tuple[Loc, str]] = []
    for index, rule in enumerate(constraints):
        if not isinstance(rule, Stability):
            continue
        first_entry: dict[tuple[str, str], int] = {}
        for k, entry in enumerate(rule.params.baseline):
            pair = (entry.task_id, entry.resource_id)
            if pair in first_entry:
                message = (
                    f"duplicate baseline assignment of resource {entry.resource_id!r} to task "
                    f"{entry.task_id!r} (first at /constraints/{index}/params/baseline/"
                    f"{first_entry[pair]})"
                )
                errors.append((("constraints", index, "params", "baseline", k), message))
            else:
                first_entry[pair] = k
    return errors
