"""Data side of the Decision Model: CSV bindings, time windows and local datetimes."""

from datetime import datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from potakt.model.common import (
    ATTRIBUTE_NAME_PATTERN,
    RESERVED_ATTRIBUTE_NAMES,
    Loc,
    StrictModel,
    custom_error,
    raise_errors,
)

ColumnType = Literal["str", "int", "float", "bool", "list"]
DEFAULT_LIST_SEPARATOR = ";"


def check_local_datetime(value: datetime) -> datetime:
    """A DM datetime is local (no UTC offset) and falls on a whole minute (ADR-0004)."""
    if value.tzinfo is not None:
        raise custom_error("must be a local time without a UTC offset (the DM timezone applies)")
    if value.second or value.microsecond:
        raise custom_error("must be a whole minute (no seconds)")
    return value


def check_source_path(value: str) -> str:
    """Data sources are relative paths that cannot escape the model directory."""
    windows, posix = PureWindowsPath(value), PurePosixPath(value)
    escapes = (
        not value
        or windows.is_absolute()
        or bool(windows.drive)
        or bool(windows.root)
        or posix.is_absolute()
        or ".." in windows.parts
    )
    if escapes:
        raise custom_error(f"source {value!r} must be a relative path inside the model directory")
    return value


LocalDateTime = Annotated[datetime, AfterValidator(check_local_datetime)]
SourcePath = Annotated[str, AfterValidator(check_source_path)]
ColumnName = Annotated[str, Field(min_length=1)]


class TimeWindow(StrictModel):
    """A local time interval ``[start, end)`` during which a resource is unavailable."""

    resource_id: str = Field(min_length=1)
    start: LocalDateTime
    end: LocalDateTime

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end <= self.start:
            raise_errors("TimeWindow", [(("end",), "end must be after start")])
        return self


class AttributeBinding(StrictModel):
    """Maps one CSV column to a typed attribute."""

    column: ColumnName
    type: ColumnType
    separator: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _check_separator(self) -> Self:
        if self.separator is not None and self.type != "list":
            raise_errors(
                "AttributeBinding",
                [(("separator",), "separator is only allowed for 'list' columns")],
            )
        return self

    @property
    def list_separator(self) -> str:
        return self.separator or DEFAULT_LIST_SEPARATOR


class _EntityBinding(StrictModel):
    source: SourcePath
    id: ColumnName
    attributes: dict[str, AttributeBinding] = Field(default_factory=dict[str, AttributeBinding])

    @model_validator(mode="after")
    def _check_attribute_names(self) -> Self:
        errors: list[tuple[Loc, str]] = []
        for name in self.attributes:
            if name in RESERVED_ATTRIBUTE_NAMES:
                errors.append((("attributes", name), f"attribute name {name!r} is reserved"))
            elif not ATTRIBUTE_NAME_PATTERN.fullmatch(name):
                errors.append((("attributes", name), f"attribute name {name!r} must be snake_case"))
        if errors:
            raise_errors(type(self).__name__, errors)
        return self


class ResourceBinding(_EntityBinding):
    """CSV binding for resources: id column plus typed attribute columns."""


class TaskBinding(_EntityBinding):
    """CSV binding for tasks: id, start, end and demand columns plus typed attribute columns."""

    start: ColumnName
    end: ColumnName
    demand: ColumnName


class WindowsBinding(StrictModel):
    """CSV binding for unavailability windows (one row per window)."""

    source: SourcePath
    resource_id: ColumnName
    start: ColumnName
    end: ColumnName


class DataBindings(StrictModel):
    """Where the resources and tasks of a DM come from."""

    resources: ResourceBinding
    tasks: TaskBinding
