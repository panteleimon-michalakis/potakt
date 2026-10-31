"""Data side of the Decision Model: CSV bindings, resources, tasks and the problem instance."""

from datetime import UTC, datetime
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, Self
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, ConfigDict, Field, WithJsonSchema, model_validator

from potakt.model.common import (
    ATTRIBUTE_NAME_PATTERN,
    RESERVED_ATTRIBUTE_NAMES,
    AttributeValue,
    Loc,
    StrictModel,
    custom_error,
    raise_errors,
)

ColumnType = Literal["str", "int", "float", "bool", "list"]
DEFAULT_LIST_SEPARATOR = ";"
CsvDelimiter = Literal[",", ";", "\t"]
CsvDecimal = Literal[".", ","]
CsvEncoding = Literal["utf-8", "windows-1253", "iso-8859-7"]


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


# Not "format": "date-time": RFC 3339 date-times require a UTC offset, DM datetimes forbid one.
LOCAL_DATETIME_JSON_SCHEMA = {
    "type": "string",
    "pattern": r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:00(\.0+)?)?$",
    "description": "Local date-time in the DM timezone, without UTC offset, e.g. 2026-11-02T07:00",
}
LocalDateTime = Annotated[
    datetime, AfterValidator(check_local_datetime), WithJsonSchema(LOCAL_DATETIME_JSON_SCHEMA)
]
SourcePath = Annotated[str, AfterValidator(check_source_path)]
ColumnName = Annotated[str, Field(min_length=1)]


class _FrozenModel(StrictModel):
    """Value objects that cannot change after validation."""

    model_config = ConfigDict(frozen=True)


class TimeWindow(_FrozenModel):
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


class CsvDialect(StrictModel):
    """How a CSV file is written (ADR-0006 §5): delimiter, decimal separator and encoding.

    The defaults read a plain UTF-8 file (BOM tolerated). ``decimal`` applies to ``float`` cells
    only; Greek-locale Excel writes ``;``, ``,`` and Windows-1253.
    """

    delimiter: CsvDelimiter = ","
    decimal: CsvDecimal = "."
    encoding: CsvEncoding = "utf-8"

    @model_validator(mode="after")
    def _check_decimal(self) -> Self:
        if self.decimal == "," and self.delimiter == ",":
            message = "decimal ',' needs a delimiter other than ','"
            raise_errors("CsvDialect", [(("decimal",), message)])
        return self


class _EntityBinding(StrictModel):
    source: SourcePath
    id: ColumnName
    attributes: dict[str, AttributeBinding] = Field(default_factory=dict[str, AttributeBinding])
    csv: CsvDialect = Field(default_factory=CsvDialect)

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
    csv: CsvDialect = Field(default_factory=CsvDialect)


class DataBindings(StrictModel):
    """Where the resources and tasks of a DM come from."""

    resources: ResourceBinding
    tasks: TaskBinding


# --- loaded data -----------------------------------------------------------------------------

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def to_utc(local: datetime, zone: ZoneInfo) -> datetime:
    """Interpret a local naive datetime in ``zone``; ambiguous times mean the first occurrence."""
    return local.replace(tzinfo=zone).astimezone(UTC)


def local_time_exists(local: datetime, zone: ZoneInfo) -> bool:
    """False for local times skipped by a DST change (spring-forward gap)."""
    return to_utc(local, zone).astimezone(zone).replace(tzinfo=None) == local


def duration_minutes(start: datetime, end: datetime, zone: ZoneInfo) -> int:
    """Real elapsed minutes between two local datetimes, DST changes included."""
    # Subtracting aware datetimes that share a tzinfo ignores the offset change; go via UTC.
    return int((to_utc(end, zone) - to_utc(start, zone)).total_seconds()) // 60


class Resource(_FrozenModel):
    """Something that can be assigned to tasks (e.g. an employee)."""

    id: str = Field(min_length=1)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict[str, AttributeValue])


class Task(_FrozenModel):
    """A unit of work given as data (e.g. a shift) that needs ``demand`` resources."""

    id: str = Field(min_length=1)
    start: LocalDateTime
    end: LocalDateTime
    demand: int = Field(ge=1)
    attributes: dict[str, AttributeValue] = Field(default_factory=dict[str, AttributeValue])

    def duration_minutes(self, zone: ZoneInfo) -> int:
        return duration_minutes(self.start, self.end, zone)

    def derived_attributes(self, zone: ZoneInfo) -> dict[str, AttributeValue]:
        """``weekday``, ``start_hour``, ``duration_hours`` and ``date``, from the local start."""
        return {
            "weekday": WEEKDAYS[self.start.weekday()],
            "start_hour": self.start.hour,
            "duration_hours": self.duration_minutes(zone) / 60,
            "date": self.start.date().isoformat(),
        }


class ProblemInstance(_FrozenModel):
    """The data of one DM, immutable after loading.

    ``windows`` maps the id of every ``unavailable`` rule to its windows: inline windows first,
    then the ones loaded from the rule's CSV source. Models are frozen, but the contained dicts
    and lists are plain containers: treat them as read-only.
    """

    resources: tuple[Resource, ...]
    tasks: tuple[Task, ...]
    windows: dict[str, tuple[TimeWindow, ...]] = Field(
        default_factory=dict[str, tuple[TimeWindow, ...]]
    )
