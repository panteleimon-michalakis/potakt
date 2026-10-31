"""Loading Decision Models from JSON and problem instances from CSV bindings (ADR-0004)."""

import csv
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

from pydantic import ValidationError

from potakt.model.catalog import Unavailable
from potakt.model.common import AttributeValue, Loc, StrictModel
from potakt.model.data import (
    AttributeBinding,
    ProblemInstance,
    Resource,
    ResourceBinding,
    Task,
    TaskBinding,
    TimeWindow,
    WindowsBinding,
    check_local_datetime,
)
from potakt.model.decision_model import DecisionModel
from potakt.model.solution import Solution
from potakt.model.validation import (
    DecisionModelError,
    ValidationIssue,
    issue_at,
    issues_from_pydantic,
)

ROOT = "/"

_INT = re.compile(r"[+-]?\d+", re.ASCII)
_FLOAT = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?", re.ASCII)
_TRUE = frozenset({"true", "yes", "1"})
_FALSE = frozenset({"false", "no", "0"})
_DEVICE_SUFFIXES = "123456789\N{SUPERSCRIPT ONE}\N{SUPERSCRIPT TWO}\N{SUPERSCRIPT THREE}"
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{port}{suffix}" for port in ("COM", "LPT") for suffix in _DEVICE_SUFFIXES}
)


def is_reserved_device_name(source: str) -> bool:
    """True if the file name of ``source`` is a Windows device name (CON, NUL, COM1, ...).

    Checked on every platform so a DM behaves the same everywhere: the base name counts with or
    without extensions and with trailing dots or spaces, case-insensitively (``nul.csv``,
    ``Aux.``, ``sub/com1.tar.gz``).
    """
    name = PureWindowsPath(source).name.rstrip(". ")
    stem = name.split(".", 1)[0].rstrip(" ")
    return stem.upper() in _RESERVED_DEVICE_NAMES


# --- Decision Model JSON ---------------------------------------------------------------------


def _parse_json[M: StrictModel](text: str, model: type[M]) -> M:
    try:
        document: object = json.loads(text)
    except json.JSONDecodeError as error:
        message = f"invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        raise DecisionModelError([ValidationIssue(ROOT, message)]) from None
    try:
        return model.model_validate_json(text)
    except ValidationError as error:
        raise DecisionModelError(issues_from_pydantic(error, document)) from None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise DecisionModelError([ValidationIssue(ROOT, f"cannot read {path}: {error}")]) from None


def parse_decision_model(text: str) -> DecisionModel:
    """Parse and structurally validate DM JSON text; raise ``DecisionModelError`` on issues."""
    return _parse_json(text, DecisionModel)


def load_decision_model(path: Path) -> DecisionModel:
    """Read a DM JSON file (UTF-8, BOM tolerated) and structurally validate it."""
    return parse_decision_model(_read_text(path))


def parse_solution(text: str) -> Solution:
    """Parse and structurally validate solution JSON text; issue paths point into the solution."""
    return _parse_json(text, Solution)


def load_solution(path: Path) -> Solution:
    """Read a solution JSON file (UTF-8, BOM tolerated) and structurally validate it."""
    return parse_solution(_read_text(path))


# --- cell parsing ----------------------------------------------------------------------------


def split_list(text: str, separator: str) -> list[str]:
    """Split a ``list`` cell: items are trimmed and empty items dropped."""
    return [item.strip() for item in text.split(separator) if item.strip()]


def parse_cell(text: str, binding: AttributeBinding) -> AttributeValue | None:
    """Parse one CSV cell for ``binding``; ``None`` means the attribute is absent (empty cell)."""
    value = text.strip()
    if binding.type == "list":
        return split_list(value, binding.list_separator)
    if not value:
        return None
    match binding.type:
        case "str":
            return value
        case "int":
            return _parse_int(value)
        case "float":
            return _parse_float(value)
        case "bool":
            return _parse_bool(value)


def _parse_int(value: str) -> int:
    if not _INT.fullmatch(value):
        raise ValueError(f"{value!r} is not an integer")
    return int(value)


def _parse_float(value: str) -> float:
    if not _FLOAT.fullmatch(value):
        raise ValueError(f"{value!r} is not a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{value!r} is not a finite number")
    return number


def _parse_bool(value: str) -> bool:
    if value.lower() in _TRUE:
        return True
    if value.lower() in _FALSE:
        return False
    raise ValueError(f"{value!r} is not a boolean (use true/false, yes/no or 1/0)")


def _parse_local_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{value!r} is not an ISO-8601 local datetime") from None
    try:
        return check_local_datetime(parsed)
    except ValueError as error:
        raise ValueError(f"{value!r} {error}") from None


# --- CSV tables ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Table:
    name: str
    columns: dict[str, int]
    duplicates: frozenset[str]
    width: int
    rows: list[tuple[int, list[str]]]


class _Reader:
    """Reads the CSV tables of one DM and collects every problem as an issue."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()
        self.issues: list[ValidationIssue] = []

    def report(self, loc: Loc, message: str) -> None:
        self.issues.append(issue_at(loc, message))

    def table(self, source: str, loc: Loc) -> _Table | None:
        source_loc = (*loc, "source")
        if is_reserved_device_name(source):
            # Whether Windows opens these as devices depends on its version; reject them anywhere.
            self.report(source_loc, f"cannot read {source}: reserved device name")
            return None
        path = (self.base_dir / source).resolve()
        if not path.is_relative_to(self.base_dir):
            self.report(source_loc, f"{source} resolves outside the model directory")
            return None
        if path.exists() and not path.is_file():
            # Directories, FIFOs and other non-files would block or fail when opened.
            self.report(source_loc, f"cannot read {source}: not a regular file")
            return None
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle)
                records = [(reader.line_num, row) for row in reader if row]
        except FileNotFoundError:
            self.report(source_loc, f"cannot read {source}: file not found")
            return None
        except UnicodeDecodeError:
            self.report(source_loc, f"cannot read {source}: not valid UTF-8")
            return None
        except (OSError, csv.Error) as error:
            self.report(source_loc, f"cannot read {source}: {error}")
            return None
        if not records:
            self.report(source_loc, f"{source} has no header row")
            return None
        header = [name.strip() for name in records[0][1]]
        columns: dict[str, int] = {}
        duplicates: set[str] = set()
        for index, name in enumerate(header):
            if name in columns:
                duplicates.add(name)
            columns.setdefault(name, index)
        return _Table(source, columns, frozenset(duplicates), len(header), records[1:])

    def column(self, table: _Table, column: str, loc: Loc) -> int | None:
        if column not in table.columns:
            self.report(loc, f"column {column!r} not found in {table.name}")
            return None
        if column in table.duplicates:
            self.report(loc, f"column {column!r} appears more than once in {table.name}")
            return None
        return table.columns[column]

    def rows(
        self, table: _Table, loc: Loc, fields: dict[str, int | None]
    ) -> Iterator[tuple[int, dict[str, str]]]:
        """Yield ``(line, cells)`` per data row; nothing if a column is missing."""
        if any(index is None for index in fields.values()):
            return
        for line, row in table.rows:
            if len(row) > table.width:
                self.report(
                    (*loc, "source"), f"{table.name} line {line}: more fields than the header"
                )
                continue
            yield (
                line,
                {
                    field: row[index].strip() if index < len(row) else ""
                    for field, index in fields.items()
                    if index is not None
                },
            )

    def cell_error(self, loc: Loc, table: _Table, line: int, column: str, message: str) -> None:
        self.report(loc, f"{table.name} line {line}, column {column!r}: {message}")


def load_instance(dm: DecisionModel, base_dir: Path) -> ProblemInstance:
    """Load the CSV data bound by ``dm`` (paths relative to ``base_dir``) into an instance.

    Raises ``DecisionModelError`` with every file, column and cell problem found. Data-level
    rules (unique ids, task times, references) are checked separately by ``validate_data``.
    """
    reader = _Reader(base_dir)
    resources = _load_resources(reader, dm.data.resources)
    tasks = _load_tasks(reader, dm.data.tasks)
    windows = _load_windows(reader, dm)
    if reader.issues:
        raise DecisionModelError(reader.issues)
    return ProblemInstance(resources=tuple(resources), tasks=tuple(tasks), windows=windows)


def _attribute_fields(
    reader: _Reader, table: _Table, binding: ResourceBinding | TaskBinding, loc: Loc
) -> dict[str, int | None]:
    return {
        name: reader.column(table, attribute.column, (*loc, "attributes", name, "column"))
        for name, attribute in binding.attributes.items()
    }


def _parse_attributes(
    reader: _Reader,
    table: _Table,
    binding: ResourceBinding | TaskBinding,
    loc: Loc,
    line: int,
    cells: dict[str, str],
) -> dict[str, AttributeValue] | None:
    attributes: dict[str, AttributeValue] = {}
    ok = True
    for name, attribute in binding.attributes.items():
        try:
            value = parse_cell(cells[name], attribute)
        except ValueError as error:
            reader.cell_error((*loc, "attributes", name), table, line, attribute.column, str(error))
            ok = False
            continue
        if value is not None:
            attributes[name] = value
    return attributes if ok else None


def _required(
    reader: _Reader, table: _Table, loc: Loc, line: int, column: str, value: str
) -> str | None:
    if not value:
        reader.cell_error(loc, table, line, column, "a value is required")
        return None
    return value


def _load_resources(reader: _Reader, binding: ResourceBinding) -> list[Resource]:
    loc: Loc = ("data", "resources")
    table = reader.table(binding.source, loc)
    if table is None:
        return []
    fields = {"_id": reader.column(table, binding.id, (*loc, "id"))}
    fields |= _attribute_fields(reader, table, binding, loc)
    resources: list[Resource] = []
    for line, cells in reader.rows(table, loc, fields):
        resource_id = _required(reader, table, (*loc, "id"), line, binding.id, cells["_id"])
        attributes = _parse_attributes(reader, table, binding, loc, line, cells)
        if resource_id is not None and attributes is not None:
            resources.append(Resource(id=resource_id, attributes=attributes))
    return resources


def _load_tasks(reader: _Reader, binding: TaskBinding) -> list[Task]:
    loc: Loc = ("data", "tasks")
    table = reader.table(binding.source, loc)
    if table is None:
        return []
    core = {"id": binding.id, "start": binding.start, "end": binding.end, "demand": binding.demand}
    fields = {
        f"_{field}": reader.column(table, column, (*loc, field)) for field, column in core.items()
    }
    fields |= _attribute_fields(reader, table, binding, loc)
    tasks: list[Task] = []
    for line, cells in reader.rows(table, loc, fields):
        values: dict[str, object] = {}
        for field, column in core.items():
            text = _required(reader, table, (*loc, field), line, column, cells[f"_{field}"])
            if text is None:
                continue
            try:
                values[field] = _parse_task_field(field, text)
            except ValueError as error:
                reader.cell_error((*loc, field), table, line, column, str(error))
        attributes = _parse_attributes(reader, table, binding, loc, line, cells)
        if len(values) == len(core) and attributes is not None:
            tasks.append(Task.model_validate({**values, "attributes": attributes}))
    return tasks


def _parse_task_field(field: str, text: str) -> str | datetime | int:
    if field in ("start", "end"):
        return _parse_local_datetime(text)
    if field == "demand":
        demand = _parse_int(text)
        if demand < 1:
            raise ValueError(f"demand must be at least 1, got {demand}")
        return demand
    return text


def _load_windows(reader: _Reader, dm: DecisionModel) -> dict[str, tuple[TimeWindow, ...]]:
    windows: dict[str, tuple[TimeWindow, ...]] = {}
    for index, rule in enumerate(dm.constraints):
        if not isinstance(rule, Unavailable):
            continue
        loaded: list[TimeWindow] = []
        if rule.params.source is not None:
            loc: Loc = ("constraints", index, "params", "source")
            loaded = _load_window_table(reader, rule.params.source, loc)
        windows[rule.id] = (*rule.params.windows, *loaded)
    return windows


def _load_window_table(reader: _Reader, binding: WindowsBinding, loc: Loc) -> list[TimeWindow]:
    table = reader.table(binding.source, loc)
    if table is None:
        return []
    columns = {"resource_id": binding.resource_id, "start": binding.start, "end": binding.end}
    fields = {
        field: reader.column(table, column, (*loc, field)) for field, column in columns.items()
    }
    windows: list[TimeWindow] = []
    for line, cells in reader.rows(table, loc, fields):
        resource_id = _required(
            reader, table, (*loc, "resource_id"), line, binding.resource_id, cells["resource_id"]
        )
        times: dict[str, datetime] = {}
        for field in ("start", "end"):
            column = columns[field]
            text = _required(reader, table, (*loc, field), line, column, cells[field])
            if text is None:
                continue
            try:
                times[field] = _parse_local_datetime(text)
            except ValueError as error:
                reader.cell_error((*loc, field), table, line, column, str(error))
        if resource_id is None or len(times) != 2:
            continue
        try:
            windows.append(TimeWindow(resource_id=resource_id, **times))
        except ValidationError as error:
            message = "; ".join(detail["msg"] for detail in error.errors(include_url=False))
            reader.report(loc, f"{table.name} line {line}: {message}")
    return windows
