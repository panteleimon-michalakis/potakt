"""Validation issues, their conversion from Pydantic errors and data-level checks (ADR-0004)."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from potakt.model.catalog import (
    Apart,
    Constraint,
    RequiresAttribute,
    Stability,
    Together,
    Unavailable,
)
from potakt.model.common import CUSTOM_ERROR_TYPE, DERIVED_TASK_ATTRIBUTES, Loc
from potakt.model.data import ProblemInstance, Task, TimeWindow, local_time_exists
from potakt.model.decision_model import DecisionModel

MAX_TASK_MINUTES = 24 * 60
# Error types whose location names a field of the document even when it is absent there.
_ABSENT_FIELD_ERRORS = frozenset({"missing", CUSTOM_ERROR_TYPE})


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem in a Decision Model: a JSON pointer into the DM and a readable message."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


class DecisionModelError(Exception):
    """Raised when a Decision Model or its data cannot be loaded; carries every issue found."""

    def __init__(self, issues: Iterable[ValidationIssue]) -> None:
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


def json_pointer(parts: Iterable[str | int]) -> str:
    """Render path segments as an RFC 6901 JSON pointer ("/" for the document root)."""
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def issue_at(loc: Loc, message: str) -> ValidationIssue:
    return ValidationIssue(json_pointer(loc), message)


# --- structural issues from Pydantic ---------------------------------------------------------


def issues_from_pydantic(error: ValidationError, document: object) -> list[ValidationIssue]:
    """Convert Pydantic errors into issues whose paths point into ``document``.

    Pydantic locations also contain union tags (the ``template`` of a constraint) and union
    member names; walking the input document drops those so the path is a real JSON pointer.
    Messages of errors that land on the same path (alternatives of a union) are merged.
    """
    merged: dict[str, list[str]] = {}
    for detail in error.errors(include_url=False):
        path = json_pointer(_document_path(detail["loc"], detail["type"], document))
        messages = merged.setdefault(path, [])
        if detail["msg"] not in messages:
            messages.append(detail["msg"])
    return [ValidationIssue(path, "; ".join(messages)) for path, messages in merged.items()]


def _document_path(loc: Loc, error_type: str, document: object) -> list[str | int]:
    parts: list[str | int] = []
    node: object = document
    for position, item in enumerate(loc):
        is_last = position == len(loc) - 1
        if isinstance(node, dict):
            mapping = cast("dict[str, Any]", node)
            if isinstance(item, str) and item in mapping:
                parts.append(item)
                node = mapping[item]
            elif mapping.get("template") == item:
                continue  # discriminated-union tag
            elif is_last and error_type in _ABSENT_FIELD_ERRORS:
                parts.append(item)  # a required field that is absent from the document
                node = None
        elif isinstance(node, list) and isinstance(item, int):
            items = cast("list[Any]", node)
            if 0 <= item < len(items):
                parts.append(item)
                node = items[item]
        # Anything else is an internal marker (union member, validator name): not in the document.
    return parts


# --- data-level validation -------------------------------------------------------------------


def validate_data(dm: DecisionModel, instance: ProblemInstance) -> list[ValidationIssue]:
    """Data-level checks of a loaded instance against its DM (ADR-0004, level 2).

    Ids are unique, task times are valid and start inside the horizon, and every attribute,
    resource id and task id referenced by scopes and parameters exists in the data.
    """
    resources, tasks = dm.data.resources, dm.data.tasks
    issues = _duplicate_ids("resource", resources.source, [r.id for r in instance.resources])
    issues += _duplicate_ids("task", tasks.source, [t.id for t in instance.tasks])
    for task in instance.tasks:
        issues += _task_time_issues(dm, task)
    references = _References(
        zone=dm.zone,
        timezone=dm.timezone,
        resource_ids=frozenset(r.id for r in instance.resources),
        task_ids=frozenset(t.id for t in instance.tasks),
        resource_attributes=frozenset(resources.attributes),
        task_attributes=frozenset(tasks.attributes),
    )
    for index, rule in enumerate(dm.constraints):
        issues += references.check(("constraints", index), rule, instance)
    return issues


def _duplicate_ids(kind: str, source: str, ids: list[str]) -> list[ValidationIssue]:
    seen: set[str] = set()
    reported: set[str] = set()
    issues: list[ValidationIssue] = []
    for item_id in ids:
        if item_id in seen and item_id not in reported:
            message = f"duplicate {kind} id {item_id!r} in {source}"
            issues.append(issue_at(("data", f"{kind}s", "id"), message))
            reported.add(item_id)
        seen.add(item_id)
    return issues


def _format_time(value: datetime) -> str:
    return value.isoformat(timespec="minutes")


def _dst_gap(label: str, value: datetime, timezone: str) -> str:
    return f"{label}: local time {_format_time(value)} does not exist in {timezone} (DST change)"


def _task_time_issues(dm: DecisionModel, task: Task) -> list[ValidationIssue]:
    zone, label = dm.zone, f"task {task.id!r}"
    issues = [
        issue_at(("data", "tasks", field), _dst_gap(label, value, dm.timezone))
        for field, value in (("start", task.start), ("end", task.end))
        if not local_time_exists(value, zone)
    ]
    if issues:
        return issues
    minutes = task.duration_minutes(zone)
    if minutes <= 0:
        start, end = _format_time(task.start), _format_time(task.end)
        message = f"{label}: end {end} must be after start {start}"
        issues.append(issue_at(("data", "tasks", "end"), message))
    elif minutes > MAX_TASK_MINUTES:
        message = f"{label}: lasts {minutes / 60:g} h, more than the maximum of 24 h"
        issues.append(issue_at(("data", "tasks", "end"), message))
    day, horizon = task.start.date(), dm.horizon
    if not horizon.start <= day <= horizon.end:
        message = f"{label}: starts on {day}, outside the horizon {horizon.start} to {horizon.end}"
        issues.append(issue_at(("data", "tasks", "start"), message))
    return issues


def _unknown_attribute(kind: str, name: str, known: frozenset[str]) -> str:
    return f"unknown {kind} attribute {name!r}; known: {', '.join(sorted(known))}"


@dataclass(frozen=True)
class _References:
    """What scopes and parameters may refer to in one loaded instance."""

    zone: ZoneInfo
    timezone: str
    resource_ids: frozenset[str]
    task_ids: frozenset[str]
    resource_attributes: frozenset[str]
    task_attributes: frozenset[str]

    def check(
        self, base: Loc, rule: Constraint, instance: ProblemInstance
    ) -> list[ValidationIssue]:
        issues = self._scope(base, rule)
        params: Loc = (*base, "params")
        match rule:
            case RequiresAttribute():
                issues += self._attribute(
                    (*params, "task_attribute"), "task", rule.params.task_attribute
                )
                issues += self._attribute(
                    (*params, "resource_attribute"), "resource", rule.params.resource_attribute
                )
            case Together() | Apart():
                for k, resource_id in enumerate(rule.params.resource_ids):
                    issues += self._resource((*params, "resource_ids", k), resource_id)
            case Stability():
                for k, assignment in enumerate(rule.params.baseline):
                    entry: Loc = (*params, "baseline", k)
                    issues += self._task((*entry, "task_id"), assignment.task_id)
                    issues += self._resource((*entry, "resource_id"), assignment.resource_id)
            case Unavailable():
                issues += self._windows(params, rule, instance)
            case _:
                pass
        return issues

    def _scope(self, base: Loc, rule: Constraint) -> list[ValidationIssue]:
        resource_known = self.resource_attributes | {"id"}
        task_known = self.task_attributes | DERIVED_TASK_ATTRIBUTES | {"id"}
        issues: list[ValidationIssue] = []
        for kind, conditions, known in (
            ("resource", rule.scope.resources, resource_known),
            ("task", rule.scope.tasks, task_known),
        ):
            for j, condition in enumerate(conditions):
                if condition.attribute not in known:
                    loc: Loc = (*base, "scope", f"{kind}s", j, "attribute")
                    message = _unknown_attribute(kind, condition.attribute, known)
                    issues.append(issue_at(loc, message))
        return issues

    def _attribute(self, loc: Loc, kind: str, name: str) -> list[ValidationIssue]:
        known = self.task_attributes if kind == "task" else self.resource_attributes
        return [] if name in known else [issue_at(loc, _unknown_attribute(kind, name, known))]

    def _resource(self, loc: Loc, resource_id: str, where: str = "") -> list[ValidationIssue]:
        if resource_id in self.resource_ids:
            return []
        return [issue_at(loc, f"unknown resource id {resource_id!r}{where}")]

    def _task(self, loc: Loc, task_id: str) -> list[ValidationIssue]:
        return [] if task_id in self.task_ids else [issue_at(loc, f"unknown task id {task_id!r}")]

    def _windows(
        self, params: Loc, rule: Unavailable, instance: ProblemInstance
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        inline = rule.params.windows
        for k, window in enumerate(inline):
            loc: Loc = (*params, "windows", k)
            issues += self._resource((*loc, "resource_id"), window.resource_id)
            issues += self._window_times(loc, window, f"window of {window.resource_id!r}")
        source = rule.params.source
        if source is not None:
            # ProblemInstance.windows lists the inline windows first, then the CSV ones.
            for window in instance.windows.get(rule.id, ())[len(inline) :]:
                loc = (*params, "source")
                where = f" in {source.source}"
                issues += self._resource((*loc, "resource_id"), window.resource_id, where)
                label = f"window of {window.resource_id!r}{where}"
                issues += self._window_times(loc, window, label)
        return issues

    def _window_times(self, loc: Loc, window: TimeWindow, label: str) -> list[ValidationIssue]:
        return [
            issue_at((*loc, field), _dst_gap(label, value, self.timezone))
            for field, value in (("start", window.start), ("end", window.end))
            if not local_time_exists(value, self.zone)
        ]
