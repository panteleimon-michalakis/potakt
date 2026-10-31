"""What evaluators read: real instants, periods, scopes and who works what (ADR-0005 §1-3)."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from potakt.checker.scope import in_scope
from potakt.model import DecisionModel, Horizon, ProblemInstance, Resource, Scope, Solution, Task
from potakt.model.common import AttributeValue, Period
from potakt.model.data import TimeWindow, to_utc

# --- time ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TimedTask:
    """A task with its real start and end instants (UTC) and the date of its local start."""

    task: Task
    start: datetime
    end: datetime
    minutes: int
    day: date

    @property
    def id(self) -> str:
        return self.task.id

    def overlaps(self, start: datetime, end: datetime) -> bool:
        """Half-open intervals ``[start, end)``: touching is not overlapping."""
        return self.start < end and start < self.end


def timed(task: Task, zone: ZoneInfo) -> TimedTask:
    start, end = to_utc(task.start, zone), to_utc(task.end, zone)
    minutes = int((end - start).total_seconds()) // 60
    return TimedTask(task, start, end, minutes, task.start.date())


def rest_minutes(before: TimedTask, after: TimedTask) -> int:
    """Real minutes from the end of ``before`` to the start of ``after`` (negative: overlap)."""
    return int((after.start - before.end).total_seconds()) // 60


def to_minutes(hours: float) -> int:
    """Hour parameters are whole minutes (ADR-0004)."""
    return round(hours * 60)


def started_hours(minutes: int) -> int:
    """``ceil(minutes / 60)``: every started hour counts."""
    return -(-minutes // 60)


def format_minutes(minutes: int) -> str:
    """``8 h``, ``45 min`` or ``10 h 59 min``."""
    hours, rest = divmod(minutes, 60)
    if not rest:
        return f"{hours} h"
    return f"{rest} min" if not hours else f"{hours} h {rest} min"


# --- periods ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeriodSpan:
    """A day, an ISO week or the horizon: its label and its first and last dates."""

    label: str
    first: date
    last: date

    def contains(self, day: date) -> bool:
        return self.first <= day <= self.last


def interval_label(first: date, last: date) -> str:
    """An ISO 8601 interval of dates, ``2026-11-02/2026-11-08``."""
    return f"{first.isoformat()}/{last.isoformat()}"


def period_spans(period: Period, horizon: Horizon) -> list[PeriodSpan]:
    """The periods of the horizon: its days, the ISO weeks that intersect it, or itself."""
    match period:
        case "day":
            days = (horizon.end - horizon.start).days + 1
            dates = [horizon.start + timedelta(days=offset) for offset in range(days)]
            return [PeriodSpan(day.isoformat(), day, day) for day in dates]
        case "week":
            spans: list[PeriodSpan] = []
            monday = horizon.start - timedelta(days=horizon.start.weekday())
            while monday <= horizon.end:
                year, week, _ = monday.isocalendar()
                spans.append(PeriodSpan(f"{year}-W{week:02d}", monday, monday + timedelta(days=6)))
                monday += timedelta(weeks=1)
            return spans
        case "horizon":
            label = interval_label(horizon.start, horizon.end)
            return [PeriodSpan(label, horizon.start, horizon.end)]


def consecutive_runs(days: Iterable[date]) -> list[tuple[date, ...]]:
    """Maximal runs of consecutive dates, in order."""
    runs: list[list[date]] = []
    for day in sorted(set(days)):
        if runs and runs[-1][-1] + timedelta(days=1) == day:
            runs[-1].append(day)
        else:
            runs.append([day])
    return [tuple(run) for run in runs]


# --- the solution as facts -------------------------------------------------------------------


class Facts:
    """The instance and the solution indexed for evaluation; shared by every rule."""

    def __init__(self, dm: DecisionModel, instance: ProblemInstance, solution: Solution) -> None:
        self.zone = dm.zone
        self.horizon = dm.horizon
        self.windows = instance.windows
        self.resources = {resource.id: resource for resource in instance.resources}
        self.tasks = {task.id: timed(task, self.zone) for task in instance.tasks}
        self.resource_attributes: dict[str, dict[str, AttributeValue]] = {
            resource.id: {**resource.attributes, "id": resource.id}
            for resource in instance.resources
        }
        self.task_attributes: dict[str, dict[str, AttributeValue]] = {
            task.id: {**task.attributes, **task.derived_attributes(self.zone), "id": task.id}
            for task in instance.tasks
        }
        pairs = {(a.task_id, a.resource_id) for a in solution.assignments}
        unknown = sorted(
            f"({task_id!r}, {resource_id!r})"
            for task_id, resource_id in pairs
            if task_id not in self.tasks or resource_id not in self.resources
        )
        if unknown:
            raise ValueError(
                f"solution assigns unknown ids {', '.join(unknown)}; run validate_solution first"
            )
        self.assigned = frozenset(pairs)

    def view(self, scope: Scope) -> "RuleContext":
        resources = tuple(
            self.resources[resource_id]
            for resource_id in sorted(self.resources)
            if in_scope(scope.resources, self.resource_attributes[resource_id])
        )
        tasks = tuple(
            sorted(
                (
                    t
                    for t in self.tasks.values()
                    if in_scope(scope.tasks, self.task_attributes[t.id])
                ),
                key=lambda t: (t.start, t.id),
            )
        )
        return RuleContext(self, resources, tasks)


class RuleContext:
    """What one rule sees: its in-scope resources and tasks and the assignments between them."""

    def __init__(
        self, facts: Facts, resources: tuple[Resource, ...], tasks: tuple[TimedTask, ...]
    ) -> None:
        self.facts = facts
        self.resources = resources  # by id
        self.tasks = tasks  # by start, then id
        self._resource_ids = frozenset(resource.id for resource in resources)
        self._task_ids = frozenset(task.id for task in tasks)

    @property
    def horizon(self) -> Horizon:
        return self.facts.horizon

    def has_resource(self, resource_id: str) -> bool:
        return resource_id in self._resource_ids

    def has_task(self, task_id: str) -> bool:
        return task_id in self._task_ids

    def works(self, resource_id: str, task_id: str) -> bool:
        """The in-scope resource works the in-scope task."""
        return (
            self.has_resource(resource_id)
            and self.has_task(task_id)
            and (task_id, resource_id) in self.facts.assigned
        )

    def tasks_of(self, resource_id: str) -> tuple[TimedTask, ...]:
        """The in-scope tasks the resource works, by start; empty if it is out of scope."""
        return tuple(task for task in self.tasks if self.works(resource_id, task.id))

    def resources_of(self, task_id: str) -> tuple[str, ...]:
        """The in-scope resources working the task, by id."""
        return tuple(r.id for r in self.resources if self.works(r.id, task_id))

    def assignments(self) -> Iterator[tuple[Resource, TimedTask]]:
        """Every in-scope assignment, by resource id then task start."""
        for resource in self.resources:
            for task in self.tasks_of(resource.id):
                yield resource, task

    def windows(self, rule_id: str) -> tuple[TimeWindow, ...]:
        return self.facts.windows.get(rule_id, ())

    def instant(self, local: datetime) -> datetime:
        return to_utc(local, self.facts.zone)
