"""Vacuous constraints (TASK-005b D1): rules that no plan on the scenario data could violate.

A truth rule without any effect on the data would match every other rule without effect, so the
corpus keeps such rules out of every truth. The checks follow the checker's semantics (ADR-0005
§4, ADR-0006 §3): a rule is vacuous when no plan, not even one that puts every resource on every
task, gives it a violation.
"""

from itertools import combinations, permutations

from potakt.checker.context import (
    TimedTask,
    consecutive_runs,
    format_minutes,
    period_spans,
    rest_minutes,
    timed,
    to_minutes,
)
from potakt.evaluation.compare import effective_limit, forbidden_pairs, scope_extension
from potakt.model import (
    Apart,
    Balance,
    Constraint,
    DecisionModel,
    MaxConsecutiveDays,
    MaxPerPeriod,
    MinPerPeriod,
    MinRest,
    NoOverlap,
    ProblemInstance,
    RequiresAttribute,
    Stability,
    Together,
    Unavailable,
)


def is_vacuous(constraint: Constraint, dm: DecisionModel, instance: ProblemInstance) -> str | None:
    """Why no plan on the data could violate ``constraint``, or ``None`` if some plan does.

    The inputs must be valid (``validate_data`` reports no issues). A rule without in-scope
    resources or tasks is vacuous; a ``min_per_period`` rule without in-scope tasks is reported
    too, although every plan violates it, because it cannot tell two plans apart. ``coverage``
    needs only an in-scope task: every demand is at least 1.
    """
    resources, task_ids = scope_extension(constraint, dm, instance)
    if not task_ids:
        return "no in-scope task"
    if constraint.template == "coverage":
        return None
    if not resources:
        return "no in-scope resource"
    tasks = [timed(task, dm.zone) for task in instance.tasks if task.id in task_ids]
    match constraint:
        case RequiresAttribute() | Unavailable():
            if forbidden_pairs(constraint, dm, instance):
                return None
            return "forbids no (resource, task) pair"
        case NoOverlap():
            if any(a.overlaps(b.start, b.end) for a, b in combinations(tasks, 2)):
                return None
            return "no two in-scope tasks overlap"
        case MinRest():
            minimum = to_minutes(constraint.params.hours)
            if any(0 <= rest_minutes(a, b) < minimum for a, b in permutations(tasks, 2)):
                return None
            return f"no two in-scope tasks are less than {format_minutes(minimum)} apart"
        case MaxPerPeriod():
            lowest = min(_limits(constraint, instance, resources))
            if _highest_period_total(constraint, dm, tasks) > lowest:
                return None
            return "no period has more in-scope tasks or hours than the lowest limit"
        case MinPerPeriod():
            if any(_limits(constraint, instance, resources)):
                return None
            return "the limit is 0 for every in-scope resource"
        case MaxConsecutiveDays():
            days = constraint.params.days
            if max(len(run) for run in consecutive_runs(task.day for task in tasks)) > days:
                return None
            return f"no run of more than {days} consecutive days has in-scope tasks"
        case Balance():
            return None if len(resources) >= 2 else "fewer than 2 in-scope resources"
        case Together() | Apart():
            outside = [rid for rid in constraint.params.resource_ids if rid not in resources]
            if not outside:
                return None
            return (
                f"inert: {', '.join(outside)} {'is' if len(outside) == 1 else 'are'} not in scope"
            )
        case Stability():
            if any(
                entry.task_id in task_ids and entry.resource_id in resources
                for entry in constraint.params.baseline
            ):
                return None
            return "the in-scope baseline is empty"
        case _:  # preference: every in-scope assignment counts
            return None


def _limits(
    rule: MaxPerPeriod | MinPerPeriod, instance: ProblemInstance, resources: frozenset[str]
) -> list[int]:
    """The effective limit of every in-scope resource (tasks, or minutes for ``hours``)."""
    return [
        effective_limit(rule, resource.attributes)
        for resource in instance.resources
        if resource.id in resources
    ]


def _highest_period_total(rule: MaxPerPeriod, dm: DecisionModel, tasks: list[TimedTask]) -> int:
    """The largest number of in-scope tasks (``count``) or minutes (``hours``) in one period."""
    count = rule.params.measure == "count"
    return max(
        sum(1 if count else task.minutes for task in tasks if span.contains(task.day))
        for span in period_spans(rule.params.period, dm.horizon)
    )
