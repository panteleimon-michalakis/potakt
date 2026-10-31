"""One evaluator per catalog template: what counts as a violation and how large it is.

This module is the readable definition of ADR-0005 §4. Each evaluator receives its rule and a
``RuleContext`` that already hides everything outside the rule's scope, and returns the rule's
violations (each with a positive integer ``amount``) in any order; scoring sorts and sums them.
"""

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Any

from potakt.checker.context import (
    PeriodSpan,
    RuleContext,
    TimedTask,
    consecutive_runs,
    format_minutes,
    interval_label,
    period_spans,
    rest_minutes,
    started_hours,
    to_minutes,
)
from potakt.checker.result import Violation
from potakt.model import (
    Apart,
    AttributeRef,
    Balance,
    Coverage,
    MaxConsecutiveDays,
    MaxPerPeriod,
    MinPerPeriod,
    MinRest,
    NoOverlap,
    Preference,
    RequiresAttribute,
    Resource,
    Stability,
    Together,
    Unavailable,
)
from potakt.model.catalog import PeriodLimitParams
from potakt.model.common import AttributeValue, Measure
from potakt.model.scope import contains_all, values_equal

Evaluator = Callable[[Any, RuleContext], list[Violation]]


def _ids(tasks: list[TimedTask] | tuple[TimedTask, ...]) -> tuple[str, ...]:
    return tuple(sorted(task.id for task in tasks))


def _shown(value: AttributeValue) -> str:
    return json.dumps(value, ensure_ascii=False)


def coverage(rule: Coverage, ctx: RuleContext) -> list[Violation]:
    """Per task: how far the number of assigned resources is from ``demand``."""
    violations: list[Violation] = []
    for task in ctx.tasks:
        assigned, demand = len(ctx.resources_of(task.id)), task.task.demand
        if assigned != demand:
            message = f"task {task.id}: {assigned} assigned, demand {demand}"
            violations.append(Violation(None, (task.id,), None, abs(assigned - demand), message))
    return violations


def requires_attribute(rule: RequiresAttribute, ctx: RuleContext) -> list[Violation]:
    """Per assignment whose resource does not match the task's requirement."""
    p = rule.params
    violations: list[Violation] = []
    for resource, task in ctx.assignments():
        required = task.task.attributes.get(p.task_attribute)
        if required is None or required == []:
            continue  # an absent or empty requirement restricts nobody (ADR-0006 §3)
        offered = resource.attributes.get(p.resource_attribute, [])
        matches = (
            contains_all(offered, required)
            if p.match == "contains"
            else values_equal(offered, required)
        )
        if not matches:
            verb = "contain" if p.match == "contains" else "equal"
            message = (
                f"{resource.id} on {task.id}: {p.resource_attribute} does not {verb} "
                f"{p.task_attribute} {_shown(required)}"
            )
            violations.append(Violation(resource.id, (task.id,), None, 1, message))
    return violations


def no_overlap(rule: NoOverlap, ctx: RuleContext) -> list[Violation]:
    """Per resource, per pair of tasks it works that overlap in time."""
    violations: list[Violation] = []
    for resource in ctx.resources:
        for a, b in combinations(ctx.tasks_of(resource.id), 2):
            if a.overlaps(b.start, b.end):
                first, second = _ids([a, b])
                message = f"{resource.id}: {first} overlaps {second}"
                violations.append(Violation(resource.id, (first, second), None, 1, message))
    return violations


def min_rest(rule: MinRest, ctx: RuleContext) -> list[Violation]:
    """Per resource, per pair ``a`` before ``b`` with a rest gap shorter than ``hours``."""
    minimum = to_minutes(rule.params.hours)
    violations: list[Violation] = []
    for resource in ctx.resources:
        for before, after in permutations(ctx.tasks_of(resource.id), 2):
            gap = rest_minutes(before, after)
            if 0 <= gap < minimum:
                message = (
                    f"{resource.id}: {format_minutes(gap)} rest between {before.id} and "
                    f"{after.id}, minimum {format_minutes(minimum)}"
                )
                violations.append(Violation(resource.id, _ids([before, after]), None, 1, message))
    return violations


def _measured(tasks: tuple[TimedTask, ...], measure: Measure) -> int:
    """The number of tasks (``count``) or their total real minutes (``hours``)."""
    return len(tasks) if measure == "count" else sum(task.minutes for task in tasks)


@dataclass(frozen=True, slots=True)
class _PeriodTotal:
    resource: Resource
    period: PeriodSpan
    tasks: tuple[TimedTask, ...]
    value: int  # count, or minutes for ``hours``
    limit: int  # count, or minutes for ``hours``


def _limit_of(params: PeriodLimitParams, resource: Resource) -> int:
    """The resource's limit in tasks (``count``) or minutes (``hours``): the literal, or its
    own value of the named attribute (ADR-0006 §5)."""
    limit = params.limit
    if isinstance(limit, AttributeRef):
        value = resource.attributes.get(limit.attribute)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(
                f"resource {resource.id!r} has no numeric {limit.attribute!r}; "
                "run validate_data first"
            )
        limit = value
    return int(limit) if params.measure == "count" else to_minutes(limit)


def _period_totals(rule: MaxPerPeriod | MinPerPeriod, ctx: RuleContext) -> Iterator[_PeriodTotal]:
    """The measure of every in-scope resource in every period of the horizon."""
    p = rule.params
    for resource in ctx.resources:
        limit = _limit_of(p, resource)
        worked = ctx.tasks_of(resource.id)
        for period in period_spans(p.period, ctx.horizon):
            tasks = tuple(task for task in worked if period.contains(task.day))
            yield _PeriodTotal(resource, period, tasks, _measured(tasks, p.measure), limit)


def _period_violation(rule: MaxPerPeriod | MinPerPeriod, total: _PeriodTotal) -> Violation:
    """A violation sized by the distance between the value and the limit."""
    distance = abs(total.value - total.limit)
    bound = "limit" if isinstance(rule, MaxPerPeriod) else "minimum"
    if rule.params.measure == "count":
        amount, shown = distance, f"{total.value} tasks, {bound} {total.limit}"
    else:
        amount = started_hours(distance)
        shown = f"{format_minutes(total.value)} worked, {bound} {format_minutes(total.limit)}"
    message = f"{total.resource.id} in {total.period.label}: {shown}"
    return Violation(total.resource.id, _ids(total.tasks), total.period.label, amount, message)


def max_per_period(rule: MaxPerPeriod, ctx: RuleContext) -> list[Violation]:
    """Per resource, per period in which it works more than ``limit``."""
    totals = _period_totals(rule, ctx)
    return [_period_violation(rule, total) for total in totals if total.value > total.limit]


def min_per_period(rule: MinPerPeriod, ctx: RuleContext) -> list[Violation]:
    """Per resource, per period (even one without work) in which it works less than ``limit``."""
    totals = _period_totals(rule, ctx)
    return [_period_violation(rule, total) for total in totals if total.value < total.limit]


def max_consecutive_days(rule: MaxConsecutiveDays, ctx: RuleContext) -> list[Violation]:
    """Per resource, per run of consecutive working days longer than ``days``."""
    limit = rule.params.days
    violations: list[Violation] = []
    for resource in ctx.resources:
        worked = ctx.tasks_of(resource.id)
        for run in consecutive_runs(task.day for task in worked):
            if len(run) > limit:
                first, last = run[0], run[-1]
                tasks = [task for task in worked if first <= task.day <= last]
                message = (
                    f"{resource.id}: {len(run)} consecutive working days {first} to {last}, "
                    f"limit {limit}"
                )
                label = interval_label(first, last)
                violations.append(
                    Violation(resource.id, _ids(tasks), label, len(run) - limit, message)
                )
    return violations


def unavailable(rule: Unavailable, ctx: RuleContext) -> list[Violation]:
    """Per assignment overlapping one of the resource's windows (counted once)."""
    violations: list[Violation] = []
    for resource, task in ctx.assignments():
        hit = next(
            (
                window
                for window in ctx.windows(rule.id)
                if window.resource_id == resource.id
                and task.overlaps(ctx.instant(window.start), ctx.instant(window.end))
            ),
            None,
        )
        if hit is not None:
            start = hit.start.isoformat(timespec="minutes")
            end = hit.end.isoformat(timespec="minutes")
            message = f"{resource.id} on {task.id}: overlaps unavailability {start} to {end}"
            violations.append(Violation(resource.id, (task.id,), None, 1, message))
    return violations


def preference(rule: Preference, ctx: RuleContext) -> list[Violation]:
    """Per in-scope assignment; for ``prefer`` scoring turns the amount into a reward."""
    kind = "preferred" if rule.params.kind == "prefer" else "avoided"
    return [
        Violation(
            resource.id, (task.id,), None, 1, f"{resource.id} on {task.id}: {kind} assignment"
        )
        for resource, task in ctx.assignments()
    ]


def balance(rule: Balance, ctx: RuleContext) -> list[Violation]:
    """Once: the spread (max - min) of the measure over all in-scope resources, zeros included."""
    measure = rule.params.measure
    count = measure == "count"
    values = {r.id: _measured(ctx.tasks_of(r.id), measure) for r in ctx.resources}
    if not values:
        return []
    low, high = min(values.values()), max(values.values())
    amount = high - low if count else started_hours(high - low)
    if amount == 0:
        return []

    def shown(value: int) -> str:
        holders = ", ".join(rid for rid, v in values.items() if v == value)
        return f"{value if count else format_minutes(value)} ({holders})"

    message = (
        f"{'tasks' if count else 'hours'} per resource range from {shown(low)} to {shown(high)}"
    )
    return [Violation(None, (), None, amount, message)]


def together(rule: Together, ctx: RuleContext) -> list[Violation]:
    """Per in-scope task worked by exactly one of the two resources."""
    first, second = rule.params.resource_ids
    if not (ctx.has_resource(first) and ctx.has_resource(second)):
        return []
    violations: list[Violation] = []
    for task in ctx.tasks:
        if ctx.works(first, task.id) != ctx.works(second, task.id):
            alone, other = (first, second) if ctx.works(first, task.id) else (second, first)
            message = f"{alone} works {task.id} without {other}"
            violations.append(Violation(alone, (task.id,), None, 1, message))
    return violations


def apart(rule: Apart, ctx: RuleContext) -> list[Violation]:
    """Per in-scope task worked by both resources."""
    first, second = rule.params.resource_ids
    return [
        Violation(None, (task.id,), None, 1, f"{first} and {second} both work {task.id}")
        for task in ctx.tasks
        if ctx.works(first, task.id) and ctx.works(second, task.id)
    ]


def stability(rule: Stability, ctx: RuleContext) -> list[Violation]:
    """Per in-scope baseline pair (each pair once) that the solution no longer contains."""
    baseline = dict.fromkeys((entry.task_id, entry.resource_id) for entry in rule.params.baseline)
    return [
        Violation(
            resource_id, (task_id,), None, 1, f"{resource_id} no longer works {task_id} (baseline)"
        )
        for task_id, resource_id in baseline
        if ctx.has_task(task_id)
        and ctx.has_resource(resource_id)
        and not ctx.works(resource_id, task_id)
    ]


EVALUATORS: Mapping[str, Evaluator] = {
    "coverage": coverage,
    "requires_attribute": requires_attribute,
    "no_overlap": no_overlap,
    "min_rest": min_rest,
    "max_per_period": max_per_period,
    "min_per_period": min_per_period,
    "max_consecutive_days": max_consecutive_days,
    "unavailable": unavailable,
    "preference": preference,
    "balance": balance,
    "together": together,
    "apart": apart,
    "stability": stability,
}
