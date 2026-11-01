"""Predicates for the mandatory unsupported requests of the corpus (ADR-0007 §4).

A request that the catalog cannot express is not encoded in the truth. When the request states it
as mandatory, the reference solution must still honour it, so a faithful plan is possible. Each
function returns the breaches of ``solution.json`` as readable strings; an empty list means the
reference honours the request. ``None`` marks a scenario whose unsupported requests are all wishes.
"""

import csv
import math
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, time

from potakt.evaluation import LoadedScenario

FRIDAY = 4  # datetime.weekday(): Monday 0 ... Sunday 6


def assigned(scenario: LoadedScenario) -> dict[str, set[str]]:
    """Task id -> the resource ids that ``solution.json`` assigns to it."""
    people: dict[str, set[str]] = defaultdict(set)
    for assignment in scenario.solution.assignments:
        people[assignment.task_id].add(assignment.resource_id)
    return people


def spanish_on_every_weekday_shift(scenario: LoadedScenario) -> list[str]:
    # "We also need at least one Spanish speaker on every weekday shift." (weekend shifts excluded)
    speakers = {
        r.id
        for r in scenario.instance.resources
        if isinstance(skills := r.attributes.get("skills"), list) and "spanish" in skills
    }
    people = assigned(scenario)
    return [
        f"{task.id}: no assigned resource has the skill spanish"
        for task in scenario.instance.tasks
        if task.start.weekday() <= FRIDAY and not people[task.id] & speakers
    ]


def licence_holder_on_every_evening_floor(scenario: LoadedScenario) -> list[str]:
    # "every evening we need at least one personal licence holder on the floor for the bar"
    # The truth does not bind the column, so staff.csv is read here; our code fixes the encoding.
    with (scenario.directory / "staff.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    holders = {row["staff_id"] for row in rows if row["personal_licence"] == "yes"}
    people = assigned(scenario)
    return [
        f"{task.id}: no assigned resource holds a personal licence"
        for task in scenario.instance.tasks
        if task.attributes.get("area") == "floor"
        and task.attributes.get("service") == "dinner"
        and not people[task.id] & holders
    ]


NURSERY_CHILDREN = {"βρεφικό": 14, "νηπιακό": 22}
NURSERY_WINDOW = (time(9, 0), time(14, 0))
CHILDREN_PER_ADULT = 8


def one_adult_per_eight_children(scenario: LoadedScenario) -> list[str]:
    # The request (RUF003 flags two short Greek words made only of Latin look-alike letters):
    # «Θα ήθελα επίσης να τηρείται σε κάθε τμήμα η αναλογία ενός ενήλικα  # noqa: RUF003
    # για κάθε οκτώ παιδιά, όσο είναι μέσα όλα τα παιδιά, δηλαδή από τις 9:00 έως  # noqa: RUF003
    # τις 14:00· φέτος έχουμε 14 βρέφη και 22 νήπια.»
    people = assigned(scenario)
    breaches: list[str] = []
    for section, children in NURSERY_CHILDREN.items():
        needed = math.ceil(children / CHILDREN_PER_ADULT)
        tasks = [t for t in scenario.instance.tasks if t.attributes.get("section") == section]
        for day in sorted({t.start.date() for t in tasks if t.start.weekday() <= FRIDAY}):
            opens, closes = (datetime.combine(day, moment) for moment in NURSERY_WINDOW)
            # The count can only change at a task start or end, so these instants are enough.
            instants = {opens} | {
                moment for t in tasks for moment in (t.start, t.end) if opens <= moment < closes
            }
            for instant in sorted(instants):
                present = {r for t in tasks if t.start <= instant < t.end for r in people[t.id]}
                if len(present) < needed:
                    breaches.append(
                        f"{section} {instant.isoformat(timespec='minutes')}: "
                        f"{len(present)} adults for {children} children, {needed} needed"
                    )
    return breaches


CHECKS: dict[str, Callable[[LoadedScenario], list[str]] | None] = {
    "03-call-centre": spanish_on_every_weekday_shift,
    "04-restaurant": licence_holder_on_every_evening_floor,
    "06-ambulance-crews": None,  # the night-off rule is a wish (TASK-005b review, F2)
    "10-municipal-nursery": one_adult_per_eight_children,
}
