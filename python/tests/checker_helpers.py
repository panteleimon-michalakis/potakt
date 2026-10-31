"""In-memory builders for checker tests: DMs, instances and solutions without CSV files."""

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from helpers import minimal_dm

from potakt.checker import CheckResult, RuleResult, check
from potakt.model import (
    Assignment,
    DecisionModel,
    ProblemInstance,
    Resource,
    Solution,
    Task,
    TimeWindow,
    Unavailable,
    parse_decision_model,
)
from potakt.model.common import AttributeValue

WEEK = ("2026-11-02", "2026-11-08")  # Mon-Sun, ISO week 45
DST_WEEK = ("2026-10-19", "2026-10-25")  # Mon-Sun, ISO week 43; Athens falls back on 25 Oct


def rule(
    template: str, hardness: str = "hard", weight: int | None = None, **fields: Any
) -> dict[str, Any]:
    """A constraint document; ``weight`` defaults to 1 for soft rules."""
    doc: dict[str, Any] = {
        "id": fields.pop("id", template),
        "template": template,
        "hardness": hardness,
        "rationale": "test rule",
        "provenance": {"kind": "default"},
    }
    if hardness == "soft":
        doc["weight"] = 1 if weight is None else weight
    doc.update(fields)
    return doc


def dm_document(
    *rules: dict[str, Any],
    horizon: tuple[str, str] = WEEK,
    timezone: str = "Europe/Athens",
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A DM document: minimal data bindings, a coverage rule (or ``coverage``) and ``rules``."""
    doc = minimal_dm()
    doc["timezone"] = timezone
    doc["horizon"] = {"start": horizon[0], "end": horizon[1]}
    if coverage is not None:
        doc["constraints"] = [coverage]
    doc["constraints"].extend(rules)
    return doc


def make_dm(
    *rules: dict[str, Any],
    horizon: tuple[str, str] = WEEK,
    timezone: str = "Europe/Athens",
    coverage: dict[str, Any] | None = None,
) -> DecisionModel:
    """The DM of ``dm_document``, structurally validated."""
    doc = dm_document(*rules, horizon=horizon, timezone=timezone, coverage=coverage)
    return parse_decision_model(json.dumps(doc, ensure_ascii=False))


def res(resource_id: str, **attributes: AttributeValue) -> Resource:
    return Resource(id=resource_id, attributes=attributes)


def task(task_id: str, start: str, end: str, demand: int = 1, **attributes: AttributeValue) -> Task:
    return Task(
        id=task_id,
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        demand=demand,
        attributes=attributes,
    )


def make_instance(
    dm: DecisionModel, resources: Sequence[Resource], tasks: Sequence[Task]
) -> ProblemInstance:
    """An instance whose windows are the inline windows of the DM's ``unavailable`` rules."""
    windows: dict[str, tuple[TimeWindow, ...]] = {
        rule.id: tuple(rule.params.windows)
        for rule in dm.constraints
        if isinstance(rule, Unavailable)
    }
    return ProblemInstance(resources=tuple(resources), tasks=tuple(tasks), windows=windows)


def make_solution(
    dm: DecisionModel, assignments: Mapping[str, Sequence[str]] | Sequence[tuple[str, str]]
) -> Solution:
    """A solution from ``{task_id: [resource_id, ...]}`` or ``[(task_id, resource_id), ...]``."""
    pairs = (
        [(t, r) for t, rs in assignments.items() for r in rs]
        if isinstance(assignments, Mapping)
        else list(assignments)
    )
    return Solution(
        schema_version="0.1",
        model_id=dm.id,
        model_version=dm.version,
        assignments=tuple(Assignment(task_id=t, resource_id=r) for t, r in pairs),
    )


def run(
    dm: DecisionModel,
    resources: Sequence[Resource],
    tasks: Sequence[Task],
    assignments: Mapping[str, Sequence[str]] | Sequence[tuple[str, str]],
) -> CheckResult:
    return check(dm, make_instance(dm, resources, tasks), make_solution(dm, assignments))


def outcome(result: RuleResult) -> list[tuple[str | None, tuple[str, ...], str | None, int]]:
    """The violations of one rule as ``(resource_id, task_ids, period, amount)`` tuples."""
    return [(v.resource_id, v.task_ids, v.period, v.amount) for v in result.violations]
