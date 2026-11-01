"""Vacuous constraints (TASK-005b D1): rules that no plan on the scenario data could violate."""

import json
from pathlib import Path
from typing import Any

import pytest
from checker_helpers import rule

from potakt.checker import EVALUATORS
from potakt.evaluation import LoadedModel, is_vacuous, load_model

# Tasks on Mon, Tue, Thu, Sat and Sun (longest run of days: 2); at most 2 tasks a day; 7 x 8 h =
# 56 h. mon-am and mon-pm touch (half-open, no overlap); tue-am and tue-mid overlap.
STAFF = """id,name,contract,max_hours,min_hours,skills
a1,Ana,full,40,0,opening;closing
a2,Ben,full,40,0,opening
a3,Cleo,part,24,10,closing
a4,Dev,part,16,8,
"""
SHIFTS = """id,start,end,demand,kind,needs
mon-am,2026-11-02T07:00,2026-11-02T15:00,1,am,opening
mon-pm,2026-11-02T15:00,2026-11-02T23:00,1,pm,closing
tue-am,2026-11-03T07:00,2026-11-03T15:00,1,am,opening
tue-mid,2026-11-03T10:00,2026-11-03T18:00,1,mid,
thu-am,2026-11-05T07:00,2026-11-05T15:00,1,am,
sat-am,2026-11-07T09:00,2026-11-07T17:00,1,am,
sun-am,2026-11-08T09:00,2026-11-08T17:00,1,am,
"""

FULL_TIME = {"resources": [{"attribute": "contract", "op": "eq", "value": "full"}]}
NIGHTS = {"tasks": [{"attribute": "kind", "op": "eq", "value": "night"}]}
WEEKEND = {"tasks": [{"attribute": "weekday", "op": "in", "value": ["sat", "sun"]}]}
SKILLS = {"task_attribute": "needs", "resource_attribute": "skills", "match": "contains"}
BASELINE = [{"task_id": "mon-am", "resource_id": "a1"}, {"task_id": "tue-am", "resource_id": "a2"}]


def only(*ids: str) -> dict[str, Any]:
    return {"resources": [{"attribute": "id", "op": "in", "value": list(ids)}]}


def kinds(*names: str) -> dict[str, Any]:
    return {"tasks": [{"attribute": "kind", "op": "in", "value": list(names)}]}


def per(measure: str, period: str, limit: object) -> dict[str, Any]:
    return {"measure": measure, "period": period, "limit": limit}


def window(day: str) -> dict[str, Any]:
    return {"windows": [{"resource_id": "a1", "start": f"{day}T00:00", "end": f"{day}T23:00"}]}


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "staff.csv").write_text(STAFF, encoding="utf-8")
    (tmp_path / "shifts.csv").write_text(SHIFTS, encoding="utf-8")
    return tmp_path


def loaded(rule_doc: dict[str, Any], directory: Path) -> LoadedModel:
    document = {
        "schema_version": "0.2",
        "id": "vacuity-test",
        "version": 1,
        "status": "draft",
        "title": "Vacuity test",
        "language": "en",
        "archetype": "assignment",
        "timezone": "Europe/London",
        "horizon": {"start": "2026-11-02", "end": "2026-11-08"},
        "data": {
            "resources": {
                "source": "staff.csv",
                "id": "id",
                "attributes": {
                    "contract": {"column": "contract", "type": "str"},
                    "max_hours": {"column": "max_hours", "type": "int"},
                    "min_hours": {"column": "min_hours", "type": "int"},
                    "skills": {"column": "skills", "type": "list", "separator": ";"},
                },
            },
            "tasks": {
                "source": "shifts.csv",
                "id": "id",
                "start": "start",
                "end": "end",
                "demand": "demand",
                "attributes": {
                    "kind": {"column": "kind", "type": "str"},
                    "needs": {"column": "needs", "type": "str"},
                },
            },
        },
        "constraints": [rule_doc]
        if rule_doc["template"] == "coverage"
        else [
            rule("coverage", id="coverage"),
            rule_doc,
        ],
    }
    result = load_model(json.dumps(document), directory)
    assert isinstance(result, LoadedModel), result
    return result


def vacuity(rule_doc: dict[str, Any], directory: Path) -> str | None:
    model = loaded(rule_doc, directory)
    (constraint,) = [c for c in model.dm.constraints if c.id == rule_doc["id"]]
    return is_vacuous(constraint, model.dm, model.instance)


# One vacuous case per template, with the reason it gives.
VACUOUS: dict[str, tuple[dict[str, Any], str]] = {
    "coverage": (rule("coverage", id="coverage", scope=NIGHTS), "no in-scope task"),
    "requires_attribute": (
        rule("requires_attribute", params=SKILLS, scope=only("a1")),
        "forbids no (resource, task) pair",
    ),
    "no_overlap": (rule("no_overlap", scope=kinds("am", "pm")), "no two in-scope tasks overlap"),
    "min_rest": (
        rule("min_rest", params={"hours": 12}, scope=kinds("am")),
        "no two in-scope tasks are less than 12 h apart",
    ),
    "max_per_period": (
        rule("max_per_period", params=per("count", "day", 2)),
        "no period has more in-scope tasks or hours than the lowest limit",
    ),
    "min_per_period": (
        rule("min_per_period", "soft", params=per("count", "week", 0)),
        "the limit is 0 for every in-scope resource",
    ),
    "max_consecutive_days": (
        rule("max_consecutive_days", params={"days": 2}),
        "no run of more than 2 consecutive days has in-scope tasks",
    ),
    "unavailable": (
        rule("unavailable", params=window("2026-11-04")),
        "forbids no (resource, task) pair",
    ),
    "preference": (
        rule("preference", "soft", params={"kind": "avoid"}, scope=NIGHTS),
        "no in-scope task",
    ),
    "balance": (
        rule("balance", "soft", params={"measure": "count"}, scope=only("a1")),
        "fewer than 2 in-scope resources",
    ),
    "together": (
        rule("together", "soft", params={"resource_ids": ["a1", "a3"]}, scope=FULL_TIME),
        "inert: a3 is not in scope",
    ),
    "apart": (
        rule("apart", params={"resource_ids": ["a4", "a1"]}, scope=FULL_TIME),
        "inert: a4 is not in scope",
    ),
    "stability": (
        rule("stability", "soft", params={"baseline": BASELINE}, scope=only("a3")),
        "the in-scope baseline is empty",
    ),
}

# One non-vacuous case per template.
NON_VACUOUS: dict[str, dict[str, Any]] = {
    "coverage": rule("coverage", id="coverage"),
    "requires_attribute": rule("requires_attribute", params=SKILLS),
    "no_overlap": rule("no_overlap"),
    "min_rest": rule("min_rest", params={"hours": 12}),
    "max_per_period": rule("max_per_period", params=per("count", "day", 1)),
    "min_per_period": rule("min_per_period", "soft", params=per("count", "week", 1)),
    "max_consecutive_days": rule("max_consecutive_days", params={"days": 1}),
    "unavailable": rule("unavailable", params=window("2026-11-03")),
    "preference": rule("preference", "soft", params={"kind": "prefer"}, scope=kinds("am")),
    "balance": rule("balance", "soft", params={"measure": "count"}),
    "together": rule("together", "soft", params={"resource_ids": ["a1", "a2"]}),
    "apart": rule("apart", params={"resource_ids": ["a1", "a2"]}),
    "stability": rule("stability", "soft", params={"baseline": BASELINE}),
}


def test_every_template_has_a_vacuous_and_a_non_vacuous_case() -> None:
    assert sorted(VACUOUS) == sorted(NON_VACUOUS) == sorted(EVALUATORS)


@pytest.mark.parametrize(("rule_doc", "reason"), VACUOUS.values(), ids=VACUOUS)
def test_a_vacuous_rule_gives_its_reason(data: Path, rule_doc: dict[str, Any], reason: str) -> None:
    assert vacuity(rule_doc, data) == reason


@pytest.mark.parametrize("rule_doc", NON_VACUOUS.values(), ids=NON_VACUOUS)
def test_a_rule_that_some_plan_violates_is_not_vacuous(
    data: Path, rule_doc: dict[str, Any]
) -> None:
    assert vacuity(rule_doc, data) is None


# Edge cases of the definitions.
EDGES: dict[str, tuple[dict[str, Any], str | None]] = {
    "no in-scope resource": (
        rule(
            "min_rest",
            params={"hours": 11},
            scope={"resources": [{"attribute": "contract", "op": "eq", "value": "temp"}]},
        ),
        "no in-scope resource",
    ),
    "coverage without in-scope resources is always violated, not vacuous": (
        rule("coverage", id="coverage", scope=only()),
        None,
    ),
    "min_per_period without in-scope tasks cannot tell plans apart": (
        rule("min_per_period", "soft", params=per("count", "week", 1), scope=NIGHTS),
        "no in-scope task",
    ),
    "overlapping tasks have no rest gap (the checker's min_rest)": (
        rule(
            "min_rest",
            params={"hours": 12},
            scope={"tasks": [{"attribute": "id", "op": "in", "value": ["tue-am", "tue-mid"]}]},
        ),
        "no two in-scope tasks are less than 12 h apart",
    ),
    "a rest gap equal to the minimum is enough": (
        rule("min_rest", params={"hours": 16}, scope=kinds("am")),
        "no two in-scope tasks are less than 16 h apart",
    ),
    "a rest gap one hour short is not": (
        rule("min_rest", params={"hours": 17}, scope=kinds("am")),
        None,
    ),
    "an hours limit equal to the week's total": (
        rule("max_per_period", params=per("hours", "week", 56)),
        "no period has more in-scope tasks or hours than the lowest limit",
    ),
    "an hours limit below the week's total": (
        rule("max_per_period", params=per("hours", "week", 55)),
        None,
    ),
    "per-resource limits that cover every period": (
        rule(
            "max_per_period", params=per("hours", "week", {"attribute": "max_hours"}), scope=WEEKEND
        ),
        "no period has more in-scope tasks or hours than the lowest limit",
    ),
    "a per-resource limit below one period's total": (
        rule("max_per_period", params=per("hours", "week", {"attribute": "max_hours"})),
        None,
    ),
    "per-resource minimums that are all 0": (
        rule(
            "min_per_period",
            "soft",
            params=per("hours", "week", {"attribute": "min_hours"}),
            scope=FULL_TIME,
        ),
        "the limit is 0 for every in-scope resource",
    ),
    "a per-resource minimum above 0": (
        rule("min_per_period", "soft", params=per("hours", "week", {"attribute": "min_hours"})),
        None,
    ),
    "a pair rule with both members in scope but no task": (
        rule("apart", params={"resource_ids": ["a1", "a2"]}, scope=NIGHTS),
        "no in-scope task",
    ),
    "a baseline pair whose task is out of scope": (
        rule("stability", "soft", params={"baseline": BASELINE[:1]}, scope=kinds("pm")),
        "the in-scope baseline is empty",
    ),
}


@pytest.mark.parametrize(("rule_doc", "reason"), EDGES.values(), ids=EDGES)
def test_edge_cases_follow_the_checker_semantics(
    data: Path, rule_doc: dict[str, Any], reason: str | None
) -> None:
    assert vacuity(rule_doc, data) == reason
