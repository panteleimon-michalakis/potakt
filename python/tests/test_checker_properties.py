"""Properties of the checker on small random instances (DESIGN §8.3, TASK-003)."""

import copy
import json
from datetime import datetime, timedelta
from typing import Any

from checker_helpers import dm_document, res, rule, run, task
from hypothesis import given, settings
from hypothesis import strategies as st

from potakt.model import DecisionModel, Resource, Task, parse_decision_model

# Fri 23 to Mon 26 Oct 2026: includes the Europe/Athens fall-back night (25 Oct, 04:00 -> 03:00).
HORIZON = ("2026-10-23", "2026-10-26")
FIRST_DAY = datetime(2026, 10, 23)
MONOTONE = ("no_overlap", "min_rest", "apart", "unavailable")

Pair = tuple[str, str]
Case = tuple[dict[str, Any], list[Resource], list[Task], list[Pair]]


def local(moment: datetime) -> str:
    return moment.isoformat(timespec="minutes")


@st.composite
def resources(draw: st.DrawFn) -> list[Resource]:
    count = draw(st.integers(2, 4))
    return [
        res(
            f"r{i}",
            skills=draw(st.lists(st.sampled_from(["als", "open"]), unique=True)),
            contract=draw(st.sampled_from(["full", "part"])),
        )
        for i in range(count)
    ]


@st.composite
def tasks(draw: st.DrawFn) -> list[Task]:
    drawn: list[Task] = []
    for i in range(draw(st.integers(2, 6))):
        offset = timedelta(
            days=draw(st.integers(0, 3)),
            hours=draw(st.integers(0, 23)),
            minutes=draw(st.sampled_from([0, 30])),
        )
        start = FIRST_DAY + offset
        end = start + timedelta(hours=draw(st.integers(1, 12)))
        attributes: dict[str, Any] = {"type": draw(st.sampled_from(["am", "night"]))}
        if draw(st.booleans()):
            attributes["required_skill"] = "als"
        demand = draw(st.integers(1, 2))
        drawn.append(task(f"t{i}", local(start), local(end), demand, **attributes))
    return drawn


@st.composite
def rules(draw: st.DrawFn, pairs: list[Pair]) -> list[dict[str, Any]]:
    def either(template: str, **fields: Any) -> dict[str, Any]:
        hardness = draw(st.sampled_from(["hard", "soft"]))
        return rule(template, hardness, draw(st.integers(1, 5)), **fields)

    def soft(template: str, **fields: Any) -> dict[str, Any]:
        return rule(template, "soft", draw(st.integers(1, 5)), **fields)

    windows = [
        {
            "resource_id": draw(st.sampled_from(["r0", "r1"])),
            "start": local(FIRST_DAY + timedelta(hours=start)),
            "end": local(FIRST_DAY + timedelta(hours=start + length)),
        }
        for start, length in draw(
            st.lists(st.tuples(st.integers(0, 90), st.integers(1, 12)), min_size=1, max_size=2)
        )
    ]
    baseline = draw(st.lists(st.sampled_from(pairs), unique=True, max_size=4))
    per = {"measure": "count", "period": "day", "limit": 1}
    return [
        either(
            "requires_attribute",
            params={
                "task_attribute": "required_skill",
                "resource_attribute": "skills",
                "match": "contains",
            },
        ),
        either("no_overlap"),
        either("min_rest", params={"hours": 11}),
        either("max_per_period", params=per, id="max-day"),
        either(
            "max_per_period", params={"measure": "hours", "period": "week", "limit": 20}, id="max-w"
        ),
        either("min_per_period", params={"measure": "hours", "period": "horizon", "limit": 8}),
        either("max_consecutive_days", params={"days": 2}),
        either("unavailable", params={"windows": windows}),
        soft(
            "preference",
            id="prefer-am",
            params={"kind": "prefer"},
            scope={"tasks": [{"attribute": "type", "op": "eq", "value": "am"}]},
        ),
        soft(
            "preference",
            id="avoid-weekend",
            params={"kind": "avoid"},
            scope={"tasks": [{"attribute": "weekday", "op": "in", "value": ["sat", "sun"]}]},
        ),
        soft("balance", id="balance-count", params={"measure": "count"}),
        soft(
            "balance",
            id="balance-hours",
            params={"measure": "hours"},
            scope={"resources": [{"attribute": "contract", "op": "eq", "value": "full"}]},
        ),
        soft("together", params={"resource_ids": ["r0", "r1"]}),
        either("apart", params={"resource_ids": ["r0", "r1"]}),
        soft(
            "stability",
            params={"baseline": [{"task_id": t, "resource_id": r} for t, r in baseline]},
        ),
    ]


@st.composite
def cases(draw: st.DrawFn) -> Case:
    people = draw(resources())
    work = draw(tasks())
    pairs = [(t.id, r.id) for t in work for r in people]
    doc = dm_document(*draw(rules(pairs)), horizon=HORIZON)
    assignments = draw(st.lists(st.sampled_from(pairs), unique=True))
    return doc, people, work, assignments


def model(doc: dict[str, Any]) -> DecisionModel:
    return parse_decision_model(json.dumps(doc))


@settings(max_examples=60, deadline=None)
@given(cases(), st.data())
def test_assignment_order_never_changes_the_result(case: Case, data: st.DataObject) -> None:
    doc, people, work, assignments = case
    shuffled = data.draw(st.permutations(assignments))
    dm = model(doc)

    original = run(dm, people, work, assignments)
    permuted = run(dm, people, work, shuffled)

    assert permuted == original
    assert permuted.to_json() == original.to_json()


@settings(max_examples=60, deadline=None)
@given(cases(), st.integers(2, 5))
def test_scaling_soft_weights_scales_soft_and_keeps_hard(case: Case, k: int) -> None:
    doc, people, work, assignments = case
    scaled = copy.deepcopy(doc)
    for constraint in scaled["constraints"]:
        if constraint["hardness"] == "soft":
            constraint["weight"] *= k

    base = run(model(doc), people, work, assignments).score
    multiplied = run(model(scaled), people, work, assignments).score

    assert multiplied.soft == k * base.soft
    assert multiplied.hard == base.hard


@settings(max_examples=60, deadline=None)
@given(cases(), st.data())
def test_exactly_demand_distinct_resources_per_task_gives_zero_coverage(
    case: Case, data: st.DataObject
) -> None:
    doc, people, work, _ = case
    ids = [r.id for r in people]
    assignments = [(t.id, r) for t in work for r in data.draw(st.permutations(ids))[: t.demand]]

    result = run(model(doc), people, work, assignments)

    assert result.rule("coverage").amount == 0


@settings(max_examples=60, deadline=None)
@given(cases(), st.data())
def test_removing_an_assignment_never_increases_conflict_amounts(
    case: Case, data: st.DataObject
) -> None:
    doc, people, work, assignments = case
    if not assignments:
        return
    index = data.draw(st.integers(0, len(assignments) - 1))
    fewer = assignments[:index] + assignments[index + 1 :]
    dm = model(doc)

    before = run(dm, people, work, assignments)
    after = run(dm, people, work, fewer)

    for rule_id in MONOTONE:
        assert after.rule(rule_id).amount <= before.rule(rule_id).amount, rule_id
