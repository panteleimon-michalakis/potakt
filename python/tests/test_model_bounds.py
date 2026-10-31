"""Bounds for dates and numeric values (ADR-0006 "Bounds", TASK-004b F1).

Out-of-range values are issues, found before any timezone conversion or arithmetic, so nothing can
overflow. The last test is a property: inside or just outside the bounds, loading, data-level
validation and checking never raise.
"""

import json
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from helpers import RESOURCES_CSV, constraint, minimal_dm, structural_issues, write_scenario
from hypothesis import given, settings
from hypothesis import strategies as st
from test_model_data import load_issues, only_issue, with_constraints

from potakt.checker import CheckResult, check
from potakt.model import (
    Assignment,
    DecisionModelError,
    Solution,
    decision_model_json_schema,
    load_decision_model,
    load_instance,
    validate_data,
    validate_solution,
)
from potakt.model.common import check_whole_minutes, is_whole

AFTER_MAX_TIME = "is after the maximum of 3000-01-01T00:00"


def issue_strings(doc: dict[str, Any]) -> list[str]:
    return [str(issue) for issue in structural_issues(doc)]


# --- dates -----------------------------------------------------------------------------------


def horizon(start: str, end: str) -> dict[str, Any]:
    doc = minimal_dm()
    doc["horizon"] = {"start": start, "end": end}
    return doc


def test_horizon_at_the_bounds_is_accepted() -> None:
    assert issue_strings(horizon("1970-01-01", "1970-01-07")) == []
    assert issue_strings(horizon("2999-12-25", "2999-12-31")) == []


def test_horizon_outside_the_bounds_is_rejected_at_the_field() -> None:
    assert issue_strings(horizon("1969-12-31", "1970-01-06")) == [
        "/horizon/start: start 1969-12-31 is before the minimum of 1970-01-01"
    ]
    assert issue_strings(horizon("2999-12-26", "3000-01-01")) == [
        "/horizon/end: end 3000-01-01 is after the maximum of 2999-12-31"
    ]


def test_late_9999_horizon_is_an_issue_not_a_crash() -> None:
    # TASK-004 probe: the ISO week of 9999-12-31 ran past datetime's maximum.
    assert issue_strings(horizon("9999-12-25", "9999-12-31")) == [
        "/horizon/start: start 9999-12-25 is after the maximum of 2999-12-31",
        "/horizon/end: end 9999-12-31 is after the maximum of 2999-12-31",
    ]


def window_rule(start: str, end: str) -> dict[str, Any]:
    window = {"resource_id": "r1", "start": start, "end": end}
    return constraint("unavailable", hardness="hard", params={"windows": [window]})


def test_inline_window_near_9999_in_new_york_is_an_issue_not_a_crash() -> None:
    # TASK-004 probe: converting 9999-12-31T20:00 New York time to UTC overflowed.
    doc = with_constraints(window_rule("9999-12-31T20:00", "9999-12-31T23:00"))
    doc["timezone"] = "America/New_York"

    assert issue_strings(doc) == [
        f"/constraints/1/params/windows/0/start: 9999-12-31T20:00 {AFTER_MAX_TIME}",
        f"/constraints/1/params/windows/0/end: 9999-12-31T23:00 {AFTER_MAX_TIME}",
    ]


def test_inline_window_bounds() -> None:
    assert (
        issue_strings(with_constraints(window_rule("1970-01-01T00:00", "3000-01-01T00:00"))) == []
    )
    assert issue_strings(with_constraints(window_rule("1969-12-31T23:00", "1970-01-01T08:00"))) == [
        "/constraints/1/params/windows/0/start: 1969-12-31T23:00 is before the minimum of "
        "1970-01-01T00:00"
    ]


def test_csv_window_near_9999_is_a_data_level_issue(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})
    doc = with_constraints(rule)
    doc["timezone"] = "America/New_York"
    leave = "who,from,to\nr1,9999-12-31T20:00,9999-12-31T23:00\n"

    issues = load_issues(write_scenario(tmp_path, doc, leave_csv=leave))

    assert [(i.path, i.message) for i in issues] == [
        (
            "/constraints/1/params/source/start",
            f"leave.csv line 2, column 'from': 9999-12-31T20:00 {AFTER_MAX_TIME}",
        ),
        (
            "/constraints/1/params/source/end",
            f"leave.csv line 2, column 'to': 9999-12-31T23:00 {AFTER_MAX_TIME}",
        ),
    ]


def test_task_ending_after_the_datetime_maximum_is_a_data_level_issue(tmp_path: Path) -> None:
    doc = horizon("2999-12-25", "2999-12-31")
    tasks = "id,start,end,demand,type,required_skill\nt1,2999-12-31T23:00,3000-01-01T07:00,1,a,\n"

    issue = only_issue(write_scenario(tmp_path, doc, tasks=tasks))

    assert issue.path == "/data/tasks/end"
    assert issue.message == f"tasks.csv line 2, column 'end': 3000-01-01T07:00 {AFTER_MAX_TIME}"


# --- numbers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule", "path", "message"),
    [
        (
            constraint("no_overlap", hardness="soft", weight=10_001),
            "/constraints/1/weight",
            "weight 10001 is above the maximum of 10000",
        ),
        (
            constraint("preference", hardness="soft", weight=10_001, params={"kind": "avoid"}),
            "/constraints/1/weight",
            "weight 10001 is above the maximum of 10000",
        ),
        (
            constraint("no_overlap", hardness="soft", weight=0),
            "/constraints/1/weight",
            "weight 0 is below the minimum of 1",
        ),
        (
            constraint("min_rest", hardness="hard", params={"hours": 8785}),
            "/constraints/1/params/hours",
            "hours 8785 is above the maximum of 8784",
        ),
        (
            constraint("min_rest", hardness="hard", params={"hours": 8784.5}),
            "/constraints/1/params/hours",
            "hours 8784.5 is above the maximum of 8784",
        ),
        (
            constraint("min_rest", hardness="hard", params={"hours": 1e308}),
            "/constraints/1/params/hours",
            "hours 1e+308 is above the maximum of 8784",
        ),
        (
            constraint("max_consecutive_days", hardness="hard", params={"days": 367}),
            "/constraints/1/params/days",
            "days 367 is above the maximum of 366",
        ),
    ],
)
def test_numbers_outside_the_bounds_name_the_value_and_the_bound(
    rule: dict[str, Any], path: str, message: str
) -> None:
    assert issue_strings(with_constraints(rule)) == [f"{path}: {message}"]


def test_numbers_at_the_bounds_are_accepted() -> None:
    doc = with_constraints(
        constraint("no_overlap", id="w-max", hardness="soft", weight=10_000),
        constraint("min_rest", hardness="hard", params={"hours": 8784}),
        constraint("max_consecutive_days", hardness="hard", params={"days": 366}),
    )

    assert issue_strings(doc) == []


TASK_HEADER = "id,start,end,demand,type,required_skill"


def test_task_demand_of_10000_is_accepted(tmp_path: Path) -> None:
    tasks = f"{TASK_HEADER}\nt1,2026-11-02T07:00,2026-11-02T15:00,10000,a,\n"

    assert load_issues(write_scenario(tmp_path, tasks=tasks)) == []


def test_task_demand_above_10000_is_a_data_level_issue(tmp_path: Path) -> None:
    tasks = f"{TASK_HEADER}\nt1,2026-11-02T07:00,2026-11-02T15:00,10001,a,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/demand"
    assert issue.message == (
        "tasks.csv line 2, column 'demand': demand 10001 is above the maximum of 10000"
    )


def test_schema_shows_the_numeric_bounds() -> None:
    defs = decision_model_json_schema()["$defs"]

    assert defs["Balance"]["properties"]["weight"] == {
        "maximum": 10000,
        "minimum": 1,
        "title": "Weight",
        "type": "integer",
    }
    assert defs["MaxConsecutiveDaysParams"]["properties"]["days"]["maximum"] == 366
    assert [
        member["maximum"] for member in defs["MinRestParams"]["properties"]["hours"]["anyOf"]
    ] == [
        8784,
        8784,
    ]
    assert "1970-01-01" in defs["Horizon"]["properties"]["start"]["description"]


# --- overflow-safe helpers (defence in depth for hand-built values) --------------------------


def test_helpers_never_overflow() -> None:
    assert is_whole(10**400)
    assert check_whole_minutes(10**400) == 10**400
    with pytest.raises(ValueError, match="1e\\+308 hours is too large"):
        check_whole_minutes(1e308)


# --- property: no traceback inside or just outside the bounds ------------------------------

TIMEZONES = [
    "Europe/Athens",
    "America/New_York",
    "Pacific/Kiritimati",
    "Pacific/Pago_Pago",
    "Asia/Tokyo",
    "UTC",
]
LOW, HIGH = datetime(1970, 1, 1), datetime(3000, 1, 1)


def near(moment: datetime) -> st.SearchStrategy[datetime]:
    return st.integers(-48, 48).map(lambda hours: moment + timedelta(hours=hours))


@st.composite
def bounded_scenarios(draw: st.DrawFn) -> tuple[dict[str, Any], str]:
    days = draw(st.integers(1, 366))
    first = draw(st.dates(date(1970, 1, 1), date(2999, 12, 31) - timedelta(days=days - 1)))
    last = first + timedelta(days=days - 1)
    starts = datetime.combine(first, datetime.min.time())
    window_start = draw(st.one_of(near(LOW), near(HIGH), near(starts)))
    window_end = window_start + timedelta(hours=draw(st.integers(1, 30)))
    hours_limit = draw(st.sampled_from([0, 1.25, 8784, 8784.25, 8785, 1e308, 10**400]))
    count_limit = draw(st.sampled_from([0, 3, 100_000, 100_001, 10**400]))
    doc = minimal_dm()
    doc["timezone"] = draw(st.sampled_from(TIMEZONES))
    doc["horizon"] = {"start": first.isoformat(), "end": last.isoformat()}
    doc["constraints"].extend(
        [
            constraint(
                "max_per_period",
                id="hours",
                hardness="soft",
                weight=draw(st.sampled_from([0, 1, 10_000, 10_001])),
                params={"measure": "hours", "period": "week", "limit": hours_limit},
            ),
            constraint(
                "min_per_period",
                id="count",
                hardness="hard",
                params={"measure": "count", "period": "day", "limit": count_limit},
            ),
            constraint(
                "max_consecutive_days",
                hardness="hard",
                params={"days": draw(st.sampled_from([0, 1, 366, 367]))},
            ),
            window_rule(
                window_start.isoformat(timespec="minutes"),
                window_end.isoformat(timespec="minutes"),
            ),
        ]
    )
    rows = []
    for k in range(draw(st.integers(1, 3))):
        start = starts + timedelta(
            days=draw(st.integers(0, days - 1)), hours=draw(st.integers(0, 23))
        )
        end = start + timedelta(hours=draw(st.integers(1, 24)))
        demand = draw(st.sampled_from([1, 2, 10_000, 10_001]))
        rows.append(
            f"t{k},{start.isoformat(timespec='minutes')},{end.isoformat(timespec='minutes')},"
            f"{demand},a,"
        )
    tasks = "\n".join(["id,start,end,demand,type,required_skill", *rows]) + "\n"
    return doc, tasks


@settings(max_examples=120, deadline=None)
@given(bounded_scenarios())
def test_loading_validating_and_checking_never_raise(scenario: tuple[dict[str, Any], str]) -> None:
    doc, tasks = scenario
    with tempfile.TemporaryDirectory() as folder:
        directory = Path(folder)
        (directory / "model.json").write_text(json.dumps(doc), encoding="utf-8")
        (directory / "resources.csv").write_text(RESOURCES_CSV, encoding="utf-8")
        (directory / "tasks.csv").write_text(tasks, encoding="utf-8")
        try:
            dm = load_decision_model(directory / "model.json")
            instance = load_instance(dm, directory)
        except DecisionModelError:
            return  # out-of-range values are reported as issues
    if validate_data(dm, instance):
        return
    solution = Solution(
        schema_version="0.1",
        model_id=dm.id,
        model_version=dm.version,
        assignments=tuple(Assignment(task_id=t.id, resource_id="r1") for t in instance.tasks),
    )
    assert validate_solution(dm, instance, solution) == []
    assert isinstance(check(dm, instance, solution), CheckResult)
