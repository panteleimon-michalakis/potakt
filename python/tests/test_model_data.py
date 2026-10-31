"""Loading problem instances from CSV bindings and data-level validation (ADR-0004, level 2)."""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from helpers import (
    RESOURCES_CSV,
    TASKS_CSV,
    constraint,
    minimal_dm,
    structural_issues,
    write_scenario,
)
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from potakt.model import (
    AttributeBinding,
    DecisionModelError,
    ProblemInstance,
    Task,
    TimeWindow,
    ValidationIssue,
    load_decision_model,
    load_instance,
    validate_data,
)
from potakt.model.loader import parse_cell, split_list

ATHENS = ZoneInfo("Europe/Athens")


def load(model_file: Path) -> tuple[ProblemInstance, list[ValidationIssue]]:
    dm = load_decision_model(model_file)
    instance = load_instance(dm, model_file.parent)
    return instance, validate_data(dm, instance)


def load_issues(model_file: Path) -> list[ValidationIssue]:
    """Issues from loading (if loading fails) or else from data-level validation."""
    try:
        return load(model_file)[1]
    except DecisionModelError as error:
        return list(error.issues)


def only_issue(model_file: Path) -> ValidationIssue:
    issues = load_issues(model_file)
    assert len(issues) == 1, issues
    return issues[0]


def with_constraints(*extra: dict[str, Any]) -> dict[str, Any]:
    doc = minimal_dm()
    doc["constraints"].extend(extra)
    return doc


# --- loading ---------------------------------------------------------------------------------


def test_valid_scenario_loads_typed_resources_and_tasks(tmp_path: Path) -> None:
    instance, issues = load(write_scenario(tmp_path))

    assert issues == []
    assert [r.id for r in instance.resources] == ["r1", "r2", "r3"]
    assert instance.resources[0].attributes == {
        "skills": ["opening", "closing"],
        "contract": "full_time",
    }
    assert instance.resources[2].attributes["skills"] == ["closing", "als"]
    task = instance.tasks[1]
    assert task.start == datetime(2026, 11, 2, 23, 0)
    assert task.end == datetime(2026, 11, 3, 7, 0)
    assert task.demand == 1
    assert task.attributes == {"type": "νύχτα", "required_skill": "als"}


def test_empty_cells_mean_absent_attributes_and_empty_lists(tmp_path: Path) -> None:
    instance, _ = load(write_scenario(tmp_path))

    assert instance.resources[1].attributes == {"skills": [], "contract": "part_time"}
    assert "required_skill" not in instance.tasks[0].attributes


def test_instance_is_immutable(tmp_path: Path) -> None:
    instance, _ = load(write_scenario(tmp_path))

    with pytest.raises(ValidationError):
        instance.tasks[0].demand = 5
    with pytest.raises(ValidationError):
        instance.resources = ()
    assert isinstance(instance.tasks, tuple)


def test_csv_with_bom_and_crlf_is_accepted(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "tasks.csv").write_bytes(b"\xef\xbb\xbf" + TASKS_CSV.replace("\n", "\r\n").encode())

    instance, issues = load(model_file)

    assert issues == []
    assert instance.tasks[0].id == "t1"


def test_missing_attribute_column_is_reported(tmp_path: Path) -> None:
    tasks = "id,start,end,demand,required_skill\nt1,2026-11-02T07:00,2026-11-02T15:00,1,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/attributes/type/column"
    assert issue.message == "column 'type' not found in tasks.csv"


def test_missing_core_column_is_reported(tmp_path: Path) -> None:
    tasks = "id,start,end,type,required_skill\nt1,2026-11-02T07:00,2026-11-02T15:00,a,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/demand"
    assert issue.message == "column 'demand' not found in tasks.csv"


def test_missing_data_file_is_reported(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "resources.csv").unlink()

    issue = only_issue(model_file)

    assert issue.path == "/data/resources/source"
    assert issue.message == "cannot read resources.csv: file not found"


def test_data_file_must_be_utf8(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    greek_rows = "\n".join(["id", "Χαρά", ""])
    (tmp_path / "resources.csv").write_bytes(greek_rows.encode("cp1253"))

    issue = only_issue(model_file)

    assert issue.path == "/data/resources/source"
    assert issue.message == "cannot read resources.csv: not valid utf-8"


def test_data_file_needs_a_header_row(tmp_path: Path) -> None:
    issue = only_issue(write_scenario(tmp_path, resources=""))

    assert issue.path == "/data/resources/source"
    assert issue.message == "resources.csv has no header row"


@pytest.mark.parametrize(
    ("row", "path", "message"),
    [
        (
            "t9,2026-11-02T07:00,2026-11-02T15:00,two,a,",
            "/data/tasks/demand",
            "tasks.csv line 3, column 'demand': 'two' is not an integer",
        ),
        (
            "t9,2026-11-02T07:00,2026-11-02T15:00,0,a,",
            "/data/tasks/demand",
            "tasks.csv line 3, column 'demand': demand must be at least 1, got 0",
        ),
        (
            "t9,2/11/2026 07:00,2026-11-02T15:00,1,a,",
            "/data/tasks/start",
            "tasks.csv line 3, column 'start': '2/11/2026 07:00' is not an ISO-8601 local datetime",
        ),
        (
            "t9,2026-11-02T07:00+02:00,2026-11-02T15:00,1,a,",
            "/data/tasks/start",
            "tasks.csv line 3, column 'start': '2026-11-02T07:00+02:00' must be a local time "
            "without a UTC offset (the DM timezone applies)",
        ),
        (
            ",2026-11-02T07:00,2026-11-02T15:00,1,a,",
            "/data/tasks/id",
            "tasks.csv line 3, column 'id': a value is required",
        ),
        (
            "t9,2026-11-02T07:00,2026-11-02T15:00,1,a,,surplus",
            "/data/tasks/source",
            "tasks.csv line 3: more fields than the header",
        ),
    ],
)
def test_invalid_task_rows_are_reported_with_file_and_line(
    tmp_path: Path, row: str, path: str, message: str
) -> None:
    tasks = "id,start,end,demand,type,required_skill\n" + TASKS_CSV.splitlines()[1] + "\n" + row
    issue = only_issue(write_scenario(tmp_path, tasks=tasks + "\n"))

    assert issue.path == path
    assert issue.message == message


def test_all_row_problems_are_reported_together(tmp_path: Path) -> None:
    tasks = "id,start,end,demand,type,required_skill\nt1,x,y,z,a,\n"

    issues = load_issues(write_scenario(tmp_path, tasks=tasks))

    assert [i.path for i in issues] == [
        "/data/tasks/start",
        "/data/tasks/end",
        "/data/tasks/demand",
    ]


def test_typed_attribute_columns(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["attributes"] = {
        "senior": {"column": "senior", "type": "bool"},
        "hours": {"column": "hours", "type": "float"},
        "level": {"column": "level", "type": "int"},
        "tags": {"column": "tags", "type": "list", "separator": "|"},
    }
    resources = "id,senior,hours,level,tags\nr1,Yes,37.5,3,a | b\nr2,0,20,-1,\n"

    instance, issues = load(write_scenario(tmp_path, doc, resources=resources))

    assert issues == []
    assert instance.resources[0].attributes == {
        "senior": True,
        "hours": 37.5,
        "level": 3,
        "tags": ["a", "b"],
    }
    assert instance.resources[1].attributes == {
        "senior": False,
        "hours": 20.0,
        "level": -1,
        "tags": [],
    }


@pytest.mark.parametrize(
    ("column_type", "text", "expected"),
    [
        ("bool", "TRUE", True),
        ("bool", "no", False),
        ("int", "+7", 7),
        ("float", "1e2", 100.0),
        ("float", ".5", 0.5),
        ("str", "  πρωί  ", "πρωί"),
        ("str", "   ", None),
    ],
)
def test_parse_cell_accepts(column_type: Any, text: str, expected: object) -> None:
    binding = AttributeBinding(column="c", type=column_type)

    assert parse_cell(text, binding) == expected


@pytest.mark.parametrize(
    ("column_type", "text"),
    [
        ("bool", "maybe"),
        ("int", "1.5"),
        ("int", "1_000"),
        ("float", "nan"),
        ("float", "inf"),
        ("float", "7,5"),
    ],
)
def test_parse_cell_rejects(column_type: Any, text: str) -> None:
    binding = AttributeBinding(column="c", type=column_type)

    with pytest.raises(ValueError, match="is not"):
        parse_cell(text, binding)


@given(
    separator=st.sampled_from([";", "|", ","]),
    items=st.lists(st.text(alphabet=st.characters(exclude_characters=";|,"), max_size=8)),
)
def test_list_parsing_preserves_non_empty_trimmed_items(separator: str, items: list[str]) -> None:
    joined = separator.join(items)

    assert split_list(joined, separator) == [item.strip() for item in items if item.strip()]


# --- data-level validation -------------------------------------------------------------------


def test_duplicate_task_id_is_reported(tmp_path: Path) -> None:
    tasks = TASKS_CSV + "t1,2026-11-03T07:00,2026-11-03T15:00,1,πρωί,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/id"
    assert issue.message == "duplicate task id 't1' in tasks.csv"


def test_duplicate_resource_id_is_reported(tmp_path: Path) -> None:
    issue = only_issue(write_scenario(tmp_path, resources=RESOURCES_CSV + "r2,Bob,,x\n"))

    assert issue.path == "/data/resources/id"
    assert issue.message == "duplicate resource id 'r2' in resources.csv"


def test_dataset_without_resources_is_reported_at_its_source(tmp_path: Path) -> None:
    issue = only_issue(write_scenario(tmp_path, resources="id,name,skills,contract\n"))

    assert issue.path == "/data/resources/source"
    assert issue.message == "resources.csv contains no resources; at least one is required"


def test_dataset_without_tasks_is_reported_at_its_source(tmp_path: Path) -> None:
    issue = only_issue(write_scenario(tmp_path, tasks="id,start,end,demand,type,required_skill\n"))

    assert issue.path == "/data/tasks/source"
    assert issue.message == "tasks.csv contains no tasks; at least one is required"


@pytest.mark.parametrize(
    ("end", "message"),
    [
        (
            "2026-11-02T07:00",
            "task 't9': end 2026-11-02T07:00 must be after start 2026-11-02T07:00",
        ),
        (
            "2026-11-02T06:00",
            "task 't9': end 2026-11-02T06:00 must be after start 2026-11-02T07:00",
        ),
        ("2026-11-03T08:00", "task 't9': lasts 25 h, more than the maximum of 24 h"),
    ],
)
def test_task_times_must_be_valid(tmp_path: Path, end: str, message: str) -> None:
    tasks = TASKS_CSV + f"t9,2026-11-02T07:00,{end},1,πρωί,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/end"
    assert issue.message == message


def test_task_of_exactly_24_hours_is_accepted(tmp_path: Path) -> None:
    tasks = TASKS_CSV + "t9,2026-11-04T07:00,2026-11-05T07:00,1,πρωί,\n"

    assert load_issues(write_scenario(tmp_path, tasks=tasks)) == []


@pytest.mark.parametrize("start", ["2026-11-01T23:00", "2026-11-09T07:00"])
def test_task_starting_outside_the_horizon_is_reported(tmp_path: Path, start: str) -> None:
    tasks = TASKS_CSV + f"t9,{start},{start[:11]}23:30,1,πρωί,\n"

    issue = only_issue(write_scenario(tmp_path, tasks=tasks))

    assert issue.path == "/data/tasks/start"
    assert issue.message == (
        f"task 't9': starts on {start[:10]}, outside the horizon 2026-11-02 to 2026-11-08"
    )


def test_task_on_last_day_may_end_after_the_horizon(tmp_path: Path) -> None:
    tasks = TASKS_CSV + "t9,2026-11-08T23:00,2026-11-09T07:00,1,νύχτα,als\n"

    assert load_issues(write_scenario(tmp_path, tasks=tasks)) == []


def test_task_starting_in_a_dst_gap_is_reported(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["horizon"] = {"start": "2027-03-22", "end": "2027-03-28"}
    tasks = "id,start,end,demand,type,required_skill\nt1,2027-03-28T03:30,2027-03-28T09:00,1,a,\n"

    issue = only_issue(write_scenario(tmp_path, doc, tasks=tasks))

    assert issue.path == "/data/tasks/start"
    assert issue.message == (
        "task 't1': local time 2027-03-28T03:30 does not exist in Europe/Athens (DST change)"
    )


def test_scope_referring_to_unknown_attribute_is_reported(tmp_path: Path) -> None:
    rule = constraint(
        "no_overlap",
        hardness="hard",
        scope={"tasks": [{"attribute": "shift_kind", "op": "eq", "value": "night"}]},
    )

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/scope/tasks/0/attribute"
    assert issue.message == (
        "unknown task attribute 'shift_kind'; known: date, duration_hours, id, "
        "required_skill, start_hour, type, weekday"
    )


def test_scope_accepts_id_data_and_derived_attributes(tmp_path: Path) -> None:
    rule = constraint(
        "no_overlap",
        hardness="hard",
        scope={
            "resources": [
                {"attribute": "id", "op": "ne", "value": "r1"},
                {"attribute": "skills", "op": "contains", "value": "closing"},
            ],
            "tasks": [
                {"attribute": "weekday", "op": "in", "value": ["sat", "sun"]},
                {"attribute": "type", "op": "eq", "value": "νύχτα"},
            ],
        },
    )

    assert load_issues(write_scenario(tmp_path, with_constraints(rule))) == []


def test_derived_attributes_are_not_resource_attributes(tmp_path: Path) -> None:
    rule = constraint(
        "no_overlap",
        hardness="hard",
        scope={"resources": [{"attribute": "weekday", "op": "eq", "value": "mon"}]},
    )

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/scope/resources/0/attribute"
    assert issue.message == "unknown resource attribute 'weekday'; known: contract, id, skills"


def test_params_referring_to_unknown_attributes_are_reported(tmp_path: Path) -> None:
    rule = constraint(
        "requires_attribute",
        hardness="hard",
        params={"task_attribute": "skill", "resource_attribute": "skils", "match": "contains"},
    )

    issues = load_issues(write_scenario(tmp_path, with_constraints(rule)))

    assert [(i.path, i.message) for i in issues] == [
        (
            "/constraints/1/params/task_attribute",
            "unknown task attribute 'skill'; known: required_skill, type",
        ),
        (
            "/constraints/1/params/resource_attribute",
            "unknown resource attribute 'skils'; known: contract, skills",
        ),
    ]


def test_resource_id_in_params_must_exist(tmp_path: Path) -> None:
    rule = constraint("apart", hardness="hard", params={"resource_ids": ["r1", "r7"]})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/params/resource_ids/1"
    assert issue.message == "unknown resource id 'r7'"


def test_baseline_ids_must_exist(tmp_path: Path) -> None:
    baseline = [{"task_id": "t1", "resource_id": "r1"}, {"task_id": "t5", "resource_id": "r9"}]
    rule = constraint("stability", hardness="soft", weight=1, params={"baseline": baseline})

    issues = load_issues(write_scenario(tmp_path, with_constraints(rule)))

    assert [(i.path, i.message) for i in issues] == [
        ("/constraints/1/params/baseline/1/task_id", "unknown task id 't5'"),
        ("/constraints/1/params/baseline/1/resource_id", "unknown resource id 'r9'"),
    ]


def test_inline_window_resource_must_exist(tmp_path: Path) -> None:
    window = {"resource_id": "r9", "start": "2026-11-03T00:00", "end": "2026-11-04T00:00"}
    rule = constraint("unavailable", hardness="hard", params={"windows": [window]})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/params/windows/0/resource_id"
    assert issue.message == "unknown resource id 'r9'"


def test_windows_are_loaded_from_csv_after_inline_windows(tmp_path: Path) -> None:
    window = {"resource_id": "r1", "start": "2026-11-03T00:00", "end": "2026-11-04T00:00"}
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint(
        "unavailable", id="leave", hardness="hard", params={"windows": [window], "source": source}
    )
    leave = "who,from,to\nr2,2026-11-05T15:00,2026-11-05T23:00\n"

    instance, issues = load(write_scenario(tmp_path, with_constraints(rule), leave_csv=leave))

    assert issues == []
    assert instance.windows["leave"] == (
        TimeWindow(
            resource_id="r1", start=datetime(2026, 11, 3, 0, 0), end=datetime(2026, 11, 4, 0, 0)
        ),
        TimeWindow(
            resource_id="r2", start=datetime(2026, 11, 5, 15, 0), end=datetime(2026, 11, 5, 23, 0)
        ),
    )


def test_csv_windows_are_validated(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})
    leave = (
        "who,from,to\nr2,2026-11-05T15:00,2026-11-05T14:00\nr9,2026-11-05T15:00,2026-11-05T23:00\n"
    )

    issues = load_issues(write_scenario(tmp_path, with_constraints(rule), leave_csv=leave))

    assert [(i.path, i.message) for i in issues] == [
        ("/constraints/1/params/source", "leave.csv line 2: end must be after start"),
    ]


def test_csv_window_resource_must_exist(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})
    leave = "who,from,to\nr9,2026-11-05T15:00,2026-11-05T23:00\n"

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule), leave_csv=leave))

    assert issue.path == "/constraints/1/params/source/resource_id"
    assert issue.message == "unknown resource id 'r9' in leave.csv"


# --- time model ------------------------------------------------------------------------------


def night_task(start: str, end: str) -> Task:
    return Task(
        id="n",
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        demand=1,
        attributes={},
    )


def test_night_shift_across_autumn_dst_change_lasts_nine_hours() -> None:
    # Europe/Athens leaves summer time on Sunday 2026-10-25 at 04:00 (back to 03:00).
    task = night_task("2026-10-24T23:00", "2026-10-25T07:00")

    assert task.duration_minutes(ATHENS) == 9 * 60


def test_night_shift_across_spring_dst_change_lasts_seven_hours() -> None:
    task = night_task("2027-03-27T23:00", "2027-03-28T07:00")

    assert task.duration_minutes(ATHENS) == 7 * 60


def test_ambiguous_local_time_means_the_first_occurrence() -> None:
    # 03:30 happens twice on 2026-10-25; the first occurrence is still summer time (UTC+3).
    task = night_task("2026-10-25T03:30", "2026-10-25T04:30")

    assert task.duration_minutes(ATHENS) == 2 * 60


def test_derived_attributes_use_the_local_start() -> None:
    task = night_task("2026-11-08T23:00", "2026-11-09T07:00")

    assert task.derived_attributes(ATHENS) == {
        "weekday": "sun",
        "start_hour": 23,
        "duration_hours": 8.0,
        "date": "2026-11-08",
    }


def test_invalid_attribute_cell_is_reported(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["attributes"]["hours"] = {"column": "hours", "type": "float"}
    resources = "id,skills,contract,hours\nr1,,x,37.5\nr2,,x,full\n"

    issue = only_issue(write_scenario(tmp_path, doc, resources=resources))

    assert issue.path == "/data/resources/attributes/hours"
    assert issue.message == "resources.csv line 3, column 'hours': 'full' is not a number"


def test_duplicate_bound_column_is_reported(tmp_path: Path) -> None:
    resources = "id,skills,contract,skills\nr1,a,x,b\n"

    issue = only_issue(write_scenario(tmp_path, resources=resources))

    assert issue.path == "/data/resources/attributes/skills/column"
    assert issue.message == "column 'skills' appears more than once in resources.csv"


def test_duplicate_unbound_columns_are_ignored(tmp_path: Path) -> None:
    resources = "id,skills,contract,,\nr1,a,x,,\n"

    assert load_issues(write_scenario(tmp_path, resources=resources)) == []


def test_data_source_must_be_a_regular_file(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "resources.csv").unlink()
    (tmp_path / "resources.csv").mkdir()

    issue = only_issue(model_file)

    assert issue.path == "/data/resources/source"
    assert issue.message == "cannot read resources.csv: not a regular file"


@pytest.mark.skipif(sys.platform != "win32", reason="backslash is a separator only on Windows")
def test_window_source_must_be_a_regular_file_on_windows(tmp_path: Path) -> None:
    (tmp_path / "leave").mkdir()
    (tmp_path / "leave" / "nov").mkdir()
    source = {"source": "leave\\nov", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/params/source/source"
    assert issue.message == "cannot read leave\\nov: not a regular file"


@pytest.mark.parametrize("device", ["NUL", "nul.csv", "Con", "COM1.txt", "lpt9", "aux."])
def test_windows_device_names_are_rejected_on_every_platform(tmp_path: Path, device: str) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["source"] = device

    issue = only_issue(write_scenario(tmp_path, doc))

    assert issue.path == "/data/resources/source"
    assert issue.message == f"cannot read {device}: reserved device name"


@pytest.mark.parametrize(
    "device", ["sub/PRN.tar.gz", "sub\\com9.csv", "LPT¹.csv", "nul .csv", "CON. . "]
)
def test_device_names_are_recognised_in_any_form(tmp_path: Path, device: str) -> None:
    source = {"source": device, "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/params/source/source"
    assert issue.message == f"cannot read {device}: reserved device name"


@pytest.mark.parametrize(
    "device",
    ["COM0", "lpt0.csv", "CONIN$", "conout$.csv", "nul:", "Aux:", "sub/con:", "nul:.csv", "prn:x"],
)
def test_console_devices_port_zero_and_colon_forms_are_reserved(
    tmp_path: Path, device: str
) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["source"] = device

    issue = only_issue(write_scenario(tmp_path, doc))

    assert issue.path == "/data/resources/source"
    assert issue.message == f"cannot read {device}: reserved device name"


@pytest.mark.parametrize(
    "name", ["console.csv", "nullable.csv", "com10.csv", "lpt.csv", "conin.csv", "conout.csv"]
)
def test_names_that_only_resemble_devices_are_read_normally(tmp_path: Path, name: str) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["source"] = name
    model_file = write_scenario(tmp_path, doc)
    (tmp_path / name).write_text(RESOURCES_CSV, encoding="utf-8")

    assert load_issues(model_file) == []


@pytest.mark.parametrize(
    ("window", "path"),
    [
        (
            {"resource_id": "r1", "start": "2027-03-28T03:15", "end": "2027-03-28T09:00"},
            "/constraints/1/params/windows/0/start",
        ),
        (
            {"resource_id": "r1", "start": "2027-03-27T22:00", "end": "2027-03-28T03:30"},
            "/constraints/1/params/windows/0/end",
        ),
    ],
)
def test_inline_window_in_a_dst_gap_is_reported(
    tmp_path: Path, window: dict[str, str], path: str
) -> None:
    rule = constraint("unavailable", hardness="hard", params={"windows": [window]})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == path
    assert "does not exist in Europe/Athens (DST change)" in issue.message


def test_csv_window_in_a_dst_gap_is_reported(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})
    leave = "who,from,to\nr1,2027-03-28T03:15,2027-03-28T09:00\n"

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule), leave_csv=leave))

    assert issue.path == "/constraints/1/params/source/start"
    assert issue.message == (
        "window of 'r1' in leave.csv: local time 2027-03-28T03:15 does not exist in "
        "Europe/Athens (DST change)"
    )


def test_window_csv_cells_are_validated(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})
    leave = "who,from,to\n,2026-11-05T15:00,2026-11-05T23:00\nr1,soon,2026-11-05T23:00\n"

    issues = load_issues(write_scenario(tmp_path, with_constraints(rule), leave_csv=leave))

    assert [(i.path, i.message) for i in issues] == [
        (
            "/constraints/1/params/source/resource_id",
            "leave.csv line 2, column 'who': a value is required",
        ),
        (
            "/constraints/1/params/source/start",
            "leave.csv line 3, column 'from': 'soon' is not an ISO-8601 local datetime",
        ),
    ]


def test_missing_window_file_points_at_its_source_field(tmp_path: Path) -> None:
    source = {"source": "leave.csv", "resource_id": "who", "start": "from", "end": "to"}
    rule = constraint("unavailable", hardness="hard", params={"source": source})

    issue = only_issue(write_scenario(tmp_path, with_constraints(rule)))

    assert issue.path == "/constraints/1/params/source/source"
    assert issue.message == "cannot read leave.csv: file not found"


def test_non_finite_float_is_rejected() -> None:
    with pytest.raises(ValueError, match="is not a finite number"):
        parse_cell("1e999", AttributeBinding(column="c", type="float"))


# --- CSV dialect (ADR-0006 §5, C2) ---------------------------------------------------------
# The file bytes are generated here, never committed, so Git line-ending normalisation cannot
# alter them.

GREEK_EXCEL = {"delimiter": ";", "decimal": ",", "encoding": "windows-1253"}


def staff_dm(csv_block: dict[str, str] | None = None) -> dict[str, Any]:
    """The minimal DM with a ``contract_hours`` float column and an optional ``csv`` block."""
    doc = minimal_dm()
    resources = doc["data"]["resources"]
    resources["attributes"]["name"] = {"column": "name", "type": "str"}
    resources["attributes"]["contract_hours"] = {"column": "contract_hours", "type": "float"}
    if csv_block is not None:
        resources["csv"] = csv_block
    return doc


def staff_rows(newline: str = "\r\n") -> str:
    rows = [
        "id;name;skills;contract;contract_hours",
        "r1;Μαρία Παπαδοπούλου;als;full_time;32,5",
        "r2;Ελένη Γεωργίου;;part_time;16",
        "r3;Άννα Ζαχαρίου;closing;part_time;16,25",
    ]
    return newline.join(rows) + newline


def load_staff(tmp_path: Path, doc: dict[str, Any], data: bytes) -> ProblemInstance:
    model_file = write_scenario(tmp_path, doc)
    (tmp_path / "resources.csv").write_bytes(data)
    instance, issues = load(model_file)
    assert issues == []
    return instance


def test_greek_excel_csv_loads_with_names_and_decimal_comma(tmp_path: Path) -> None:
    data = staff_rows("\r\n").encode("cp1253")
    assert b"\r\n" in data and "Μαρία".encode("cp1253") in data

    instance = load_staff(tmp_path, staff_dm(GREEK_EXCEL), data)

    assert [r.attributes["name"] for r in instance.resources] == [
        "Μαρία Παπαδοπούλου",
        "Ελένη Γεωργίου",
        "Άννα Ζαχαρίου",
    ]
    assert [r.attributes["contract_hours"] for r in instance.resources] == [32.5, 16.0, 16.25]


def test_windows_1253_bytes_declared_as_utf8_cannot_be_read(tmp_path: Path) -> None:
    doc = staff_dm({"delimiter": ";", "decimal": ","})
    model_file = write_scenario(tmp_path, doc)
    (tmp_path / "resources.csv").write_bytes(staff_rows().encode("cp1253"))

    issue = only_issue(model_file)

    assert issue.path == "/data/resources/source"
    assert issue.message == "cannot read resources.csv: not valid utf-8"


def test_excel_csv_utf8_with_bom_loads(tmp_path: Path) -> None:
    data = b"\xef\xbb\xbf" + staff_rows().encode("utf-8")

    instance = load_staff(tmp_path, staff_dm({"delimiter": ";", "decimal": ","}), data)

    assert instance.resources[0].id == "r1"
    assert instance.resources[0].attributes["contract_hours"] == 32.5


def test_iso_8859_7_csv_loads_with_names_intact(tmp_path: Path) -> None:
    # 'Ά' is 0xA2 in Windows-1253 and 0xB6 in ISO-8859-7: the declared codec must be the one used.
    data = staff_rows("\n").encode("iso-8859-7")
    dialect = {"delimiter": ";", "decimal": ",", "encoding": "iso-8859-7"}

    instance = load_staff(tmp_path, staff_dm(dialect), data)

    assert instance.resources[2].attributes["name"] == "Άννα Ζαχαρίου"


def test_decimal_comma_needs_another_delimiter() -> None:
    doc = staff_dm({"delimiter": ",", "decimal": ","})

    issues = structural_issues(doc)

    assert [str(issue) for issue in issues] == [
        "/data/resources/csv/decimal: decimal ',' needs a delimiter other than ','"
    ]


@pytest.mark.parametrize("cell", ["32,5", "32.5"])
def test_int_cells_never_accept_a_decimal_separator(tmp_path: Path, cell: str) -> None:
    doc = staff_dm(GREEK_EXCEL)
    doc["data"]["resources"]["attributes"]["contract_hours"]["type"] = "int"
    rows = f"id;name;skills;contract;contract_hours\r\nr1;Μαρία;als;full_time;{cell}\r\n"
    model_file = write_scenario(tmp_path, doc)
    (tmp_path / "resources.csv").write_bytes(rows.encode("cp1253"))

    issue = only_issue(model_file)

    assert issue.path == "/data/resources/attributes/contract_hours"
    assert issue.message == (
        f"resources.csv line 2, column 'contract_hours': {cell!r} is not an integer"
    )


@pytest.mark.parametrize("cell", ["32.5", "1.234,5"])
def test_decimal_comma_rejects_points_in_float_cells(cell: str) -> None:
    # Under decimal ',' a '.' is a thousands separator in those locales: never guess.
    binding = AttributeBinding(column="c", type="float")

    with pytest.raises(ValueError, match=r"is not a number \(decimal separator ','\)"):
        parse_cell(cell, binding, decimal=",")


@settings(max_examples=200, deadline=None)
@given(st.decimals(min_value=0, max_value=1000, places=2))
def test_decimal_comma_and_point_parse_to_the_same_value(value: Decimal) -> None:
    binding = AttributeBinding(column="c", type="float")
    with_point = format(value, "f")
    with_comma = with_point.replace(".", ",")

    parsed = parse_cell(with_comma, binding, decimal=",")

    assert parsed == parse_cell(with_point, binding)
    assert parsed == float(with_point)


def test_tasks_binding_accepts_a_csv_dialect(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["data"]["tasks"]["csv"] = {"delimiter": "\t", "encoding": "windows-1253"}
    tasks = TASKS_CSV.replace(",", "\t").replace("\n", "\r\n").encode("cp1253")
    model_file = write_scenario(tmp_path, doc)
    (tmp_path / "tasks.csv").write_bytes(tasks)

    instance, issues = load(model_file)

    assert issues == []
    assert [t.attributes["type"] for t in instance.tasks] == ["πρωί", "νύχτα"]


def test_window_source_accepts_a_csv_dialect(tmp_path: Path) -> None:
    source = {
        "source": "leave.csv",
        "resource_id": "ποιος",
        "start": "από",
        "end": "έως",
        "csv": {"delimiter": ";", "encoding": "iso-8859-7"},
    }
    rule = constraint("unavailable", id="leave", hardness="hard", params={"source": source})
    model_file = write_scenario(tmp_path, with_constraints(rule))
    leave = "ποιος;από;έως\r\nr2;2026-11-05T15:00;2026-11-05T23:00\r\n"
    (tmp_path / "leave.csv").write_bytes(leave.encode("iso-8859-7"))

    instance, issues = load(model_file)

    assert issues == []
    assert [w.resource_id for w in instance.windows["leave"]] == ["r2"]
