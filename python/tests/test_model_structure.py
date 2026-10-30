"""Structural validation of the Decision Model (ADR-0004, level 1)."""

import json
from pathlib import Path
from typing import Any

import pytest
from helpers import constraint, minimal_dm, paths, structural_issues

from potakt.model import (
    Coverage,
    DecisionModel,
    DecisionModelError,
    MinRest,
    ValidationIssue,
    load_decision_model,
    parse_decision_model,
)


def only_issue(doc: dict[str, Any]) -> ValidationIssue:
    issues = structural_issues(doc)
    assert len(issues) == 1, issues
    return issues[0]


def with_constraints(*extra: dict[str, Any]) -> dict[str, Any]:
    doc = minimal_dm()
    doc["constraints"].extend(extra)
    return doc


# --- valid documents -------------------------------------------------------------------------


def test_minimal_document_parses_into_typed_model() -> None:
    dm = parse_decision_model(json.dumps(minimal_dm()))

    assert isinstance(dm, DecisionModel)
    assert isinstance(dm.constraints[0], Coverage)
    assert dm.objective == "lexicographic_hard_then_soft"
    assert dm.constraints[0].status == "proposed"


def test_discriminated_union_selects_template_model() -> None:
    doc = with_constraints(
        constraint("min_rest", hardness="soft", weight=3, params={"hours": 11}),
    )

    dm = parse_decision_model(json.dumps(doc))

    rule = dm.constraints[1]
    assert isinstance(rule, MinRest)
    assert rule.params.hours == 11


def test_round_trip_through_json_is_lossless() -> None:
    doc = with_constraints(
        constraint("min_rest", hardness="hard", params={"hours": 7.5}),
        constraint(
            "preference",
            hardness="soft",
            weight=2,
            params={"kind": "avoid"},
            scope={"tasks": [{"attribute": "weekday", "op": "in", "value": ["sat", "sun"]}]},
        ),
    )
    dm = parse_decision_model(json.dumps(doc))

    assert parse_decision_model(dm.model_dump_json()) == dm


def test_every_template_of_catalog_v0_is_accepted() -> None:
    window = {"resource_id": "r1", "start": "2026-11-03T00:00", "end": "2026-11-04T00:00"}
    doc = with_constraints(
        constraint(
            "requires_attribute",
            hardness="hard",
            params={
                "task_attribute": "required_skill",
                "resource_attribute": "skills",
                "match": "contains",
            },
        ),
        constraint("no_overlap", hardness="hard"),
        constraint("min_rest", hardness="hard", params={"hours": 11}),
        constraint(
            "max_per_period",
            hardness="hard",
            params={"measure": "hours", "period": "week", "limit": 40},
        ),
        constraint(
            "min_per_period",
            hardness="soft",
            weight=1,
            params={"measure": "count", "period": "horizon", "limit": 2},
        ),
        constraint("max_consecutive_days", hardness="soft", weight=1, params={"days": 5}),
        constraint("unavailable", hardness="hard", params={"windows": [window]}),
        constraint("preference", hardness="soft", weight=1, params={"kind": "prefer"}),
        constraint("balance", hardness="soft", weight=1, params={"measure": "count"}),
        constraint("together", hardness="soft", weight=1, params={"resource_ids": ["r1", "r2"]}),
        constraint("apart", hardness="hard", params={"resource_ids": ["r1", "r2"]}),
        constraint(
            "stability",
            hardness="soft",
            weight=1,
            params={"baseline": [{"task_id": "t1", "resource_id": "r1"}]},
        ),
    )

    assert structural_issues(doc) == []


# --- required invalid cases (TASK-002) ------------------------------------------------------


def test_duplicate_constraint_ids_are_rejected() -> None:
    doc = with_constraints(
        constraint("no_overlap", id="rule", hardness="hard"),
        constraint("min_rest", id="rule", hardness="hard", params={"hours": 11}),
    )

    issue = only_issue(doc)

    assert issue.path == "/constraints/2/id"
    assert "duplicate constraint id 'rule'" in issue.message
    assert "/constraints/1" in issue.message


def test_soft_constraint_without_weight_is_rejected() -> None:
    issue = only_issue(with_constraints(constraint("no_overlap", hardness="soft")))

    assert issue.path == "/constraints/1/weight"
    assert issue.message == "weight is required for soft constraints"


def test_hard_constraint_with_weight_is_rejected() -> None:
    issue = only_issue(with_constraints(constraint("no_overlap", hardness="hard", weight=5)))

    assert issue.path == "/constraints/1/weight"
    assert issue.message == "weight is not allowed for hard constraints"


def test_soft_only_template_marked_hard_is_rejected() -> None:
    doc = with_constraints(constraint("balance", hardness="hard", params={"measure": "count"}))

    issues = structural_issues(doc)

    assert "/constraints/1/hardness" in paths(issues)
    hardness = next(i for i in issues if i.path == "/constraints/1/hardness")
    assert hardness.message == "Input should be 'soft'"


def test_missing_coverage_is_rejected() -> None:
    doc = minimal_dm()
    doc["constraints"] = [constraint("no_overlap", hardness="hard")]

    issue = only_issue(doc)

    assert issue.path == "/constraints"
    assert issue.message == "exactly one 'coverage' constraint is required, found none"


def test_duplicated_coverage_is_rejected() -> None:
    doc = with_constraints(constraint("coverage", id="coverage-2", hardness="hard"))

    issue = only_issue(doc)

    assert issue.path == "/constraints/1"
    assert issue.message == (
        "exactly one 'coverage' constraint is allowed; another one is at /constraints/0"
    )


def test_unknown_template_is_rejected() -> None:
    issue = only_issue(with_constraints(constraint("teleport", hardness="hard")))

    assert issue.path == "/constraints/1"
    assert "'teleport'" in issue.message
    assert "does not match any of the expected tags" in issue.message


@pytest.mark.parametrize("hours", [0, -1])
def test_min_rest_hours_must_be_positive(hours: float) -> None:
    issue = only_issue(
        with_constraints(constraint("min_rest", hardness="hard", params={"hours": hours}))
    )

    assert issue.path == "/constraints/1/params/hours"
    assert issue.message == "Input should be greater than 0"


@pytest.mark.parametrize(
    ("template", "params", "field"),
    [
        ("min_rest", {"hours": 10.999}, "hours"),
        ("max_per_period", {"measure": "hours", "period": "week", "limit": 7.501}, "limit"),
        ("min_per_period", {"measure": "hours", "period": "day", "limit": 0.01}, "limit"),
    ],
)
def test_hours_must_be_a_whole_number_of_minutes(
    template: str, params: dict[str, Any], field: str
) -> None:
    issue = only_issue(with_constraints(constraint(template, hardness="hard", params=params)))

    assert issue.path == f"/constraints/1/params/{field}"
    assert "whole number of minutes" in issue.message


def test_hours_with_whole_minutes_are_accepted() -> None:
    doc = with_constraints(
        constraint("min_rest", hardness="hard", params={"hours": 0.1}),
        constraint(
            "max_per_period",
            hardness="hard",
            params={"measure": "hours", "period": "day", "limit": 7.25},
        ),
    )

    assert structural_issues(doc) == []


def test_count_limit_must_be_an_integer() -> None:
    doc = with_constraints(
        constraint(
            "max_per_period",
            hardness="hard",
            params={"measure": "count", "period": "week", "limit": 4.5},
        )
    )

    issue = only_issue(doc)

    assert issue.path == "/constraints/1/params/limit"
    assert issue.message == "limit must be a whole number when measure is 'count'"


@pytest.mark.parametrize("timezone", ["Europe/Atlantis", "europe/athens", "../etc/passwd", ""])
def test_invalid_timezone_is_rejected(timezone: str) -> None:
    doc = minimal_dm()
    doc["timezone"] = timezone

    issue = only_issue(doc)

    assert issue.path == "/timezone"
    assert issue.message == f"unknown IANA timezone {timezone!r}"


def test_horizon_end_before_start_is_rejected() -> None:
    doc = minimal_dm()
    doc["horizon"] = {"start": "2026-11-08", "end": "2026-11-02"}

    issue = only_issue(doc)

    assert issue.path == "/horizon/end"
    assert issue.message == "horizon end 2026-11-02 is before its start 2026-11-08"


def test_single_day_horizon_is_accepted() -> None:
    doc = minimal_dm()
    doc["horizon"] = {"start": "2026-11-02", "end": "2026-11-02"}

    assert structural_issues(doc) == []


@pytest.mark.parametrize("resource_ids", [["r1"], ["r1", "r2", "r3"]])
def test_together_needs_exactly_two_resources(resource_ids: list[str]) -> None:
    doc = with_constraints(
        constraint("together", hardness="soft", weight=1, params={"resource_ids": resource_ids})
    )

    issue = only_issue(doc)

    assert issue.path == "/constraints/1/params/resource_ids"
    assert "should have at" in issue.message


def test_pair_of_resources_must_be_distinct() -> None:
    doc = with_constraints(
        constraint("apart", hardness="hard", params={"resource_ids": ["r1", "r1"]})
    )

    issue = only_issue(doc)

    assert issue.path == "/constraints/1/params/resource_ids"
    assert issue.message == "the two resource ids must be different"


@pytest.mark.parametrize("name", ["start", "id", "weekday", "duration_hours"])
def test_reserved_attribute_name_in_binding_is_rejected(name: str) -> None:
    doc = minimal_dm()
    doc["data"]["tasks"]["attributes"][name] = {"column": "x", "type": "str"}

    issue = only_issue(doc)

    assert issue.path == f"/data/tasks/attributes/{name}"
    assert issue.message == f"attribute name {name!r} is reserved"


# --- further structural rules from ADR-0004 --------------------------------------------------


@pytest.mark.parametrize("name", ["Skills", "2nd", "first-name", "ώρες"])
def test_attribute_names_must_be_snake_case(name: str) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["attributes"][name] = {"column": "x", "type": "str"}

    issue = only_issue(doc)

    assert issue.path == f"/data/resources/attributes/{name}"
    assert issue.message == f"attribute name {name!r} must be snake_case"


def test_separator_is_only_allowed_for_list_columns() -> None:
    doc = minimal_dm()
    doc["data"]["tasks"]["attributes"]["type"]["separator"] = ","

    issue = only_issue(doc)

    assert issue.path == "/data/tasks/attributes/type/separator"
    assert issue.message == "separator is only allowed for 'list' columns"


@pytest.mark.parametrize(
    "source", ["../secrets.csv", "data/../../x.csv", "/etc/passwd", "C:\\x.csv", "..\\x.csv", ""]
)
def test_data_sources_must_stay_inside_the_model_directory(source: str) -> None:
    doc = minimal_dm()
    doc["data"]["resources"]["source"] = source

    issue = only_issue(doc)

    assert issue.path == "/data/resources/source"
    assert "relative path inside the model directory" in issue.message


def test_values_are_not_coerced() -> None:
    doc = minimal_dm()
    doc["version"] = "1"

    issue = only_issue(doc)

    assert issue.path == "/version"
    assert issue.message == "Input should be a valid integer"


def test_unknown_fields_are_rejected() -> None:
    doc = minimal_dm()
    doc["constraints"][0]["priority"] = "high"

    issue = only_issue(doc)

    assert issue.path == "/constraints/0/priority"
    assert issue.message == "Extra inputs are not permitted"


def test_missing_required_field_points_at_the_field() -> None:
    doc = minimal_dm()
    del doc["constraints"][0]["rationale"]

    issue = only_issue(doc)

    assert issue.path == "/constraints/0/rationale"
    assert issue.message == "Field required"


def test_invalid_json_is_reported_at_the_root() -> None:
    with pytest.raises(DecisionModelError) as caught:
        parse_decision_model('{"schema_version": ')

    [issue] = caught.value.issues
    assert issue.path == "/"
    assert issue.message.startswith("invalid JSON")


def test_document_must_be_a_json_object() -> None:
    with pytest.raises(DecisionModelError) as caught:
        parse_decision_model("[1, 2]")

    [issue] = caught.value.issues
    assert issue.path == "/"


def test_parent_version_must_precede_version() -> None:
    doc = minimal_dm()
    doc["parent_version"] = 1

    issue = only_issue(doc)

    assert issue.path == "/parent_version"
    assert issue.message == "parent_version (1) must be lower than version (1)"


def test_assumption_ids_must_be_unique() -> None:
    doc = minimal_dm()
    doc["assumptions"] = [
        {"id": "a1", "text": "first", "status": "proposed"},
        {"id": "a1", "text": "second", "status": "confirmed"},
    ]

    issue = only_issue(doc)

    assert issue.path == "/assumptions/1/id"
    assert "duplicate assumption id 'a1'" in issue.message


def test_utterance_provenance_requires_a_quote() -> None:
    rule = constraint("no_overlap", hardness="hard", provenance={"kind": "utterance"})

    issue = only_issue(with_constraints(rule))

    assert issue.path == "/constraints/1/provenance/quote"
    assert issue.message == "quote is required when kind is 'utterance'"


def test_data_provenance_requires_a_source() -> None:
    rule = constraint("no_overlap", hardness="hard", provenance={"kind": "data"})

    issue = only_issue(with_constraints(rule))

    assert issue.path == "/constraints/1/provenance/source"
    assert issue.message == "source is required when kind is 'data'"


@pytest.mark.parametrize(
    ("op", "value", "message"),
    [
        ("in", "sat", "value must be a list for op 'in'"),
        ("not_in", 3, "value must be a list for op 'not_in'"),
        ("contains", ["a"], "value must be a single value for op 'contains'"),
        ("gte", True, "value must be a number or a string for op 'gte'"),
        ("lte", ["a"], "value must be a number or a string for op 'lte'"),
    ],
)
def test_condition_value_must_fit_the_operator(op: str, value: Any, message: str) -> None:
    rule = constraint(
        "no_overlap",
        hardness="hard",
        scope={"tasks": [{"attribute": "type", "op": op, "value": value}]},
    )

    issue = only_issue(with_constraints(rule))

    assert issue.path == "/constraints/1/scope/tasks/0/value"
    assert issue.message == message


def test_unavailable_needs_windows_or_a_source() -> None:
    issue = only_issue(with_constraints(constraint("unavailable", hardness="hard", params={})))

    assert issue.path == "/constraints/1/params"
    assert issue.message == "at least one of 'windows' or 'source' is required"


@pytest.mark.parametrize(
    ("start", "end", "field", "message"),
    [
        ("2026-11-03T08:00", "2026-11-03T08:00", "end", "end must be after start"),
        ("2026-11-03T08:00+02:00", "2026-11-03T09:00", "start", "must be a local time"),
        ("2026-11-03T08:00:30", "2026-11-03T09:00", "start", "whole minute"),
    ],
)
def test_inline_windows_are_validated(start: str, end: str, field: str, message: str) -> None:
    window = {"resource_id": "r1", "start": start, "end": end}
    rule = constraint("unavailable", hardness="hard", params={"windows": [window]})

    issue = only_issue(with_constraints(rule))

    assert issue.path == f"/constraints/1/params/windows/0/{field}"
    assert message in issue.message


def test_all_independent_problems_are_reported_together() -> None:
    doc = minimal_dm()
    doc["timezone"] = "Mars/Olympus"
    doc["language"] = "fr"

    assert sorted(paths(structural_issues(doc))) == ["/language", "/timezone"]


def test_issue_renders_as_path_and_message() -> None:
    assert str(ValidationIssue(path="/a/b", message="broken")) == "/a/b: broken"


def test_pointer_segments_are_escaped() -> None:
    doc = minimal_dm()
    doc["data"]["tasks"]["attributes"]["a/b~c"] = {"column": "x", "type": "str"}

    issue = only_issue(doc)

    assert issue.path == "/data/tasks/attributes/a~1b~0c"


def test_model_file_is_read_as_utf8_with_optional_bom(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["title"] = "Βάρδιες νοσηλευτών"
    model_file = tmp_path / "model.json"
    model_file.write_bytes(b"\xef\xbb\xbf" + json.dumps(doc, ensure_ascii=False).encode("utf-8"))

    assert load_decision_model(model_file).title == "Βάρδιες νοσηλευτών"


def test_unreadable_model_file_is_reported_at_the_root(tmp_path: Path) -> None:
    with pytest.raises(DecisionModelError) as caught:
        load_decision_model(tmp_path / "missing.json")

    [issue] = caught.value.issues
    assert issue.path == "/"
    assert issue.message.startswith("cannot read")


@pytest.mark.parametrize(
    ("rule", "path"),
    [
        (
            constraint("min_rest", hardness="hard", params={"hours": {"x": 1}}),
            "/constraints/1/params/hours",
        ),
        (
            constraint(
                "no_overlap",
                hardness="hard",
                scope={"tasks": [{"attribute": "type", "op": "eq", "value": {"a": 1}}]},
            ),
            "/constraints/1/scope/tasks/0/value",
        ),
    ],
)
def test_union_errors_point_at_the_field_in_the_document(rule: dict[str, Any], path: str) -> None:
    issue = only_issue(with_constraints(rule))

    assert issue.path == path
