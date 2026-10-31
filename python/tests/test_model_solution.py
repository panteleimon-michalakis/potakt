"""Solutions: structural validation, data-level validation against a DM, schema drift."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from helpers import paths, write_scenario
from pydantic import ValidationError

from potakt.model import (
    DecisionModel,
    DecisionModelError,
    ProblemInstance,
    ValidationIssue,
    load_decision_model,
    load_instance,
    load_solution,
    parse_solution,
    render_solution_json_schema,
    solution_json_schema,
    validate_solution,
)

SCHEMA_FILE = Path(__file__).resolve().parents[2] / "spec" / "solution.schema.json"

_MINIMAL_SOLUTION: dict[str, Any] = {
    "schema_version": "0.1",
    "model_id": "dm-test",
    "model_version": 1,
    "assignments": [
        {"task_id": "t1", "resource_id": "r1"},
        {"task_id": "t1", "resource_id": "r2"},
        {"task_id": "t2", "resource_id": "r3"},
    ],
}


def minimal_solution() -> dict[str, Any]:
    return copy.deepcopy(_MINIMAL_SOLUTION)


def solution_issues(doc: dict[str, Any]) -> list[ValidationIssue]:
    try:
        parse_solution(json.dumps(doc))
    except DecisionModelError as error:
        return list(error.issues)
    return []


@pytest.fixture
def scenario(tmp_path: Path) -> tuple[DecisionModel, ProblemInstance]:
    model_file = write_scenario(tmp_path)
    dm = load_decision_model(model_file)
    return dm, load_instance(dm, model_file.parent)


def data_issues(
    scenario: tuple[DecisionModel, ProblemInstance], doc: dict[str, Any]
) -> list[ValidationIssue]:
    dm, instance = scenario
    return validate_solution(dm, instance, parse_solution(json.dumps(doc)))


# --- structure -------------------------------------------------------------------------------


def test_minimal_solution_parses_into_frozen_assignments() -> None:
    solution = parse_solution(json.dumps(minimal_solution()))

    assert solution.model_id == "dm-test"
    assert solution.model_version == 1
    assert [(a.task_id, a.resource_id) for a in solution.assignments] == [
        ("t1", "r1"),
        ("t1", "r2"),
        ("t2", "r3"),
    ]
    assert solution.metadata is None
    with pytest.raises(ValidationError):
        solution.assignments[0].task_id = "t2"  # type: ignore[misc]


def test_metadata_fields_are_optional_text() -> None:
    doc = minimal_solution()
    doc["metadata"] = {"produced_by": "reviewer", "notes": "σημειώσεις"}

    solution = parse_solution(json.dumps(doc, ensure_ascii=False))

    assert solution.metadata is not None
    assert solution.metadata.produced_by == "reviewer"
    assert solution.metadata.created_at is None
    assert solution.metadata.notes == "σημειώσεις"


def test_empty_assignment_list_is_a_valid_solution() -> None:
    doc = minimal_solution()
    doc["assignments"] = []

    assert solution_issues(doc) == []


@pytest.mark.parametrize(
    ("change", "path"),
    [
        (lambda d: d.update(extra=1), "/extra"),
        (lambda d: d["assignments"][0].update(shift="t1"), "/assignments/0/shift"),
        (lambda d: d.update(metadata={"author": "x"}), "/metadata/author"),
        (lambda d: d.update(schema_version="0.2"), "/schema_version"),
        (lambda d: d.update(model_version="1"), "/model_version"),
        (lambda d: d.update(model_version=0), "/model_version"),
        (lambda d: d.update(model_id=""), "/model_id"),
        (lambda d: d["assignments"][1].update(resource_id=""), "/assignments/1/resource_id"),
        (lambda d: d["assignments"][2].pop("task_id"), "/assignments/2/task_id"),
    ],
)
def test_structural_errors_point_into_the_solution(change: Any, path: str) -> None:
    doc = minimal_solution()
    change(doc)

    assert paths(solution_issues(doc)) == [path]


def test_duplicate_assignment_pairs_are_rejected() -> None:
    doc = minimal_solution()
    doc["assignments"].append({"task_id": "t1", "resource_id": "r2"})

    issues = solution_issues(doc)

    assert [str(issue) for issue in issues] == [
        "/assignments/3: duplicate assignment of resource 'r2' to task 't1' "
        "(first at /assignments/1)"
    ]


def test_weight_in_a_solution_keeps_the_generic_message() -> None:
    doc = minimal_solution()
    doc["assignments"][0]["weight"] = 2

    assert [str(issue) for issue in solution_issues(doc)] == [
        "/assignments/0/weight: Extra inputs are not permitted"
    ]


def test_invalid_json_is_reported_at_the_root() -> None:
    with pytest.raises(DecisionModelError) as caught:
        parse_solution("{")

    assert paths(list(caught.value.issues)) == ["/"]
    assert caught.value.issues[0].message.startswith("invalid JSON at line 1")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[" * 100_000 + "]" * 100_000, "JSON too deeply nested"),
        ('{"model_version": ' + "9" * 5000 + "}", "JSON number too large"),
    ],
)
def test_pathological_json_is_an_issue_at_the_root(text: str, message: str) -> None:
    with pytest.raises(DecisionModelError) as caught:
        parse_solution(text)

    assert [str(issue) for issue in caught.value.issues] == [f"/: {message}"]


def test_load_solution_reads_utf8_files(tmp_path: Path) -> None:
    target = tmp_path / "solution.json"
    target.write_text(json.dumps(minimal_solution()), encoding="utf-8-sig")

    assert load_solution(target).model_id == "dm-test"


def test_load_solution_reports_an_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(DecisionModelError) as caught:
        load_solution(tmp_path / "missing.json")

    issue = caught.value.issues[0]
    assert issue.path == "/"
    assert issue.message.startswith("cannot read ")


# --- data level ------------------------------------------------------------------------------


def test_matching_solution_has_no_data_issues(
    scenario: tuple[DecisionModel, ProblemInstance],
) -> None:
    assert data_issues(scenario, minimal_solution()) == []


def test_unassigned_tasks_are_allowed(scenario: tuple[DecisionModel, ProblemInstance]) -> None:
    doc = minimal_solution()
    doc["assignments"] = [{"task_id": "t1", "resource_id": "r1"}]

    assert data_issues(scenario, doc) == []


def test_solution_must_name_the_model_id_and_version(
    scenario: tuple[DecisionModel, ProblemInstance],
) -> None:
    doc = minimal_solution()
    doc["model_id"] = "other"
    doc["model_version"] = 2

    assert [str(issue) for issue in data_issues(scenario, doc)] == [
        "/model_id: solution is for model 'other', not 'dm-test'",
        "/model_version: solution is for model version 2, not 1",
    ]


def test_unknown_task_and_resource_ids_are_reported(
    scenario: tuple[DecisionModel, ProblemInstance],
) -> None:
    doc = minimal_solution()
    doc["assignments"][1] = {"task_id": "t9", "resource_id": "r1"}
    doc["assignments"][2] = {"task_id": "t2", "resource_id": "r9"}

    assert [str(issue) for issue in data_issues(scenario, doc)] == [
        "/assignments/1/task_id: unknown task id 't9'",
        "/assignments/2/resource_id: unknown resource id 'r9'",
    ]


# --- schema ----------------------------------------------------------------------------------


def test_committed_solution_schema_matches_generated_schema() -> None:
    committed = SCHEMA_FILE.read_text(encoding="utf-8")

    assert committed == render_solution_json_schema(), (
        "spec/solution.schema.json is out of date; regenerate it (from python/) with\n"
        'uv run python -c "from potakt.model import render_solution_json_schema as r; '
        "open('../spec/solution.schema.json', 'w', encoding='utf-8', newline='\\n')"
        '.write(r())"'
    )


def test_solution_schema_declares_dialect_and_forbids_unknown_fields() -> None:
    schema = solution_json_schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["schema_version", "model_id", "model_version", "assignments"]
    assert json.loads(render_solution_json_schema()) == schema
