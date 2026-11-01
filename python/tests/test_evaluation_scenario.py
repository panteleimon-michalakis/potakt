"""Scenario metadata and the corpus loader (ADR-0007 §2, TASK-005a D2)."""

import json
from pathlib import Path

import pytest
from evaluation_helpers import REQUEST, scenario_metadata, write_json, write_scenario_dir
from helpers import minimal_dm

from potakt.evaluation import SCENARIO_TAGS, ScenarioError, ScenarioIssue, load_scenario


def issues_of(directory: Path) -> list[str]:
    with pytest.raises(ScenarioError) as raised:
        load_scenario(directory)
    return [str(issue) for issue in raised.value.issues]


def test_a_valid_scenario_loads_every_part(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)

    loaded = load_scenario(directory)

    assert loaded.directory == directory
    assert loaded.scenario.id == "01-test"
    assert loaded.scenario.language == "en"
    assert loaded.scenario.reference_score.hard == 0
    assert loaded.scenario.disruption.resource_id == "r1"
    assert loaded.scenario.unsupported == []
    assert loaded.scenario.notes is None
    assert loaded.request == REQUEST
    assert loaded.truth.id == "dm-test"
    assert [r.id for r in loaded.instance.resources] == ["r1", "r2", "r3"]
    assert len(loaded.solution.assignments) == 3


def test_request_line_endings_are_read_as_newlines(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "request.md").write_bytes(b"line one\r\nline two\r\n")

    assert load_scenario(directory).request == "line one\nline two\n"


def test_optional_notes_and_unsupported_requests_are_kept(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["notes"] = "built by hand"
    metadata["unsupported"] = [{"quote": "the two shifts", "reason": "not a rule"}]
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    scenario = load_scenario(directory).scenario

    assert scenario.notes == "built by hand"
    assert [(u.quote, u.reason) for u in scenario.unsupported] == [("the two shifts", "not a rule")]


def test_tags_are_optional(tmp_path: Path) -> None:
    assert load_scenario(write_scenario_dir(tmp_path)).scenario.tags == []


def test_the_tags_are_a_closed_set() -> None:
    assert SCENARIO_TAGS == (
        "numbers-in-words",
        "irrelevant-information",
        "per-resource-limit",
        "non-default-csv",
        "windows-1253",
        "stability-baseline",
        "owner-written",
        "owner-edited",
        "web-grounded",
    )


def test_known_tags_are_kept_in_order(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["tags"] = list(reversed(SCENARIO_TAGS))
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert load_scenario(directory).scenario.tags == list(reversed(SCENARIO_TAGS))


@pytest.mark.parametrize(
    ("tags", "path"),
    [(["web-grounded", "greek"], "/tags/1"), ("web-grounded", "/tags"), ([""], "/tags/0")],
)
def test_an_unknown_tag_is_an_issue(tmp_path: Path, tags: object, path: str) -> None:
    metadata = scenario_metadata()
    metadata["tags"] = tags
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert any(issue.startswith(f"scenario.json#{path}: ") for issue in issues_of(directory))


def test_a_tag_listed_twice_is_an_issue(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["tags"] = ["web-grounded", "windows-1253", "web-grounded"]
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert issues_of(directory) == ["scenario.json#/tags/2: web-grounded is listed twice"]


@pytest.mark.parametrize(
    ("field", "value", "path"),
    [
        ("language", "de", "/language"),
        ("difficulty", "extreme", "/difficulty"),
        ("style", "letter", "/style"),
        ("id", "1-test", "/id"),
        ("id", "01_test", "/id"),
        ("title", "", "/title"),
        ("csv_files", [], "/csv_files"),
        ("csv_files", ["../tasks.csv"], "/csv_files/0"),
        ("csv_files", ["tasks.txt"], "/csv_files/0"),
        ("reference_score", {"hard": "0", "soft": 0}, "/reference_score/hard"),
        ("unsupported", [{"quote": "x"}], "/unsupported/0/reason"),
        ("notes", "", "/notes"),
        ("colour", "blue", "/colour"),
    ],
)
def test_metadata_is_strict(tmp_path: Path, field: str, value: object, path: str) -> None:
    metadata = scenario_metadata()
    metadata[field] = value
    directory = write_scenario_dir(tmp_path, name="01-test", metadata=metadata)

    issues = issues_of(directory)

    assert any(issue.startswith(f"scenario.json#{path}: ") for issue in issues), issues


@pytest.mark.parametrize("field", ["title", "domain", "csv_files", "disruption", "unsupported"])
def test_required_metadata_fields(tmp_path: Path, field: str) -> None:
    metadata = scenario_metadata()
    del metadata[field]
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert f"scenario.json#/{field}: Field required" in issues_of(directory)


def test_invalid_json_is_an_issue_at_the_root(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "scenario.json").write_text("{not json", encoding="utf-8")

    (issue,) = issues_of(directory)

    assert issue.startswith("scenario.json#/: invalid JSON at line 1, column 2")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[" * 100_000, "JSON nested more than 64 levels deep"),
        ("[" * 100_000 + "]" * 100_000, "JSON nested more than 64 levels deep"),
        ("1" * 5_000, "JSON number too large"),
    ],
    ids=["unbalanced", "balanced", "huge-integer"],
)
def test_pathological_json_is_an_issue_not_a_crash(tmp_path: Path, text: str, message: str) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "scenario.json").write_text(text, encoding="utf-8")

    assert issues_of(directory) == [f"scenario.json#/: {message}"]


def test_metadata_with_a_utf8_byte_order_mark_loads(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    path = directory / "scenario.json"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())

    assert load_scenario(directory).scenario.id == "01-test"


def test_a_request_that_is_not_utf8_is_an_issue(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "request.md").write_bytes("Καλημέρα".encode("cp1253"))

    assert issues_of(directory) == ["request.md#/: not valid UTF-8"]


def test_a_file_that_cannot_be_read_is_an_issue(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "request.md").unlink()
    (directory / "request.md").mkdir()

    (issue,) = issues_of(directory)

    # The OS reason differs (Windows: permission, Linux: is a directory); our prefix does not.
    assert issue.startswith("request.md#/: cannot read: ")


def test_the_id_must_equal_the_directory_name(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path, name="02-other")

    assert issues_of(directory) == [
        "scenario.json#/id: id '01-test' must equal the directory name '02-other'"
    ]


def test_listed_csv_files_must_be_exactly_the_csv_files_present(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["csv_files"] = ["resources.csv", "missing.csv"]
    directory = write_scenario_dir(tmp_path, metadata=metadata)
    (directory / "extra.CSV").write_text("id\n", encoding="utf-8")  # not ".csv": not a CSV file

    assert issues_of(directory) == [
        "scenario.json#/csv_files: listed but not present: missing.csv; "
        "present but not listed: tasks.csv"
    ]


def test_csv_files_must_not_repeat(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["csv_files"] = ["resources.csv", "tasks.csv", "tasks.csv"]
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert issues_of(directory) == ["scenario.json#/csv_files/2: tasks.csv is listed twice"]


@pytest.mark.parametrize(
    ("disruption", "issue"),
    [
        (
            {"resource_id": "r9", "start": "2026-11-02T00:00", "end": "2026-11-03T00:00"},
            "scenario.json#/disruption/resource_id: unknown resource id 'r9'",
        ),
        (
            {"resource_id": "r1", "start": "2026-11-01T23:00", "end": "2026-11-02T08:00"},
            "scenario.json#/disruption/start: 2026-11-01T23:00 is before the horizon start "
            "2026-11-02T00:00",
        ),
        (
            {"resource_id": "r1", "start": "2026-11-08T20:00", "end": "2026-11-09T00:01"},
            "scenario.json#/disruption/end: 2026-11-09T00:01 is after the horizon end "
            "2026-11-09T00:00",
        ),
    ],
)
def test_the_disruption_names_a_resource_and_stays_inside_the_horizon(
    tmp_path: Path, disruption: dict[str, str], issue: str
) -> None:
    metadata = scenario_metadata()
    metadata["disruption"] = disruption
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert issues_of(directory) == [issue]


def test_the_disruption_window_must_not_be_empty(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["disruption"]["end"] = metadata["disruption"]["start"]
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert issues_of(directory) == ["scenario.json#/disruption/end: end must be after start"]


def test_the_disruption_uses_local_whole_minute_datetimes(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["disruption"]["start"] = "2026-11-02T00:00+02:00"
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    (issue,) = issues_of(directory)

    assert issue.startswith("scenario.json#/disruption/start: ")
    assert "without a UTC offset" in issue


def test_language_must_equal_the_truth_language(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["language"] = "el"
    directory = write_scenario_dir(tmp_path, metadata=metadata)

    assert issues_of(directory) == [
        "scenario.json#/language: language 'el' differs from the truth language 'en'"
    ]


def test_every_missing_file_is_reported(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    for name in ("request.md", "scenario.json", "truth.json", "solution.json"):
        (directory / name).unlink()

    assert issues_of(directory) == [
        "scenario.json#/: file not found",
        "request.md#/: file not found",
        "truth.json#/: file not found",
        "solution.json#/: file not found",
    ]


def test_an_empty_request_is_an_issue(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path, request=" \n")

    assert issues_of(directory) == ["request.md#/: the request is empty"]


def test_a_directory_that_does_not_exist_is_an_issue(tmp_path: Path) -> None:
    assert issues_of(tmp_path / "03-none") == ["03-none#/: not a scenario directory"]


def test_truth_issues_name_the_truth_file(tmp_path: Path) -> None:
    truth = minimal_dm()
    truth["constraints"][0]["hardness"] = "soft"
    directory = write_scenario_dir(tmp_path, truth=truth)

    issues = issues_of(directory)

    assert issues == [
        "truth.json#/constraints/0/hardness: template 'coverage' allows only hard rules"
    ]


def test_truth_data_issues_are_reported(tmp_path: Path) -> None:
    truth = minimal_dm()
    truth["constraints"].append(
        {
            "id": "apart",
            "template": "apart",
            "hardness": "hard",
            "params": {"resource_ids": ["r1", "r9"]},
            "rationale": "test",
            "provenance": {"kind": "default"},
        }
    )
    directory = write_scenario_dir(tmp_path, truth=truth)

    assert issues_of(directory) == [
        "truth.json#/constraints/1/params/resource_ids/1: unknown resource id 'r9'"
    ]


def test_csv_issues_of_the_truth_are_reported(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    (directory / "tasks.csv").write_text("id,start\n", encoding="utf-8")

    issues = issues_of(directory)

    assert "truth.json#/data/tasks/end: column 'end' not found in tasks.csv" in issues


def test_solution_issues_name_the_solution_file(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    solution = json.loads((directory / "solution.json").read_text(encoding="utf-8"))
    solution["assignments"].append({"task_id": "t9", "resource_id": "r1"})
    write_json(directory / "solution.json", solution)

    assert issues_of(directory) == ["solution.json#/assignments/3/task_id: unknown task id 't9'"]


def test_structurally_invalid_solution_is_reported(tmp_path: Path) -> None:
    directory = write_scenario_dir(tmp_path)
    write_json(directory / "solution.json", {"schema_version": "0.1"})

    issues = issues_of(directory)

    assert "solution.json#/model_id: Field required" in issues


def test_issues_of_several_files_are_reported_together(tmp_path: Path) -> None:
    metadata = scenario_metadata()
    metadata["style"] = "letter"
    directory = write_scenario_dir(tmp_path, metadata=metadata, request="")
    write_json(directory / "solution.json", {"schema_version": "0.1"})

    files = {issue.split("#", 1)[0] for issue in issues_of(directory)}

    assert files == {"scenario.json", "request.md", "solution.json"}


def test_scenario_error_message_lists_every_issue() -> None:
    issues = [ScenarioIssue("a.json", "/x", "bad"), ScenarioIssue("b.md", "/", "worse")]

    error = ScenarioError(issues)

    assert error.issues == tuple(issues)
    assert str(error) == "a.json#/x: bad\nb.md#/: worse"
