import io
import json
import sys
from importlib.metadata import version
from pathlib import Path

import pytest
from helpers import TASKS_CSV, minimal_dm, write_scenario
from typer.testing import CliRunner

from potakt.cli import app, entry_point, use_utf8
from potakt.model import render_json_schema, render_solution_json_schema

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"potakt {version('potakt')}\n"


# --- schema ----------------------------------------------------------------------------------


def test_schema_prints_the_json_schema() -> None:
    result = runner.invoke(app, ["schema"])

    assert result.exit_code == 0
    assert result.stdout == render_json_schema()


def test_schema_writes_the_json_schema_with_lf_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "out" / "decision-model.schema.json"

    result = runner.invoke(app, ["schema", "--out", str(target)])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert target.read_bytes() == render_json_schema().encode("utf-8")


def test_schema_solution_prints_the_solution_schema() -> None:
    result = runner.invoke(app, ["schema", "--solution"])

    assert result.exit_code == 0
    assert result.stdout == render_solution_json_schema()


def test_schema_solution_writes_the_solution_schema(tmp_path: Path) -> None:
    target = tmp_path / "solution.schema.json"

    result = runner.invoke(app, ["schema", "--solution", "--out", str(target)])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert target.read_bytes() == render_solution_json_schema().encode("utf-8")


# --- validate --------------------------------------------------------------------------------


def test_validate_reports_ok_with_counts(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["title"] = "Βάρδιες κλινικής"
    model_file = write_scenario(tmp_path, doc)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 0
    assert result.stdout == "OK: Βάρδιες κλινικής — 3 resources, 2 tasks, 1 constraints\n"
    assert result.stderr == ""


def test_validate_reports_structural_issues_one_per_line_on_stderr(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["timezone"] = "Mars/Olympus"
    doc["language"] = "fr"
    model_file = write_scenario(tmp_path, doc)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert sorted(result.stderr.splitlines()) == [
        "/language: Input should be 'en' or 'el'",
        "/timezone: unknown IANA timezone 'Mars/Olympus'",
    ]


def test_validate_reports_data_loading_issues_on_stderr(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "tasks.csv").unlink()

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "/data/tasks/source: cannot read tasks.csv: file not found\n"


def test_validate_reports_data_level_issues_on_stderr(tmp_path: Path) -> None:
    duplicate = TASKS_CSV + "t2,2026-11-03T23:00,2026-11-04T07:00,1,νύχτα,als\n"
    model_file = write_scenario(tmp_path, tasks=duplicate)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "/data/tasks/id: duplicate task id 't2' in tasks.csv\n"


def test_validate_reports_a_missing_model_file_on_stderr(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith("/: cannot read ")


# --- output streams (A2) ---------------------------------------------------------------------


def test_output_streams_are_switched_to_utf8() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")

    use_utf8(stream)
    stream.write("Βάρδιες")
    stream.flush()

    assert raw.getvalue() == "Βάρδιες".encode()


def test_lf_is_forced_even_when_the_stream_is_already_utf8() -> None:
    # newline="\r\n" translates on every OS, so this needs no Windows to show the translation.
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", newline="\r\n")

    text = "\n".join(["Βάρδιες", "Νύχτα", ""])

    use_utf8(stream)
    stream.write(text)
    stream.flush()

    assert raw.getvalue() == text.encode()
    assert b"\r" not in raw.getvalue()


def test_the_stream_error_handler_is_kept() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="backslashreplace")

    use_utf8(stream)

    assert (stream.encoding, stream.errors) == ("utf-8", "backslashreplace")


def test_entry_point_switches_the_streams_before_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    # --version is handled while arguments are parsed, before any command callback runs.
    raw = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(raw, encoding="cp1252", newline="\r\n"))
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    monkeypatch.setattr(sys, "argv", ["potakt", "--version"])

    with pytest.raises(SystemExit) as stopped:
        entry_point()

    sys.stdout.flush()
    assert stopped.value.code == 0
    assert raw.getvalue() == f"potakt {version('potakt')}\n".encode()


# --- check -----------------------------------------------------------------------------------


def write_solution(directory: Path, assignments: list[tuple[str, str]], **fields: object) -> Path:
    doc: dict[str, object] = {
        "schema_version": "0.1",
        "model_id": "dm-test",
        "model_version": 1,
        "assignments": [{"task_id": t, "resource_id": r} for t, r in assignments],
    }
    doc.update(fields)
    target = directory / "solution.json"
    target.write_text(json.dumps(doc), encoding="utf-8")
    return target


FEASIBLE = [("t1", "r1"), ("t1", "r2"), ("t2", "r3")]


def test_check_prints_the_report_and_exits_0_when_feasible(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    solution_file = write_solution(tmp_path, FEASIBLE)

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exit_code == 0
    assert result.stdout == (
        "Test model (dm-test v1)\n"
        "Score: hard 0, soft 0 (feasible)\n"
        "\n"
        "Rules without violations: coverage\n"
    )
    assert result.stderr == ""


def test_check_exits_3_when_infeasible(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    solution_file = write_solution(tmp_path, [("t1", "r1"), ("t2", "r3")])

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exit_code == 3
    assert "Score: hard 1, soft 0 (infeasible)\n" in result.stdout
    assert "  - task t1: 1 assigned, demand 2\n" in result.stdout
    assert result.stderr == ""


def test_check_json_is_valid_and_deterministic(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    solution_file = write_solution(tmp_path, [("t1", "r1"), ("t2", "r3")])
    args = ["check", str(model_file), str(solution_file), "--json"]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 3
    assert first.stdout_bytes == second.stdout_bytes
    document = json.loads(first.stdout)
    assert document["score"] == {"hard": 1, "soft": 0, "feasible": False}
    assert document["rules"][0]["violations"][0]["task_ids"] == ["t1"]


def test_check_help_documents_the_exit_codes() -> None:
    result = runner.invoke(app, ["check", "--help"], terminal_width=200)

    assert result.exit_code == 0
    text = " ".join(result.stdout.split())
    assert "Exit code 0: feasible; 1: invalid input; 3: infeasible" in text
    assert "2: usage error" in text


def test_check_reports_model_issues_on_stderr_and_exits_1(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["timezone"] = "Mars/Olympus"
    model_file = write_scenario(tmp_path, doc)
    solution_file = write_solution(tmp_path, FEASIBLE)

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "/timezone: unknown IANA timezone 'Mars/Olympus'\n"


def test_check_reports_solution_issues_with_the_solution_file(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    solution_file = write_solution(tmp_path, [*FEASIBLE, ("t9", "r1")], model_version=2)

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.splitlines() == [
        f"{solution_file}#/model_version: solution is for model version 2, not 1",
        f"{solution_file}#/assignments/3/task_id: unknown task id 't9'",
    ]


def test_check_reports_model_and_solution_structure_issues_together(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "tasks.csv").unlink()
    solution_file = write_solution(tmp_path, [("t1", "r1"), ("t1", "r1")])

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "/data/tasks/source: cannot read tasks.csv: file not found",
        f"{solution_file}#/assignments/1: duplicate assignment of resource 'r1' to task 't1' "
        "(first at /assignments/0)",
    ]


def test_check_reports_deeply_nested_files_without_a_traceback(tmp_path: Path) -> None:
    nested = "[" * 100_000 + "]" * 100_000
    model_file = write_scenario(tmp_path)
    model_file.write_text(nested, encoding="utf-8")
    solution_file = tmp_path / "solution.json"
    solution_file.write_text(nested, encoding="utf-8")

    result = runner.invoke(app, ["check", str(model_file), str(solution_file)])

    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.exit_code == 1
    assert result.stderr.splitlines() == [
        "/: JSON too deeply nested",
        f"{solution_file}#/: JSON too deeply nested",
    ]


def test_check_reports_a_missing_solution_file(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    missing = tmp_path / "nope.json"

    result = runner.invoke(app, ["check", str(model_file), str(missing)])

    assert result.exit_code == 1
    assert result.stderr.startswith(f"{missing}#/: cannot read ")
