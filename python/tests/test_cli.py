import io
from importlib.metadata import version
from pathlib import Path

from helpers import TASKS_CSV, minimal_dm, write_scenario
from typer.testing import CliRunner

from potakt.cli import app, use_utf8
from potakt.model import render_json_schema

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


# --- validate --------------------------------------------------------------------------------


def test_validate_reports_ok_with_counts(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["title"] = "Βάρδιες κλινικής"
    model_file = write_scenario(tmp_path, doc)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 0
    assert result.stdout == "OK: Βάρδιες κλινικής — 3 resources, 2 tasks, 1 constraints\n"


def test_validate_reports_structural_issues_one_per_line(tmp_path: Path) -> None:
    doc = minimal_dm()
    doc["timezone"] = "Mars/Olympus"
    doc["language"] = "fr"
    model_file = write_scenario(tmp_path, doc)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert sorted(result.stdout.splitlines()) == [
        "/language: Input should be 'en' or 'el'",
        "/timezone: unknown IANA timezone 'Mars/Olympus'",
    ]


def test_validate_reports_data_loading_issues(tmp_path: Path) -> None:
    model_file = write_scenario(tmp_path)
    (tmp_path / "tasks.csv").unlink()

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert result.stdout == "/data/tasks/source: cannot read tasks.csv: file not found\n"


def test_validate_reports_data_level_issues(tmp_path: Path) -> None:
    duplicate = TASKS_CSV + "t2,2026-11-03T23:00,2026-11-04T07:00,1,νύχτα,als\n"
    model_file = write_scenario(tmp_path, tasks=duplicate)

    result = runner.invoke(app, ["validate", str(model_file)])

    assert result.exit_code == 1
    assert result.stdout == "/data/tasks/id: duplicate task id 't2' in tasks.csv\n"


def test_validate_reports_a_missing_model_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["validate", str(tmp_path / "nope.json")])

    assert result.exit_code == 1
    assert result.stdout.startswith("/: cannot read ")


def test_output_streams_are_switched_to_utf8() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")

    use_utf8(stream)
    stream.write("Βάρδιες")
    stream.flush()

    assert raw.getvalue() == "Βάρδιες".encode()
