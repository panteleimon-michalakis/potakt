"""The installed ``potakt`` command, run as a user would: output bytes and exit codes (A1, A2).

Every run gets an environment whose defaults would break UTF-8 output (``PYTHONIOENCODING`` set to
Latin-1, UTF-8 mode off), so the bytes are decided by Potakt, not by the OS or Python defaults.
CI runs these tests on Linux and Windows; the golden file must match byte for byte on both.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"
CLINIC_MODEL = REPO / "examples" / "clinic-el" / "model.json"
CLINIC_SOLUTION = REPO / "examples" / "clinic-el" / "solution.json"
CLINIC_TITLE = json.loads(CLINIC_MODEL.read_text(encoding="utf-8"))["title"]


def potakt_command() -> Path:
    """The console script installed next to the running interpreter (the venv's Scripts/bin)."""
    folder = Path(sys.executable).parent
    for name in ("potakt.exe", "potakt"):
        if (folder / name).is_file():
            return folder / name
    raise AssertionError(f"the potakt console script is not installed in {folder}")


def hostile_environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
    env["PYTHONIOENCODING"] = "latin-1"
    env["PYTHONUTF8"] = "0"
    return env


def potakt(*args: str | Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(potakt_command()), *map(str, args)],
        capture_output=True,
        env=hostile_environment(),
        timeout=120,
        check=False,
    )


# --- bytes (A2) ------------------------------------------------------------------------------


def test_check_json_matches_the_golden_file_byte_for_byte() -> None:
    result = potakt("check", "--json", CLINIC_MODEL, CLINIC_SOLUTION)

    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert result.stdout == (GOLDEN / "clinic-el.check.json").read_bytes()


def test_text_report_is_utf8_with_lf_line_endings() -> None:
    # The JSON report is ASCII; the text report starts with the Greek title of the model.
    result = potakt("check", CLINIC_MODEL, CLINIC_SOLUTION)

    assert result.returncode == 0, result.stderr
    assert b"\r\n" not in result.stdout
    assert result.stdout.startswith(f"{CLINIC_TITLE} (clinic-el-2026-w45 v1)\n".encode())
    assert b"\nScore: hard 0, soft 57 (feasible)\n" in result.stdout


def test_issues_on_stderr_are_utf8_with_lf_line_endings(tmp_path: Path) -> None:
    doc = json.loads(CLINIC_MODEL.read_text(encoding="utf-8"))
    doc["timezone"] = "Ευρώπη/Αθήνα"
    doc["language"] = "gr"
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    result = potakt("validate", model_file)

    assert result.returncode == 1
    assert result.stdout == b""
    assert b"\r\n" not in result.stderr
    assert sorted(result.stderr.decode("utf-8").splitlines()) == [
        "/language: Input should be 'en' or 'el'",
        "/timezone: unknown IANA timezone 'Ευρώπη/Αθήνα'",
    ]


@pytest.mark.parametrize("args", [("--version",), ("--help",), ("check", "--bogus")])
def test_help_version_and_usage_errors_use_lf(args: tuple[str, ...]) -> None:
    result = potakt(*args)

    assert result.stdout or result.stderr
    assert b"\r\n" not in result.stdout
    assert b"\r\n" not in result.stderr


@pytest.mark.parametrize(
    ("args", "committed"),
    [((), "decision-model.schema.json"), (("--solution",), "solution.schema.json")],
)
def test_schema_output_equals_the_committed_spec(args: tuple[str, ...], committed: str) -> None:
    result = potakt("schema", *args)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (REPO / "spec" / committed).read_bytes()


# --- exit codes (A1) -------------------------------------------------------------------------


def test_feasible_solution_exits_0() -> None:
    assert potakt("check", CLINIC_MODEL, CLINIC_SOLUTION).returncode == 0


def test_infeasible_solution_exits_3(tmp_path: Path) -> None:
    solution = json.loads(CLINIC_SOLUTION.read_text(encoding="utf-8"))
    del solution["assignments"][0]  # one task is now short of its demand: a hard violation
    infeasible = tmp_path / "solution.json"
    infeasible.write_text(json.dumps(solution), encoding="utf-8")

    result = potakt("check", CLINIC_MODEL, infeasible)

    assert result.returncode == 3
    assert b"(infeasible)\n" in result.stdout


def test_invalid_input_exits_1(tmp_path: Path) -> None:
    invalid = tmp_path / "solution.json"
    invalid.write_text("{", encoding="utf-8")

    result = potakt("check", CLINIC_MODEL, invalid)

    assert result.returncode == 1
    assert result.stdout == b""
    assert b"#/: invalid JSON" in result.stderr


def test_unknown_option_is_the_framework_usage_error_2() -> None:
    result = potakt("check", "--bogus", CLINIC_MODEL, CLINIC_SOLUTION)

    assert result.returncode == 2
    assert result.stdout == b""
    assert b"--bogus" in result.stderr
