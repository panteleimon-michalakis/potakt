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
RETAIL_MODEL = REPO / "examples" / "retail-en" / "model.json"
RETAIL_SOLUTION = REPO / "examples" / "retail-en" / "solution.json"
# Two moved shifts put one full-timer and one part-timer under their own contract hours.
RETAIL_MOVES = {("tue-open", "s1"): "s6", ("sat-mid", "s7"): "s2"}


def retail_variant(directory: Path) -> Path:
    """The retail reference solution with ``RETAIL_MOVES`` applied, written to ``directory``."""
    solution = json.loads(RETAIL_SOLUTION.read_text(encoding="utf-8"))
    for assignment in solution["assignments"]:
        key = (assignment["task_id"], assignment["resource_id"])
        assignment["resource_id"] = RETAIL_MOVES.get(key, assignment["resource_id"])
    target = directory / "solution.json"
    target.write_text(json.dumps(solution), encoding="utf-8")
    return target


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


def potakt(
    *args: str | Path, extra_env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [str(potakt_command()), *map(str, args)],
        capture_output=True,
        env=hostile_environment() | (extra_env or {}),
        timeout=120,
        check=False,
    )


# --- bytes (A2) ------------------------------------------------------------------------------

FORCED_COLOUR = {"GITHUB_ACTIONS": "true", "FORCE_COLOR": "1"}


def test_machine_output_has_no_ansi_codes_even_when_colour_is_forced(tmp_path: Path) -> None:
    # Help and usage errors may be coloured; reports, JSON and issues never are.
    as_json = potakt("check", "--json", CLINIC_MODEL, CLINIC_SOLUTION, extra_env=FORCED_COLOUR)
    as_text = potakt("check", CLINIC_MODEL, CLINIC_SOLUTION, extra_env=FORCED_COLOUR)
    valid = potakt("validate", CLINIC_MODEL, extra_env=FORCED_COLOUR)
    broken = tmp_path / "model.json"
    broken.write_text('{"schema_version": "0.1"}', encoding="utf-8")
    invalid = potakt("validate", broken, extra_env=FORCED_COLOUR)

    assert as_json.stdout == (GOLDEN / "clinic-el.check.json").read_bytes()
    assert as_text.stdout == (GOLDEN / "clinic-el.check.txt").read_bytes()
    assert (valid.returncode, invalid.returncode) == (0, 1)
    assert valid.stdout.startswith(b"OK: ")
    assert invalid.stderr.startswith(b"/")
    for result in (as_json, as_text, valid, invalid):
        assert b"\x1b" not in result.stdout + result.stderr


def test_check_json_matches_the_golden_file_byte_for_byte() -> None:
    result = potakt("check", "--json", CLINIC_MODEL, CLINIC_SOLUTION)

    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert result.stdout == (GOLDEN / "clinic-el.check.json").read_bytes()


def test_retail_json_with_per_person_limits_matches_the_golden_file(tmp_path: Path) -> None:
    result = potakt("check", "--json", RETAIL_MODEL, retail_variant(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout == (GOLDEN / "retail-en.check.json").read_bytes()
    assert b"24 h worked, minimum 32 h" in result.stdout  # s1, full-time
    assert b"12 h worked, minimum 16 h" in result.stdout  # s7, part-time


def test_text_report_matches_the_golden_file_byte_for_byte() -> None:
    # The JSON golden is ASCII; this one starts with the Greek title, so CI compares Greek bytes.
    result = potakt("check", CLINIC_MODEL, CLINIC_SOLUTION)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (GOLDEN / "clinic-el.check.txt").read_bytes()


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


# --- eval (TASK-005c1 D5) --------------------------------------------------------------------

SCENARIOS = REPO / "docs" / "validation" / "scenarios"
HOTEL = SCENARIOS / "02-hotel-reception"
PHARMACY = SCENARIOS / "01-pharmacy"
OPUS_RUNS = Path(__file__).resolve().parent / "fixtures" / "test_a_runs" / "claude-opus-5-5"


def test_eval_prompt_matches_the_golden_byte_for_byte_even_when_colour_is_forced() -> None:
    result = potakt("eval", "prompt", HOTEL, "--model", "claude-opus-5-5", extra_env=FORCED_COLOUR)

    assert result.returncode == 0, result.stderr
    assert result.stderr == b""
    assert result.stdout == (GOLDEN / "eval-prompt-02-claude-opus-5-5.json").read_bytes()


def test_eval_summary_matches_the_golden_byte_for_byte() -> None:
    result = potakt("eval", "summary", OPUS_RUNS, "--corpus", SCENARIOS)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (GOLDEN / "test-a-summary.txt").read_bytes()


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("prompt-ok", 0),
        ("prompt-gemini", 0),
        ("prompt-invalid-scenario", 1),
        ("prompt-unknown-claude-model", 2),
        ("compare-valid", 0),
        ("compare-invalid-candidate", 0),
        ("compare-missing-candidate", 1),
        ("compare-invalid-scenario", 1),
        ("summary-ok", 0),
        ("summary-json", 0),
        ("summary-missing-directory", 1),
    ],
)
def test_eval_exit_codes(case: str, expected: int, tmp_path: Path) -> None:
    truth = tmp_path / "truth-candidate.txt"
    truth.write_bytes((PHARMACY / "truth.json").read_bytes())
    junk = tmp_path / "junk.txt"
    junk.write_bytes(b"Sorry, no model today.\n")
    args: dict[str, tuple[str | Path, ...]] = {
        "prompt-ok": ("prompt", HOTEL),
        "prompt-gemini": ("prompt", HOTEL, "--model", "gemini-3.8-flash"),
        "prompt-invalid-scenario": ("prompt", tmp_path),
        "prompt-unknown-claude-model": ("prompt", HOTEL, "--model", "claude-sonnet-5-5"),
        "compare-valid": ("compare", PHARMACY, truth),
        "compare-invalid-candidate": ("compare", PHARMACY, junk),
        "compare-missing-candidate": ("compare", PHARMACY, tmp_path / "nope.txt"),
        "compare-invalid-scenario": ("compare", tmp_path, truth),
        "summary-ok": ("summary", OPUS_RUNS),
        "summary-json": ("summary", OPUS_RUNS, "--json"),
        "summary-missing-directory": ("summary", tmp_path / "nope"),
    }

    result = potakt("eval", *args[case])

    assert result.returncode == expected, result.stderr
    assert b"\r\n" not in result.stdout + result.stderr
    if expected == 0:
        assert result.stderr == b""
        assert result.stdout
    else:
        assert result.stdout == b""
        assert result.stderr
