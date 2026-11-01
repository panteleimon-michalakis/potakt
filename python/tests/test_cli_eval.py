"""The offline ``potakt eval`` commands (TASK-005c1 D5): prompt, compare and summary."""

import json
from pathlib import Path

import pytest
from evaluation_helpers import scenario_metadata, write_scenario_dir
from helpers import constraint, minimal_dm
from typer.testing import CliRunner

import potakt.evaluation.test_a as test_a_module
from potakt.cli import app
from potakt.evaluation import CORPUS_PATH

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "docs" / "validation" / "scenarios"
HOTEL = CORPUS / "02-hotel-reception"
PHARMACY = CORPUS / "01-pharmacy"
GOLDEN = Path(__file__).resolve().parent / "golden"
OPUS_RUNS = Path(__file__).resolve().parent / "fixtures" / "test_a_runs" / "claude-opus-5-5"

runner = CliRunner()


def help_text(*command: str) -> str:
    """The help as one line: Rich panel borders removed, whitespace collapsed (Rich may wrap an
    option's help over several rows of its panel)."""
    result = runner.invoke(app, [*command, "--help"], terminal_width=200)
    assert result.exit_code == 0
    return " ".join(result.stdout.replace("\N{BOX DRAWINGS LIGHT VERTICAL}", " ").split())


# --- eval prompt -----------------------------------------------------------------------------


def test_prompt_prints_the_opus_payload_of_02_as_the_golden() -> None:
    result = runner.invoke(app, ["eval", "prompt", str(HOTEL), "--model", "claude-opus-5-5"])

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout_bytes == (GOLDEN / "eval-prompt-02-claude-opus-5-5.json").read_bytes()


def test_prompt_defaults_to_the_strong_model() -> None:
    default = runner.invoke(app, ["eval", "prompt", str(HOTEL)])
    opus = runner.invoke(app, ["eval", "prompt", str(HOTEL), "--model", "claude-opus-5-5"])

    assert default.exit_code == 0
    assert default.stdout_bytes == opus.stdout_bytes


def test_prompt_for_gemini_uses_the_gemini_shape() -> None:
    result = runner.invoke(app, ["eval", "prompt", str(PHARMACY), "--model", "gemini-3.8-flash"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert sorted(payload) == ["contents", "generationConfig", "systemInstruction"]
    assert payload["generationConfig"] == {"maxOutputTokens": 32000}


def test_prompt_refuses_a_claude_model_without_settings_as_a_usage_error() -> None:
    result = runner.invoke(app, ["eval", "prompt", str(HOTEL), "--model", "claude-sonnet-5-5"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "no pre-registered Test A settings" in result.stderr


def test_prompt_reports_an_invalid_scenario_and_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "prompt", str(tmp_path)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "scenario.json#/: file not found" in result.stderr.splitlines()


def absences_rule(rule_id: str, **dialect: str) -> dict[str, object]:
    source: dict[str, object] = {
        "source": "absences.csv",
        "resource_id": "person",
        "start": "from",
        "end": "to",
    }
    if dialect:
        source["csv"] = dialect
    return constraint("unavailable", id=rule_id, hardness="hard", params={"source": source})


def test_prompt_reports_a_file_bound_with_two_dialects_and_exits_1(tmp_path: Path) -> None:
    # An ASCII file decodes under both encodings, so the scenario itself loads.
    truth = minimal_dm()
    truth["constraints"] += [absences_rule("away"), absences_rule("again", encoding="iso-8859-7")]
    metadata = scenario_metadata()
    metadata["csv_files"] = ["resources.csv", "tasks.csv", "absences.csv"]
    directory = write_scenario_dir(tmp_path, metadata=metadata, truth=truth)
    (directory / "absences.csv").write_text("person,from,to\n", encoding="utf-8")

    result = runner.invoke(app, ["eval", "prompt", str(directory)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "absences.csv is bound with two different CSV dialects\n"


def test_prompt_reports_a_missing_instruction_file_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Outside a source checkout the prompt file is absent: an error message, not a traceback.
    missing = tmp_path / "test-a.md"
    monkeypatch.setattr(test_a_module, "INSTRUCTION_PATH", missing)

    result = runner.invoke(app, ["eval", "prompt", str(HOTEL)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.startswith(f"cannot read the instruction file {missing}: ")


def test_prompt_help_states_the_exit_codes() -> None:
    text = help_text("eval", "prompt")

    assert "Exit code 0: payload printed; 1: invalid scenario; 2: usage error" in text
    assert "no credentials" in text


# --- eval compare ----------------------------------------------------------------------------


def write_candidate(tmp_path: Path, text: str) -> Path:
    target = tmp_path / "candidate.txt"
    target.write_bytes(text.encode("utf-8"))
    return target


def test_compare_prints_the_comparison_of_a_valid_candidate(tmp_path: Path) -> None:
    candidate = write_candidate(tmp_path, (PHARMACY / "truth.json").read_text(encoding="utf-8"))

    result = runner.invoke(app, ["eval", "compare", str(PHARMACY), str(candidate)])

    assert result.exit_code == 0
    assert result.stdout_bytes == (OPUS_RUNS / "01-pharmacy" / "comparison.json").read_bytes()


def test_compare_prints_the_comparison_of_an_invalid_candidate_and_exits_0(tmp_path: Path) -> None:
    candidate = write_candidate(tmp_path, '{"source": "truth.json"}')

    result = runner.invoke(app, ["eval", "compare", str(PHARMACY), str(candidate)])

    assert result.exit_code == 0
    document = json.loads(result.stdout)
    assert (document["valid"], document["recall"]) == (False, 0.0)


def test_compare_reports_an_unreadable_candidate_and_exits_1(tmp_path: Path) -> None:
    not_utf8 = tmp_path / "latin.txt"
    not_utf8.write_bytes(b"{\xcc}")  # a windows-1253 capital mu: not UTF-8

    missing = runner.invoke(app, ["eval", "compare", str(PHARMACY), str(tmp_path / "nope.txt")])
    broken = runner.invoke(app, ["eval", "compare", str(PHARMACY), str(not_utf8)])

    assert (missing.exit_code, broken.exit_code) == (1, 1)
    assert (missing.stdout, broken.stdout) == ("", "")
    assert missing.stderr.startswith(f"{tmp_path / 'nope.txt'}: cannot read")
    assert broken.stderr == f"{not_utf8}: not valid UTF-8\n"


def test_compare_reports_an_invalid_scenario_and_exits_1(tmp_path: Path) -> None:
    candidate = write_candidate(tmp_path, "{}")

    result = runner.invoke(app, ["eval", "compare", str(tmp_path / "nope"), str(candidate)])

    assert result.exit_code == 1
    assert result.stderr == "nope#/: not a scenario directory\n"


def test_compare_help_states_the_exit_codes() -> None:
    text = help_text("eval", "compare")

    assert "Exit code 0: comparison printed, for valid and invalid candidates alike" in text
    assert "1: invalid scenario or unreadable candidate file; 2: usage error" in text


# --- eval summary ----------------------------------------------------------------------------


def test_summary_prints_the_text_form() -> None:
    result = runner.invoke(app, ["eval", "summary", str(OPUS_RUNS), "--corpus", str(CORPUS)])

    assert result.exit_code == 0
    assert result.stdout_bytes == (GOLDEN / "test-a-summary.txt").read_bytes()


def test_summary_json_and_the_default_corpus() -> None:
    result = runner.invoke(app, ["eval", "summary", str(OPUS_RUNS), "--json"])

    assert result.exit_code == 0
    assert result.stdout_bytes == (GOLDEN / "test-a-summary.json").read_bytes()


@pytest.mark.parametrize("which", ["runs", "corpus"])
def test_summary_reports_a_missing_directory_and_exits_1(tmp_path: Path, which: str) -> None:
    missing = tmp_path / "nope"
    runs, corpus = (missing, CORPUS) if which == "runs" else (OPUS_RUNS, missing)

    result = runner.invoke(app, ["eval", "summary", str(runs), "--corpus", str(corpus)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"{missing} is not a directory\n"


def test_summary_help_states_the_exit_codes() -> None:
    text = help_text("eval", "summary")

    assert "Exit code 0: summary printed; 1: missing directory; 2: usage error" in text


def test_summary_help_names_the_default_corpus_without_a_machine_path() -> None:
    text = help_text("eval", "summary")

    assert "default: the repository's docs/validation/scenarios" in text
    assert str(CORPUS_PATH) not in text
    assert "RUNS_DIR" in text


def test_eval_group_says_it_makes_no_model_calls() -> None:
    assert "no model calls" in help_text("eval")
