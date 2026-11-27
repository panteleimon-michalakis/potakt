"""The offline ``potakt eval`` commands (TASK-005c1 D5): prompt, compare and summary."""

import json
import shutil
from pathlib import Path

import pytest
from evaluation_helpers import scenario_metadata, write_scenario_dir
from helpers import constraint, minimal_dm
from hypothesis import example, given, settings
from hypothesis import strategies as st
from runner_helpers import (
    ALL_SCENARIOS,
    anthropic_message,
    truth_text,
    write_draws_fixture,
    write_repairs_fixture,
)
from typer.testing import CliRunner, Result

import potakt.evaluation.test_a as test_a_module
from potakt.cli import app
from potakt.evaluation import CORPUS_PATH, load_scenario, write_answer
from potakt.evaluation.draws import draws_report, scenario_draws
from potakt.evaluation.repairs_report import repairs_report

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
    assert payload["generationConfig"] == {"maxOutputTokens": 64000}  # TASK-005c3g G8


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gemini-2.5-flash"])
def test_prompt_refuses_an_unregistered_model_as_a_usage_error(model: str) -> None:
    result = runner.invoke(app, ["eval", "prompt", str(HOTEL), "--model", model])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == (
        f"no pre-registered Test A settings for {model!r}; "
        "use claude-haiku-5-5, gemini-3.8-flash or claude-opus-5-5\n"
    )


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
    assert "also an unregistered model id" in text
    assert "claude-haiku-5-5, gemini-3.8-flash, claude-opus-5-5" in text
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


# --- eval rescore (TASK-005c2 E5) ------------------------------------------------------------


def runs_copy(tmp_path: Path, model: str = "claude-opus-5-5") -> Path:
    """A writable copy of the fixture runs: rescoring never writes into the fixtures."""
    target = tmp_path / model
    shutil.copytree(OPUS_RUNS, target)
    return target


def rescore(runs: Path, *extra: str) -> Result:
    return runner.invoke(app, ["eval", "rescore", str(runs), "--corpus", str(CORPUS), *extra])


def test_rescore_output_after_one_deleted_file_matches_the_golden(tmp_path: Path) -> None:
    runs = runs_copy(tmp_path)
    (runs / "03-call-centre" / "comparison.json").unlink()

    result = rescore(runs)

    assert result.exit_code == 0, result.stderr
    assert result.stderr == ""
    assert result.stdout_bytes == (GOLDEN / "eval-rescore.txt").read_bytes()
    restored = (runs / "03-call-centre" / "comparison.json").read_bytes()
    assert restored == (OPUS_RUNS / "03-call-centre" / "comparison.json").read_bytes()


def test_rescore_of_unchanged_runs_and_a_run_without_a_response(tmp_path: Path) -> None:
    runs = runs_copy(tmp_path)
    (runs / "02-hotel-reception").mkdir()  # a run that never got an answer

    result = rescore(runs)

    assert result.exit_code == 0, result.stderr
    assert result.stdout == (
        "01-pharmacy: unchanged\n"
        "02-hotel-reception: skipped (no response.json)\n"
        "03-call-centre: unchanged\n"
        "04-restaurant: unchanged\n"
    )


def test_rescore_reports_both_derived_files_when_both_changed(tmp_path: Path) -> None:
    runs = runs_copy(tmp_path)
    (runs / "04-restaurant" / "candidate.txt").write_bytes(b"{}")
    (runs / "04-restaurant" / "comparison.json").unlink()

    result = rescore(runs)

    assert result.exit_code == 0
    assert "04-restaurant: updated (candidate.txt, comparison.json)\n" in result.stdout


def test_rescore_of_the_current_directory_uses_its_resolved_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Review finding: "." gave the model "" and was refused as unregistered.
    monkeypatch.chdir(runs_copy(tmp_path))

    result = runner.invoke(app, ["eval", "rescore", ".", "--corpus", str(CORPUS)])

    assert result.exit_code == 0, result.stderr
    assert result.stdout.count(": unchanged\n") == 3


def test_rescore_uses_the_default_corpus(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "rescore", str(runs_copy(tmp_path))])

    assert result.exit_code == 0
    assert result.stdout.count(": unchanged\n") == 3


def test_rescore_prints_a_scenario_outside_the_corpus_continues_and_exits_1(
    tmp_path: Path,
) -> None:
    runs = runs_copy(tmp_path)
    shutil.copytree(runs / "01-pharmacy", runs / "00-unknown")

    result = rescore(runs)

    assert result.exit_code == 1
    assert result.stdout == (
        "00-unknown: not a scenario of the corpus\n"
        "01-pharmacy: unchanged\n"
        "03-call-centre: unchanged\n"
        "04-restaurant: unchanged\n"
    )
    assert result.stderr == ""


@pytest.mark.parametrize("which", ["runs", "corpus"])
def test_rescore_reports_a_missing_directory_and_exits_1(tmp_path: Path, which: str) -> None:
    missing = tmp_path / "nope"
    runs = runs_copy(tmp_path)
    runs, corpus = (missing, CORPUS) if which == "runs" else (runs, missing)

    result = runner.invoke(app, ["eval", "rescore", str(runs), "--corpus", str(corpus)])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"{missing} is not a directory\n"


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "runs"])
def test_rescore_refuses_a_directory_not_named_after_a_registered_model(
    tmp_path: Path, model: str
) -> None:
    runs = runs_copy(tmp_path, model)
    before = {path: path.read_bytes() for path in runs.rglob("*") if path.is_file()}

    result = rescore(runs)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == (
        f"no pre-registered Test A settings for {model!r}; "
        "use claude-haiku-5-5, gemini-3.8-flash or claude-opus-5-5\n"
    )
    assert {path: path.read_bytes() for path in runs.rglob("*") if path.is_file()} == before


def test_rescore_help_states_the_lines_and_exit_codes() -> None:
    text = help_text("eval", "rescore")

    assert "from response.json alone" in text
    assert "no request is sent" in text
    assert "unchanged, updated (<files>) or skipped (no response.json)" in text
    assert "Exit code 0: all directories processed" in text
    assert (
        "1: missing runs or corpus directory, a runs directory not named after a registered "
        "model, or a scenario not in the corpus (its line is printed and processing continues)"
    ) in text
    assert "RUNS_DIR" in text


# --- eval repairs (TASK-005e5 decision 7, ADR-0008 D7) -----------------------------------------

HAIKU = "claude-haiku-5-5"


def repairs(runs: Path, *extra: str, corpus: Path = CORPUS) -> Result:
    args = ["eval", "repairs", "--model", HAIKU, "--runs", str(runs), "--corpus", str(corpus)]
    return runner.invoke(app, [*args, *extra])


def test_the_repairs_report_matches_the_goldens_through_the_api(tmp_path: Path) -> None:
    write_repairs_fixture(tmp_path)

    report = repairs_report(HAIKU, tmp_path, CORPUS)

    assert report.to_text() == (GOLDEN / "test-a-repairs.txt").read_text(encoding="utf-8")
    assert report.to_json() == (GOLDEN / "test-a-repairs.json").read_text(encoding="utf-8")


def test_the_repairs_command_prints_the_goldens(tmp_path: Path) -> None:
    write_repairs_fixture(tmp_path)

    text = repairs(tmp_path)
    as_json = repairs(tmp_path, "--json")

    assert (text.exit_code, as_json.exit_code) == (0, 0), text.stderr + as_json.stderr
    assert (text.stderr, as_json.stderr) == ("", "")
    assert text.stdout_bytes == (GOLDEN / "test-a-repairs.txt").read_bytes()
    assert as_json.stdout_bytes == (GOLDEN / "test-a-repairs.json").read_bytes()


def test_the_golden_numbers_are_the_fixtures_counts(tmp_path: Path) -> None:
    # The truth counts of the corpus: 01 5, 02 8, 03 8, 04 12, 05 8, 06 11, 07 5, 08 6, 09 9,
    # 10 7 (79). Valid first: 01, 02, 10 (20 matched). Valid after one repair: 04, 05 (+20).
    write_repairs_fixture(tmp_path)

    document = json.loads(repairs_report(HAIKU, tmp_path, CORPUS).to_json())

    assert document["recall"] == {"matched": 5 + 8 + 7, "truth": 79, "recall": 20 / 79}
    assert document["recall_after_repair"] == {"matched": 40, "truth": 79, "recall": 40 / 79}
    counts = (document["recorded"], document["without_answer"], document["not_sent"])
    assert counts == (3, 1, 2)
    assert document["missing"] == ["09-care-home"]


def test_only_a_repair_that_is_valid_adds_its_matched_count(tmp_path: Path) -> None:
    # A crafted record: 03's repair is invalid but lists matched pairs; they never count.
    write_repairs_fixture(tmp_path)
    crafted = tmp_path / "repair" / HAIKU / "03-call-centre" / "comparison.json"
    valid = json.loads((tmp_path / HAIKU / "01-pharmacy" / "comparison.json").read_bytes())
    document = json.loads(crafted.read_bytes())
    document["matched"] = valid["matched"]
    document["unmatched_truth"] = document["unmatched_truth"][len(valid["matched"]) :]
    crafted.write_text(json.dumps(document), encoding="utf-8")

    report = repairs_report(HAIKU, tmp_path, CORPUS)

    assert json.loads(report.to_json())["recall_after_repair"]["matched"] == 40


def test_a_first_run_without_an_invalid_candidate_has_none(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    for name in ("01-pharmacy", "02-hotel-reception"):
        shutil.copytree(CORPUS / name, corpus / name)
        message = anthropic_message(HAIKU, truth_text(name))
        write_answer(tmp_path / "runs" / HAIKU / name, load_scenario(corpus / name), HAIKU, message)

    result = repairs(tmp_path / "runs", corpus=corpus)

    assert result.exit_code == 0, result.stderr
    assert result.stdout == (
        "Repairs: claude-haiku-5-5 (ADR-0008 D7, not the gate)\n"
        "First run: 2 scored, 2 valid, 0 invalid\n"
        "Repair run: 0 recorded, 0 without an answer, 0 not sent, 0 missing\n"
        "Complete: yes\n"
        "Recall: 1.000 (13/13)\n"
        "Recall after one repair (not the gate): 1.000 (13/13); 0 of 0 invalid candidates valid "
        "after one repair\n"
        "Per invalid candidate:\n"
        "  none\n"
    )
    document = json.loads(repairs(tmp_path / "runs", "--json", corpus=corpus).stdout)
    assert (document["candidates"], document["complete"]) == ([], True)


def test_a_repair_run_with_every_candidate_recorded_is_complete(tmp_path: Path) -> None:
    write_repairs_fixture(tmp_path)
    name = "09-care-home"
    write_answer(
        tmp_path / "repair" / HAIKU / name,
        load_scenario(CORPUS / name),
        HAIKU,
        anthropic_message(HAIKU, truth_text(name)),
    )

    lines = repairs(tmp_path).stdout.splitlines()

    assert "Complete: yes" in lines
    assert "Repair run: 4 recorded, 1 without an answer, 2 not sent, 0 missing" in lines
    assert "  09-care-home: first invalid (JSON or fence), repair valid, matched 9 of 9" in lines


def test_repairs_of_an_incomplete_first_run_exit_1(tmp_path: Path) -> None:
    write_repairs_fixture(tmp_path)
    (tmp_path / HAIKU / "02-hotel-reception" / "comparison.json").unlink()
    shutil.rmtree(tmp_path / HAIKU / "10-municipal-nursery")

    result = repairs(tmp_path)

    assert result.exit_code == 1
    assert result.stdout == ""
    # As check_order lists them: the summary's missing (in corpus order), then the not scored.
    assert result.stderr == (
        "claude-haiku-5-5 has no complete run (missing: 02-hotel-reception, 10-municipal-nursery)\n"
    )
    shutil.rmtree(tmp_path / HAIKU)
    nothing = repairs(tmp_path)
    assert nothing.exit_code == 1
    assert nothing.stderr.startswith("claude-haiku-5-5 has no complete run (missing: 01-pharmacy,")


@pytest.mark.parametrize("which", ["runs", "corpus"])
def test_repairs_report_a_missing_directory_and_exit_1(tmp_path: Path, which: str) -> None:
    missing = tmp_path / "nope"
    runs, corpus = (missing, CORPUS) if which == "runs" else (tmp_path, missing)

    result = repairs(runs, corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == f"{missing} is not a directory\n"


def test_repairs_of_an_unregistered_model_are_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "repairs", "--model", "claude-sonnet-5-5"])

    assert result.exit_code == 2
    assert "claude-sonnet-5-5" in result.stderr


# --- eval draws (TASK-005e5 decision 9, ADR-0008 D8) ------------------------------------------


def draws(runs: Path, *extra: str, corpus: Path = CORPUS) -> Result:
    args = ["eval", "draws", "--runs", str(runs), "--corpus", str(corpus)]
    return runner.invoke(app, [*args, *extra])


def test_the_draws_report_matches_the_goldens_through_the_api(tmp_path: Path) -> None:
    write_draws_fixture(tmp_path)

    report = draws_report(tmp_path, CORPUS)

    assert report.to_text() == (GOLDEN / "test-a-draws.txt").read_text(encoding="utf-8")
    assert report.to_json() == (GOLDEN / "test-a-draws.json").read_text(encoding="utf-8")


def test_the_draws_command_prints_the_goldens(tmp_path: Path) -> None:
    write_draws_fixture(tmp_path)

    text, as_json = draws(tmp_path), draws(tmp_path, "--json")

    assert (text.exit_code, as_json.exit_code) == (0, 0), text.stderr + as_json.stderr
    assert (text.stderr, as_json.stderr) == ("", "")
    assert text.stdout_bytes == (GOLDEN / "test-a-draws.txt").read_bytes()
    assert as_json.stdout_bytes == (GOLDEN / "test-a-draws.json").read_bytes()


def test_the_draws_golden_numbers_are_the_fixtures_counts(tmp_path: Path) -> None:
    # 79 truth constraints; 03 matches 5 of 8 in draws 1 and 2 and is unanswered in draw 3; 02
    # (8) is invalid in draw 2. Draw 1: 79 - 3 = 76; draw 2: 76 - 8 = 68; draw 3: 79 - 8 = 71.
    write_draws_fixture(tmp_path)

    document = json.loads(draws_report(tmp_path, CORPUS).to_json())

    assert [(d["matched"], d["truth"]) for d in document["draws"]] == [(76, 79), (68, 79), (71, 79)]
    by_name = {row["scenario"]: row for row in document["scenarios"]}
    assert by_name["02-hotel-reception"]["recalls"] == [1.0, 0.0, 1.0]
    assert by_name["03-call-centre"]["recalls"] == [5 / 8, 5 / 8, 0.0]
    assert by_name["03-call-centre"]["mean"] == (5 / 8 + 5 / 8 + 0) / 3


recall_counts = st.integers(min_value=1, max_value=60).flatmap(
    lambda truth: st.tuples(st.integers(min_value=0, max_value=truth), st.just(truth))
)


@settings(deadline=None, max_examples=300)
@given(st.lists(recall_counts, min_size=3, max_size=3))
@example([(1, 10), (1, 10), (1, 10)])  # 0.1 three times: a float sum would exceed the maximum
@example([(0, 1), (1, 2), (3, 3)])
def test_the_mean_of_a_scenario_is_the_mean_of_its_three_recalls(
    counts: list[tuple[int, int]],
) -> None:
    recalls = [matched / truth for matched, truth in counts]

    row = scenario_draws("x", counts)

    assert row.recalls == tuple(recalls)
    assert row.min is not None and row.mean is not None and row.max is not None
    assert row.min <= row.mean <= row.max
    assert (row.min, row.max) == (min(recalls), max(recalls))
    assert row.mean == pytest.approx(sum(recalls) / 3, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("counts", [[(0, 0), (1, 1), (1, 2)], []])
def test_a_draw_without_truth_constraints_has_no_mean(counts: list[tuple[int, int]]) -> None:
    row = scenario_draws("x", counts)

    assert (row.mean, row.min, row.max) == (None, None, None)
    if counts:
        assert row.recalls == (None, 1.0, 0.5)
        assert row.to_text() == "x: n/a, 1.000, 0.500; mean n/a, min n/a, max n/a"


def test_the_mean_is_not_the_pooled_recall() -> None:
    row = scenario_draws("x", [(0, 1), (1, 2), (3, 3)])

    assert row.mean == pytest.approx(0.5)  # (0 + 0.5 + 1) / 3, while 4 of 6 would be 0.667


def test_a_draw_without_its_directory_is_not_complete(tmp_path: Path) -> None:
    write_draws_fixture(tmp_path)
    shutil.rmtree(tmp_path / "repeat-3")

    result = draws(tmp_path)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == f"draw 3 is not complete (missing: {', '.join(ALL_SCENARIOS)})\n"


def test_the_first_incomplete_draw_is_named(tmp_path: Path) -> None:
    write_draws_fixture(tmp_path)
    draw_2 = tmp_path / "repeat-2" / "claude-opus-5-5"
    (draw_2 / "05-supermarket-checkouts" / "comparison.json").unlink()
    shutil.rmtree(tmp_path / "repeat-3")

    result = draws(tmp_path)

    assert result.exit_code == 1
    assert result.stderr == "draw 2 is not complete (missing: 05-supermarket-checkouts)\n"


@pytest.mark.parametrize("which", ["runs", "corpus"])
def test_draws_report_a_missing_directory_and_exit_1(tmp_path: Path, which: str) -> None:
    missing = tmp_path / "nope"
    runs, corpus = (missing, CORPUS) if which == "runs" else (tmp_path, missing)

    result = draws(runs, corpus=corpus)

    assert result.exit_code == 1
    assert result.stderr == f"{missing} is not a directory\n"


def test_draws_help_states_the_exit_codes() -> None:
    text = help_text("eval", "draws")

    assert "never the gate" in text
    assert (
        "Exit code 0: report printed; 1: missing runs or corpus directory, or a draw not "
        "complete; 2: usage error"
    ) in text


def test_repairs_help_states_the_exit_codes() -> None:
    text = help_text("eval", "repairs")

    assert "never the gate" in text
    assert (
        "Exit code 0: report printed; 1: missing runs or corpus directory, or the model's first "
        "run not complete; 2: usage error (also an unregistered model id)"
    ) in text
